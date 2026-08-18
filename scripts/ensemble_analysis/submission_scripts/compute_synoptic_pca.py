"""Compute the synoptic-PCA diagnostic for one mode's ensemble.

Decomposes the ensemble's *free-end* state into principal components on a
synoptic-scale precursor field (z500 by default; jointly if several
variables are configured), then, for each leading EOF, picks the ensemble
members whose PC score sits closest to a set of fixed percentiles
(5/25/50/75/95).  Those percentile members are the diagnostic's headline
samples: their full trajectories on the precursor field show how distinct
synoptic precursors evolve, and a downstream *impact* diagnostic shows how
that precursor diversity manifests in the event itself --

    * scalar impact (e.g. a heatwave): per-member bbox-mean of an impact
      variable (t2m, land-masked) over lead time.
    * track impact (a tropical cyclone): each member's storm track, read
      from the cached TempestExtremes parquet.

There is intentionally **no clustering** here: over the short windows this
pipeline targets, the free-end state is consistently a single cluster, so
the continuous PCA / percentile-member view is the useful one.

Free-end frame:
    end   -> frame 0   (the precursor)
    start -> last frame (the outcome)
The ``both`` mode is rejected at the dispatcher level.

The PCA does **not** subtract a climatology: PCA's own per-feature
centering removes any constant per-pixel offset (which is exactly what a
per-pixel climatology is), so subtracting it first would be invisible to
the PC scores and EOF patterns.

Saved fields are **raw** (no anomaly subtraction): the aggregator forms
``ens_mean - ERA5`` (bias) and ``member - ens_mean`` (diversity) at plot
time.  HEALPix pixels are equal-area, so no ``cos(lat)`` weighting is
needed in the PCA covariance.

Outputs (in --output-dir):
    synoptic_pca_<mode>.npz   per-mode PCA result + percentile-member fields

Usage:
    python compute_synoptic_pca.py \\
        --ensemble-zarr      /path/to/ensemble_f0_n1000.zarr \\
        --era5-zarr          /path/to/era5_reference.zarr \\
        --output-dir         /path/to/diagnostics/synoptic_pca \\
        --mode               end \\
        --case-name          "Hurricane Sandy" \\
        --variable z500 --variable msl \\
        --lon-min -100 --lon-max -30 --lat-min 25 --lat-max 75 \\
        --impact-kind track --tc-parquet /path/tc_tracks_end.parquet \\
        --tc-tracker tempest --tc-domain -85 -60 25 45 --landfall-frame 11
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_ANALYSIS_DIR = SCRIPT_DIR.parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(ENSEMBLE_ANALYSIS_DIR))

from _dispatch_lib import lead_hours  # noqa: E402
from tc_track_targets import free_end_target_arrays  # noqa: E402
from tc_tracks_io import load_tracks_parquet, restrict_to_domain  # noqa: E402

from custom_utils.diagnostics import (  # noqa: E402
    domain_hpx_indices,
    hpx_land_mask_from_era5,
    hpx_pixel_centers,
)
from custom_utils.ensemble_loading import variable_index  # noqa: E402
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402

# EOF-percentile-member figure: per-EOF PC percentiles at which to pick the
# closest ensemble member, and the default cap on how many leading EOFs to
# bake into the npz.  PC4+ usually carries little physically distinct signal.
PERCENTILE_VALUES: Tuple[int, ...] = (5, 25, 50, 75, 95)
EOF_PERCENTILE_MAX_ROWS: int = 3

# Domain-sensitivity ladder: linear-size scale factors applied to the impact
# bbox about its centre.  The aggregator regresses the free-end impact scalar
# over each on the PCs -- a controlled de-attenuation test (centre fixed, area
# varying) showing whether a bigger averaging box sharpens the precursor ->
# impact R^2 by cutting spatial-sampling noise, without tuning the domain.
IMPACT_LADDER_SCALES: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.3)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _open_ensemble(path: str) -> zarr.Group:
    root = zarr.open_group(path, mode="r")
    if root["data"].ndim != 5 or "hpx" not in root.array_keys():
        raise ValueError(
            f"{path}: expected HEALPix ensemble layout (ndim=5 with 'hpx' "
            f"coord); got ndim={root['data'].ndim}, "
            f"arrays={sorted(root.array_keys())}"
        )
    return root


def _open_era5(path: str) -> zarr.Group:
    root = zarr.open_group(path, mode="r")
    if root["data"].ndim != 5:
        raise ValueError(
            f"{path}: expected ERA5 layout (ndim=5), got ndim={root['data'].ndim}"
        )
    return root


def _hpx_level_from_root(root: zarr.Group) -> int:
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    n_pix_arr = int(root["hpx"].shape[0])
    return int(np.log2(n_pix_arr // 12) / 2)


# ---------------------------------------------------------------------------
# Domain mask on HEALPix
# ---------------------------------------------------------------------------


# These two were the original implementation; they now live in
# ``custom_utils.diagnostics`` so the spread/RMSE/CRPS battery masks exactly
# the pixels this pipeline masks.  Kept as thin aliases rather than inlined
# at the call sites to leave the surrounding code untouched.
_hpx_pixel_centers = hpx_pixel_centers
_domain_hpx_indices = domain_hpx_indices


# ---------------------------------------------------------------------------
# Free-end frame selection
# ---------------------------------------------------------------------------


def _pick_free_end_frame(mode: str) -> int:
    """Frame index of the unconditioned end for one mode.

    cBottle always generates 12-frame windows in this pipeline, so the
    free end of a start-conditioned run is fixed at frame 11.  If the
    window length ever changes, update this function rather than
    silently inferring from ``n_leads``.
    """
    if mode == "end":
        return 0
    if mode == "start":
        return 11
    raise ValueError(
        f"synoptic_pca does not support mode '{mode}'. "
        f"Eligible modes: 'start', 'end'."
    )


# ---------------------------------------------------------------------------
# Feature build: free-end frame -> domain mask -> z-score -> stack
# ---------------------------------------------------------------------------


def _load_ensemble_frame(
    ens_root: zarr.Group,
    frame: int,
    var_indices: Sequence[int],
) -> torch.Tensor:
    """Load all members at one frame for the given variables.

    Returns ``[N_members, n_vars, n_pix_hpx]``.
    """
    data = ens_root["data"]  # [N, 1, n_leads, n_vars, n_pix]
    n_members = int(data.shape[0])
    n_pix = int(data.shape[-1])
    out = np.empty((n_members, len(var_indices), n_pix), dtype=np.float32)
    # Variables are NOT contiguous on disk along the var axis in zarr's
    # natural chunking, so iterate per-variable to avoid pulling extra
    # variables into the read.
    for j, vi in enumerate(var_indices):
        out[:, j, :] = np.array(data[:, 0, frame, vi, :], dtype=np.float32)
    return torch.from_numpy(out)


def _load_member_trajectories_hpx(
    ens_root: zarr.Group,
    member_indices: Sequence[int],
    var_indices: Sequence[int],
) -> torch.Tensor:
    """Load the full trajectory of selected ensemble members.

    Returns ``[n_selected, n_leads, n_vars, n_pix_hpx]``.  Duplicate
    indices produce duplicate rows.  Reads member-major and per-variable
    to mirror ``_load_ensemble_frame``'s avoidance of extra-variable I/O.
    """
    data = ens_root["data"]  # [N, 1, n_leads, n_vars, n_pix]
    n_leads = int(data.shape[2])
    n_pix = int(data.shape[-1])
    member_indices_arr = np.asarray(member_indices, dtype=np.int64)
    n_selected = int(member_indices_arr.size)
    out = np.empty((n_selected, n_leads, len(var_indices), n_pix), dtype=np.float32)
    for s, m in enumerate(member_indices_arr):
        m_int = int(m)
        for j, vi in enumerate(var_indices):
            out[s, :, j, :] = np.array(data[m_int, 0, :, vi, :], dtype=np.float32)
    return torch.from_numpy(out)


def _ensemble_mean_hpx(
    ens_root: zarr.Group,
    var_indices: Sequence[int],
) -> torch.Tensor:
    """Per-frame ensemble mean field over all members, on full HPX.

    Returns ``[n_leads, n_vars, n_pix_hpx]`` float32, computed frame-by-
    frame so peak memory stays at one frame's worth of full-N data.  The
    aggregator uses this as the anomaly reference: ``member - ens_mean``
    (diversity) and ``ens_mean - ERA5`` (bias).
    """
    data = ens_root["data"]  # [N, 1, n_leads, n_vars, n_pix]
    _, _, n_leads, _, n_pix = data.shape
    n_vars = len(var_indices)
    out = torch.zeros(n_leads, n_vars, n_pix, dtype=torch.float32)
    for f in range(n_leads):
        for j, vi in enumerate(var_indices):
            frame = np.array(data[:, 0, f, vi, :], dtype=np.float32)  # [N, n_pix]
            out[f, j] = torch.from_numpy(frame.mean(axis=0))
    return out


# ---------------------------------------------------------------------------
# Scalar-bbox helpers for the impact time-series.  Duplicated from
# ``compute_free_end_states.py``: the two diagnostics consume the same bbox
# and per-variable mask config (case YAML regions.impact +
# diagnostics.free_end_states.mask, fed identically by both dispatchers) so
# the impact scalar matches the free_end_states time-series scalar
# pixel-for-pixel.  Keep in sync.
# ---------------------------------------------------------------------------


def _hpx_bbox_mask(
    pix_lat: np.ndarray,
    pix_lon: np.ndarray,
    bbox: Sequence[float],
) -> np.ndarray:
    """Boolean mask over HPX pixels whose centers lie inside ``bbox``.

    ``bbox`` is ``[lon_min, lon_max, lat_min, lat_max]``.  Longitudes are
    normalized to [-180, 180] so both store conventions ([-180,180] or
    [0,360]) work; dateline-crossing boxes are rejected.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    if lon_min >= lon_max:
        raise SystemExit(
            f"ERROR: impact bbox lon_min ({lon_min}) >= lon_max ({lon_max}); "
            f"dateline-crossing boxes are not supported."
        )
    if lat_min >= lat_max:
        raise SystemExit(
            f"ERROR: impact bbox lat_min ({lat_min}) >= lat_max ({lat_max})."
        )
    pix_lon_norm = ((np.asarray(pix_lon) + 180.0) % 360.0) - 180.0
    lon_min_n = ((lon_min + 180.0) % 360.0) - 180.0
    lon_max_n = ((lon_max + 180.0) % 360.0) - 180.0
    mask = (
        (pix_lat >= lat_min)
        & (pix_lat <= lat_max)
        & (pix_lon_norm >= lon_min_n)
        & (pix_lon_norm <= lon_max_n)
    )
    if not mask.any():
        raise SystemExit(
            f"ERROR: impact bbox {list(bbox)} matched zero HPX pixel centers; "
            f"box may be smaller than the pixel size."
        )
    return mask


def _scaled_bbox(bbox: Sequence[float], factor: float) -> List[float]:
    """Scale an impact bbox about its centre by ``factor`` (lon/lat extents).

    ``bbox`` is ``[lon_min, lon_max, lat_min, lat_max]``.  Builds the nested
    domain-sensitivity ladder: ``factor < 1`` shrinks toward the centre, ``> 1``
    grows it, holding the centre fixed.  Latitude is clipped to just inside the
    poles; longitude is left unwrapped (the ladder targets small regional boxes).
    """
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in bbox)
    lon_c = 0.5 * (lon_min + lon_max)
    lat_c = 0.5 * (lat_min + lat_max)
    lon_half = 0.5 * (lon_max - lon_min) * factor
    lat_half = 0.5 * (lat_max - lat_min) * factor
    return [
        lon_c - lon_half,
        lon_c + lon_half,
        max(-89.5, lat_c - lat_half),
        min(89.5, lat_c + lat_half),
    ]


# This was the original implementation; it now lives in
# ``custom_utils.diagnostics`` so this pipeline, free_end_states, and the
# spread/RMSE/CRPS battery cannot disagree about which pixels are land --
# `pnw_heatwave.yaml` requires this diagnostic's t2m impact average to match
# the free_end_states scalar pixel-for-pixel.  Kept as a thin alias rather
# than inlined at the call site to leave the surrounding code untouched.
_hpx_land_mask_from_sst = hpx_land_mask_from_era5


def _resolve_impact_mask(
    bbox_mask: np.ndarray,
    mask_kind: str,
    era_root: zarr.Group,
    hpx_level: int,
) -> np.ndarray:
    """Bbox mask, optionally intersected with land/sea, for the impact var."""
    if mask_kind == "none":
        m = bbox_mask.copy()
    elif mask_kind in ("land", "sea"):
        land_mask = _hpx_land_mask_from_sst(era_root, hpx_level)
        m = bbox_mask & land_mask if mask_kind == "land" else bbox_mask & (~land_mask)
    else:
        raise SystemExit(
            f"ERROR: unsupported impact mask kind {mask_kind!r}; "
            f"expected one of: none, land, sea."
        )
    if not m.any():
        raise SystemExit(
            f"ERROR: impact bbox ∩ {mask_kind}-mask matched zero HPX pixels."
        )
    return m


def _compute_member_scalar_hpx(
    ens_root: zarr.Group,
    var_index: int,
    var_mask: np.ndarray,
) -> np.ndarray:
    """Per-member, per-lead bbox domain-mean scalar for one variable.

    Returns ``[N, n_leads]`` float32.  Reads the zarr frame-by-frame to
    keep peak memory at one frame's worth of full-N data.
    """
    data = ens_root["data"]  # [N, 1, n_leads, n_vars, n_pix]
    N = int(data.shape[0])
    n_leads = int(data.shape[2])
    pix_idx = np.where(var_mask)[0]
    out = np.zeros((N, n_leads), dtype=np.float32)
    for f in range(n_leads):
        frame = np.array(data[:, 0, f, var_index, :], dtype=np.float32)  # [N, n_pix]
        out[:, f] = frame[:, pix_idx].mean(axis=1)
    return out


def _build_feature_matrix(
    ens_freeend: torch.Tensor,
    domain_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask to domain, z-score per variable, flatten -- ready for PCA.

    Inputs
    ------
    ens_freeend : ``[N, n_vars, n_pix_hpx]``
    domain_idx : ``[n_pix_in_domain]``

    Returns
    -------
    X : ``[N, n_vars * n_pix_in_domain]`` z-scored features, ready for PCA.
    per_var_mean : ``[n_vars]`` per-variable mean (across members & domain
        pixels) of the raw field; used to undo the z-score on the EOF
        patterns at save time, and saved for inspection.
    per_var_std : ``[n_vars]`` per-variable std (same population) used to
        rescale columns so multi-variable PCA isn't unit-dominated (e.g.
        z500 in m vs MSL in Pa).
    """
    fields = ens_freeend[:, :, domain_idx]  # [N, n_vars, n_pix_in_domain]
    N, n_vars, n_dpix = fields.shape

    per_var_mean = np.empty(n_vars, dtype=np.float32)
    per_var_std = np.empty(n_vars, dtype=np.float32)
    for v in range(n_vars):
        slab = fields[:, v, :]
        per_var_mean[v] = float(slab.mean())
        std = float(slab.std())
        # Guard against a fully-zero variable (extremely unlikely in
        # practice, but explicit is better).
        per_var_std[v] = std if std > 0 else 1.0
        fields[:, v, :] = (slab - per_var_mean[v]) / per_var_std[v]

    X = fields.reshape(N, n_vars * n_dpix).cpu().numpy().astype(np.float32)
    return X, per_var_mean, per_var_std


# ---------------------------------------------------------------------------
# PCA via SVD on the centered feature matrix
# ---------------------------------------------------------------------------


def _pca(
    X: np.ndarray, max_d: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-``max_d`` PCs of ``X`` (already z-scored).

    Returns
    -------
    scores : ``[N, max_d]`` PC scores ( = U * S, sign-fixed for repeatability)
    components : ``[max_d, n_features]`` EOF patterns (rows of V^T)
    explained_variance_ratio : ``[n_components]`` EVR for the FULL spectrum.
    feature_mean : ``[n_features]`` the per-feature mean subtracted before SVD
    eigenvalues : ``[n_components]`` squared singular values (proportional
        to PCA eigenvalues).
    """
    feature_mean = X.mean(axis=0)
    Xc = X - feature_mean
    # full_matrices=False keeps shapes [N, k], [k], [k, n_features] with
    # k = min(N, n_features) -- cheap and what we want.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigenvalues = (S**2).astype(np.float64)  # full spectrum
    k = min(max_d, S.size)
    scores = U[:, :k] * S[:k]
    components = Vt[:k]
    total_var = float(eigenvalues.sum())
    if total_var > 0:
        evr_full = (eigenvalues / total_var).astype(np.float32)
    else:
        evr_full = np.zeros_like(eigenvalues, dtype=np.float32)
    # Sign-flip each PC so its loading max-absolute element is positive --
    # makes the diagnostic byte-identical across re-runs (SVD signs are
    # arbitrary up to a unitary rotation in the degenerate cases).
    for j in range(k):
        peak = components[j, np.argmax(np.abs(components[j]))]
        if peak < 0:
            components[j] *= -1
            scores[:, j] *= -1
    return (
        scores.astype(np.float32),
        components.astype(np.float32),
        evr_full,
        feature_mean.astype(np.float32),
        eigenvalues,
    )


# ---------------------------------------------------------------------------
# HPX -> lat/lon-domain regridding
# ---------------------------------------------------------------------------


def _hpx_to_latlon_domain(
    hpx_field: torch.Tensor,
    hpx_level: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid an HPX field ``[..., n_pix]`` to a lat/lon domain box.

    Returns ``(field_box, domain_lat, domain_lon)`` with ``field_box``
    shaped ``[..., n_lat_in_box, n_lon_in_box]`` (axis order preserved).
    """
    # CBottleUtils' regridder is hard-wired to 721x1440 output; reshape to
    # match its expected coord layout, regrid, then slice to the domain.
    original_shape = hpx_field.shape[:-1]
    flat = hpx_field.reshape(-1, hpx_field.shape[-1])  # [M, n_pix]
    # Wrap into [1, 1, M, n_pix] so the regridder's broadcast pattern works.
    wrapped = flat.unsqueeze(0).unsqueeze(0)
    dummy_coords = OrderedDict(
        {
            "time": np.array([0]),
            "lead_time": np.array([0]),
            "variable": np.array([f"v{i}" for i in range(flat.shape[0])]),
            "hpx": np.arange(hpx_field.shape[-1]),
        }
    )
    latlon, latlon_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        wrapped, dummy_coords, hpx_level=hpx_level
    )
    full_lat = np.asarray(latlon_coords["lat"])
    full_lon = np.asarray(latlon_coords["lon"])

    # Wrap lon to [-180, 180) so user-supplied bounds compare cleanly.
    lon_w = np.where(full_lon > 180, full_lon - 360, full_lon)
    if lon_min <= lon_max:
        lon_mask = (lon_w >= lon_min) & (lon_w <= lon_max)
    else:
        lon_mask = (lon_w >= lon_min) | (lon_w <= lon_max)
    lat_mask = (full_lat >= lat_min) & (full_lat <= lat_max)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    out = latlon[0, 0][:, lat_idx, :][:, :, lon_idx]  # [M, n_lat_box, n_lon_box]
    out = out.reshape(*original_shape, lat_idx.size, lon_idx.size)
    return (
        out.cpu().numpy().astype(np.float32),
        full_lat[lat_idx].astype(np.float32),
        lon_w[lon_idx].astype(np.float32),
    )


def _era5_vars_to_box(
    era_root: zarr.Group,
    era_lat: np.ndarray,
    era_lon: np.ndarray,
    var_names: Sequence[str],
    n_leads: int,
    hpx_level: int,
    box: Tuple[float, float, float, float],
) -> np.ndarray:
    """ERA5 per-frame trajectory for ``var_names``, on the domain box.

    Mirrors the precursor ERA5 round-trip (lat/lon -> HPX -> lat/lon box) so the
    returned fields land on the same ``(domain_lat, domain_lon)`` grid as every
    other saved field.  Returns ``[n_leads, n_vars, nlat, nlon]``.  Unlike the
    precursor path it keeps no HPX intermediate -- these are overlay-only fields
    (raw winds for the steering arrows), with no PC projection to feed.
    """
    idxs = [variable_index(era_root, v) for v in var_names]
    full = np.empty((n_leads, len(idxs), era_lat.size, era_lon.size), dtype=np.float32)
    for j, vi in enumerate(idxs):
        full[:, j] = np.array(era_root["data"][:, 0, vi, :, :], dtype=np.float32)
    coords = OrderedDict(
        {
            "time": np.arange(n_leads),
            "lead_time": np.array([0]),
            "variable": np.array(list(var_names)),
            "lat": era_lat,
            "lon": era_lon,
        }
    )
    hpx, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
        torch.from_numpy(full).unsqueeze(1), coords, hpx_level=hpx_level
    )
    box_field, _, _ = _hpx_to_latlon_domain(
        hpx[:, 0], hpx_level, box[0], box[1], box[2], box[3]
    )
    return box_field.astype(np.float32)


def _percentile_member_fields_box(
    ens_root: zarr.Group,
    member_idx_flat: np.ndarray,
    var_indices: Sequence[int],
    n_eof: int,
    n_pct: int,
    n_leads: int,
    hpx_level: int,
    box: Tuple[float, float, float, float],
) -> np.ndarray:
    """Percentile members' trajectories for ``var_indices``, on the domain box.

    Loads the flat percentile-member set, regrids to the box, and reshapes back
    to ``[n_eof, n_pct, n_leads, n_vars, nlat, nlon]``.  Shared by the raw-wind
    overlay and the video field, which both need the same percentile-member
    trajectories on the precursor grid -- only the variable set differs.
    """
    hpx = _load_member_trajectories_hpx(
        ens_root, member_indices=member_idx_flat, var_indices=var_indices
    )  # [n_eof * n_pct, n_leads, n_vars, n_pix]
    box_flat, _, _ = _hpx_to_latlon_domain(
        hpx, hpx_level, box[0], box[1], box[2], box[3]
    )
    return box_flat.reshape(
        n_eof,
        n_pct,
        n_leads,
        len(var_indices),
        box_flat.shape[-2],
        box_flat.shape[-1],
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Impact: TC track lookup from the cached TempestExtremes parquet
# ---------------------------------------------------------------------------


def _load_member_tracks_from_parquet(
    parquet_path: str,
    n_members: int,
    n_leads: int,
    tc_domain: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Primary-path (lat, lon) tracks for every member + the ERA5 track.

    Reads the per-mode tc_tracks parquet (written by the tc_tracks
    diagnostic), restricts to ``tc_domain`` (out-of-domain fixes -> NaN),
    and returns ``(member_track [N, n_leads, 2], era5_track [n_leads, 2],
    available)``.  When the parquet is missing -- the tc_tracks diagnostic
    has not run yet -- returns NaN-filled arrays with ``available=False``
    so the aggregator can render the precursor maps without the track
    panel rather than failing the whole run.
    """
    member_track = np.full((n_members, n_leads, 2), np.nan, dtype=np.float32)
    era5_track = np.full((n_leads, 2), np.nan, dtype=np.float32)
    if not Path(parquet_path).exists():
        print(
            f"[syn-pca] WARNING: tc_tracks parquet not found: {parquet_path}\n"
            f"          run the tc_tracks diagnostic first for the impact "
            f"track panel; precursor maps will still be produced."
        )
        return member_track, era5_track, False

    tracks = load_tracks_parquet(str(parquet_path))
    lon_min, lon_max, lat_min, lat_max = tc_domain
    ens = tracks.get("ensemble")
    if ens is not None:
        ens = restrict_to_domain(ens, lon_min, lon_max, lat_min, lat_max)
        T = min(int(ens.shape[2]), n_leads)
        M = min(int(ens.shape[0]), n_members)
        member_track[:M, :T, 0] = ens[:M, 0, :T, 0]  # tc_lat
        member_track[:M, :T, 1] = ens[:M, 0, :T, 1]  # tc_lon
    era = tracks.get("era5")
    if era is not None:
        era = restrict_to_domain(era, lon_min, lon_max, lat_min, lat_max)
        T = min(int(era.shape[2]), n_leads)
        era5_track[:T, 0] = era[0, 0, :T, 0]
        era5_track[:T, 1] = era[0, 0, :T, 1]
    return member_track, era5_track, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensemble-zarr", required=True)
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end"],
        help="Conditioning mode; encoded in the output filename. The 'both' "
        "mode is excluded -- see dispatch_synoptic_pca.py docstring.",
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Ensemble IC as an ISO 8601 UTC string (e.g. 2012-10-27T06:00:00Z). "
        "Saved into the npz for downstream local-time axis formatting.",
    )
    parser.add_argument("--timezone", type=str, default="UTC")
    parser.add_argument(
        "--variable",
        action="append",
        required=True,
        dest="variables",
        help="Synoptic precursor variable for the PCA (repeatable; default "
        "z500).  The first variable is the one rendered in the precursor maps.",
    )
    parser.add_argument(
        "--wind-variable",
        action="append",
        default=None,
        dest="wind_variables",
        help="Raw (u, v) wind components saved for the aggregator's steering "
        "arrows (repeatable; e.g. --wind-variable u500 --wind-variable "
        "v500).  Overlay-only -- not part of the PCA.  Best-effort: if the "
        "store lacks them the diagnostic still completes without winds.",
    )
    parser.add_argument(
        "--video-variable",
        type=str,
        default=None,
        help="Variable for the aggregator's per-EOF percentile-member animation "
        "(e.g. msl, where the hurricane low is clearest).  Overlay-only; "
        "best-effort.",
    )
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument(
        "--max-d",
        type=int,
        default=10,
        help="Number of leading PCs to compute and store (PCA truncation). "
        "Default 10.",
    )
    parser.add_argument(
        "--n-eof",
        type=int,
        default=EOF_PERCENTILE_MAX_ROWS,
        help=f"Number of leading EOFs to bake percentile-member fields for "
        f"(default {EOF_PERCENTILE_MAX_ROWS}). Capped at --max-d.",
    )
    parser.add_argument(
        "--conditioning-frame",
        type=int,
        action="append",
        default=None,
        dest="conditioning_frames",
        help="Frame index pinned to ERA5 in this run (repeatable). Saved for "
        "the aggregator's bookkeeping vertical lines.",
    )
    # --- Impact configuration -------------------------------------------
    parser.add_argument(
        "--impact-kind",
        choices=["scalar", "track", "none"],
        default="none",
        help="Downstream impact shown alongside the precursor maps: "
        "'scalar' (bbox-mean of an impact variable, e.g. t2m), 'track' (TC "
        "storm track from the cached parquet), or 'none'.",
    )
    parser.add_argument(
        "--impact-variable",
        type=str,
        default=None,
        help="Impact variable for --impact-kind scalar (e.g. t2m).",
    )
    parser.add_argument(
        "--impact-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="bbox the scalar impact is reduced over (domain mean).",
    )
    parser.add_argument(
        "--impact-mask",
        type=str,
        choices=["none", "land", "sea"],
        default="none",
        help="Mask applied to the scalar-impact bbox reduction.",
    )
    parser.add_argument(
        "--tc-parquet",
        type=str,
        default=None,
        help="Path to the per-mode tc_tracks parquet for --impact-kind track.",
    )
    parser.add_argument("--tc-tracker", type=str, default=None)
    parser.add_argument(
        "--tc-domain",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Track domain box (tracks outside it are dropped to NaN).",
    )
    parser.add_argument("--landfall-frame", type=int, default=-1)

    args = parser.parse_args()

    variables: List[str] = list(args.variables)
    wind_variables: List[str] = list(args.wind_variables or [])
    video_variable: str = args.video_variable or ""
    max_d: int = int(args.max_d)
    conditioning_frames = list(args.conditioning_frames or [])

    print(f"[syn-pca] case        : {args.case_name}")
    print(f"[syn-pca] mode        : {args.mode}")
    print(f"[syn-pca] ensemble    : {args.ensemble_zarr}")
    print(f"[syn-pca] era5        : {args.era5_zarr}")
    print(f"[syn-pca] variables   : {variables}")
    print(f"[syn-pca] wind vars   : {wind_variables or '(none)'}")
    print(f"[syn-pca] video var   : {video_variable or '(none)'}")
    print(
        f"[syn-pca] domain      : lon ∈ [{args.lon_min:g}, {args.lon_max:g}], "
        f"lat ∈ [{args.lat_min:g}, {args.lat_max:g}]"
    )
    print(f"[syn-pca] impact      : kind={args.impact_kind}")

    # --- Open zarr stores ------------------------------------------------
    ens_root = _open_ensemble(args.ensemble_zarr)
    era_root = _open_era5(args.era5_zarr)

    hpx_level = _hpx_level_from_root(ens_root)
    n_leads = int(ens_root["data"].shape[2])
    lead_h = lead_hours(np.array(ens_root["lead_time"][:]))
    free_end_frame = _pick_free_end_frame(args.mode)
    n_members = int(ens_root["data"].shape[0])

    print(f"[syn-pca] hpx level   : {hpx_level} ({12 * 4**hpx_level} pixels)")
    print(f"[syn-pca] n_leads     : {n_leads}, free_end_frame = {free_end_frame}")
    print(f"[syn-pca] n_members   : {n_members}")

    # --- Domain mask -----------------------------------------------------
    domain_idx = _domain_hpx_indices(
        hpx_level, args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    if domain_idx.size == 0:
        raise SystemExit(
            f"ERROR: domain mask is empty for lon ∈ [{args.lon_min}, {args.lon_max}], "
            f"lat ∈ [{args.lat_min}, {args.lat_max}] at hpx_level={hpx_level}."
        )
    print(f"[syn-pca] domain pix  : {domain_idx.size} (of {12 * 4**hpx_level})")

    # --- Free-end load + feature build -----------------------------------
    var_indices = [variable_index(ens_root, v) for v in variables]
    t0 = time.time()
    ens_freeend = _load_ensemble_frame(ens_root, free_end_frame, var_indices)
    print(
        f"[syn-pca] free-end load: {time.time() - t0:.1f}s  "
        f"(shape {tuple(ens_freeend.shape)})"
    )

    # Physical free-end spread, per variable, in storage units squared --
    # the amplitude of the precursor diversity in real units (the eigenvalue
    # spectrum below is on z-scored features, so it carries the *shape* /
    # dimensionality of the diversity but not its physical amplitude).
    # Computed on the raw field BEFORE the z-score, on the same (members,
    # domain pixels) cube the PCA sees.  Equal-area HPX pixels => the
    # per-pixel-mean variance is a clean ``units^2`` diversity measure.
    fe_domain = ens_freeend[:, :, domain_idx]  # [N, n_vars, n_dpix]
    per_pixel_var = fe_domain.var(dim=0, unbiased=True)  # [n_vars, n_dpix]
    physical_variance_per_var = (
        per_pixel_var.sum(dim=-1).cpu().numpy().astype(np.float64)
    )
    physical_variance_mean_per_var = (
        per_pixel_var.mean(dim=-1).cpu().numpy().astype(np.float64)
    )
    del fe_domain, per_pixel_var
    print(
        f"[syn-pca] phys variance: per-pixel-mean (storage units²) = "
        f"{physical_variance_mean_per_var.tolist()}"
    )

    X, per_var_mean, per_var_std = _build_feature_matrix(ens_freeend, domain_idx)
    print(f"[syn-pca] feature mat : {X.shape}  ({X.dtype})")
    print(f"[syn-pca] per-var mean: {per_var_mean.tolist()}")
    print(f"[syn-pca] per-var std : {per_var_std.tolist()}")

    # --- PCA -------------------------------------------------------------
    t0 = time.time()
    scores, components, evr, feature_mean, eigenvalues = _pca(X, max_d=max_d)
    n_eof_show = min(int(args.n_eof), scores.shape[1])
    print(
        f"[syn-pca] PCA         : {time.time() - t0:.1f}s  "
        f"(top {scores.shape[1]} PCs, EVR sum {evr[:scores.shape[1]].sum():.3f}, "
        f"showing {n_eof_show} EOFs)"
    )
    print(
        f"[syn-pca] leading EVR : "
        f"{[f'EOF{j + 1}={100 * evr[j]:.1f}%' for j in range(n_eof_show)]}"
    )

    # --- EOF percentile members: actual member snapshots along each PC ---
    # For each of the top ``n_eof_show`` EOFs, pick the ensemble member
    # whose PC score lies closest to each of ``PERCENTILE_VALUES``
    # percentiles of that PC's score distribution.
    n_pct = len(PERCENTILE_VALUES)
    percentile_member_idx = np.empty((n_eof_show, n_pct), dtype=np.int64)
    for j in range(n_eof_show):
        pcj = scores[:, j]
        for p_i, p in enumerate(PERCENTILE_VALUES):
            target = float(np.percentile(pcj, p))
            percentile_member_idx[j, p_i] = int(np.argmin(np.abs(pcj - target)))
    print(
        f"[syn-pca] EOF pct members: percentiles={list(PERCENTILE_VALUES)}, "
        f"member indices=\n{percentile_member_idx.tolist()}"
    )

    # Load the full trajectories of these percentile members so the
    # aggregator can plot precursor-map trajectories.  The set is small
    # (n_eof_show * n_pct ~ 15 members), so a member-major read is cheap.
    t0 = time.time()
    percentile_trajectory_hpx = _load_member_trajectories_hpx(
        ens_root,
        member_indices=percentile_member_idx.reshape(-1),
        var_indices=var_indices,
    )  # [n_eof_show * n_pct, n_leads, n_vars, n_pix]
    print(
        f"[syn-pca] pct member trajectories load: {time.time() - t0:.1f}s "
        f"(shape {tuple(percentile_trajectory_hpx.shape)})"
    )

    # --- Ensemble mean trajectory (precursor vars), on HPX ---------------
    t0 = time.time()
    ensemble_mean_hpx = _ensemble_mean_hpx(ens_root, var_indices)
    print(
        f"[syn-pca] ensemble mean (HPX): {time.time() - t0:.1f}s  "
        f"(shape {tuple(ensemble_mean_hpx.shape)})"
    )

    # --- Regrid percentile trajectories + ensemble mean to lat/lon box ---
    t0 = time.time()
    percentile_trajectory_latlon_flat, domain_lat, domain_lon = _hpx_to_latlon_domain(
        percentile_trajectory_hpx,
        hpx_level,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    percentile_trajectory_latlon = percentile_trajectory_latlon_flat.reshape(
        n_eof_show,
        n_pct,
        n_leads,
        len(variables),
        percentile_trajectory_latlon_flat.shape[-2],
        percentile_trajectory_latlon_flat.shape[-1],
    )
    ensemble_mean_latlon, _, _ = _hpx_to_latlon_domain(
        ensemble_mean_hpx,
        hpx_level,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    print(
        f"[syn-pca] HPX -> latlon box: {time.time() - t0:.1f}s "
        f"(pct trajectories {percentile_trajectory_latlon.shape}, "
        f"ensemble mean {ensemble_mean_latlon.shape})"
    )
    del percentile_trajectory_hpx

    # --- ERA5 truth trajectory: round-trip through HPX so it lands on ----
    # the same lat/lon-domain grid as every other saved field.
    t0 = time.time()
    era_lat = np.array(era_root["lat"][:])
    era_lon = np.array(era_root["lon"][:])
    era_var_indices = [variable_index(era_root, v) for v in variables]
    n_era_leads = int(era_root["data"].shape[0])
    if n_era_leads != n_leads:
        raise SystemExit(
            f"ERROR: ERA5 reference has {n_era_leads} frames; expected "
            f"{n_leads} to match the ensemble's lead-time axis."
        )
    era_traj_full = np.empty(
        (n_era_leads, len(variables), era_lat.size, era_lon.size), dtype=np.float32
    )
    for j, vi in enumerate(era_var_indices):
        era_traj_full[:, j] = np.array(
            era_root["data"][:, 0, vi, :, :], dtype=np.float32
        )

    era_traj_tensor = torch.from_numpy(era_traj_full)
    era_coords = OrderedDict(
        {
            "time": np.arange(n_era_leads),
            "lead_time": np.array([0]),
            "variable": np.array(variables),
            "lat": era_lat,
            "lon": era_lon,
        }
    )
    era_traj_hpx, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
        era_traj_tensor.unsqueeze(1),  # [T, 1, n_vars, n_lat, n_lon]
        era_coords,
        hpx_level=hpx_level,
    )
    era_traj_hpx_flat = era_traj_hpx[:, 0]  # [T, n_vars, n_pix]
    del era_traj_full, era_traj_tensor, era_traj_hpx

    era_traj_latlon, _, _ = _hpx_to_latlon_domain(
        era_traj_hpx_flat,
        hpx_level,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    print(
        f"[syn-pca] ERA5 traj box: {time.time() - t0:.1f}s  "
        f"(shape {era_traj_latlon.shape})  (lat/lon -> HPX -> lat/lon, raw)"
    )

    # --- Raw-wind overlay fields (optional) ------------------------------
    # The aggregator draws raw-flow steering arrows on the precursor panels to
    # make the steering explicit (why one member's storm goes one way and
    # another's goes elsewhere).  Save the percentile members' and ERA5's raw
    # winds (e.g. u500/v500) on the same (domain_lat, domain_lon) grid as the
    # precursor field.  Best-effort: a store missing the requested wind variables
    # degrades to "no arrows" rather than failing the whole diagnostic.
    wind_variables_saved: List[str] = []
    percentile_member_wind_latlon = np.zeros(0, dtype=np.float32)
    era5_wind_latlon = np.zeros(0, dtype=np.float32)
    if wind_variables:
        try:
            wind_var_indices = [variable_index(ens_root, v) for v in wind_variables]
            # Validate ERA5 carries them too before any heavy load.
            for v in wind_variables:
                variable_index(era_root, v)
            t0 = time.time()
            percentile_member_wind_latlon = _percentile_member_fields_box(
                ens_root,
                percentile_member_idx.reshape(-1),
                wind_var_indices,
                n_eof_show,
                n_pct,
                n_leads,
                hpx_level,
                (args.lon_min, args.lon_max, args.lat_min, args.lat_max),
            )
            era5_wind_latlon = _era5_vars_to_box(
                era_root,
                era_lat,
                era_lon,
                wind_variables,
                n_leads,
                hpx_level,
                (args.lon_min, args.lon_max, args.lat_min, args.lat_max),
            )
            wind_variables_saved = list(wind_variables)
            print(
                f"[syn-pca] raw winds   : {time.time() - t0:.1f}s  "
                f"vars={wind_variables} (members "
                f"{percentile_member_wind_latlon.shape}, era5 {era5_wind_latlon.shape})"
            )
        except ValueError as exc:
            print(f"[syn-pca] raw winds   : skipped -- {exc}")

    # --- Video field (optional) ------------------------------------------
    # The aggregator renders a short per-EOF animation of the percentile members'
    # storm field (MSL, where the hurricane low is clearest) so the trajectory
    # can be watched frame by frame.  Save those members' trajectories on the
    # precursor grid; best-effort, like the winds above.
    video_variable_saved = ""
    percentile_member_video_latlon = np.zeros(0, dtype=np.float32)
    if video_variable:
        try:
            vid_var_indices = [variable_index(ens_root, video_variable)]
            t0 = time.time()
            video_box = _percentile_member_fields_box(
                ens_root,
                percentile_member_idx.reshape(-1),
                vid_var_indices,
                n_eof_show,
                n_pct,
                n_leads,
                hpx_level,
                (args.lon_min, args.lon_max, args.lat_min, args.lat_max),
            )  # [n_eof_show, n_pct, n_leads, 1, nlat, nlon]
            percentile_member_video_latlon = video_box[:, :, :, 0].astype(np.float32)
            print(
                f"[syn-pca] video field : {time.time() - t0:.1f}s  "
                f"var={video_variable} (shape {percentile_member_video_latlon.shape})"
            )
            video_variable_saved = video_variable
        except ValueError as exc:
            print(f"[syn-pca] video field : skipped -- {exc}")

    # --- ERA5 PC projection: where does truth sit in the ensemble's PC space? ---
    era_fe_hpx = era_traj_hpx_flat[free_end_frame].cpu().numpy().astype(np.float32)
    era_fe_domain = era_fe_hpx[:, domain_idx].copy()  # [n_vars, n_dpix]
    for v in range(len(variables)):
        era_fe_domain[v] = (era_fe_domain[v] - per_var_mean[v]) / per_var_std[v]
    era_fe_centered = era_fe_domain.reshape(-1) - feature_mean  # [n_features]
    era5_pc_score = (era_fe_centered @ components[:n_eof_show].T).astype(np.float32)
    era5_pc_percentile = np.empty(n_eof_show, dtype=np.float32)
    for j in range(n_eof_show):
        era5_pc_percentile[j] = float(
            100.0 * (scores[:, j] <= era5_pc_score[j]).sum() / n_members
        )
    print(
        f"[syn-pca] ERA5 PC pcts : "
        f"{[f'PC{j + 1}={era5_pc_percentile[j]:.0f}' for j in range(n_eof_show)]}"
    )

    # --- EOF patterns: undo z-score, embed into full HPX, regrid to box ---
    eof_unscaled = (
        components[:n_eof_show]
        .reshape(n_eof_show, len(variables), domain_idx.size)
        .copy()
    )
    for v in range(len(variables)):
        eof_unscaled[:, v, :] *= per_var_std[v]
    eof_hpx_full = torch.zeros(
        n_eof_show, len(variables), 12 * 4**hpx_level, dtype=torch.float32
    )
    eof_hpx_full[:, :, domain_idx] = torch.from_numpy(eof_unscaled)
    eof_latlon_box, _, _ = _hpx_to_latlon_domain(
        eof_hpx_full,
        hpx_level,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )

    # --- Impact ----------------------------------------------------------
    impact_kind = args.impact_kind
    impact_variable = args.impact_variable or ""
    impact_mask_kind = args.impact_mask
    impact_bbox = np.zeros(0, dtype=np.float32)
    # Pixels inside the impact bbox that the land/sea mask *drops* (ocean when
    # mask='land', land when mask='sea'), on the precursor (domain_lat,
    # domain_lon) grid.  Saved so the aggregator can hatch the unused part of
    # the dashed impact box and the reader does not read it as a plain area
    # mean.  Stays empty when no mask is applied.  Mirrors
    # compute_free_end_states.py's excluded_mask_box.
    impact_excluded_mask_box = np.zeros((0, 0), dtype=bool)
    member_impact_scalar = np.zeros((0, 0), dtype=np.float32)
    era5_impact_scalar = np.zeros(0, dtype=np.float32)
    ensemble_mean_impact_scalar = np.zeros(0, dtype=np.float32)
    # Warm/cool-decile composite of the free-end precursor field (model-free
    # twin of the regression composite): the mean precursor field over the
    # warmest vs coolest decile of members, ranked by the free-end impact
    # scalar.  The aggregator forms warm - cool to show the precursor pattern
    # that accompanies an extreme impact without assuming linearity.  Raw
    # fields on the (domain_lat, domain_lon) grid; empty unless scalar impact.
    impact_decile_warm_latlon = np.zeros((0, 0, 0), dtype=np.float32)
    impact_decile_cool_latlon = np.zeros((0, 0, 0), dtype=np.float32)
    impact_decile_frac = np.float32(0.0)
    impact_decile_warm_mean = np.float32(0.0)
    impact_decile_cool_mean = np.float32(0.0)
    # Domain-sensitivity ladder: per-member free-end impact scalar over a nested
    # set of impact boxes (headline box scaled about its centre).  Empty unless
    # scalar impact.  The aggregator regresses each on the PCs (de-attenuation
    # check: does growing the box sharpen the precursor -> impact R^2?).
    member_impact_ladder = np.zeros((0, 0), dtype=np.float32)
    era5_impact_ladder = np.zeros(0, dtype=np.float32)
    ladder_bboxes = np.zeros((0, 4), dtype=np.float32)
    ladder_scales = np.zeros(0, dtype=np.float32)
    ladder_n_pix = np.zeros(0, dtype=np.int32)
    tc_tracker = args.tc_tracker or ""
    tc_domain = np.zeros(0, dtype=np.float32)
    landfall_frame = int(args.landfall_frame)
    member_track_latlon = np.zeros((0, 0, 0), dtype=np.float32)
    era5_track_latlon = np.zeros((0, 0), dtype=np.float32)
    # Baked free-end track targets (path_id=0), anchored via the shared
    # tc_track_targets extraction, so aggregate_synoptic_pca can draw the
    # track-regression figure and aggregate_tc_tracks can annotate the anchored
    # ERA5 MSL percentile from one source of truth.  Empty unless
    # impact_kind == "track"; populated in the track branch below.
    track_target_fe = np.zeros((0, 4), dtype=np.float32)  # [N, (lat, lon, dist, msl)]
    track_path0_anchored = np.zeros(0, dtype=bool)  # [N]
    track_era5_fe = np.zeros(0, dtype=np.float32)  # [3] (lat, lon, msl_hpa)
    track_anchor_frame = np.int32(-1)
    track_anchor_counts = np.zeros(0, dtype=np.int32)  # [3] anchored/alt_path/lost
    track_era5_fe_msl_pct = np.float32(np.nan)  # unanchored
    track_era5_fe_msl_pct_anchored = np.float32(np.nan)
    track_n_fallback = np.int32(0)

    if impact_kind == "scalar":
        if not impact_variable or args.impact_bbox is None:
            raise SystemExit(
                "ERROR: --impact-kind scalar requires --impact-variable and "
                "--impact-bbox."
            )
        t0 = time.time()
        impact_bbox = np.asarray(args.impact_bbox, dtype=np.float32)
        pix_lat, pix_lon = _hpx_pixel_centers(hpx_level)
        bbox_mask = _hpx_bbox_mask(pix_lat, pix_lon, args.impact_bbox)
        impact_mask = _resolve_impact_mask(
            bbox_mask, impact_mask_kind, era_root, hpx_level
        )
        # Pixels inside the bbox but excluded by the land/sea mask
        # (``bbox_mask & ~impact_mask`` is exactly the set the scalar mean below
        # leaves out, so the hatch stays provably consistent with the average).
        # Regrid that 0/1 set onto the precursor (domain_lat, domain_lon) grid
        # the same way every saved field is regridded, so the aggregator can
        # hatch it inside the dashed impact box.
        if impact_mask_kind in ("land", "sea"):
            excluded_hpx = (bbox_mask & ~impact_mask).astype(np.float32)
            excluded_box, _, _ = _hpx_to_latlon_domain(
                torch.from_numpy(excluded_hpx)[None, :],
                hpx_level,
                args.lon_min,
                args.lon_max,
                args.lat_min,
                args.lat_max,
            )
            impact_excluded_mask_box = excluded_box[0] > 0.5
        impact_var_idx = variable_index(ens_root, impact_variable)
        member_impact_scalar = _compute_member_scalar_hpx(
            ens_root, impact_var_idx, impact_mask
        )  # [N, n_leads]
        ensemble_mean_impact_scalar = member_impact_scalar.mean(axis=0).astype(
            np.float32
        )
        # ERA5 impact var: regrid to HPX (it is not a precursor var) and reduce.
        era_impact_idx = variable_index(era_root, impact_variable)
        era_impact_latlon = np.array(
            era_root["data"][:, 0, era_impact_idx, :, :], dtype=np.float32
        )  # [n_leads, lat, lon]
        era_impact_coords = OrderedDict(
            {
                "time": np.arange(n_leads),
                "lead_time": np.array([0]),
                "variable": np.array([impact_variable]),
                "lat": era_lat,
                "lon": era_lon,
            }
        )
        era_impact_hpx, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
            torch.from_numpy(era_impact_latlon)
            .unsqueeze(1)
            .unsqueeze(2),  # [T, 1, 1, lat, lon]
            era_impact_coords,
            hpx_level=hpx_level,
        )
        era_impact_hpx_np = era_impact_hpx[:, 0, 0].cpu().numpy()  # [n_leads, n_pix]
        era5_impact_scalar = (
            era_impact_hpx_np[:, impact_mask].mean(axis=1).astype(np.float32)
        )
        print(
            f"[syn-pca] impact scalar: {time.time() - t0:.1f}s  var={impact_variable} "
            f"mask={impact_mask_kind} n_pix={int(impact_mask.sum())}"
        )

        # --- Warm/cool-decile precursor composite (model-free twin) -------
        # Rank members by their free-end impact scalar; composite the free-end
        # precursor field (all PCA variables) over the warmest vs coolest
        # decile.  ens_freeend still holds the raw [N, n_vars, n_pix] free-end
        # fields (the in-place z-score in _build_feature_matrix operated on a
        # fancy-indexed copy, not this tensor), so the composite is the true
        # field, not a PCA-subspace reconstruction.  The warm - cool difference
        # (formed at plot time) is the precursor pattern that accompanies an
        # extreme impact, assuming no functional form for precursor -> impact.
        t0 = time.time()
        y_free_end = member_impact_scalar[:, free_end_frame]
        order = np.argsort(y_free_end)
        n_dec = max(1, int(round(0.1 * n_members)))
        cool_sel = order[:n_dec]
        warm_sel = order[-n_dec:]
        decile_hpx = torch.stack(
            [
                ens_freeend[torch.as_tensor(warm_sel)].mean(dim=0),
                ens_freeend[torch.as_tensor(cool_sel)].mean(dim=0),
            ],
            dim=0,
        )  # [2, n_vars, n_pix]
        decile_box, _, _ = _hpx_to_latlon_domain(
            decile_hpx,
            hpx_level,
            args.lon_min,
            args.lon_max,
            args.lat_min,
            args.lat_max,
        )  # [2, n_vars, nlat, nlon]
        impact_decile_warm_latlon = decile_box[0].astype(np.float32)
        impact_decile_cool_latlon = decile_box[1].astype(np.float32)
        impact_decile_frac = np.float32(n_dec / n_members)
        impact_decile_warm_mean = np.float32(float(y_free_end[warm_sel].mean()))
        impact_decile_cool_mean = np.float32(float(y_free_end[cool_sel].mean()))
        print(
            f"[syn-pca] decile composite: {time.time() - t0:.1f}s  "
            f"n_dec={n_dec} (frac={float(impact_decile_frac):.2f}) "
            f"warm_mean={float(impact_decile_warm_mean):.2f} "
            f"cool_mean={float(impact_decile_cool_mean):.2f}"
        )

        # --- Domain-sensitivity ladder ------------------------------------
        # Recompute the free-end impact scalar over a nested set of impact
        # boxes (headline box scaled about its centre).  Holding the centre
        # fixed and growing the area is a controlled de-attenuation test: the
        # aggregator regresses each on the PCs, and a precursor -> impact R^2
        # that sharpens with box size (while the composite pattern stays put)
        # is the signature of noise reduction, not a domain tuned to the fit.
        # One free-end frame read, then average over each box's land mask.
        t0 = time.time()
        impact_fe_hpx = np.array(
            ens_root["data"][:, 0, free_end_frame, impact_var_idx, :],
            dtype=np.float32,
        )  # [N, n_pix]
        era_impact_fe_hpx = era_impact_hpx_np[free_end_frame]  # [n_pix]
        ladder_scales_acc: List[float] = []
        ladder_bboxes_acc: List[np.ndarray] = []
        ladder_member_acc: List[np.ndarray] = []
        ladder_era5_acc: List[float] = []
        ladder_npix_acc: List[int] = []
        for scale in IMPACT_LADDER_SCALES:
            box = _scaled_bbox(args.impact_bbox, scale)
            try:
                box_mask = _hpx_bbox_mask(pix_lat, pix_lon, box)
                lad_mask = _resolve_impact_mask(
                    box_mask, impact_mask_kind, era_root, hpx_level
                )
            except SystemExit as exc:
                print(f"[syn-pca] ladder scale {scale:g}: skipped -- {exc}")
                continue
            ladder_scales_acc.append(float(scale))
            ladder_bboxes_acc.append(np.asarray(box, dtype=np.float32))
            ladder_member_acc.append(
                impact_fe_hpx[:, lad_mask].mean(axis=1).astype(np.float32)
            )
            ladder_era5_acc.append(float(era_impact_fe_hpx[lad_mask].mean()))
            ladder_npix_acc.append(int(lad_mask.sum()))
        if ladder_member_acc:
            member_impact_ladder = np.stack(ladder_member_acc, axis=1).astype(
                np.float32
            )  # [N, n_ladder]
            ladder_bboxes = np.stack(ladder_bboxes_acc, axis=0).astype(np.float32)
            era5_impact_ladder = np.asarray(ladder_era5_acc, dtype=np.float32)
            ladder_scales = np.asarray(ladder_scales_acc, dtype=np.float32)
            ladder_n_pix = np.asarray(ladder_npix_acc, dtype=np.int32)
            print(
                f"[syn-pca] domain ladder: {time.time() - t0:.1f}s  "
                f"scales={ladder_scales_acc} n_pix={ladder_npix_acc}"
            )
    elif impact_kind == "track":
        if not args.tc_parquet or args.tc_domain is None:
            raise SystemExit(
                "ERROR: --impact-kind track requires --tc-parquet and --tc-domain."
            )
        t0 = time.time()
        tc_domain = np.asarray(args.tc_domain, dtype=np.float32)
        member_track_latlon, era5_track_latlon, available = (
            _load_member_tracks_from_parquet(
                args.tc_parquet, n_members, n_leads, args.tc_domain
            )
        )
        if not available:
            # Parquet missing -> fall back to maps-only; aggregator skips panel.
            impact_kind = "none"
            member_track_latlon = np.zeros((0, 0, 0), dtype=np.float32)
            era5_track_latlon = np.zeros((0, 0), dtype=np.float32)
        else:
            n_tracked = int(np.isfinite(member_track_latlon[:, :, 0]).any(axis=1).sum())
            print(
                f"[syn-pca] impact track : {time.time() - t0:.1f}s  "
                f"tracker={tc_tracker} members with track={n_tracked}/{n_members}"
            )
            # Bake the anchored free-end targets from the raw all-paths parquet
            # via the shared extraction (the same load_track_targets the
            # track-stats report uses) so the aggregators quote one source of
            # truth.  Anchor = the pinned end: frame 0 for start-conditioning,
            # the landfall frame (last frame if unset) for end-conditioning.
            # A baking failure degrades to maps-only rather than sinking the run.
            anchor_frame = (
                0
                if args.mode == "start"
                else (landfall_frame if 0 <= landfall_frame < n_leads else n_leads - 1)
            )
            try:
                baked = free_end_target_arrays(
                    args.tc_parquet,
                    free_end_frame,
                    n_members,
                    anchor_frame=anchor_frame,
                )
                track_target_fe = baked["track_target_fe"]
                track_path0_anchored = baked["track_path0_anchored"]
                track_era5_fe = baked["track_era5_fe"]
                track_anchor_frame = baked["track_anchor_frame"]
                track_anchor_counts = baked["track_anchor_counts"]
                track_era5_fe_msl_pct = baked["track_era5_fe_msl_pct"]
                track_era5_fe_msl_pct_anchored = baked["track_era5_fe_msl_pct_anchored"]
                track_n_fallback = baked["track_n_fallback"]
                a = track_anchor_counts
                print(
                    f"[syn-pca] track targets: anchor frame {anchor_frame}, "
                    f"anchored/alt/lost={a[0]}/{a[1]}/{a[2]}, ERA5 free-end MSL "
                    f"pct anchored={float(track_era5_fe_msl_pct_anchored):.1f}"
                )
            except Exception as exc:  # noqa: BLE001 - degrade, do not crash
                print(
                    f"[syn-pca] WARNING: could not bake free-end track targets "
                    f"({exc}); the aggregator will skip the track-regression "
                    f"figure and the anchored percentile annotation."
                )

    # --- Save -----------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"synoptic_pca_{args.mode}.npz"
    np.savez(
        npz_path,
        # Identity / config
        case_name=np.array(args.case_name),
        mode=np.array(args.mode),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
        variables=np.array(variables),
        lon_min=np.float32(args.lon_min),
        lon_max=np.float32(args.lon_max),
        lat_min=np.float32(args.lat_min),
        lat_max=np.float32(args.lat_max),
        lead_hours=np.asarray(lead_h, dtype=np.float64),
        n_members=np.int32(n_members),
        free_end_frame=np.int32(free_end_frame),
        conditioning_frames=np.asarray(conditioning_frames, dtype=np.int32),
        max_d=np.int32(max_d),
        # PCA (full spectrum saved; aggregator accesses positionally up to n_eof_show)
        pc_scores=scores,  # [N, max_d]
        pc_components_latlon=eof_latlon_box,  # [n_eof_show, n_vars, nlat, nlon]
        explained_variance_ratio=evr,  # [n_components] (full spectrum)
        eigenvalues=eigenvalues.astype(np.float64),  # [n_components] = S^2
        per_var_mean=per_var_mean,  # [n_vars]
        per_var_std=per_var_std,  # [n_vars]
        # Physical free-end spread (storage units^2) for the diversity figure;
        # the per-pixel mean is the area-weighted, unit-clean amplitude.
        physical_variance_per_var=physical_variance_per_var,  # [n_vars] (sum over domain)
        physical_variance_mean_per_var=physical_variance_mean_per_var,  # [n_vars]
        n_eof_show=np.int32(n_eof_show),
        # EOF percentile members: trajectories on the precursor field.  The
        # aggregator renders ``member - ens_mean`` (diversity) against
        # ``ens_mean - ERA5`` (bias) on a shared diverging colorbar, and
        # slices ``[..., free_end_frame, ...]`` for the free-end snapshot.
        percentile_member_trajectory_latlon=percentile_trajectory_latlon,
        percentile_member_idx=percentile_member_idx.astype(
            np.int32
        ),  # [n_eof_show, n_pct]
        percentile_values=np.asarray(PERCENTILE_VALUES, dtype=np.int32),  # [n_pct]
        era5_pc_score=era5_pc_score,  # [n_eof_show]
        era5_pc_percentile=era5_pc_percentile,  # [n_eof_show] in 0..100
        # Reference fields (raw) on the shared (domain_lat, domain_lon) grid.
        era5_trajectory_latlon=era_traj_latlon,  # [n_leads, n_vars, nlat, nlon]
        ensemble_mean_trajectory_latlon=ensemble_mean_latlon,  # [n_leads, n_vars, nlat, nlon]
        domain_lat=domain_lat,
        domain_lon=domain_lon,
        # Raw-wind overlay fields for the aggregator's steering arrows
        # (empty arrays when --wind-variable was not supplied, or the store
        # lacked the requested components).
        wind_variables=np.array(wind_variables_saved),
        # [n_eof_show, n_pct, n_leads, n_wind, nlat, nlon]
        percentile_member_wind_latlon=percentile_member_wind_latlon,
        era5_wind_latlon=era5_wind_latlon,  # [n_leads, n_wind, nlat, nlon]
        # Video field for the per-EOF percentile-member animation (empty when
        # --video-variable was not supplied or the store lacked it).
        video_variable=np.array(video_variable_saved),
        # [n_eof_show, n_pct, n_leads, nlat, nlon]
        percentile_member_video_latlon=percentile_member_video_latlon,
        # Impact (kind selects which arrays are populated; the rest are empty).
        impact_kind=np.array(impact_kind),
        impact_variable=np.array(impact_variable),
        impact_mask_kind=np.array(impact_mask_kind),
        impact_bbox=impact_bbox,  # [4] or empty
        # [nlat, nlon] bool: True where a bbox pixel is excluded by the
        # land/sea mask (empty when no mask).  Hatched by the aggregator.
        impact_excluded_mask_box=np.asarray(impact_excluded_mask_box, dtype=bool),
        member_impact_scalar=member_impact_scalar.astype(
            np.float32
        ),  # [N, n_leads] or empty
        era5_impact_scalar=era5_impact_scalar.astype(np.float32),  # [n_leads] or empty
        ensemble_mean_impact_scalar=ensemble_mean_impact_scalar.astype(np.float32),
        # Warm/cool-decile precursor composite (model-free twin); empty unless
        # scalar impact.  [n_vars, nlat, nlon] raw fields on the precursor grid.
        impact_decile_warm_latlon=impact_decile_warm_latlon,
        impact_decile_cool_latlon=impact_decile_cool_latlon,
        impact_decile_frac=impact_decile_frac,
        impact_decile_warm_mean=impact_decile_warm_mean,
        impact_decile_cool_mean=impact_decile_cool_mean,
        # Domain-sensitivity ladder (empty unless scalar impact).
        member_impact_ladder=member_impact_ladder,  # [N, n_ladder]
        era5_impact_ladder=era5_impact_ladder,  # [n_ladder]
        ladder_bboxes=ladder_bboxes,  # [n_ladder, 4]
        ladder_scales=ladder_scales,  # [n_ladder]
        ladder_n_pix=ladder_n_pix,  # [n_ladder]
        tc_tracker=np.array(tc_tracker),
        tc_domain=tc_domain,  # [4] or empty
        landfall_frame=np.int32(landfall_frame),
        member_track_latlon=member_track_latlon.astype(
            np.float32
        ),  # [N, n_leads, 2] or empty
        era5_track_latlon=era5_track_latlon.astype(np.float32),  # [n_leads, 2] or empty
        # Baked anchored free-end targets (empty unless impact_kind == "track");
        # one source of truth for the track-regression figure + anchored ERA5
        # MSL percentile.  See tc_track_targets.free_end_target_arrays.
        track_target_fe=track_target_fe,  # [N, 4] (lat, lon, dist_km, msl_hpa)
        track_path0_anchored=track_path0_anchored,  # [N] bool
        track_era5_fe=track_era5_fe,  # [3] (lat, lon, msl_hpa)
        track_anchor_frame=track_anchor_frame,
        track_anchor_counts=track_anchor_counts,  # [3] anchored/alt_path/lost
        track_era5_fe_msl_pct=track_era5_fe_msl_pct,  # unanchored
        track_era5_fe_msl_pct_anchored=track_era5_fe_msl_pct_anchored,
        track_n_fallback=track_n_fallback,
    )
    print(f"[syn-pca] wrote {npz_path}")
    print("[syn-pca] done.")


if __name__ == "__main__":
    main()
