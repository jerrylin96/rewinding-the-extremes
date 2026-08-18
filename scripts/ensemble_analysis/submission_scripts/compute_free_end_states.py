"""Compute free-end-state rankings for one (mode, variable) of an ensemble.

The "free end" is the lead-time frame *opposite* the conditioning frame:
for ``start`` conditioning the free end is the last frame (e.g. frame 11),
for ``end`` conditioning the free end is frame 0.  The diagnostic ranks
every ensemble member by its *signed* box-mean departure from ERA5 *at
that free end* -- member box-mean minus ERA5 box-mean, over HEALPix
pixels whose centers fall inside a configurable lat/lon box -- and records
the K members at each tail: the ``low`` tail (most negative bias:
coolest/lowest) and the ``high`` tail (most positive bias:
warmest/highest).  The signed bias keeps the *direction* of the departure
(for a heatwave, whether a member runs hotter or cooler than observed),
which an unsigned RMSE discards; RMSE is still computed and saved as a
secondary field.  The ``both`` mode has no free end and is rejected.

The reduction lives in HEALPix space (matching ``diagnostics_from_zarr``
used by ``compute_spread_rmse_crps.py``): the ensemble stays on its
native HPX grid and ERA5 is regridded lat/lon → HPX once.  Equal-area
HPX pixels mean uniform per-pixel weights, no cos-lat correction needed.
Only the K low-tail and K high-tail member HPX fields (plus the HPX ERA5)
are regridded back to lat/lon for the saved snapshots that
``aggregate_free_end_states.py`` plots.  Those snapshots are cropped to a
``--view-bbox`` field of view (defaulting to the impact ``--bbox``); a
wider view lets the aggregator show synoptic context around the dashed
impact box, matching the synoptic-PCA precursor panels, while the bias
ranking and the bbox-mean timeseries still reduce over ``--bbox``.

``--mask`` optionally narrows the ranking + bbox-mean reductions to land
or sea pixels.  The mask is derived from ERA5's own SST field: SST is
NaN over land in the reference store, so ``isnan(sst_hpx)`` directly
identifies land pixels in the same grid the diagnostic already uses,
with the same land/sea convention as the reference being compared
against.  For surface-temperature variables on regional bboxes that
include substantial ocean, masking to land prevents the near-constant
SST side from diluting the ranking — ocean t2m is essentially SST and
barely moves on a multi-day lead window, so unmasked pixels there can
make a member that nails the land signal look similar to one that
misses it.  Defaults to ``none`` (no masking) to keep the diagnostic
general; opt in only for variables where the distinction matters.

This script writes the numerics + field snapshots for ONE conditioning
mode.  The cross-mode aggregated figure is produced separately by
``aggregate_free_end_states.py`` once every per-mode job has finished.

Outputs (in ``--output-dir``)::

    free_end_states_<mode>.npz
        member_bias           : (N,)             float32    — signed box-mean (member-ERA5) for every member
        member_rmse           : (N,)             float32    — HPX RMSE for every member (secondary)
        low_idx               : (K,)             int32      — member indices, ascending bias (coolest/lowest first)
        high_idx              : (K,)             int32      — member indices, descending bias (warmest/highest first)
        low_bias              : (K,)             float32    — signed bias of the low (coolest/lowest) members
        high_bias             : (K,)             float32    — signed bias of the high (warmest/highest) members
        low_rmse              : (K,)             float32    — RMSE of those members (secondary)
        high_rmse             : (K,)             float32
        low_fields            : (K, ny, nx)      float32    — low (coolest/lowest) member fields in view box (lat/lon)
        high_fields           : (K, ny, nx)      float32    — high (warmest/highest) member fields in view box (lat/lon)
        era5_field            : (ny, nx)         float32    — ERA5 in view box (lat/lon, HPX-roundtripped)
        excluded_mask_box     : (ny, nx)         bool       — only when mask != 'none';
                                                              True where the pixel is *excluded* from
                                                              RMSE + bbox-mean (i.e. ocean when
                                                              mask='land', land when mask='sea').
                                                              Confined to the impact bbox (False
                                                              everywhere outside it) and laid on the
                                                              same view-box lat/lon grid as the member
                                                              fields, so the aggregator hatches only
                                                              the box's excluded pixels, not the whole
                                                              synoptic view.  Round-tripped from the
                                                              same ERA5 SST field that defines the HPX
                                                              metric mask.
        member_scalar         : (N, n_leads)     float32    — per-member bbox domain-mean at every lead
        era5_scalar           : (n_leads,)       float32    — ERA5 bbox domain-mean at every lead
        ensemble_mean_scalar  : (n_leads,)       float32    — mean over members at every lead
        lead_hours_all        : (n_leads,)       float64    — lead-time hours for x-axis
        lat                   : (ny,)            float64    — view-box latitudes
        lon                   : (nx,)            float64    — view-box longitudes
        bbox                  : (4,)             float64    — impact box [lon_min, lon_max, lat_min, lat_max]
                                                              (RMSE + bbox-mean domain; drawn dashed)
        view_bbox             : (4,)             float64    — plotting field of view
                                                              [lon_min, lon_max, lat_min, lat_max]
        n_pix_in_box          : ()               int64      — HPX pixels used in the RMSE (post-mask)
        mask_kind             : ()               str        — 'none' / 'land' / 'sea'
        hpx_level             : ()               int32
        free_end_frame        : ()               int32
        free_end_hours        : ()               float64
        conditioning_frames   : object           list
        n_members             : ()               int64
        top_k                 : ()               int32
        var_name              : ()               str
        case_name             : ()               str
        mode                  : ()               str
        start_time            : ()               str        — ISO 8601 UTC
        timezone              : ()               str

Usage::

    python compute_free_end_states.py \\
        --ensemble-zarr /path/to/ensemble_f0_n1000.zarr \\
        --era5-zarr     /path/to/era5_reference.zarr \\
        --output-dir    /path/to/diagnostics/free_end_states/<variable> \\
        --mode          start \\
        --var-name      z500 \\
        --bbox          -85 -60 25 45

For surface-temperature diagnostics over a land-heavy event, restrict
to land pixels::

    python compute_free_end_states.py \\
        ... \\
        --var-name t2m --bbox -140 -110 38 55 --mask land
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import MODE_FRAMES, lead_hours  # noqa: E402

from custom_utils.diagnostics import hpx_land_mask_from_sst_hpx  # noqa: E402
from custom_utils.ensemble_loading import variable_index  # noqa: E402


def _free_end_frame(mode: str, n_leads: int) -> int:
    """Return the frame index opposite the conditioning frame for ``mode``."""
    if mode == "start":
        return n_leads - 1
    if mode == "end":
        return 0
    raise ValueError(
        f"mode '{mode}' has no free end (both ends are conditioned); "
        f"this diagnostic only runs on 'start' and 'end'."
    )


def _hpx_level_from_root(root) -> int:
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    return int(np.log2(root["hpx"].shape[0] // 12) / 2)


def _require_hpx_ensemble(root, zarr_path: str) -> None:
    if "hpx" not in root.array_keys():
        raise SystemExit(
            f"ERROR: {zarr_path}: expected HEALPix ensemble layout "
            f"(with 'hpx' coord); got arrays {sorted(root.array_keys())}.  "
            f"This diagnostic only supports HEALPix ensembles to keep the "
            f"distance metric in the same space as compute_spread_rmse_crps."
        )


def _load_ensemble_hpx_all_frames(
    ens_zarr_path: str,
    var_name: str,
    n_members: int | None,
) -> torch.Tensor:
    """Load ``[N, n_leads, n_pix]`` HPX field for one variable, all leads."""
    root = zarr.open_group(ens_zarr_path, mode="r")
    _require_hpx_ensemble(root, ens_zarr_path)
    total = root["data"].shape[0]
    n = total if n_members is None else min(n_members, total)
    var_idx = variable_index(root, var_name)
    # data shape: [N, 1, n_leads, n_vars, n_pix]
    arr = np.array(root["data"][:n, 0, :, var_idx, :])
    return torch.from_numpy(arr).float()


def _regrid_era5_latlon_to_hpx_all_frames(
    era5_zarr_path: str,
    var_name: str,
    hpx_level: int,
) -> torch.Tensor:
    """Regrid the ERA5 field for ``var_name`` to HPX across every lead frame.

    Returns ``[n_leads, n_pix]``.  Regridding all frames in a single call
    amortises the regridder build cost.
    """
    from custom_utils.interpolation_utils import CBottleUtils

    era5_root = zarr.open_group(era5_zarr_path, mode="r")
    var_idx = variable_index(era5_root, var_name)
    # ERA5 data shape: [n_leads, 1, n_vars, n_lat, n_lon]
    era5 = torch.from_numpy(
        np.array(era5_root["data"][:, 0, var_idx, :, :])
    ).float()  # [n_leads, n_lat, n_lon]
    n_leads = era5.shape[0]

    t5 = era5.unsqueeze(0).unsqueeze(2)  # [1, n_leads, 1, n_lat, n_lon]
    coords = OrderedDict(
        {
            "time": np.array([0]),
            "lead_time": np.arange(n_leads),
            "variable": np.array([var_name]),
            "lat": np.array(era5_root["lat"][:]),
            "lon": np.array(era5_root["lon"][:]),
        }
    )
    hpx_data, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
        t5,
        coords,
        hpx_level=hpx_level,
    )
    return hpx_data[0, :, 0, :].cpu().float()  # [n_leads, n_pix]


def _hpx_pixel_centers(hpx_level: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) center coordinates for every HEALPix NEST pixel."""
    import earth2grid

    grid = earth2grid.healpix.Grid(
        hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    return np.asarray(grid.lat), np.asarray(grid.lon)


def _hpx_bbox_mask(
    pix_lat: np.ndarray,
    pix_lon: np.ndarray,
    bbox: List[float],
) -> np.ndarray:
    """Boolean mask over HPX pixels whose centers lie inside ``bbox``.

    ``bbox`` is ``[lon_min, lon_max, lat_min, lat_max]``.  Longitudes are
    normalized to [-180, 180] so both store conventions ([-180,180] or
    [0,360]) work; dateline-crossing boxes are rejected.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    if lon_min >= lon_max:
        raise SystemExit(
            f"ERROR: --bbox lon_min ({lon_min}) >= lon_max ({lon_max}); "
            f"dateline-crossing boxes are not supported."
        )
    if lat_min >= lat_max:
        raise SystemExit(f"ERROR: --bbox lat_min ({lat_min}) >= lat_max ({lat_max}).")

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
            f"ERROR: bbox {bbox} matched zero HPX pixel centers; "
            f"box may be smaller than the pixel size."
        )
    return mask


# This was the original implementation; it now lives in
# ``custom_utils.diagnostics`` so this diagnostic, the synoptic-PCA scalar
# impact, and the spread/RMSE/CRPS battery cannot disagree about which pixels
# are land.  Kept as a thin alias rather than inlined at the call site to
# leave the surrounding code untouched.  See the canonical docstring for the
# ERA5-SST-NaN convention and the conservative coastal treatment.
_hpx_land_mask_from_sst = hpx_land_mask_from_sst_hpx


def _hpx_rmse(
    members: torch.Tensor,
    era5: torch.Tensor,
    mask: np.ndarray,
) -> torch.Tensor:
    """Per-member RMSE over the masked HPX pixels (uniform weights).

    Equal-area pixels mean no cos-lat correction is needed — every pixel
    inside the bbox contributes equally.
    """
    pix_idx = torch.from_numpy(np.where(mask)[0]).long()
    members_box = members.index_select(dim=1, index=pix_idx)  # [N, n_box]
    era5_box = era5.index_select(dim=0, index=pix_idx)  # [n_box]
    diff_sq = (members_box - era5_box.unsqueeze(0)) ** 2
    return torch.sqrt(diff_sq.mean(dim=1))


def _hpx_domain_mean(
    fields: torch.Tensor,
    mask: np.ndarray,
) -> torch.Tensor:
    """Mean of ``fields`` over masked HPX pixels (uniform-weight, equal-area).

    Works on any leading shape — ``fields[..., n_pix]`` is reduced over the
    trailing pixel axis to ``fields[...]``.
    """
    pix_idx = torch.from_numpy(np.where(mask)[0]).long()
    return fields.index_select(dim=-1, index=pix_idx).mean(dim=-1)


def _hpx_fields_to_latlon_bbox(
    hpx_fields: torch.Tensor,
    hpx_level: int,
    bbox: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid HPX fields ``[K, n_pix]`` back to lat/lon and clip to ``bbox``.

    Returns ``(fields_box, lat_box, lon_box)``.  ``fields_box`` has shape
    ``[K, ny, nx]``; ``lat_box`` and ``lon_box`` are the bbox-cropped
    coordinate vectors of the round-tripped lat/lon grid.
    """
    from custom_utils.interpolation_utils import CBottleUtils

    k = hpx_fields.shape[0]
    t4 = hpx_fields.unsqueeze(1).unsqueeze(2)  # [K, 1, 1, n_pix]
    # The regridder hard-requires "time"/"lead_time"/"variable" coord keys and
    # forwards them into latlon_coords; their values are unused downstream so
    # we synthesise minimal placeholders.
    coords = OrderedDict(
        {
            "time": np.arange(k),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(["_"]),
            "hpx": np.arange(hpx_fields.shape[-1]),
        }
    )
    latlon, latlon_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        t4,
        coords,
        hpx_level=hpx_level,
    )
    fields = latlon[:, 0, 0, :, :].cpu().numpy().astype(np.float32)  # [K, ny, nx]
    lat = np.asarray(latlon_coords["lat"])
    lon = np.asarray(latlon_coords["lon"])

    # Normalize lon to [-180, 180] so the saved vector matches the bbox /
    # cartopy convention.  Sort the matching indices by the normalized lon
    # so the output is monotonic even when the bbox straddles lon=0 (where
    # the regridder's [0, 360) wraps around).
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_min_n = ((lon_min + 180.0) % 360.0) - 180.0
    lon_max_n = ((lon_max + 180.0) % 360.0) - 180.0
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_hits = np.where((lon_norm >= lon_min_n) & (lon_norm <= lon_max_n))[0]
    lon_idx = lon_hits[np.argsort(lon_norm[lon_hits])]
    fields_box = fields[:, lat_idx[:, None], lon_idx[None, :]]
    return fields_box, lat[lat_idx], lon_norm[lon_idx]


def _latlon_in_bbox_mask(
    lat: np.ndarray, lon: np.ndarray, bbox: List[float]
) -> np.ndarray:
    """``(ny, nx)`` bool mask of a lat/lon grid's pixels inside ``bbox``.

    ``lat`` / ``lon`` are 1-D coordinate vectors and ``bbox`` is
    ``[lon_min, lon_max, lat_min, lat_max]``.  Longitudes are normalized to
    [-180, 180] so the test is independent of the grid's lon convention.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_arr = np.asarray(lat)
    lon_norm = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    lon_min_n = ((lon_min + 180.0) % 360.0) - 180.0
    lon_max_n = ((lon_max + 180.0) % 360.0) - 180.0
    in_lat = (lat_arr >= lat_min) & (lat_arr <= lat_max)
    in_lon = (lon_norm >= lon_min_n) & (lon_norm <= lon_max_n)
    return in_lat[:, None] & in_lon[None, :]


def _validate_view_bbox(view_bbox: List[float], impact_bbox: List[float]) -> None:
    """Sanity-check the plotting view box against the impact box.

    ``view_bbox`` is the field of view the saved snapshots are cropped to;
    ``impact_bbox`` is the RMSE / timeseries domain drawn as a dashed
    rectangle inside it.  We require a well-ordered, non-dateline-crossing
    view and only *warn* when it does not fully contain the impact box,
    since a view that clips the box still renders — the dashed rectangle
    just runs off-panel.
    """
    v_lon_min, v_lon_max, v_lat_min, v_lat_max = view_bbox
    if v_lon_min >= v_lon_max:
        raise SystemExit(
            f"ERROR: --view-bbox lon_min ({v_lon_min}) >= lon_max ({v_lon_max}); "
            f"dateline-crossing views are not supported."
        )
    if v_lat_min >= v_lat_max:
        raise SystemExit(
            f"ERROR: --view-bbox lat_min ({v_lat_min}) >= lat_max ({v_lat_max})."
        )
    i_lon_min, i_lon_max, i_lat_min, i_lat_max = impact_bbox
    if (
        v_lon_min > i_lon_min
        or v_lon_max < i_lon_max
        or v_lat_min > i_lat_min
        or v_lat_max < i_lat_max
    ):
        print(
            f"[free_end_states] WARNING: --view-bbox {view_bbox} does not fully "
            f"contain the impact --bbox {list(impact_bbox)}; the dashed box may "
            f"extend past the panel edge."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensemble-zarr", required=True)
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end"],
        help="Conditioning mode (start or end); 'both' has no free end and "
        "is not a valid input for this diagnostic.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="msl",
        help="Variable to analyse (default: msl).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Impact box in degrees (cartopy convention): the RMSE ranking, "
        "the bbox-mean timeseries, and the dashed box on the plots all use "
        "this region.",
    )
    parser.add_argument(
        "--view-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Plotting field of view in degrees (cartopy convention).  The "
        "saved member / ERA5 snapshots are cropped to this wider box so the "
        "aggregator can show synoptic context around the impact box (matching "
        "the synoptic-PCA precursor panels); the RMSE + timeseries reductions "
        "still use --bbox.  Defaults to --bbox (panels clip to the impact box).",
    )
    parser.add_argument(
        "--mask",
        choices=("none", "land", "sea"),
        default="none",
        help=(
            "Optionally restrict the bbox RMSE + bbox-mean reductions to a "
            "subset of HPX pixels: 'land' keeps only centers on land (useful "
            "for surface-temperature variables on bboxes with substantial "
            "ocean — ocean t2m is ~SST and barely moves on the lead window, "
            "diluting the ranking), 'sea' keeps only ocean pixels, 'none' "
            "(default) uses every pixel inside the bbox."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of low (coolest/lowest) and high (warmest/highest) members "
        "to record per tail (default: 3).",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="",
        help="Display name forwarded to the aggregator for plot titles.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Ensemble IC as ISO 8601 UTC (forwarded to the aggregator).",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for the lead-axis (forwarded to the aggregator).",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=None,
        help="Number of members to process (default: all in the zarr).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The plotting field of view defaults to the impact box (panels clip to
    # it, the original behaviour); when a wider --view-bbox is given the saved
    # snapshots are cropped to it instead so the aggregator can show synoptic
    # context around the dashed impact box.
    view_bbox = list(args.view_bbox) if args.view_bbox is not None else list(args.bbox)
    _validate_view_bbox(view_bbox, args.bbox)

    print(f"[free_end_states] case        : {args.case_name}")
    print(f"[free_end_states] mode        : {args.mode}")
    print(f"[free_end_states] ensemble    : {args.ensemble_zarr}")
    print(f"[free_end_states] era5        : {args.era5_zarr}")
    print(f"[free_end_states] output dir  : {output_dir}")
    print(f"[free_end_states] var_name    : {args.var_name}")
    print(f"[free_end_states] bbox        : {args.bbox}")
    print(f"[free_end_states] view bbox   : {view_bbox}")
    print(f"[free_end_states] mask        : {args.mask}")
    print(f"[free_end_states] top-k       : {args.top_k}")

    if not os.path.isdir(args.ensemble_zarr):
        raise SystemExit(f"ERROR: Ensemble zarr does not exist: {args.ensemble_zarr}")
    if not os.path.isdir(args.era5_zarr):
        raise SystemExit(f"ERROR: ERA5 zarr does not exist: {args.era5_zarr}")

    ens_root = zarr.open_group(args.ensemble_zarr, mode="r")
    _require_hpx_ensemble(ens_root, args.ensemble_zarr)
    lead_times = np.array(ens_root["lead_time"][:])
    n_leads = len(lead_times)
    free_end = _free_end_frame(args.mode, n_leads)
    free_end_h = float(lead_hours(lead_times)[free_end])
    cond_frames = list(
        ens_root.attrs.get("conditioning_frames", MODE_FRAMES[args.mode])
    )
    if free_end in cond_frames:
        raise SystemExit(
            f"ERROR: derived free-end frame {free_end} is in the "
            f"zarr's conditioning_frames {cond_frames}; refusing to proceed."
        )
    hpx_level = _hpx_level_from_root(ens_root)
    print(f"[free_end_states] free end    : frame {free_end} (+{free_end_h:g}h)")
    print(f"[free_end_states] cond frames : {cond_frames}")
    print(f"[free_end_states] hpx_level   : {hpx_level}")

    t0 = time.time()

    print("[free_end_states] loading ensemble HPX field (all leads) ...", flush=True)
    members_all = _load_ensemble_hpx_all_frames(
        args.ensemble_zarr,
        args.var_name,
        args.n_members,
    )  # [N, n_leads, n_pix]
    n_members = int(members_all.shape[0])
    n_pix = int(members_all.shape[-1])
    members = members_all[:, free_end, :]  # [N, n_pix] at the free-end frame

    print(
        "[free_end_states] regridding ERA5 lat/lon -> HPX (all leads) ...", flush=True
    )
    era5_hpx_all = _regrid_era5_latlon_to_hpx_all_frames(
        args.era5_zarr,
        args.var_name,
        hpx_level,
    )  # [n_leads, n_pix]
    era5_hpx = era5_hpx_all[free_end]  # [n_pix] at the free-end frame

    print("[free_end_states] building HPX bbox mask ...", flush=True)
    pix_lat, pix_lon = _hpx_pixel_centers(hpx_level)
    mask = _hpx_bbox_mask(pix_lat, pix_lon, args.bbox)
    # Hold onto the SST HPX tensor when masking is active so we can
    # round-trip a single frame back to lat/lon alongside the member
    # fields below (giving the aggregator a bbox-clipped lat/lon
    # excluded-region mask consistent with the HPX metric mask, with
    # no second regridder build).
    sst_hpx_full: torch.Tensor | None = None
    if args.mask != "none":
        print(
            f"[free_end_states] regridding ERA5 SST for {args.mask} mask "
            f"(ERA5 SST NaN) ...",
            flush=True,
        )
        sst_hpx_full = _regrid_era5_latlon_to_hpx_all_frames(
            args.era5_zarr, "sst", hpx_level
        )
        land = _hpx_land_mask_from_sst(sst_hpx_full)
        mask = mask & (land if args.mask == "land" else ~land)
        if not mask.any():
            raise SystemExit(
                f"ERROR: bbox {args.bbox} ∩ {args.mask}-mask matched zero "
                f"HPX pixel centers; widen the bbox or use --mask none."
            )
    n_pix_in_box = int(mask.sum())
    print(
        f"[free_end_states] n_pix in box: {n_pix_in_box}/{n_pix} "
        f"({100.0 * n_pix_in_box / n_pix:.2f}%) [mask={args.mask}]"
    )

    print(
        "[free_end_states] computing per-member bbox domain-mean (all leads) ...",
        flush=True,
    )
    member_scalar = _hpx_domain_mean(members_all, mask)  # [N, n_leads]
    era5_scalar = _hpx_domain_mean(era5_hpx_all, mask)  # [n_leads]
    member_scalar_np = member_scalar.cpu().numpy().astype(np.float32)
    era5_scalar_np = era5_scalar.cpu().numpy().astype(np.float32)
    ensemble_mean_scalar_np = member_scalar_np.mean(axis=0).astype(np.float32)

    # Rank members by their *signed* box-mean departure from ERA5 at the free
    # end (member box-mean minus ERA5 box-mean), not by RMSE.  The two answer
    # different questions: RMSE is an unsigned distance (a verification metric,
    # already covered by the spread/RMSE/CRPS suite), whereas the signed bias
    # keeps the *direction* of the departure -- for a heatwave, whether a
    # member runs hotter or cooler than observed.  Selecting the low tail (most
    # negative bias: coolest/lowest) and high tail (most positive bias:
    # warmest/highest) gives a consistent physical axis across conditioning
    # modes and makes no "closest = best match" claim, so it cannot be fooled
    # by the spatial cancellation that a near-zero-bias-but-structurally-wrong
    # member would slip past |bias|.  member_scalar already holds the masked
    # box-mean at every lead, so the free-end bias is just a subtraction.
    print("[free_end_states] ranking members by signed free-end bias ...", flush=True)
    member_bias_np = (member_scalar_np[:, free_end] - era5_scalar_np[free_end]).astype(
        np.float32
    )
    order_asc = np.argsort(member_bias_np)  # ascending: coolest (most negative) first
    k = min(args.top_k, n_members // 2) if n_members >= 2 else n_members
    if k != args.top_k:
        print(
            f"[free_end_states] WARNING: capped top-k from {args.top_k} to {k} "
            f"(N={n_members})"
        )
    low_idx = order_asc[:k].astype(np.int32)  # coolest / lowest (most negative bias)
    high_idx = order_asc[::-1][:k].astype(np.int32)  # warmest / highest (most positive)

    # RMSE is still computed and saved as a secondary unsigned-distance field
    # so the figure can report it and any RMSE consumer keeps working.
    print(
        "[free_end_states] computing per-member RMSE on HPX (secondary) ...", flush=True
    )
    rmse = _hpx_rmse(members, era5_hpx, mask)
    rmse_np = rmse.cpu().numpy().astype(np.float32)

    # Regrid the selected K low-tail / K high-tail member HPX fields and the
    # HPX ERA5 back to lat/lon for plotting, cropped to the (wider) view box.
    # Doing it after ranking keeps the regrid cost bounded by ~(2K + 1)
    # fields rather than N.  When masking is active, we tack the SST frame
    # onto the same batch so a single regridder build handles both the
    # plotted fields and the lat/lon excluded-region mask the aggregator
    # hatches.
    print("[free_end_states] regridding selected HPX fields -> lat/lon ...", flush=True)
    selected_pieces = [
        members.index_select(dim=0, index=torch.from_numpy(low_idx).long()),
        members.index_select(dim=0, index=torch.from_numpy(high_idx).long()),
        era5_hpx.unsqueeze(0),
    ]
    if sst_hpx_full is not None:
        selected_pieces.append(sst_hpx_full[0:1])
    selected_hpx = torch.cat(selected_pieces, dim=0)  # [2K + 1 (+1), n_pix]
    selected_latlon, lat_box, lon_box = _hpx_fields_to_latlon_bbox(
        selected_hpx, hpx_level, view_bbox
    )
    low_fields = selected_latlon[:k]
    high_fields = selected_latlon[k : 2 * k]
    era5_field = selected_latlon[2 * k]
    excluded_mask_box: np.ndarray | None = None
    if sst_hpx_full is not None:
        sst_box = selected_latlon[2 * k + 1]
        # SST is NaN over land in ERA5; the lat/lon → HPX → lat/lon
        # round-trip propagates NaN at every step, so isnan(sst_box)
        # gives a land mask on the *same* (view) grid the saved fields live
        # on.  Excluded region = pixels NOT used in the metric:
        #   mask=land → metric uses land    → excluded = ocean = ~is_land
        #   mask=sea  → metric uses ocean   → excluded = land  =  is_land
        is_land_box = np.isnan(sst_box)
        excluded = ~is_land_box if args.mask == "land" else is_land_box
        # The metric only spans the impact box, so confine the hatched
        # "excluded" overlay to it: outside the box every pixel is already
        # outside the metric (the dashed rectangle conveys that), and hatching
        # the whole synoptic view would swamp the panel.
        in_impact = _latlon_in_bbox_mask(lat_box, lon_box, list(args.bbox))
        excluded_mask_box = (excluded & in_impact).astype(bool)
    print(
        f"[free_end_states] view lat/lon shape: "
        f"{low_fields.shape[1]} x {low_fields.shape[2]}"
    )

    elapsed = time.time() - t0
    print(f"[free_end_states] done in {elapsed:.1f}s")
    print(
        f"[free_end_states] bias min/median/max  : "
        f"{member_bias_np.min():+.4g} / {np.median(member_bias_np):+.4g} / "
        f"{member_bias_np.max():+.4g}"
    )
    print(
        f"[free_end_states] RMSE min/median/max  : "
        f"{rmse_np.min():.4g} / {np.median(rmse_np):.4g} / {rmse_np.max():.4g}"
    )

    lead_hours_all = lead_hours(lead_times).astype(np.float64)

    npz_path = output_dir / f"free_end_states_{args.mode}.npz"
    save_kwargs: dict = dict(
        # Signed free-end bias is the headline ranking metric; member_rmse is
        # kept as a secondary unsigned distance.  'low' = coolest / lowest
        # (most negative member-ERA5 bias), 'high' = warmest / highest.
        member_bias=member_bias_np,
        member_rmse=rmse_np,
        low_idx=low_idx,
        high_idx=high_idx,
        low_bias=member_bias_np[low_idx].astype(np.float32),
        high_bias=member_bias_np[high_idx].astype(np.float32),
        low_rmse=rmse_np[low_idx].astype(np.float32),
        high_rmse=rmse_np[high_idx].astype(np.float32),
        low_fields=low_fields,
        high_fields=high_fields,
        era5_field=era5_field,
        member_scalar=member_scalar_np,
        era5_scalar=era5_scalar_np,
        ensemble_mean_scalar=ensemble_mean_scalar_np,
        lead_hours_all=lead_hours_all,
        lat=lat_box.astype(np.float64),
        lon=lon_box.astype(np.float64),
        bbox=np.asarray(args.bbox, dtype=np.float64),
        view_bbox=np.asarray(view_bbox, dtype=np.float64),
        n_pix_in_box=np.int64(n_pix_in_box),
        mask_kind=np.array(args.mask),
        hpx_level=np.int32(hpx_level),
        free_end_frame=np.int32(free_end),
        free_end_hours=np.float64(free_end_h),
        conditioning_frames=np.array(cond_frames, dtype=object),
        n_members=np.int64(n_members),
        top_k=np.int32(k),
        var_name=np.array(args.var_name),
        case_name=np.array(args.case_name),
        mode=np.array(args.mode),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
    )
    if excluded_mask_box is not None:
        save_kwargs["excluded_mask_box"] = excluded_mask_box
    np.savez(npz_path, **save_kwargs)
    print(f"[free_end_states] wrote {npz_path}")
    print("[free_end_states] done.")


if __name__ == "__main__":
    main()
