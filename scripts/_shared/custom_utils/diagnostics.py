"""Diagnostic utilities for analyzing atmospheric fields.

Functions here operate on single fields (not prediction-vs-truth comparisons)
and produce rich outputs like spectra.  They are meant to be model-agnostic
and reusable across notebooks and scripts.

The primary data convention follows the rest of ``custom_utils``:
  * **data** – ``torch.Tensor``
  * **coords** – ``collections.OrderedDict`` with string keys
    (e.g. ``time``, ``lead_time``, ``variable``, ``lat``, ``lon``)

Spectrum convention
-------------------
:func:`zonal_spectrum` uses ``norm="forward"``, which divides FFT
coefficients by N, so ``|fft|² = |X|²/N²``.  Only the longitude
axis is transformed (it is genuinely periodic); the meridional
direction is omitted because latitude is a non-periodic bounded
domain for which FFT is ill-posed.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import zarr

from earth2studio.statistics import crps as _es_crps
from earth2studio.statistics import rmse as _es_rmse

EARTH_RADIUS_M = 6.371e6


# ---------------------------------------------------------------------------
# Zonal power spectrum (1-D along longitude, averaged over latitude)
# ---------------------------------------------------------------------------


def zonal_spectrum(
    data: torch.Tensor,
    coords: OrderedDict,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
) -> tuple[np.ndarray, torch.Tensor, OrderedDict]:
    """Compute the 1-D zonal power spectrum averaged over latitude.

    The spectrum is computed via ``rfft`` along the longitude axis (which is
    genuinely periodic), scaled by Earth's circumference at each latitude
    following the WeatherBench2 convention, and then averaged uniformly over
    latitude.  The circumference factor ``2πR cos(lat)`` provides an effective
    ``cos_lat`` weighting without a redundant second application.

    Longitude is the natural axis for this computation because it is
    physically periodic.  Latitude is a bounded non-periodic domain, making
    FFT-based meridional spectra ill-posed; they are not computed here.

    Parameters
    ----------
    data : torch.Tensor
        Input field.  The last two axes must correspond to *spatial_dims*.
    coords : OrderedDict
        Coordinate system describing *data*.
    spatial_dims : tuple[str, str], optional
        Names of the two spatial dimensions (default ``("lat", "lon")``).

    Returns
    -------
    wavenumbers : np.ndarray
        1-D array of zonal wavenumber bins ``k = 0, 1, …, nx//2``.
        Units are dimensionless cycles per grid-point.
    spectrum : torch.Tensor
        Squared magnitude of the forward-normalized (÷ ``nx``) rfft along
        longitude, multiplied by circumference ``2πR cos(lat)`` and averaged
        uniformly over latitude.  Equals ``(PSD_zonal / nx) × circumference``;
        using ``norm="ortho"`` instead would give true ``PSD × circumference``.
        Shape ``(*batch_dims, n_kx)``.  **Not k-weighted; do not pass to**
        :func:`spectral_slope`.
    out_coords : OrderedDict
        New coordinate system with spatial dims replaced by ``wavenumber``.
    """
    dim_names = list(coords.keys())
    assert dim_names[-2:] == list(spatial_dims), (
        f"Expected spatial dims {list(spatial_dims)} as the last two axes, "
        f"got {dim_names[-2:]}"
    )

    nx = data.shape[-1]
    lat_name, lon_name = spatial_dims

    # rfft along lon (periodic axis)
    fft_zonal = torch.fft.rfft(data, dim=-1, norm="forward")
    power_zonal = fft_zonal.real**2 + fft_zonal.imag**2

    # Factor-of-2 correction for interior rfft bins
    n_kx = power_zonal.shape[-1]
    doubling_x = torch.ones(n_kx, device=data.device)
    doubling_x[1 : n_kx - (1 if nx % 2 == 0 else 0)] = 2.0
    power_zonal = power_zonal * doubling_x

    # Circumference at each latitude: C(lat) = 2π R cos(lat).
    # Multiplying by circumference converts to physical units (m · field_unit²)
    # and gives an effective cos_lat weighting when averaged over latitude.
    lat_vals = coords[lat_name]
    cos_lat = torch.tensor(
        np.cos(np.deg2rad(np.asarray(lat_vals))),
        dtype=data.dtype,
        device=data.device,
    )
    circumference = 2 * np.pi * EARTH_RADIUS_M * cos_lat  # (ny,)
    power_zonal = power_zonal * circumference.unsqueeze(-1)  # (..., ny, n_kx)

    # Uniform average over lat: circumference already encodes cos_lat weighting.
    # A separate cos_lat averaging weight would produce cos²_lat suppression.
    spectrum = power_zonal.mean(dim=-2)

    wavenumbers = np.arange(n_kx)

    out_coords = OrderedDict()
    for k, v in coords.items():
        if k not in spatial_dims:
            out_coords[k] = v
    out_coords["wavenumber"] = wavenumbers

    return wavenumbers, spectrum, out_coords


# ---------------------------------------------------------------------------
# Domain mask on HEALPix
# ---------------------------------------------------------------------------
#
# Canonical implementation for the whole repo.  ``compute_synoptic_pca.py``
# delegates to these so the EOF analysis and the spread/RMSE/CRPS battery
# mask the *same* pixels for a given case region -- if the two ever drift,
# the free-end spread ratios computed by the two pipelines stop being
# comparable, which is a keypoint-level problem (keypoint 2 quotes them).


def hpx_pixel_centers(hpx_level: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) of NEST-ordered HEALPix pixel centers in degrees.

    Lon is wrapped to match user-supplied bounds.  The wrap is
    ``lon > 180 -> lon - 360``, so the half-open interval is ``(-180, 180]``:
    a pixel sitting exactly on the antimeridian stays at ``+180`` and will
    NOT be picked up by a box whose western edge is ``-180``.  None of the
    case regions in ``scripts/_shared/cases/`` touches the antimeridian, so
    this is latent rather than active -- but a box with an edge at +-180
    should use the wrapped form (``lon_min > lon_max``) to be safe.

    Behaviour is preserved verbatim from the original implementation in
    ``compute_synoptic_pca.py``; do not "fix" the boundary without
    re-checking the EOF numbers §3 of PAPER_PLAN.md locks, which were
    produced with this exact masking.
    """
    import earth2grid  # local import keeps top-of-file deps cheap

    grid = earth2grid.healpix.Grid(
        hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    lat = np.asarray(grid.lat).astype(np.float64)
    lon = np.asarray(grid.lon).astype(np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon


def domain_hpx_indices(
    hpx_level: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Pixel indices (NEST ordering) inside the lat/lon box, in degrees."""
    pix_lat, pix_lon = hpx_pixel_centers(hpx_level)
    if lon_min <= lon_max:
        lon_mask = (pix_lon >= lon_min) & (pix_lon <= lon_max)
    else:
        # Wrapped box (e.g. [170, -170] for the dateline): union the two halves.
        lon_mask = (pix_lon >= lon_min) | (pix_lon <= lon_max)
    lat_mask = (pix_lat >= lat_min) & (pix_lat <= lat_max)
    return np.where(lon_mask & lat_mask)[0]


# ---------------------------------------------------------------------------
# Land / sea mask on HEALPix
# ---------------------------------------------------------------------------
#
# Canonical implementation for the whole repo, alongside the box helpers
# above.  ``compute_free_end_states.py`` and ``compute_synoptic_pca.py`` both
# delegate here.  These three diagnostics are *required* to agree
# pixel-for-pixel: `pnw_heatwave.yaml` reduces t2m over ``regions.impact``
# with ``mask: {t2m: land}`` in free_end_states AND averages the same pixels
# for the synoptic-PCA scalar impact ("one shared averaging, never two" --
# scripts/_shared/README.md), and both numbers reach the manuscript.  Two
# copies of this logic is two chances for those figures to disagree.


def hpx_land_mask_from_sst_hpx(sst_hpx: torch.Tensor) -> np.ndarray:
    """Boolean HPX land mask from an already-regridded ERA5 SST field.

    ERA5 reports SST as NaN over land (SST is undefined there); the
    land/sea geometry is time-invariant so any single regridded frame
    suffices, and this reads frame 0.  Sourcing the mask from the ERA5
    reference itself keeps the land/sea convention provably consistent
    with what the diagnostic compares against -- there is no way for the
    mask to disagree with the ground truth, as a separate
    Natural-Earth-style source could.

    The earth2grid lat/lon -> HPX regridder is a sparse linear
    interpolator, so a NaN in any contributing source cell propagates to
    NaN in the HPX output.  Coastal HPX pixels whose stencil touches even
    one land ERA5 cell therefore classify as land -- a *conservative*
    boundary treatment that is the right call here (coastal pixels are
    physically more land-like for a heatwave diagnostic, and the ``sea``
    complement just trims that same strip).

    Takes the regridded tensor rather than a store path so a caller that
    already built the lat/lon -> HPX regridder can reuse it, and so
    ``compute_free_end_states.py`` can round-trip the same tensor back to
    lat/lon for its hatched-region overlay without building the regridder
    a second time.  Accepts ``[n_pix]``, ``[n_leads, n_pix]``, or the
    ``[1, 1, 1, n_pix]`` shape the regridder returns for a single frame.
    """
    frame = sst_hpx if sst_hpx.ndim == 1 else sst_hpx[0]
    return torch.isnan(frame).reshape(-1).cpu().numpy()


def hpx_land_mask_from_era5(era5_root: zarr.Group, hpx_level: int) -> np.ndarray:
    """Boolean HPX land mask derived from an ERA5 store's SST NaN pattern.

    Convenience wrapper for callers that have the store but no regridded
    SST in hand: regrids a single ERA5 SST frame to HPX and delegates to
    :func:`hpx_land_mask_from_sst_hpx`, which documents the convention.
    """
    # Lazy imports: both pull in the heavy regridding stack, which we don't
    # want loaded just to read this module.
    from custom_utils.ensemble_loading import variable_index
    from custom_utils.interpolation_utils import CBottleUtils

    sst_idx = variable_index(era5_root, "sst")
    sst_latlon = np.array(era5_root["data"][0, 0, sst_idx, :, :], dtype=np.float32)
    sst_tensor = torch.from_numpy(sst_latlon)[None, None, None]  # [1,1,1,lat,lon]
    coords = OrderedDict(
        {
            "time": np.array([0]),
            "lead_time": np.array([0]),
            "variable": np.array(["sst"]),
            "lat": np.array(era5_root["lat"][:]),
            "lon": np.array(era5_root["lon"][:]),
        }
    )
    sst_hpx, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
        sst_tensor, coords, hpx_level=hpx_level
    )
    return hpx_land_mask_from_sst_hpx(sst_hpx)


# ---------------------------------------------------------------------------
# Ensemble diagnostics from zarr
# ---------------------------------------------------------------------------


def diagnostics_from_zarr(
    ensemble_zarr_path: str,
    era5_zarr_path: str,
    var_name: str,
    n_members: int | None = None,
    swap_era5_on_conditioned: bool = False,
    region: tuple[float, float, float, float] | None = None,
    mask: str = "none",
) -> dict:
    """Compute pointwise ensemble diagnostics on the native HEALPix grid.

    The ensemble store is expected to be on HEALPix; ERA5 (lat/lon) is
    regridded once to the same HEALPix grid so the comparison happens in
    the model's training space, with equal-area weights (uniform per
    pixel).  This is the fairness fix for the ensemble-vs-ERA5 resolution
    mismatch: ERA5's information content is projected once onto the
    HPX-resolvable subspace cBottle was trained against, and then both
    fields are reduced over the ``hpx`` dimension with the natural
    pixel-area measure.

    All quantities are reduced over ``hpx`` with uniform (equal-area)
    weights.  Sqrt operations are applied **after** averaging so the
    returned scalars are not depressed by Jensen's inequality.

    Domain.  With ``region=None`` the reduction is global (all 12*4**level
    pixels), which is what this function did before the ``region``
    parameter existed.  Passing a box restricts every reduction to the
    pixels inside it, via :func:`domain_hpx_indices` -- the same masking
    the EOF pipeline uses -- so a regional run is directly comparable with
    the synoptic-PCA numbers.  Global and regional answers differ
    systematically for a conditioned ensemble: outside the case region the
    conditioning frames pin the field tightly while the model's own error
    persists, which depresses the global spread/skill ratio relative to
    the region where the event actually lives.

    ``mask`` further restricts the domain to land or sea pixels, using the
    same ERA5-SST-derived mask ``compute_free_end_states.py`` and
    ``compute_synoptic_pca.py`` apply, so a masked run here is comparable
    with theirs pixel-for-pixel.  This matters for any surface variable
    whose paper-facing average is masked: `pnw_heatwave.yaml` reduces t2m
    over land only, because ocean t2m is essentially the prescribed
    monthly-mean SST -- see the degenerate-pixel guard below for why that
    is not merely a cosmetic difference in a *spread* diagnostic.

    Degenerate pixels.  A pixel whose across-member variance is ~0 at
    every unconditioned lead contributes nothing to ``spread`` but full
    weight to the pixel mean, so including such pixels pulls the
    spread/skill ratio toward zero.  The usual cause is a field that is a
    prescribed, member-identical boundary condition over part of the
    domain: ``sst`` everywhere, and ``t2m`` over ocean, where it is
    essentially that same SST.  This function counts those pixels, warns
    when it finds any, raises when the whole domain is degenerate, and
    returns the count so the caller can record it.

    Diagnostics:
      * **spread** — sqrt of the mean-over-pixels of the member-wise
        unbiased sample variance (``correction=1``).
      * **skill** — RMSE of the ensemble mean vs ERA5; computed via
        ``earth2studio.statistics.rmse`` with ``ensemble_dimension``.
        Used only as the denominator of the spread/skill ratio in the
        aggregator (not plotted on its own panel).
      * **member_rmse** — Average per-member RMSE: joint RMS over
        ``(member, pixel)`` errors vs ERA5.  This is the operational
        typical-member skill plotted as a headline magnitude in the
        aggregator (an ensemble mean is not itself a forecast, so the
        per-member quantity is more representative of what a single
        realization looks like).
      * **mean_error** — Signed pixel-mean error of the ensemble mean against
        ERA5 (``mean_pix(ens_mean - era5)``).  One scalar per lead-time;
        negative when the ensemble mean is colder/drier/lower than ERA5
        on average over the domain.
      * **crps** — ``earth2studio.statistics.crps`` (empirical CDF),
        reducing over ``hpx``.

    Calibration target.  For a reliable ensemble of N members iid with
    truth, the spread/skill ratio converges to ``sqrt(N/(N+1))``, which
    tends to 1 for large N.  See Fortin et al. (MWR 2014), *Why Should
    Ensemble Spread Match the RMSE of the Ensemble Mean?*

    The bias-variance identity remains exact:
    ``member_rmse^2 = ((N-1)/N) * spread^2 + skill^2``.

    Parameters
    ----------
    ensemble_zarr_path : str
        Path to the ensemble zarr store.  Expected layout:
        ``[N, 1, n_leads, n_vars, n_pix]`` with ``hpx`` coord and
        ``hpx_level`` attr.
    era5_zarr_path : str
        Path to the ERA5 reference zarr store on lat/lon.  Expected
        layout: ``[n_leads, 1, n_vars_era5, n_lat, n_lon]``.
    var_name : str
        Variable to analyse (must exist in both stores).
    n_members : int | None, optional
        Number of ensemble members to use.  If *None*, uses all available.
    swap_era5_on_conditioned : bool, optional
        If *True*, replace the model's output on conditioning frames with
        the regridded ERA5 values before computing diagnostics.  Default
        is *False*.
    region : tuple of float, optional
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees, lon wrapped
        into ``[-180, 180)``.  A wrapped box (``lon_min > lon_max``) spans
        the dateline.  Default *None* reduces globally.
    mask : {"none", "land", "sea"}, optional
        Restrict the domain to land or sea pixels, intersected with
        *region* when both are given.  Default ``"none"`` keeps every
        pixel in the box, which is what this function did before the
        parameter existed.

    Returns
    -------
    dict
        ``lead_times``, ``conditioning_frames``, ``n_members``,
        ``spread``, ``skill``, ``member_rmse``, ``mean_error`` (and ``bias`` as
        deprecated alias), ``crps``, ``region`` (the box or *None*),
        ``mask_kind``, ``n_pix_used`` and ``n_pix_degenerate``.
    """
    if mask not in ("none", "land", "sea"):
        raise ValueError(f"mask must be one of 'none', 'land', 'sea'; got {mask!r}.")
    # Lazy import: pulls in cbottle / earth2grid / cartopy, which we don't
    # want loaded just to read this module.
    from custom_utils.interpolation_utils import CBottleUtils

    ens_root = zarr.open_group(ensemble_zarr_path, mode="r")
    era5_root = zarr.open_group(era5_zarr_path, mode="r")

    if "hpx" not in ens_root.array_keys():
        raise ValueError(
            f"{ensemble_zarr_path}: expected HEALPix ensemble layout (with "
            f"'hpx' coord); got arrays {sorted(ens_root.array_keys())}"
        )
    hpx_level = ens_root.attrs.get("hpx_level")
    if hpx_level is None:
        # Older HPX zarrs may not carry this attr; derive from pixel count.
        n_pix_arr = int(ens_root["hpx"].shape[0])
        hpx_level = int(np.log2(n_pix_arr // 12) / 2)

    total_members = ens_root["data"].shape[0]
    N = min(n_members, total_members) if n_members is not None else total_members

    ens_vars = list(np.array(ens_root["variable"][:]))
    era5_vars = list(np.array(era5_root["variable"][:]))
    ens_var_idx = ens_vars.index(var_name)
    era5_var_idx = era5_vars.index(var_name)

    lead_times = np.array(ens_root["lead_time"][:])
    conditioning_frames = ens_root.attrs.get("conditioning_frames", None)

    # ens_data: [N, n_leads, n_pix]
    ens_data = torch.from_numpy(
        np.array(ens_root["data"][:N, 0, :, ens_var_idx, :])
    ).float()

    # era5_latlon: [n_leads, n_lat, n_lon]
    era5_latlon = torch.from_numpy(
        np.array(era5_root["data"][:, 0, era5_var_idx, :, :])
    ).float()
    era5_lat = np.array(era5_root["lat"][:])
    era5_lon = np.array(era5_root["lon"][:])

    # Regrid ERA5 to HPX once.  ``regrid_latlon_to_healpix_nest`` expects
    # ``[..., nlat, nlon]`` and a coords dict with ``time`` / ``lead_time`` /
    # ``variable`` keys; we synthesise minimal coords because they are only
    # forwarded into the returned hpx_coords (which we don't use).
    era5_for_regrid = era5_latlon.unsqueeze(0).unsqueeze(
        2
    )  # [1, n_leads, 1, n_lat, n_lon]
    era5_coords = OrderedDict(
        {
            "time": np.array([0]),
            "lead_time": lead_times,
            "variable": np.array([var_name]),
            "lat": era5_lat,
            "lon": era5_lon,
        }
    )
    era5_hpx_4d, _ = CBottleUtils.regrid_latlon_to_healpix_nest(
        era5_for_regrid,
        era5_coords,
        hpx_level=hpx_level,
    )
    # Squeeze synthetic time/variable dims back out: [n_leads, n_pix]
    era5 = era5_hpx_4d[0, :, 0, :].cpu()

    if swap_era5_on_conditioned and conditioning_frames is not None:
        for f in conditioning_frames:
            ens_data[:, f] = era5[f]

    # Domain mask.  Applied once, to both fields, before any reduction --
    # every metric below then reduces over the surviving pixels only.
    # Equal-area pixels mean the masked reduction stays a plain mean, so
    # none of the metric math changes.
    sel: np.ndarray | None = None
    if region is not None:
        sel = domain_hpx_indices(hpx_level, *region)
        if sel.size == 0:
            raise ValueError(
                f"region {region} selects no HEALPix pixels at level "
                f"{hpx_level}; check that lon is in [-180, 180)."
            )
    if mask != "none":
        land = hpx_land_mask_from_era5(era5_root, hpx_level)
        keep = land if mask == "land" else ~land
        sel = np.where(keep)[0] if sel is None else sel[keep[sel]]
        if sel.size == 0:
            raise ValueError(
                f"region {region} intersected with the {mask}-mask selects no "
                f"HEALPix pixels at level {hpx_level}; widen the box or use "
                f"mask='none'."
            )
    if sel is not None:
        sel_t = torch.from_numpy(sel).long()
        ens_data = ens_data.index_select(-1, sel_t)
        era5 = era5.index_select(-1, sel_t)

    # Coords for the earth2studio metrics: ensemble + lead_time + hpx for
    # the forecast, and lead_time + hpx for the reference.
    n_pix = int(ens_data.shape[-1])
    hpx_coord = np.arange(n_pix)
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(N),
            "lead_time": lead_times,
            "hpx": hpx_coord,
        }
    )
    y_coords = OrderedDict(
        {
            "lead_time": lead_times,
            "hpx": hpx_coord,
        }
    )

    # Spread: sqrt of mean-over-pixels of unbiased sample variance across
    # members.  correction=1 makes E[var] = sigma^2, treating the members
    # as samples from the predictive distribution (the framing required
    # by spread/skill calibration theory).
    ens_var = ens_data.var(dim=0, correction=1)  # [n_leads, n_pix]
    spread = ens_var.mean(dim=1).sqrt()  # [n_leads]

    # Degenerate-pixel guard (see "Degenerate pixels" in the docstring).
    # Conditioning frames are excluded: every pixel is legitimately pinned
    # there, so counting them would flag every run.  A pixel is degenerate
    # only if it stays flat across members at *every* free lead, judged
    # relative to the most variable pixel in the domain so the test carries
    # no physical units and works for any variable.
    cond_set = set(conditioning_frames or [])
    free_leads = [i for i in range(int(ens_var.shape[0])) if i not in cond_set]
    n_pix_degenerate = 0
    if free_leads:
        pix_var = ens_var[free_leads].amax(dim=0)  # [n_pix]
        ref = float(pix_var.max())
        if ref <= 0.0:
            raise ValueError(
                f"'{var_name}' has zero across-member variance at every "
                f"unconditioned lead over this domain, so spread/skill/CRPS "
                f"are all identically zero. This variable is a "
                f"member-identical field here (a prescribed boundary "
                f"condition such as 'sst'); there is nothing to verify."
            )
        n_pix_degenerate = int((pix_var <= 1e-6 * ref).sum())
        if n_pix_degenerate:
            frac = n_pix_degenerate / int(pix_var.numel())
            print(
                f"[diagnostics] WARNING: {n_pix_degenerate} of "
                f"{int(pix_var.numel())} pixels ({frac:.1%}) have ~zero "
                f"across-member variance at every unconditioned lead for "
                f"'{var_name}'. These carry full weight in the pixel mean "
                f"but contribute no spread, so they bias spread/skill toward "
                f"zero. Likely a member-identical boundary condition over "
                f"part of the domain (ocean t2m is essentially the "
                f"prescribed monthly-mean SST) -- consider mask='land' or a "
                f"tighter region.",
                flush=True,
            )

    # Skill: RMSE of the ensemble mean vs ERA5, via earth2studio.  This
    # is the literature-standard denominator of the spread/skill ratio.
    skill_metric = _es_rmse(
        reduction_dimensions=["hpx"],
        ensemble_dimension="ensemble",
    )
    skill_vals, _ = skill_metric(ens_data, x_coords, era5, y_coords)

    # Per-member RMSE: joint RMS over members and pixels.  This is the
    # headline magnitude an operational reader cares about (an ensemble
    # mean is not itself a realistic forecast).
    sq_err = (ens_data - era5.unsqueeze(0)) ** 2  # [N, n_leads, n_pix]
    member_rmse = sq_err.mean(dim=(0, 2)).sqrt()  # [n_leads]

    # Signed mean error of the ensemble mean: positive = ensemble mean exceeds
    # ERA5 on average over the domain.  Distinct from the magnitude
    # quantities above; preserves sign so the panel reveals systematic
    # over/under-prediction.
    mean_error_vals = (ens_data.mean(dim=0) - era5).mean(dim=1)  # [n_leads]

    # CRPS via earth2studio, reducing over hpx with no weights (each pixel
    # carries the same area, so unweighted == equal-area weighted).
    crps_metric = _es_crps(
        ensemble_dimension="ensemble",
        reduction_dimensions=["hpx"],
    )
    crps_vals, _ = crps_metric(ens_data, x_coords, era5, y_coords)

    return {
        "lead_times": lead_times,
        "conditioning_frames": conditioning_frames,
        "n_members": N,
        "spread": spread,
        "skill": skill_vals,
        "member_rmse": member_rmse,
        "mean_error": mean_error_vals,
        "bias": mean_error_vals,  # Deprecated alias for backward compatibility
        "crps": crps_vals,
        "region": region,
        "mask_kind": mask,
        "n_pix_used": n_pix,
        "n_pix_degenerate": n_pix_degenerate,
    }
