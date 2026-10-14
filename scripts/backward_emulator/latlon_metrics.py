"""Grid-agnostic lat/lon ensemble metrics: spread, skill, member-RMSE, CRPS.

This is the **shared reducer** for the spread/RMSE/CRPS comparison: both the
makani ensemble (native 0.25 deg) and the cBottle ensemble (regridded
HPX->lat/lon) feed the *same* function, so any difference in the numbers is a
difference in the models, not the analysis.  It replaces the HPX equal-area
reduction of ``custom_utils.diagnostics.diagnostics_from_zarr`` with a cos(lat)
area-weighted one, keeping every metric definition identical:

* **spread** -- sqrt of the area-weighted mean over pixels of the unbiased
  (``ddof=1``) across-member sample variance;
* **skill** -- area-weighted RMSE of the ensemble mean vs truth (denominator of
  the spread/skill ratio);
* **member_rmse** -- joint RMS over (member, pixel) errors -- the typical-member
  magnitude;
* **mean_error** -- signed area-weighted mean error of the ensemble mean;
* **crps** -- empirical CRPS (fair energy form), area-weighted over pixels.

Sqrt is applied after averaging (so Jensen's inequality doesn't depress the
scalars), and the exact bias-variance identity holds by construction:
``member_rmse**2 == ((N-1)/N) * spread**2 + skill**2``.

Numpy only -- the reduction is light and stays portable/testable; the cBottle
side hands in numpy lat/lon arrays too.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Region = Tuple[float, float, float, float]  # (lon_min, lon_max, lat_min, lat_max)


def cos_lat_weights(lat: np.ndarray) -> np.ndarray:
    """cos(lat) latitude weights (>= 0), the standard area measure on a
    regular lat/lon grid.  Not normalized -- callers normalize over the pixels
    they keep."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return np.clip(w, 0.0, None)


def _lon_to_pm180(lon: np.ndarray) -> np.ndarray:
    """Map longitudes into [-180, 180) so a box in that convention selects
    correctly regardless of whether the grid is stored [0,360) or [-180,180)."""
    lon = np.asarray(lon, dtype=np.float64)
    return np.where(lon >= 180.0, lon - 360.0, lon)


def select_pixels(
    lat: np.ndarray,
    lon: np.ndarray,
    region: Optional[Region] = None,
    weight_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_pixel_indices, weights)`` for a lat/lon grid.

    ``region`` is ``(lon_min, lon_max, lat_min, lat_max)`` in degrees with lon in
    [-180, 180); ``lon_min > lon_max`` spans the dateline.  ``weight_mask`` is an
    optional boolean ``(H, W)`` (e.g. land) that further restricts the domain.
    Weights are cos(lat), normalized to sum to 1 over the kept pixels.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    H, W = lat.shape[0], lon.shape[0]
    wlat = cos_lat_weights(lat)  # (H,)
    w2d = np.broadcast_to(wlat[:, None], (H, W)).copy()  # (H, W)

    keep = np.ones((H, W), dtype=bool)
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        plon = _lon_to_pm180(lon)  # (W,)
        if lon_min <= lon_max:
            lon_mask = (plon >= lon_min) & (plon <= lon_max)
        else:  # wrapped box across the dateline
            lon_mask = (plon >= lon_min) | (plon <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        keep &= lat_mask[:, None] & lon_mask[None, :]
    if weight_mask is not None:
        keep &= np.asarray(weight_mask, dtype=bool)

    flat = np.where(keep.reshape(-1))[0]
    if flat.size == 0:
        raise ValueError(
            f"region {region} + mask select no pixels on this {H}x{W} grid."
        )
    w = w2d.reshape(-1)[flat]
    w = w / w.sum()
    return flat, w


def _empirical_crps(ens_sel: np.ndarray, truth_sel: np.ndarray) -> np.ndarray:
    """Per-pixel empirical CRPS, vectorized -- shape ``(L, P)``.

    CRPS = mean_i |x_i - y| - 1/(2 N^2) sum_{i,j} |x_i - x_j|.  The pairwise term
    is evaluated in O(N log N) via the sorted identity
    ``sum_{i,j}|x_i-x_j| = 2 sum_i (2i - N - 1) x_(i)``.
    """
    N = ens_sel.shape[0]
    term1 = np.abs(ens_sel - truth_sel[None]).mean(axis=0)  # (L, P)
    xs = np.sort(ens_sel, axis=0)                           # ascending over N
    i = np.arange(1, N + 1, dtype=np.float64)[:, None, None]  # (N,1,1)
    term2 = ((2.0 * i - N - 1.0) * xs).sum(axis=0) / (N * N)  # (L, P)
    return term1 - term2


def spread_rmse_crps_latlon(
    ens: np.ndarray,
    truth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    region: Optional[Region] = None,
    weight_mask: Optional[np.ndarray] = None,
    conditioning_frames: Optional[np.ndarray] = None,
) -> Dict:
    """Compute the spread/RMSE/CRPS suite for one variable, per lead.

    Parameters
    ----------
    ens : (N, L, H, W) ensemble on lat/lon (physical units).
    truth : (L, H, W) reference on the same grid.
    lat, lon : 1-D degree coordinates.
    region, weight_mask : see :func:`select_pixels`.
    conditioning_frames : lead indices to exclude from the degenerate-pixel
        guard (pinned frames are legitimately zero-variance).

    Returns a dict of ``(L,)`` arrays plus ``n_pix``, ``n_pix_degenerate``.
    """
    ens = np.asarray(ens, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if ens.ndim != 4:
        raise ValueError(f"ens must be (N,L,H,W); got {ens.shape}")
    N, L, H, W = ens.shape
    if truth.shape != (L, H, W):
        raise ValueError(f"truth must be {(L, H, W)}; got {truth.shape}")

    flat, w = select_pixels(lat, lon, region, weight_mask)
    ens_sel = ens.reshape(N, L, H * W)[:, :, flat]     # (N, L, P)
    truth_sel = truth.reshape(L, H * W)[:, flat]        # (L, P)
    return reduce_selected(ens_sel, truth_sel, w,
                           conditioning_frames=conditioning_frames)


def reduce_selected(
    ens_sel: np.ndarray,
    truth_sel: np.ndarray,
    weights: np.ndarray,
    *,
    conditioning_frames: Optional[np.ndarray] = None,
    alive: Optional[np.ndarray] = None,
) -> Dict:
    """Core reduction on pre-selected pixels: ``ens_sel (N,L,P)``, ``truth_sel
    (L,P)``, ``weights (P,)`` (summing to 1).  Shared by the full-array entry
    point and the memory-safe source path so both models reduce identically.

    ``alive`` is an optional ``(N, L)`` boolean survivor mask (makani's censoring:
    a diverged member is dead from its first failing lead onward).  When given,
    every statistic at lead ``l`` is over the members alive at ``l`` only, and the
    per-lead ``survival`` fraction is returned.  cBottle passes ``alive=None`` (all
    members valid), recovering the plain all-member reduction bit-for-bit.
    """
    ens_sel = np.asarray(ens_sel, dtype=np.float64)
    truth_sel = np.asarray(truth_sel, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    N, L, P = ens_sel.shape

    # Degenerate-pixel guard is a domain/variable property, not a divergence one,
    # so it is judged over ALL members (mirrors diagnostics_from_zarr).
    sample_var_all = ens_sel.var(axis=0, ddof=1)  # (L, P)
    cond = set(int(c) for c in (conditioning_frames if conditioning_frames is not None else []))
    free = [i for i in range(L) if i not in cond]
    n_deg = 0
    if free:
        pix_var = sample_var_all[free].max(axis=0)
        ref = float(pix_var.max())
        if ref > 0:
            n_deg = int((pix_var <= 1e-6 * ref).sum())

    if alive is None:
        ens_mean = ens_sel.mean(axis=0)
        spread = np.sqrt((w * sample_var_all).sum(axis=1))
        skill = np.sqrt((w * (ens_mean - truth_sel) ** 2).sum(axis=1))
        member_rmse = np.sqrt((w * ((ens_sel - truth_sel[None]) ** 2).mean(axis=0)).sum(axis=1))
        mean_error = (w * (ens_mean - truth_sel)).sum(axis=1)
        crps = (w * _empirical_crps(ens_sel, truth_sel)).sum(axis=1)
        n_alive = np.full(L, N, dtype=np.int64)
    else:
        alive = np.asarray(alive, dtype=bool)
        if alive.shape != (N, L):
            raise ValueError(f"alive must be {(N, L)}; got {alive.shape}")
        spread = np.full(L, np.nan)
        skill = np.full(L, np.nan)
        member_rmse = np.full(L, np.nan)
        mean_error = np.full(L, np.nan)
        crps = np.full(L, np.nan)
        n_alive = alive.sum(axis=0).astype(np.int64)
        for l in range(L):
            m = alive[:, l]
            nl = int(m.sum())
            if nl == 0:
                continue
            el = ens_sel[m, l, :]          # (nl, P)
            tl = truth_sel[l, :]           # (P,)
            mean_l = el.mean(axis=0)
            err = mean_l - tl
            skill[l] = np.sqrt((w * err ** 2).sum())
            mean_error[l] = (w * err).sum()
            member_rmse[l] = np.sqrt((w * ((el - tl[None]) ** 2).mean(axis=0)).sum())
            crps[l] = (w * _empirical_crps(el[:, None, :], tl[None, :])[0]).sum()
            if nl >= 2:
                spread[l] = np.sqrt((w * el.var(axis=0, ddof=1)).sum())

    return {
        "spread": spread,
        "skill": skill,
        "member_rmse": member_rmse,
        "mean_error": mean_error,
        "crps": crps,
        "n_members": N,
        "n_alive": n_alive,
        "survival": n_alive.astype(np.float64) / float(N),
        "n_pix": int(P),
        "n_pix_degenerate": n_deg,
    }


def spread_rmse_crps_from_source(
    source,
    variable: str,
    *,
    region: Optional[Region] = None,
    weight_mask: Optional[np.ndarray] = None,
    conditioning_frames: Optional[np.ndarray] = None,
    alive: Optional[np.ndarray] = None,
) -> Dict:
    """Memory-safe reduction for a field source (``lat``/``lon``/``load_region``).

    Resolves the region pixels + cos-lat weights on the source grid, streams only
    those pixels via ``source.load_region`` (a box is ~GB, not ~50 GB), and runs
    the identical :func:`reduce_selected`.  ``alive`` (N, L) applies survivor
    censoring (makani); ``None`` reduces over all members (cBottle).  This is the
    path the driver uses at case-study scale.
    """
    flat, w = select_pixels(source.lat, source.lon, region, weight_mask)
    ens_sel, truth_sel = source.load_region(variable, flat)
    return reduce_selected(ens_sel, truth_sel, w,
                           conditioning_frames=conditioning_frames, alive=alive)
