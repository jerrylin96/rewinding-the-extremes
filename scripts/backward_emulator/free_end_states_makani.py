"""makani free-end-states producer (paper Figure-1 analog input).

Writes a ``free_end_states_makani_<case>.npz`` in the SAME schema as the paper's
``compute_free_end_states.py`` so the paper's ``aggregate_free_end_states.make_figure``
can render a makani block indistinguishable in layout from its cBottle block (our
comparison aggregator reuses that function verbatim).  makani is native 0.25 deg
lat/lon -- no HEALPix round trip; its own resolution-matched baseline is native ERA5.

Following ``dispatch_free_end_states.py``, the spatial panels are cropped to the wider
**view** region (``regions.synoptic``) with the **impact** region (``regions.impact``)
drawn as a dashed inset, while the signed-bias ranking + the domain-mean time series
reduce over the (land-masked) impact box.  On the impact-box grid the metric is a
cos-lat land-mean; ocean pixels inside the impact box are flagged in
``excluded_mask_box`` for the aggregator's hatch.

What it bakes, at the free-end frame (forward frame 0, the 66 h-back window start):

* ``era5_field`` -- native-ERA5 field on the view grid;
* ``low_fields`` / ``high_fields`` -- the coolest / warmest ``top_k`` members' fields
  on the view grid (ranked by signed impact-box land-mean departure from ERA5);
* ``member_bias`` / ``member_rmse`` -- per-member free-end impact-box bias + RMS;
* ``member_scalar`` ``[N, T]`` -- per-member impact-box land-mean over EVERY lead frame;
* ``excluded_mask_box`` -- ocean pixels inside the impact box (hatched by the aggregator).

**Survivor censoring.** The free-end frame is where an autoregressive rollout is most
likely to have diverged, so a blown-up member would otherwise dominate the warm tail and
the ensemble mean.  With ``--survival-npz`` members are censored to those alive at the
free-end frame (``alive[:, 0]``).

Pure reductions (grid reshape, metric/excluded masks, weighted mean, signed-bias
ranking) are unit-tested; the makani/h5 read needs ``e2s-custom``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _dispatch_lib as dl  # noqa: E402

FREE_END_FRAME = 0  # forward frame 0 = the unpinned window start (end-mode precursor)


# --------------------------------------------------------------------------- #
# Pure reductions (numpy only; unit-tested)
# --------------------------------------------------------------------------- #
def grid_axes(flat: np.ndarray, width: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row/col unique axes + scatter indices for flat pixel ids on a ``H x width`` grid.

    A lat/lon box is a rectangular sub-grid, so the mapping is exact.  Returns
    ``(unique_rows, unique_cols, ri, ci)``.
    """
    flat = np.asarray(flat, dtype=np.int64)
    rows = flat // width
    cols = flat % width
    ur = np.unique(rows)
    uc = np.unique(cols)
    return ur, uc, np.searchsorted(ur, rows), np.searchsorted(uc, cols)


def scatter_to_grid(values: np.ndarray, ri: np.ndarray, ci: np.ndarray,
                    nlat: int, nlon: int) -> np.ndarray:
    """Scatter ``[..., P]`` values onto ``[..., nlat, nlon]`` (holes -> NaN)."""
    out = np.full(values.shape[:-1] + (nlat, nlon), np.nan, dtype=float)
    out[..., ri, ci] = values
    return out


def weighted_mean_last(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``sum_p arr[...,p] w[p]`` with ``w`` summing to 1 (impact-box land-mean)."""
    return (arr * w).sum(axis=-1)


def signed_bias_rank(bias: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` coolest (most negative bias) and ``k`` warmest members."""
    order = np.argsort(np.asarray(bias, float), kind="stable")
    k = int(min(k, order.size))
    return order[:k], order[::-1][:k]


def metric_and_excluded(flat_view: np.ndarray, flat_impact_land: np.ndarray,
                        flat_impact_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-view-pixel (metric_keep, excluded) masks.

    ``metric_keep`` = the view pixels that are impact-box AND land (the reduction
    domain); ``excluded`` = impact-box pixels that are NOT land (ocean carved out of
    the box, hatched by the aggregator).  Pixels outside the impact box are neither.
    """
    metric = np.isin(flat_view, flat_impact_land)
    inside = np.isin(flat_view, flat_impact_all)
    return metric, inside & ~metric


def _survivor_keep(n: int, survival_npz: Optional[str]) -> np.ndarray:
    if not survival_npz:
        return np.ones(n, bool)
    z = np.load(survival_npz)
    alive = np.asarray(z["alive"], bool)  # [N, L] forward order
    keep = alive[:, FREE_END_FRAME]
    if keep.shape[0] != n:
        print(f"[fes-mk] WARNING: survival N={keep.shape[0]} != members N={n}; "
              "not censoring.", file=sys.stderr)
        return np.ones(n, bool)
    return keep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--var-name", default="t2m", help="Impact surface field (PNW: t2m).")
    p.add_argument("--view-region", default="synoptic",
                   help="Region the spatial panels are cropped to (paper: synoptic).")
    p.add_argument("--impact-region", default="impact",
                   help="Region the bias + time series reduce over (dashed inset).")
    p.add_argument("--land-mask", default=None,
                   help="Full-grid (H,W) bool .npy; land-masks the impact reduction.")
    p.add_argument("--survival-npz", default=None,
                   help="Censor members to survivors alive at the free-end frame.")
    p.add_argument("--top-k", type=int, default=3, help="Members per tail (paper: 3).")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--timezone", default="America/Los_Angeles")
    p.add_argument("--start-time", default=None)
    args = p.parse_args(argv)

    from latlon_metrics import select_pixels  # heavy env
    from makani_source import MakaniBackwardSource

    case = dl.load_case(args.case)
    impact_box = dl.region_box(case, args.impact_region)
    view_box = dl.region_box(case, args.view_region) or impact_box
    if impact_box is None:
        raise SystemExit(f"case '{args.case}' has no regions.{args.impact_region} box.")
    if not args.start_time:
        try:
            args.start_time = dl.start_time_iso_from_case(case)
            args.timezone = dl.resolve_timezone(case)
        except Exception:  # noqa: BLE001, S110 -- optional; defaults handle absence
            pass

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames)
    W = int(makani.lon.shape[0])
    lat, lon = np.asarray(makani.lat, float), np.asarray(makani.lon, float)

    # View pixels (rectangular field grid) + impact-box reduction pixels.
    flat_view, _ = select_pixels(lat, lon, view_box)
    land_arr = (np.load(args.land_mask).astype(bool) if args.land_mask else None)
    flat_impact_land, _ = select_pixels(lat, lon, impact_box, land_arr)
    flat_impact_all, _ = select_pixels(lat, lon, impact_box)
    metric_view, excluded_view = metric_and_excluded(
        flat_view, flat_impact_land, flat_impact_all)
    if not metric_view.any():
        raise SystemExit("no impact-box land pixels inside the view; check regions/mask.")

    ur, uc, ri, ci = grid_axes(flat_view, W)
    nlat, nlon = ur.size, uc.size
    lat_view, lon_view = lat[ur], lon[uc]

    ens_view, truth_view = makani.load_region(args.var_name, flat_view)  # [N,L,P],[L,P]
    n_total = ens_view.shape[0]

    # cos(lat) land weights over the impact box, normalized to sum 1 (0 elsewhere).
    w = np.cos(np.deg2rad(lat[flat_view // W])) * metric_view
    w_scalar = w / w.sum()

    keep = _survivor_keep(n_total, args.survival_npz)
    ens_view = ens_view[keep]
    n_members = ens_view.shape[0]
    print(f"[fes-mk] members: {n_members}/{n_total} (censored={bool(args.survival_npz)})")

    member_scalar = weighted_mean_last(ens_view, w_scalar).astype(np.float64)  # [N,L]
    era5_scalar = weighted_mean_last(truth_view, w_scalar).astype(np.float64)  # [L]
    ensemble_mean_scalar = member_scalar.mean(axis=0)
    fe = FREE_END_FRAME
    member_bias = (member_scalar[:, fe] - era5_scalar[fe]).astype(np.float64)
    diff_fe = ens_view[:, fe, :] - truth_view[fe][None, :]  # [N,P]
    member_rmse = np.sqrt(weighted_mean_last(diff_fe ** 2, w_scalar)).astype(np.float64)

    low_idx, high_idx = signed_bias_rank(member_bias, args.top_k)

    ens_fe_grid = scatter_to_grid(ens_view[:, fe, :], ri, ci, nlat, nlon)  # [N,ny,nx]
    era5_field = scatter_to_grid(truth_view[fe], ri, ci, nlat, nlon)  # [ny,nx]
    excluded_grid = scatter_to_grid(excluded_view.astype(float), ri, ci, nlat, nlon)
    excluded_mask_box = np.nan_to_num(excluded_grid) > 0.5

    lead_hours_all = np.asarray(makani.lead_hours, dtype=np.float64)[:args.n_frames]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"free_end_states_makani_{args.case}.npz"
    np.savez(
        npz,
        member_bias=member_bias,
        member_rmse=member_rmse,
        low_idx=low_idx.astype(np.int64),
        high_idx=high_idx.astype(np.int64),
        low_bias=member_bias[low_idx].astype(np.float32),
        high_bias=member_bias[high_idx].astype(np.float32),
        low_rmse=member_rmse[low_idx].astype(np.float32),
        high_rmse=member_rmse[high_idx].astype(np.float32),
        low_fields=ens_fe_grid[low_idx].astype(np.float32),
        high_fields=ens_fe_grid[high_idx].astype(np.float32),
        era5_field=era5_field.astype(np.float32),
        member_scalar=member_scalar.astype(np.float32),
        era5_scalar=era5_scalar.astype(np.float32),
        ensemble_mean_scalar=ensemble_mean_scalar.astype(np.float32),
        lead_hours_all=lead_hours_all,
        lat=lat_view.astype(np.float64),
        lon=lon_view.astype(np.float64),
        bbox=np.asarray(impact_box, dtype=np.float64),
        view_bbox=np.asarray(view_box, dtype=np.float64),
        n_pix_in_box=np.int64(int(metric_view.sum())),
        mask_kind=np.array("land" if args.land_mask else "none"),
        hpx_level=np.int32(-1),  # native lat/lon (no HEALPix)
        free_end_frame=np.int32(fe),
        free_end_hours=np.float64(lead_hours_all[fe]),
        conditioning_frames=np.array([args.n_frames - 1], dtype=object),
        n_members=np.int64(n_members),
        n_members_total=np.int64(n_total),
        top_k=np.int32(args.top_k),
        var_name=np.array(args.var_name),
        case_name=np.array(case.get("display_name", args.case)),
        model=np.array("makani"),
        mode=np.array("end"),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
        excluded_mask_box=excluded_mask_box,
    )
    print(f"[fes-mk] wrote {npz}  view={nlat}x{nlon}  impact_land_pix="
          f"{int(metric_view.sum())}")
    print(f"[fes-mk] free-end bias: warmest {float(member_bias[high_idx[0]]):+.2f} K, "
          f"coolest {float(member_bias[low_idx[0]]):+.2f} K, "
          f"mean {float(member_bias.mean()):+.2f} K")


if __name__ == "__main__":
    main()
