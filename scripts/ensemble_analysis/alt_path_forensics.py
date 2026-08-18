#!/usr/bin/env python3
"""Are the multi-path members' rival tracks fragments of the storm, or other systems?

One-off forensics for the round-5 QC prose.  The paper's 144 (Sandy end) is
the ``alt_path`` population of the SUPERSEDED rule in
``tc_track_targets._anchor_check``: members where raw ``path_id=0`` fails to
reach the pinned anchor but some other path reaches it.  For each of those
members this compares the anchor-resolved storm of interest against raw
path 0 head to head:

  * temporal overlap   : frames where BOTH paths have a valid fix.
  * separation         : great-circle distance between the two paths at those
                         overlapping frames.
  * ERA5 affinity      : each path's distance to the ERA5 storm track, over
                         the frames where that path and ERA5 are both valid.

Reading of the result:

  FRAGMENTS OF ONE STORM  -> little/no temporal overlap (the paths tile the
      window in sequence), and small separation where they do overlap.
  DISTINCT SYSTEMS        -> heavy temporal overlap with LARGE separation
      throughout, and the rival path sits far from ERA5 at every frame.
  ONE STORM, TWO CENTRES  -> heavy temporal overlap but MODEST separation
      (a few hundred km), both paths near ERA5 -- the transitioning storm's
      warm-core and baroclinic centres tracked as two chains.

Writes a per-member CSV next to the parquet and prints a summary table.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Both-end conditioning pins both ends and has no free end, so there is no
# free-end population to run these forensics over; only end and start apply.
MODES = ("end", "start")

# Separation thresholds used only to label the printed verdict.
SAME_STORM_KM = 300.0
DISTINCT_KM = 800.0


def mode_frames(mode: str, landfall_frame: int, n_steps: int) -> tuple[int, int]:
    """``(anchor_frame, free_end_frame)`` for one conditioning mode.

    The anchor is the PINNED end and the free end is the other one, so both
    flip with the mode.  Matches the branch every other track_qc consumer
    uses (synoptic_pca_track_stats.py:626-634); using the landfall frame as
    the anchor under start-conditioning would score every member against the
    wrong end of its own trajectory.
    """
    landfall = landfall_frame if 0 <= landfall_frame < n_steps else n_steps - 1
    if mode == "start":
        return 0, landfall
    return landfall, 0


def _summary(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n/a"
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return (
        f"n={x.size:4d}  p5={p[0]:8.1f}  p25={p[1]:8.1f}  "
        f"p50={p[2]:8.1f}  p75={p[3]:8.1f}  p95={p[4]:8.1f}"
    )


def analyze(
    parquet: str,
    mode: str,
    out_csv: str,
    track_box: tuple[float, float, float, float],
    landfall_frame: int,
) -> None:
    from tc_track_targets import (
        ANCHOR_RADIUS_KM,
        _fix_at_frame,
        _great_circle_km,
        _norm_lon,
        anchor_min_distances,
    )
    from tc_tracks_io import _fixes_in_bbox, load_tracks_parquet
    from track_qc import compute_track_qc_arrays

    d = load_tracks_parquet(parquet)
    if "ensemble" not in d or "era5" not in d:
        raise SystemExit(f"{parquet}: needs both 'ensemble' and 'era5' sources")

    raw_ens, raw_era5 = d["ensemble"], d["era5"]
    anchor_frame, free_end_frame = mode_frames(
        mode, landfall_frame, int(raw_ens.shape[2])
    )

    qc = compute_track_qc_arrays(
        raw_ens,
        raw_era5,
        track_box=track_box,
        anchor_frame=anchor_frame,
        free_end_frame=free_end_frame,
    )
    resolved = qc.resolved_path_id  # [N], -1 where nothing resolved

    ens = raw_ens[:, :, :, :3].astype(np.float64)
    ens[:, :, :, 1] = _norm_lon(ens[:, :, :, 1])
    era5 = raw_era5[0, 0, :, :3].astype(np.float64)
    era5[:, 1] = _norm_lon(era5[:, 1])

    n_members, n_paths, n_steps, _ = ens.shape
    valid = np.isfinite(ens[..., 0]) & np.isfinite(ens[..., 1])  # [N,P,T]
    in_dom = _fixes_in_bbox(ens[..., :2], *track_box)  # [N,P,T]
    n_dom = in_dom.sum(axis=-1)  # [N,P]

    # The SUPERSEDED rule, reproduced exactly as tc_track_targets._anchor_check
    # defines it: the storm is raw ``path_id=0``, and a member is `alt_path`
    # when path 0 fails to reach the anchor but some other path does.  This --
    # not a fix-count rule -- is what produced the paper's 144 for Sandy end.
    anchor_fix, _ = _fix_at_frame(era5, anchor_frame)
    if not np.isfinite(anchor_fix[:2]).all():
        raise SystemExit("ERA5 has no valid fix at the anchor frame")
    dmin = anchor_min_distances(
        ens, float(anchor_fix[0]), float(anchor_fix[1]), anchor_frame
    )
    if dmin is None:
        raise SystemExit("anchor window lies beyond the parquet's lead axis")
    reaches = dmin <= ANCHOR_RADIUS_KM  # [N,P]
    path0_anchored = reaches[:, 0]
    old_alt_path = ~path0_anchored & reaches.any(axis=1)
    old_lost = ~reaches.any(axis=1)

    # Secondary comparator: what a "most fixes in the box" rule would pick.
    # This is restrict_to_domain's "primary-first" ordering, a different
    # heuristic from raw path 0; reported so the two can be told apart.
    fixcount_pick = np.argmax(n_dom, axis=1)

    n_indom_paths = (n_dom > 0).sum(axis=1)
    print(f"\n{'='*78}\n{mode.upper()}-CONDITIONED   {parquet}\n{'='*78}")
    print(f"members={n_members}  paths/member in array={n_paths}  steps={n_steps}")
    print(f"anchor frame={anchor_frame}  free-end frame={free_end_frame}  "
          f"track box={track_box}")
    c = qc.counts()
    print(
        f"QC: analyzable={c['analyzable']}  excluded={c['excluded']} "
        f"(no_track {c['no_track']} / unanchored {c['unanchored']} / "
        f"insufficient {c['insufficient_fixes']})"
    )

    print("\n-- how many paths place at least one fix in the track box --")
    for k in range(int(n_indom_paths.max()) + 1):
        n = int((n_indom_paths == k).sum())
        if n:
            print(f"   {k} in-domain path(s): {n:4d} members")

    print("\n-- superseded (round-2/4) taxonomy, keyed on RAW path 0 --")
    print(f"   anchored (path 0 reaches the anchor): {int(path0_anchored.sum()):4d}")
    print(f"   alt_path (another path does)        : {int(old_alt_path.sum()):4d}"
          "   <-- the paper's 144 for Sandy end")
    print(f"   lost     (no path does)             : {int(old_lost.sum()):4d}")
    print(f"   fix-count rule differs from raw path 0 in "
          f"{int(((fixcount_pick != 0) & (resolved >= 0)).sum())} members "
          "(a different heuristic; not the 144)")

    idx = np.flatnonzero(old_alt_path & (resolved >= 0))
    print(f"\n-- forensics on the {idx.size} alt_path members --")
    print("   comparing the anchor-resolved storm against raw path 0")
    if idx.size == 0:
        print("   nothing to compare; no CSV written.")
        return

    rows = []
    for m in idx:
        a = int(resolved[m])  # anchor-resolved path = the storm of interest
        b = 0  # raw path 0 = the path the superseded rule called the storm
        va, vb = valid[m, a], valid[m, b]
        both = va & vb
        sep = np.full(n_steps, np.nan)
        if both.any():
            sep[both] = _great_circle_km(
                ens[m, a, both, 0], ens[m, a, both, 1],
                ens[m, b, both, 0], ens[m, b, both, 1],
            )

        v_era5 = np.isfinite(era5[:, 0]) & np.isfinite(era5[:, 1])
        da = np.full(n_steps, np.nan)
        db = np.full(n_steps, np.nan)
        ma, mb = va & v_era5, vb & v_era5
        if ma.any():
            da[ma] = _great_circle_km(
                ens[m, a, ma, 0], ens[m, a, ma, 1], era5[ma, 0], era5[ma, 1]
            )
        if mb.any():
            db[mb] = _great_circle_km(
                ens[m, b, mb, 0], ens[m, b, mb, 1], era5[mb, 0], era5[mb, 1]
            )

        fa = np.flatnonzero(va)
        fb = np.flatnonzero(vb)
        rows.append(
            {
                "member": int(m),
                "anchor_path": a,
                "path0": b,
                "anchor_first_step": int(fa[0]) if fa.size else -1,
                "anchor_last_step": int(fa[-1]) if fa.size else -1,
                "anchor_n_valid": int(va.sum()),
                "anchor_n_domain": int(n_dom[m, a]),
                "path0_first_step": int(fb[0]) if fb.size else -1,
                "path0_last_step": int(fb[-1]) if fb.size else -1,
                "path0_n_valid": int(vb.sum()),
                "path0_n_domain": int(n_dom[m, b]),
                "n_overlap_frames": int(both.sum()),
                "sep_median_km": float(np.nanmedian(sep)) if both.any() else np.nan,
                "sep_min_km": float(np.nanmin(sep)) if both.any() else np.nan,
                "sep_max_km": float(np.nanmax(sep)) if both.any() else np.nan,
                "anchor_dist_era5_median_km": float(np.nanmedian(da)) if ma.any() else np.nan,
                "path0_dist_era5_median_km": float(np.nanmedian(db)) if mb.any() else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    ov = df["n_overlap_frames"].to_numpy()
    sep = df["sep_median_km"].to_numpy()
    print(f"\n   temporal overlap (frames both paths valid, of {n_steps}):")
    print(f"     {_summary(ov)}")
    print(f"     members with ZERO overlap (sequential, i.e. true fragments): "
          f"{int((ov == 0).sum())}/{len(df)}")
    print("\n   separation between the two paths at overlapping frames (km):")
    print(f"     {_summary(sep)}")
    print("\n   each path's median distance to the ERA5 storm track (km):")
    print(f"     anchor-resolved path : {_summary(df['anchor_dist_era5_median_km'])}")
    print(f"     raw path 0           : {_summary(df['path0_dist_era5_median_km'])}")

    med = float(np.nanmedian(sep)) if np.isfinite(sep).any() else np.nan
    zero_ov = float((ov == 0).mean())
    print("\n   VERDICT:")
    if zero_ov > 0.5:
        print("     -> FRAGMENTS: most rival paths never coexist in time with the")
        print("        storm of interest; they tile the window sequentially.")
    elif np.isfinite(med) and med < SAME_STORM_KM:
        print("     -> SAME SYSTEM, TWO CHAINS: paths coexist but stay close;")
        print("        one storm tracked as two centres, not two storms.")
    elif np.isfinite(med) and med > DISTINCT_KM:
        print("     -> DISTINCT SYSTEMS: paths coexist and stay far apart.")
    else:
        print("     -> MIXED/AMBIGUOUS: coexisting, intermediate separation.")
        print("        Inspect the CSV before choosing wording.")
    print(f"\n   per-member CSV: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default=os.path.expanduser("~/earth2studio-private"),
        help="repo root (for scripts/ensemble_analysis on sys.path)",
    )
    ap.add_argument(
        "--tracks-dir",
        default="/projectnb/eb-general/jlin404/scratch/sandy/diagnostics/tc_tracks/tempest",
        help="directory holding tc_tracks_<mode>.parquet",
    )
    ap.add_argument(
        "--case",
        default="sandy",
        help="case YAML name or path; supplies the track domain and the "
        "landfall frame so they are never hardcoded to one storm",
    )
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--out-dir", default=None, help="default: --tracks-dir")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "scripts", "ensemble_analysis"))
    sys.path.insert(0, os.path.join(args.repo, "scripts", "_shared"))
    from _dispatch_lib import load_case, require_region, resolve_landfall_frame

    case = load_case(args.case)
    track_box = require_region(case, "track")
    landfall_frame = resolve_landfall_frame(case)

    out_dir = args.out_dir or args.tracks_dir
    os.makedirs(out_dir, exist_ok=True)

    for mode in args.modes:
        pq = os.path.join(args.tracks_dir, f"tc_tracks_{mode}.parquet")
        if not os.path.exists(pq):
            print(f"SKIP {mode}: missing {pq}")
            continue
        analyze(
            pq,
            mode,
            os.path.join(out_dir, f"alt_path_forensics_{mode}.csv"),
            track_box,
            landfall_frame,
        )


if __name__ == "__main__":
    main()
