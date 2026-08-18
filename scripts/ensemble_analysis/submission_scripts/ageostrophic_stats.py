"""Turn the ageostrophic npz into the numbers the paper is allowed to quote.

A companion to ``aggregate_ageostrophic.py`` that reports, as text, every
quantity the SI and Discussion need from
``ageostrophic_fraction_<mode>.npz`` — so no number reaches a caption or
``main.tex`` by being read off a rendered panel (``paper/FIGURES.md``
makes npz confirmation a standing rule, and the npz live on scratch).

Reads every mode present under ``--output-dir`` and reports:

1.  **Metadata echo and cross-mode agreement.** Level, band, member and
    frame counts.  All three modes must share a band and a level or the
    columns of the figure are not mutually comparable; disagreement is
    reported as an ERROR rather than left for a reader to notice.

2.  **Pinned-frame values.** For each conditioned frame and each of the
    four quantities: ERA5, the ERA5 HEALPix round trip, the ensemble
    mean, its standard deviation, and the member range.  Duplicates what
    ``compute_ageostrophic.py`` already prints to its job log, as a
    cross-check that the log and the npz agree.

3.  **Free-frame values.** The same quantities over the unpinned frames,
    reported both across all free frames and across the free frames at
    least two frames from any pin — the second being the "recovered
    plateau" that a magnitude claim should be quoted from.

4.  **The conditioning pinch, quantified.** Ensemble-mean fraction at
    each pin, at one frame either side, and on the plateau, plus the
    fraction of the pin-to-plateau gap recovered after one frame.  This
    is what turns "recovers within roughly one frame" into a number.

5.  **Excess over ERA5, against both references.** The generated field
    takes one interpolation pass and the round trip takes two, so the
    correct reference for a free frame lies between raw ERA5 and the
    round trip.  The excess is therefore reported against both ends of
    that bracket, never against one alone.

6.  **Axis-pin candidates.** Per-row minima and maxima across the modes
    of this case, for building an ``ageostrophic`` block in
    ``scripts/_shared/axis_limits.yaml`` once all three cases are run.

Usage:
    python ageostrophic_stats.py \\
        --output-dir /path/to/<case>/diagnostics/ageostrophic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

QUANTITIES = [
    ("|V_a|/|V|", "ratio", ""),
    ("mean |V|", "mean_v", " m/s"),
    ("mean |V_g|", "mean_vg", " m/s"),
    ("mean |V_a|", "mean_va", " m/s"),
]
MODES = ("end", "start", "both")


def _scalar(arr):
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def _load_mode(output_dir: Path, mode: str) -> dict | None:
    path = output_dir / f"ageostrophic_fraction_{mode}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _free_frames(n_leads: int, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (all free frames, free frames >= 2 frames from any pin)."""
    all_idx = np.arange(n_leads)
    is_pin = np.isin(all_idx, cond)
    free = all_idx[~is_pin]
    if cond.size == 0:
        return free, free
    dist = np.min(np.abs(all_idx[:, None] - cond[None, :]), axis=1)
    return free, all_idx[dist >= 2]


def _report_metadata(per_mode: dict[str, dict]) -> None:
    print("=" * 78)
    first = next(iter(per_mode.values()))
    print(f"case        : {_scalar(first['case_name'])}")
    print(f"level       : {int(_scalar(first['level']))} hPa")
    print(f"band        : {_scalar(first['band_label'])}")
    print(f"modes found : {', '.join(per_mode)}")
    for mode, d in per_mode.items():
        ens = np.asarray(d["ratio_ens"])
        cond = np.asarray(d["conditioning_frames"]).ravel()
        print(
            f"  {mode:<5}  members={ens.shape[0]:<5} frames={ens.shape[1]:<3} "
            f"pinned={cond.tolist()}"
        )

    levels = {int(_scalar(d["level"])) for d in per_mode.values()}
    bands = {str(_scalar(d["band_label"])) for d in per_mode.values()}
    if len(levels) > 1 or len(bands) > 1:
        print(
            f"\n  ERROR: modes disagree on level {sorted(levels)} or band "
            f"{sorted(bands)} — the figure's columns are NOT comparable."
        )
    else:
        print("\n  modes agree on level and band (columns are comparable)")


def _report_pinned(per_mode: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("PINNED-FRAME VALUES  (cross-check against the compute job log)")
    for mode, d in per_mode.items():
        lead_h = np.asarray(d["lead_hours"])
        for cf in np.asarray(d["conditioning_frames"]).ravel().tolist():
            if not 0 <= cf < len(lead_h):
                continue
            print(f"\n-- mode={mode}  frame {cf} ({lead_h[cf]:.1f} h)")
            print(
                f"   {'quantity':<12} {'ERA5':>9} {'ERA5_RT':>9} {'EnsMean':>9} "
                f"{'MemStd':>9} {'MemMin':>9} {'MemMax':>9}"
            )
            for label, key, _u in QUANTITIES:
                ens = np.asarray(d[f"{key}_ens"])[:, cf]
                print(
                    f"   {label:<12} {float(np.asarray(d[f'{key}_era'])[cf]):>9.4f} "
                    f"{float(np.asarray(d[f'{key}_era_rt'])[cf]):>9.4f} "
                    f"{ens.mean():>9.4f} {ens.std(ddof=1):>9.4f} "
                    f"{ens.min():>9.4f} {ens.max():>9.4f}"
                )


def _report_free(per_mode: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("FREE-FRAME VALUES")
    print("  'all free'  = every unpinned frame")
    print("  'plateau'   = unpinned frames >= 2 frames from any pin")
    print("  MemStd   = member spread, averaged over the frames in the window")
    print("  FrameStd = variation of the ensemble mean ACROSS those frames")
    print("  (two different quantities -- do not quote either as 'the spread')")
    for mode, d in per_mode.items():
        n_leads = np.asarray(d["ratio_ens"]).shape[1]
        cond = np.asarray(d["conditioning_frames"]).ravel()
        free, plateau = _free_frames(n_leads, cond)
        for name, idx in (("all free", free), ("plateau", plateau)):
            if idx.size == 0:
                print(f"\n-- mode={mode}  {name}: none")
                continue
            print(f"\n-- mode={mode}  {name}  (n={idx.size} frames)")
            print(
                f"   {'quantity':<12} {'ERA5':>9} {'ERA5_RT':>9} {'EnsMean':>9} "
                f"{'MemStd':>9} {'FrameStd':>9}"
            )
            for label, key, _u in QUANTITIES:
                ens = np.asarray(d[f"{key}_ens"])[:, idx]
                mem_std = float(ens.std(axis=0, ddof=1).mean())
                frame_std = (
                    float(ens.mean(axis=0).std(ddof=1))
                    if idx.size > 1
                    else float("nan")
                )
                print(
                    f"   {label:<12} "
                    f"{float(np.asarray(d[f'{key}_era'])[idx].mean()):>9.4f} "
                    f"{float(np.asarray(d[f'{key}_era_rt'])[idx].mean()):>9.4f} "
                    f"{ens.mean():>9.4f} {mem_std:>9.4f} {frame_std:>9.4f}"
                )


def _report_pinch(per_mode: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("CONDITIONING PINCH  (ensemble-mean |V_a|/|V|)")
    print("  recovery = (value at pin+-1 - value at pin) / (plateau - value at pin)")
    for mode, d in per_mode.items():
        ratio = np.asarray(d["ratio_ens"])
        lead_h = np.asarray(d["lead_hours"])
        n_leads = ratio.shape[1]
        cond = np.asarray(d["conditioning_frames"]).ravel()
        _free, plateau_idx = _free_frames(n_leads, cond)
        if plateau_idx.size == 0:
            print(f"\n-- mode={mode}: no plateau frames, pinch not quantifiable")
            continue
        mean_by_frame = ratio.mean(axis=0)
        plateau = float(mean_by_frame[plateau_idx].mean())
        era = float(np.asarray(d["ratio_era"])[plateau_idx].mean())
        era_rt = float(np.asarray(d["ratio_era_rt"])[plateau_idx].mean())
        print(
            f"\n-- mode={mode}   plateau={plateau:.4f}  "
            f"ERA5={era:.4f}  ERA5_RT={era_rt:.4f}"
        )
        for cf in cond.tolist():
            if not 0 <= cf < n_leads:
                continue
            at_pin = float(mean_by_frame[cf])
            gap = plateau - at_pin
            print(f"   pin frame {cf} ({lead_h[cf]:.1f} h): {at_pin:.4f}")
            for off in (-1, 1):
                nb = cf + off
                if not 0 <= nb < n_leads or nb in cond.tolist():
                    continue
                val = float(mean_by_frame[nb])
                rec = (val - at_pin) / gap if abs(gap) > 1e-12 else float("nan")
                print(
                    f"     frame {nb:>2} ({lead_h[nb]:.1f} h): {val:.4f}   "
                    f"recovery {100 * rec:5.1f}% of the pin-to-plateau gap"
                )


def _report_excess(per_mode: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("EXCESS OVER ERA5, PLATEAU FRAMES  (both ends of the bracket)")
    print("  The generated field takes one interpolation pass and the round trip")
    print("  takes two, so the correct reference lies between the two columns.")
    print("  Quote the interval, not one endpoint.")
    print(f"\n   {'mode':<6} {'EnsMean':>9} {'vs ERA5':>10} {'vs ERA5_RT':>12}")
    for mode, d in per_mode.items():
        ratio = np.asarray(d["ratio_ens"])
        n_leads = ratio.shape[1]
        cond = np.asarray(d["conditioning_frames"]).ravel()
        _free, plateau_idx = _free_frames(n_leads, cond)
        if plateau_idx.size == 0:
            continue
        ens = float(ratio.mean(axis=0)[plateau_idx].mean())
        era = float(np.asarray(d["ratio_era"])[plateau_idx].mean())
        era_rt = float(np.asarray(d["ratio_era_rt"])[plateau_idx].mean())
        print(
            f"   {mode:<6} {ens:>9.4f} {100 * (ens / era - 1):>9.1f}% "
            f"{100 * (ens / era_rt - 1):>11.1f}%"
        )
    print("\n   ERA5_RT below ERA5 means regridding SUPPRESSES the fraction, so")
    print("   the excess cannot be explained by the ensemble having been regridded.")
    print("   Check the sign before relying on that argument.")


def _report_axis_pins(per_mode: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("AXIS-PIN CANDIDATES  (this case only; span all three cases before pinning)")
    print("  Covers every drawn artist: members, ERA5, and the round trip.")
    print(f"\n   {'quantity':<12} {'min':>10} {'max':>10}")
    for label, key, unit in QUANTITIES:
        lo, hi = np.inf, -np.inf
        for d in per_mode.values():
            for suffix in ("_ens", "_era", "_era_rt"):
                a = np.asarray(d[f"{key}{suffix}"], dtype=float)
                lo, hi = min(lo, float(np.nanmin(a))), max(hi, float(np.nanmax(a)))
        print(f"   {label:<12} {lo:>10.4f} {hi:>10.4f}{unit}")
    print("\n   NOTE: aggregate_ageostrophic.py floors the shared top row at 0.10")
    print("   (line 200) and autoscales everything else. If any member minimum")
    print("   above is below 0.10, that member is being silently clipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="The case's diagnostics/ageostrophic directory holding the npz.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"ERROR: --output-dir does not exist: {output_dir}")

    per_mode: dict[str, dict] = {}
    for mode in MODES:
        d = _load_mode(output_dir, mode)
        if d is None:
            print(f"WARNING: missing ageostrophic_fraction_{mode}.npz, skipping")
        else:
            per_mode[mode] = d

    if not per_mode:
        raise SystemExit(f"ERROR: no per-mode npz files found in {output_dir}")

    _report_metadata(per_mode)
    _report_pinned(per_mode)
    _report_free(per_mode)
    _report_pinch(per_mode)
    _report_excess(per_mode)
    _report_axis_pins(per_mode)
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
