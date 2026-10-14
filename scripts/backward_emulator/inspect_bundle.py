"""Morning-first sanity check on a transferred makani bundle.

Run this before any analysis, against a case directory of ``*fields*.h5`` dumps.
It pools the files through :class:`MakaniBackwardSource` (so it exercises the same
validation the diagnostics will) and prints a PASS/WARN checklist covering the
things a cross-facility transfer or a convention mismatch would break:

* all dumps present and mutually consistent (grid, channels, anchor);
* pooled member count is what we expect (~1000);
* the 8-channel comparison manifest is fully present;
* the grid is the standard ERA5 721x1440, lat 90->-90, lon in [0, 360);
* the anchor valid-time equals the cBottle window END for the case;
* the pinned frame equals truth (frame alignment), and a couple of channels are
  finite and in a physically plausible range.

Torch-free (numpy + h5py).  Example:

    python inspect_bundle.py \\
        /projectnb/eb-general/jlin404/scratch/backward_emulator_data/2021_pnw_heatwave \\
        --case pnw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from makani_source import MakaniBackwardSource  # noqa: E402
from trim_backward_fields import COMPARISON_CHANNELS  # noqa: E402

# cBottle window END = makani anchor, per case (see README / case YAMLs).
EXPECTED_ANCHOR = {
    "pnw": "2021-06-28T12",
    "ian": "2022-09-30T18",
}

# Loose physical-range sanity bounds (physical/storage units) for a spot check.
_PLAUSIBLE = {
    "t2m": (180.0, 340.0),      # K
    "msl": (85_000.0, 110_000.0),  # Pa
    "z500": (45_000.0, 60_000.0),  # m^2/s^2 (geopotential, not height)
}


def _line(ok: Optional[bool], msg: str) -> str:
    tag = "PASS" if ok else ("WARN" if ok is not None else "info")
    return f"  [{tag}] {msg}"


def inspect(case_dir: str, *, case: Optional[str], expected_members: int,
            expected_anchor: Optional[str]) -> bool:
    print(f"[inspect] {case_dir}")
    anchor_target = expected_anchor or (EXPECTED_ANCHOR.get(case) if case else None)

    try:
        src = MakaniBackwardSource(case_dir, expected_anchor_iso=None)
    except Exception as exc:  # noqa: BLE001 - this tool exists to report failures
        print(_line(False, f"could not pool the bundle: {exc}"))
        return False

    s = src.summary()
    all_ok = True

    def check(ok: bool, msg: str) -> None:
        nonlocal all_ok
        all_ok = all_ok and ok
        print(_line(ok, msg))

    print(f"  files={s['n_files']}  members={s['n_members']}  "
          f"frames={s['n_frames']}  grid={s['grid']}")
    print(f"  channels={s['channels']}")
    print(f"  anchor='{s['anchor_timestamp']}'  step_hours={s['step_hours']}  "
          f"lead_hours_forward={s['lead_hours_forward']}")

    # -- manifest channels ------------------------------------------------------
    missing = s["default_channels_missing"]
    check(not missing, f"comparison manifest present ({len(COMPARISON_CHANNELS)} "
          f"channels){'' if not missing else f'; MISSING {missing}'}")

    # -- grid orientation -------------------------------------------------------
    H, W = s["grid"]
    check((H, W) == (721, 1440), f"grid is 721x1440 (got {H}x{W})")
    lat0, lat1 = s["lat_range"]
    check(lat0 > lat1, f"lat descends 90->-90 (got {lat0}->{lat1})")
    lon0, lon1 = s["lon_range"]
    check(0.0 <= lon0 < 1.0 and lon1 > 180.0,
          f"lon in [0,360) (got {lon0}..{lon1})")

    # -- member count -----------------------------------------------------------
    check(s["n_members"] == expected_members,
          f"pooled members == {expected_members} (got {s['n_members']})")

    # -- anchor == window end ---------------------------------------------------
    if anchor_target:
        # makani writes "YYYY-MM-DD HH:MM:SS+00:00" (space separator); normalize
        # the separator so an ISO 'T' target still matches.
        anchor_norm = str(s["anchor_timestamp"]).replace(" ", "T")
        check(anchor_norm.startswith(anchor_target),
              f"anchor {anchor_norm[:16]} matches cBottle window end "
              f"'{anchor_target}'")
    else:
        print(_line(None, "no expected anchor given (pass --case or --expected-anchor)"))

    # -- pinned frame + finiteness + range on spot-check channels ---------------
    # Sampled (one file's members + truth window, ~200 MB), NEVER the full 50 GB
    # pooled ensemble.  A real survival scan over all members is separate.
    for ch in ("z500", "msl", "t2m"):
        if ch not in s["channels"]:
            continue
        try:
            src.assert_ic_pinned(ch)
            check(True, f"{ch}: pinned frame == truth (alignment ok)")
        except AssertionError as exc:
            check(False, f"{ch}: {exc}")
        truth = src.sample_truth_field(ch)          # (nf, H, W)
        pred = src.sample_member_window(ch)         # (M, nf, H, W), one file
        finite = bool(np.isfinite(truth).all() and np.isfinite(pred).all())
        check(finite, f"{ch}: sample all finite (no NaN/Inf)")
        lo, hi = float(np.nanmin(truth)), float(np.nanmax(truth))
        if ch in _PLAUSIBLE:
            plo, phi = _PLAUSIBLE[ch]
            check(plo <= lo and hi <= phi,
                  f"{ch}: truth range [{lo:.4g}, {hi:.4g}] within plausible "
                  f"[{plo:g}, {phi:g}]")
        # Blow-up smell test: max |pred| over the sampled members/window.
        pmax = float(np.nanmax(np.abs(pred))) if np.isfinite(pred).any() else float("inf")
        blow = _PLAUSIBLE.get(ch, (None, None))[1]
        if blow is not None:
            note = "" if pmax <= 5 * blow else "  <-- possible blow-up in window"
            print(_line(None, f"{ch}: sample max|pred| = {pmax:.4g}{note}"))

    print(f"[inspect] {'ALL PASS' if all_ok else 'SOME CHECKS FAILED — see WARN above'}")
    return all_ok


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("case_dir", help="Directory of *fields*.h5 dumps for one case.")
    p.add_argument("--case", choices=sorted(EXPECTED_ANCHOR),
                   help="Case name, to look up the expected anchor (window end).")
    p.add_argument("--expected-anchor", default=None,
                   help="ISO prefix of the expected anchor valid-time "
                   "(overrides --case).")
    p.add_argument("--expected-members", type=int, default=1000,
                   help="Expected pooled member count (default 1000).")
    args = p.parse_args(argv)
    ok = inspect(args.case_dir, case=args.case,
                 expected_members=args.expected_members,
                 expected_anchor=args.expected_anchor)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
