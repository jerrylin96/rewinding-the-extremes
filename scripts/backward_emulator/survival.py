"""Survivor censoring for the makani backward ensemble.

The autoregressive emulator's members go numerically unstable and blow up (one
member at 1e8 K destroys a 1000-member mean), so makani's settled methodology
computes every distributional statistic over **survivors**: a member is dead from
its first failing lead onward, and the failure is made MONOTONE (a trajectory that
has blown up does not un-blow-up).  This module produces the per-member ``alive``
mask that ``latlon_metrics.reduce_selected`` consumes, plus the survival fraction
that gets its own panel.

We reuse makani's methodology and **its exact thresholds** rather than inventing
any -- the ``range`` detector's physical interval and the monotone-alive rule come
straight from ``run-it-back/analysis/ensemble_horizon.py``.  Pass
``--makani-analysis-dir`` (or its default sibling guesses) to import that module
directly and guarantee parity; otherwise the constants below are copied verbatim
from it (kept in sync via the citation).

Detector: ``range`` (fields only, always available) -- a member fails at the first
lead where the field leaves a physical interval or goes non-finite.  cBottle needs
no censoring (its end-conditioned members are all valid), so this is a
makani-only step; the ``tail`` spectral detector (makani's earlier-firing default
when spectra are supplied) is a documented follow-up.

Torch-free (numpy + h5py via the source).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# --- makani's divergence thresholds, copied verbatim (do NOT invent) ----------
# Source: run-it-back/analysis/ensemble_horizon.py
#   RANGE_DEFAULT = {"t2m": (150.0, 350.0)}   # K; ERA5 extremes ~180-335 K, so
#     this interval cannot censor a physically plausible retrodiction -- it
#     catches numerical divergence only.
#   TAIL_CEIL_DEFAULT = 10.0                  # spectral tail-ratio ceiling.
RANGE_DEFAULT: Dict[str, Tuple[float, float]] = {"t2m": (150.0, 350.0)}
TAIL_CEIL_DEFAULT = 10.0

# t2m is the range-detector channel: divergence shows first in the surface
# temperature field (Ian's sampled max|pred| hit 383 K while z500/msl stayed in
# range).  Censoring is a per-MEMBER (whole-trajectory) property, so the mask this
# yields applies to every variable's statistics, not just t2m.
DEFAULT_DETECTOR_CHANNEL = "t2m"


def _import_makani_ensemble_horizon(makani_analysis_dir: Optional[str]):
    """Return makani's ensemble_horizon module if importable, else None.

    Tries the explicit dir, then a couple of sibling-repo guesses.  When found we
    use ITS first_true / make_monotone_alive / RANGE_DEFAULT so our censoring is
    provably makani's, not a re-implementation.
    """
    candidates = []
    if makani_analysis_dir:
        candidates.append(Path(makani_analysis_dir))
    here = Path(__file__).resolve()
    for up in (here.parents[3], here.parents[4] if len(here.parents) > 4 else here.parents[3]):
        candidates.append(up / "run-it-back" / "analysis")
        candidates.append(up / "makani-aicr" / "analysis")  # the checkout's name on our cluster
    for c in candidates:
        if (c / "ensemble_horizon.py").is_file():
            sys.path.insert(0, str(c))
            try:
                import ensemble_horizon as eh  # type: ignore
                return eh
            except Exception:  # pragma: no cover - fall back to vendored
                return None
    return None


def _first_true(flags: np.ndarray) -> Optional[int]:
    """First index where a boolean series is True; None if never (makani semantics)."""
    idx = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(idx[0]) if idx.size else None


def _monotone_alive(fail_lead: Optional[int], n_leads: int) -> np.ndarray:
    """(n_leads,) bool alive mask, monotone: dead from ``fail_lead`` on (makani)."""
    alive = np.ones(int(n_leads), dtype=bool)
    if fail_lead is not None:
        alive[int(fail_lead):] = False
    return alive


def range_alive_mask(
    source,
    *,
    channel: str = DEFAULT_DETECTOR_CHANNEL,
    bounds: Optional[Tuple[float, float]] = None,
    makani_analysis_dir: Optional[str] = None,
) -> Dict:
    """Compute the per-member survivor mask via makani's ``range`` detector.

    Streams the detector channel one member at a time over the whole grid (a
    member is a diverged *trajectory*, judged globally, not per region), flags the
    first lead where it leaves ``bounds`` or goes non-finite, and censors
    monotonically from there.  Returns ``alive (N, L)``, ``survival (L,)``,
    ``n_alive (L,)``, and metadata.
    """
    lo, hi = bounds if bounds is not None else RANGE_DEFAULT.get(channel, (150.0, 350.0))
    eh = _import_makani_ensemble_horizon(makani_analysis_dir)
    first_true = eh.first_true if eh is not None else _first_true
    monotone = eh.make_monotone_alive if eh is not None else _monotone_alive
    used_makani = eh is not None

    N, L = source.n_members, source.n_frames
    alive = np.ones((N, L), dtype=bool)
    fail_leads = np.full(N, -1, dtype=np.int64)  # -1 = never failed; in leads-into-past
    for gid, series in source.iter_member_domain_series(channel):  # series (L, P), FORWARD
        finite = np.isfinite(series).all(axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice -> nan
            mx = np.nanmax(series, axis=1)
            mn = np.nanmin(series, axis=1)
        # NaN comparisons are False, so a non-finite lead is caught by ~finite.
        flag_fwd = (~finite) | (mx > hi) | (mn < lo)  # (L,) bool, forward order
        # Detect + censor in makani's ROLLOUT order (present -> past): the present
        # (forward frame -1, the pinned IC) is index 0 there and always valid, and
        # divergence grows into the past.  Reverse the flag, apply the monotone
        # rule, then map the alive mask back to forward order for the reducer.
        fl = first_true(flag_fwd[::-1])               # lead-into-past index
        alive[gid] = np.asarray(monotone(fl, L))[::-1]
        fail_leads[gid] = -1 if fl is None else int(fl)

    n_alive = alive.sum(axis=0).astype(np.int64)
    return {
        "alive": alive,
        "survival": n_alive.astype(np.float64) / float(N),
        "n_alive": n_alive,
        "fail_leads": fail_leads,
        "n_members": N,
        "n_diverged": int((fail_leads >= 0).sum()),
        "detector": "range",
        "channel": channel,
        "bounds": (float(lo), float(hi)),
        "used_makani_module": used_makani,
    }


def main(argv=None) -> None:
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from makani_source import MakaniBackwardSource, find_field_dumps, resolve_expected_anchor  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True, help="Case dir of *fields*.h5.")
    p.add_argument("--channel", default=DEFAULT_DETECTOR_CHANNEL)
    p.add_argument("--out", required=True, help="Output .npz for the alive mask.")
    p.add_argument("--makani-analysis-dir", default=None,
                   help="Path to run-it-back/analysis to import ensemble_horizon "
                   "(for guaranteed threshold parity). Falls back to vendored "
                   "constants if omitted.")
    p.add_argument("--n-frames", type=int, default=12,
                   help="Leads to scan (12 = the 66 h comparison window). 0 = "
                   "every lead the dumps carry (the full rollout horizon, for "
                   "the stability figure).")
    p.add_argument("--case", default=None,
                   help="Case name (scripts/_shared/cases); when given, the dumps' "
                   "anchor must equal the case's window end (start_time + 66 h).")
    p.add_argument("--expected-anchor", default=None,
                   help="ISO-hour prefix the dumps' anchor must match; overrides "
                   "--case ('none' disables the check).")
    args = p.parse_args(argv)

    case = None
    if args.case is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
        import _dispatch_lib as dl  # noqa: E402
        case = dl.load_case(args.case)

    n_frames = args.n_frames
    if n_frames <= 0:
        import h5py
        with h5py.File(find_field_dumps(args.makani_dir)[0], "r") as f:
            pred = f["pred"]
            n_frames = int(pred.shape[2] if pred.ndim == 6 else pred.shape[1])
        print(f"[survival] --n-frames {args.n_frames} -> scanning all "
              f"{n_frames} leads in the dumps")

    src = MakaniBackwardSource(args.makani_dir, n_frames=n_frames,
                               expected_anchor_iso=resolve_expected_anchor(case, args.expected_anchor))
    res = range_alive_mask(src, channel=args.channel,
                           makani_analysis_dir=args.makani_analysis_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        alive=res["alive"], survival=res["survival"], n_alive=res["n_alive"],
        fail_leads=res["fail_leads"], lead_hours=np.asarray(src.lead_hours, float),
        channel=np.array(res["channel"]), bounds=np.asarray(res["bounds"]),
        detector=np.array(res["detector"]),
    )
    surv = res["survival"]
    print(f"[survival] {args.makani_dir}")
    print(f"[survival]   detector=range channel={res['channel']} "
          f"bounds={res['bounds']} makani_module={res['used_makani_module']}")
    print(f"[survival]   diverged members: {res['n_diverged']}/{res['n_members']}")
    print(f"[survival]   survival fraction by lead: "
          f"{[round(float(x), 3) for x in surv]}")
    # Forward frame -1 is the pinned present (survival 1.0 by construction);
    # frame 0 is the far end of the scanned window -- the number that matters.
    print(f"[survival]   at the window start ({float(src.lead_hours[-1]):.0f} h "
          f"before the anchor): {float(surv[0]):.3f}")
    print(f"[survival] wrote {out}")


if __name__ == "__main__":
    main()
