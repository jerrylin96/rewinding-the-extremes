"""Stability-only SI figure: backward-rollout survival curves, all cases on one axis.

The spread/RMSE/CRPS comparison figures carry the survival fraction only as a
footnote panel over the 66 h comparison window, yet two load-bearing analysis
choices rest on it: (1) every autoregressive-emulator statistic is computed over
*survivors* (``survival.py``), and (2) the comparison window is truncated to
66 h because the rollout destabilizes beyond it.  This figure makes both
auditable in one place, over the emulator's FULL rollout horizon (−180 h), and
shows that the attrition rate is regime-dependent (the slow PNW ridge vs. the
fast-evolving Ian environment) by overlaying the cases on a single panel.

Consumes the ``.npz`` files written by ``survival.py`` — run that (via
``submit_survival_scan.sh``; the t2m stream is too heavy for a login node) with
``--n-frames 31`` so the curves cover the full −180 h rollout rather than the
12-frame comparison window:

    python plot_survival_stability.py \\
        --case-npz ian=out/ian/survival_full.npz \\
        --case-npz pnw_heatwave=out/pnw/survival_full.npz \\
        --out out/survival_stability.png

The x-axis is hours from the anchor (0 = the pinned present, negative = into
the past) rather than the local wall-clock axis the other figures use, because
the two cases share the figure and only the relative axis is common to both.

Light env (numpy + matplotlib + pyyaml); re-plotting from saved ``.npz`` is a
login-node-safe, seconds-scale operation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

# The 66 h cBottle-video context window (12 frames x 6 h to the anchor).
DEFAULT_CONTEXT_HOURS = 66.0

_CASE_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


def load_survival_npz(path: Path) -> Dict:
    """Read one ``survival.py`` output; convert leads to hours-from-anchor.

    ``survival.py`` saves ``lead_hours`` on cBottle's FORWARD axis (0 at the
    window start, increasing to the anchor at the last frame), so hours from
    the anchor are ``lead_hours - lead_hours[-1]`` (0 at the present, negative
    into the past) with ``survival`` already in the matching forward order.
    """
    with np.load(path) as d:
        lead = np.asarray(d["lead_hours"], dtype=float)
        surv = np.asarray(d["survival"], dtype=float)
        info = {
            "hours_from_anchor": lead - lead[-1],
            "survival": surv,
            "n_members": int(np.asarray(d["alive"]).shape[0]) if "alive" in d else None,
            "detector": str(d["detector"].item()) if "detector" in d else "range",
            "channel": str(d["channel"].item()) if "channel" in d else "t2m",
            "bounds": tuple(float(b) for b in d["bounds"]) if "bounds" in d else None,
        }
    if surv.shape != info["hours_from_anchor"].shape:
        raise ValueError(f"{path}: survival and lead_hours shapes differ")
    if not np.all(np.diff(info["hours_from_anchor"]) > 0):
        raise ValueError(f"{path}: lead_hours must be strictly increasing")
    return info


def survival_at(hours_from_anchor: np.ndarray, survival: np.ndarray,
                hour: float) -> float:
    """Survival fraction at ``hour`` (<= 0), linearly interpolated between frames."""
    return float(np.interp(hour, hours_from_anchor, survival))


def crossing_hour(hours_from_anchor: np.ndarray, survival: np.ndarray,
                  level: float = 0.5) -> Optional[float]:
    """Hour (<= 0) where survival first drops below ``level``, scanning from the
    anchor into the past; linearly interpolated between the bracketing frames.
    None if survival never falls below ``level`` over the sampled horizon."""
    h = np.asarray(hours_from_anchor, float)
    s = np.asarray(survival, float)
    # Scan from the anchor (last index, h=0) backward in time.
    for i in range(len(h) - 1, 0, -1):
        s_near, s_far = s[i], s[i - 1]  # nearer / farther from the anchor
        if s_near >= level > s_far:
            frac = (s_near - level) / (s_near - s_far)
            return float(h[i] + frac * (h[i - 1] - h[i]))
    return None


def _display_name(case_name: str) -> str:
    """Case display name from the shared YAML; the raw name if none exists."""
    try:
        import _dispatch_lib as dl  # noqa: E402
        return str(dl.load_case(case_name).get("display_name", case_name))
    except Exception:
        return case_name


def plot(out_path: Path, cases: List[Tuple[str, Dict]], *,
         context_hours: float = DEFAULT_CONTEXT_HOURS) -> Dict[str, Dict]:
    """Draw the single-panel overlay; return the per-case headline stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps  # noqa: E402

    # The detector must be identical across cases or the curves aren't comparable.
    ref = cases[0][1]
    for name, info in cases[1:]:
        for key in ("detector", "channel", "bounds"):
            if info[key] != ref[key]:
                raise ValueError(
                    f"case '{name}' used {key}={info[key]!r} but "
                    f"'{cases[0][0]}' used {ref[key]!r} -- rerun the survival "
                    "scans with one detector configuration."
                )

    stats: Dict[str, Dict] = {}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, (name, info) in enumerate(cases):
        h, s = info["hours_from_anchor"], info["survival"]
        s_ctx = survival_at(h, s, -context_hours)
        h50 = crossing_hour(h, s, 0.5)
        stats[name] = {"survival_at_context": s_ctx, "half_life_hours": h50}
        half = (f"50% at −{-h50:.0f} h" if h50 is not None
                else f"never <50% by −{-h[0]:.0f} h")
        # Two lines so the legend box stays narrow enough to sit clear of the
        # curves in the empty upper-left of these monotone survival sigmoids.
        label = (f"{_display_name(name)}\n"
                 f"   {s_ctx:.0%} at −{context_hours:.0f} h; {half}")
        color = _CASE_COLORS[i % len(_CASE_COLORS)]
        ax.plot(h, s, "-o", color=color, lw=1.8, ms=4, label=label)

    ax.axhline(0.5, color="0.5", lw=0.9, ls=":")
    ax.axvline(-context_hours, color="0.4", lw=1.0, ls="--")
    ax.text(-context_hours, 0.02, f"cBottle-video {context_hours:.0f} h context  ",
            rotation=90, ha="right", va="bottom", color="0.4",
            fontsize=ps.TICK_LABEL_SIZE)
    ax.set_xlabel("hours from the anchor  (0 = the present; negative = into the past)",
                  fontsize=ps.AXIS_LABEL_SIZE)
    ax.set_ylabel("surviving fraction", fontsize=ps.AXIS_LABEL_SIZE)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=ps.LEGEND_SIZE, loc="upper left", framealpha=0.9)
    ps.style_axes(ax)
    ax.set_title("Autoregressive backward-rollout stability",
                 fontsize=ps.TITLE_SIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stats


def _parse_case_npz(items: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise argparse.ArgumentTypeError(
                f"--case-npz expects NAME=PATH, got {item!r}")
        out.append((name, Path(path)))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case-npz", action="append", required=True, metavar="NAME=PATH",
                   help="Case-YAML name and its survival.py .npz, e.g. "
                   "ian=out/ian/survival_full.npz. Repeat per case; legend "
                   "order follows CLI order.")
    p.add_argument("--out", required=True, help="Output figure path (.png).")
    p.add_argument("--context-hours", type=float, default=DEFAULT_CONTEXT_HOURS,
                   help="Comparison-window context to mark (default 66).")
    args = p.parse_args(argv)

    pairs = _parse_case_npz(args.case_npz)
    cases = [(name, load_survival_npz(path)) for name, path in pairs]
    stats = plot(Path(args.out), cases, context_hours=args.context_hours)

    ref = cases[0][1]
    print(f"[stability] detector={ref['detector']} channel={ref['channel']} "
          f"bounds={ref['bounds']}")
    for name, info in cases:
        st = stats[name]
        h50 = st["half_life_hours"]
        horizon = info["hours_from_anchor"][0]
        print(f"[stability] {name}: N={info['n_members']} "
              f"horizon={horizon:.0f} h  "
              f"survival at -{args.context_hours:.0f} h = "
              f"{st['survival_at_context']:.3f}  "
              f"50% crossing = {f'{h50:.1f} h' if h50 is not None else 'none'}")
    print(f"[stability] wrote {args.out}")


if __name__ == "__main__":
    main()
