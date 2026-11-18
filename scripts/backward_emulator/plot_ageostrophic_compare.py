"""Draw the ageostrophic comparison, Rewind-AR vs cBottle-video, for one or more cases.

Reads the npz that ``ageostrophic_compare.py`` writes per case and draws one figure:
a column per case, a row per diagnostic -- the ageostrophic fraction
``<|V_a|>/<|V|>`` and the mean speeds ``<|V|>``, ``<|V_g|>``, ``<|V_a|>`` over the
30-60 deg |lat| band at 500 hPa.  It follows the conventions of the original
paper's Figures S15-S17 (``aggregate_ageostrophic.py``): thin member lines under a
bold ensemble mean, ERA5 dashed black, ERA5 after the HEALPix round trip dotted
green, the conditioned frame dashed gray, and the same pinned axis ranges
(``axis_limits.yaml``, ``ageostrophic``), so this figure reads against those.

References.  Rewind-AR runs on ERA5's own 0.25 deg grid, so native ERA5 is its
exact reference.  cBottle-video is generated on HEALPix and regridded; its
reference lies between native ERA5 and the round trip (that paper's Text S5).

Rewind-AR statistics are over survivors: a member is dropped from its first
failing frame on (``makani::alive``, the comparison's survivor mask).

Also prints, per case, each model's departure of the ageostrophic fraction from
its references over the unconditioned frames: the numbers a text would quote.

Light env (numpy + matplotlib).  Usage (from this directory, after the
``dispatch_ageostrophic.py`` jobs finish):
    python plot_ageostrophic_compare.py --case ian --case pnw_heatwave --case sandy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ageostrophic_compare import npz_name  # noqa: E402
from model_labels import MODEL_COLORS, MODEL_LABELS  # noqa: E402

ROWS = (
    ("ratio", r"$|V_a| / |V|$"),
    ("mean_v", r"mean $|V|$ (m/s)"),
    ("mean_vg", r"mean $|V_g|$ (m/s)"),
    ("mean_va", r"mean $|V_a|$ (m/s)"),
)
MODELS = ("makani", "cbottle")
ERA_COLOR = "black"
ERA_RT_COLOR = "tab:green"
COND_COLOR = "gray"


def load(path: Path) -> Dict:
    """One case's npz as a dict; refuses an npz without the survivor mask."""
    if not path.is_file():
        raise SystemExit(f"ERROR: missing {path}; run dispatch_ageostrophic.py first.")
    d = dict(np.load(path, allow_pickle=False))
    if "makani::alive" not in d:
        raise SystemExit(
            f"ERROR: {path} has no survivor mask (makani::alive); it predates the "
            "mask being stored. Re-run ageostrophic_compare.py for this case.")
    return d


def member_traces(d: Dict, model: str, key: str) -> np.ndarray:
    """(members, frames) traces; Rewind-AR member-frames after failure are NaN."""
    x = np.asarray(d[f"{model}::{key}_ens"], dtype=float).copy()
    if model == "makani":
        x[~np.asarray(d["makani::alive"], bool)] = np.nan
    return x


def ensemble_mean(traces: np.ndarray) -> np.ndarray:
    """Per-frame mean over the finite (surviving) members; NaN where none survive."""
    ok = np.isfinite(traces)
    n = ok.sum(axis=0)
    total = np.where(ok, traces, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def free_frames(d: Dict) -> np.ndarray:
    """Indices of the frames that are not conditioned on ERA5."""
    n = np.asarray(d["lead_hours"]).size
    cond = set(np.asarray(d["conditioning_frames"]).ravel().tolist())
    return np.array([f for f in range(n) if f not in cond])


def departures(d: Dict) -> Dict[str, np.ndarray]:
    """Ensemble-mean ageostrophic fraction minus each reference, per free frame."""
    f = free_frames(d)
    rar, cbv = MODEL_LABELS["makani"], MODEL_LABELS["cbottle"]
    era = np.asarray(d["makani::ratio_base"], float)[f]
    out = {f"{rar} - ERA5": ensemble_mean(member_traces(d, "makani", "ratio"))[f] - era}
    if "cbottle::ratio_ens" in d:
        cb = ensemble_mean(member_traces(d, "cbottle", "ratio"))[f]
        out[f"{cbv} - ERA5"] = cb - era
        out[f"{cbv} - ERA5 round trip"] = cb - np.asarray(d["cbottle::ratio_base"], float)[f]
    return out


def report(case: str, d: Dict) -> None:
    alive = np.asarray(d["makani::alive"], bool)
    n_m, n_min = alive.shape[0], int(alive.sum(axis=0).min())
    line = (f"[ageo-plot] {case} ({d['case_name']}): {MODEL_LABELS['makani']} "
            f"{n_min} of {n_m} members survive to the free end")
    if "cbottle::ratio_ens" in d:
        line += f"; {MODEL_LABELS['cbottle']} N={d['cbottle::ratio_ens'].shape[0]}"
    print(line)
    if not bool(d.get("makani::censored", True)):
        print("[ageo-plot]   WARNING: written with --no-censor; Rewind-AR is NOT "
              "survivor-screened in this figure.")
    print(f"[ageo-plot]   |V_a|/|V| ensemble mean minus reference over the "
          f"{free_frames(d).size} unconditioned frames: mean [min, max]")
    for name, x in departures(d).items():
        print(f"[ageo-plot]     {name:<36} {np.mean(x):+.4f} "
              f"[{np.min(x):+.4f}, {np.max(x):+.4f}]")


def _check_shared(cases: Dict[str, Dict]) -> None:
    """All columns must reduce over the same level and band, or a row is not one quantity."""
    for field in ("level", "band_label"):
        vals = {str(d[field]) for d in cases.values()}
        if len(vals) > 1:
            raise SystemExit(f"ERROR: cases differ in {field}: {sorted(vals)}")


def draw(cases: Dict[str, Dict]):
    """The figure: one column per case (in order), one row per diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from axis_limits import apply_limits
    from local_time_axis import format_local_time_axis
    from plot_style import (
        AXIS_LABEL_SIZE_DENSE,
        LEGEND_SIZE_DENSE,
        SUPTITLE_SIZE_DENSE,
        TICK_LABEL_SIZE_DENSE,
        TITLE_SIZE_DENSE,
        add_subplot_labels,
    )

    _check_shared(cases)
    any_d = next(iter(cases.values()))
    level = int(any_d["level"])
    n_cols = len(cases)
    fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(5.5 * n_cols, 3.2 * len(ROWS)),
                             sharex="col", sharey="row", squeeze=False)
    row_data: List[List[np.ndarray]] = [[] for _ in ROWS]

    for j, (case, d) in enumerate(cases.items()):
        lead = np.asarray(d["lead_hours"], float)
        axes[0, j].set_title(str(d["case_name"]), fontsize=TITLE_SIZE_DENSE)
        models = [m for m in MODELS if f"{m}::ratio_ens" in d]
        for i, (key, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            for m in models:
                tr = member_traces(d, m, key)
                alpha = max(0.02, min(0.3, 30.0 / max(tr.shape[0], 1)))
                ax.plot(lead, tr.T, color=MODEL_COLORS[m], alpha=alpha, lw=0.5)
                row_data[i].append(tr[np.isfinite(tr)])
            for m in models:
                ax.plot(lead, ensemble_mean(member_traces(d, m, key)),
                        color=MODEL_COLORS[m], lw=2.0, label=MODEL_LABELS[m])
            era = np.asarray(d[f"makani::{key}_base"], float)
            ax.plot(lead, era, color=ERA_COLOR, lw=1.6, ls="--", label="ERA5")
            row_data[i].append(era)
            if "cbottle" in models:
                rt = np.asarray(d[f"cbottle::{key}_base"], float)
                ax.plot(lead, rt, color=ERA_RT_COLOR, lw=1.6, ls=":",
                        label="ERA5 (HEALPix round trip)")
                row_data[i].append(rt)
            for f in np.asarray(d["conditioning_frames"]).ravel().tolist():
                ax.axvline(lead[f], color=COND_COLOR, ls="--", lw=1.4, alpha=0.75,
                           label="Conditioned frame")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE_DENSE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_DENSE)
            ax.grid(True, alpha=0.3)
        format_local_time_axis(axes[-1, j], lead, str(d["start_time"]), str(d["timezone"]))
        axes[-1, j].xaxis.label.set_fontsize(AXIS_LABEL_SIZE_DENSE)
        axes[-1, j].tick_params(axis="x", labelsize=TICK_LABEL_SIZE_DENSE)

    axes[0, 0].legend(fontsize=LEGEND_SIZE_DENSE, loc="lower left", ncol=2)
    # sharey="row": constraining column 0 constrains the row.  Same YAML entry as
    # the original paper's S15-S17; a limit in force warns if data exceeds it.
    for i, (key, _) in enumerate(ROWS):
        apply_limits(axes[i, 0], "ageostrophic", f"uvz{level}", key,
                     y_data=np.concatenate(row_data[i]),
                     log_tag="plot_ageostrophic_compare")
    add_subplot_labels(axes, placement="inside")
    fig.suptitle(f"Ageostrophic diagnostics at {level} hPa, final conditioning\n"
                 f"band: {any_d['band_label']}", fontsize=SUPTITLE_SIZE_DENSE)
    fig.tight_layout()
    return fig


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", action="append", required=True,
                   help="Case name (repeatable); one column each, in this order.")
    p.add_argument("--out-root", default="./out",
                   help="Root the dispatcher wrote to: <root>/<case>/ageostrophic/.")
    p.add_argument("--level", type=int, default=500)
    p.add_argument("--out", default=None,
                   help="Figure path (default <out-root>/ageostrophic_compare_L<level>.png).")
    args = p.parse_args(argv)

    root = Path(args.out_root)
    cases = {c: load(root / c / "ageostrophic" / npz_name(c, args.level))
             for c in args.case}
    for case, d in cases.items():
        report(case, d)
    out = Path(args.out) if args.out else root / f"ageostrophic_compare_L{args.level}.png"
    fig = draw(cases)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[ageo-plot] wrote {out}")


if __name__ == "__main__":
    main()
