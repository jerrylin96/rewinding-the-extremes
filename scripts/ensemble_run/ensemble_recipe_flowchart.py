"""Flowchart for the README: how the Extremes on Rewind ensembles are regenerated.

Every label is checked against the paper (SI Text S1, main text Section 2 and
Data and Code Availability) and the code (case YAMLs, dispatch_ensemble.py,
ensemble_interpolation.py, cbottle_video.py) and reviewed externally. Scope: the three
main case studies; the stress tests and the reproducibility note are in the README text
around it (section "Regenerating the ensembles").

    python ensemble_recipe_flowchart.py   ->  ensemble_recipe_flowchart.png beside this file
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1f2933"
MUTED = "#52606d"
ACCENT = "#2f5d8a"
INPUT_FILL = "#e8eef5"
STEP_FILL = "#ffffff"
OUT_FILL = "#f3f5f7"

fig = plt.figure(figsize=(7.0, 5.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 78)
ax.axis("off")


def box(x, y, w, h, title, lines, code=None, fill=STEP_FILL, edge=INK):
    """A rounded box: bold title, body lines, optional script names in grey monospace."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=edge, lw=0.9))
    cx, top = x + w / 2, y + h - 2.3
    ax.text(cx, top, title, ha="center", va="center", fontsize=8.6, weight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(cx, top - 3.3 - 2.75 * i, line, ha="center", va="center", fontsize=7.1, color=INK)
    if code:
        ax.text(cx, y + 1.7, code, ha="center", va="center", fontsize=6.3,
                family="monospace", color=MUTED)


def arrow(p, q, label=None, style="arc3", lpos=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                 color=ACCENT, connectionstyle=style, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=6.6, color=ACCENT, style="italic", ha="center", va="center")


# Row 1: the three inputs, all public and pinned.
y1, h1, w1 = 56, 18, 31
box(1.5, y1, w1, h1, "ERA5 reanalysis",
    ["WeatherBench 2, read with Earth2Studio",
     "12 frames, 6-hourly, from the case's",
     "start time $t_0$ to $t_0$ + 66 h"],
    fill=INPUT_FILL, edge=ACCENT)
box(34.5, y1, w1, h1, "cBottle-video checkpoint",
    ["huggingface.co/nvidia/cbottle",
     "at revision eebd93c: the model and",
     "the AMIP mid-month SST it ships with",
     "(SST identical across members)"],
    fill=INPUT_FILL, edge=ACCENT)
box(67.5, y1, w1, h1, "Case file",
    ["start time $t_0$, conditioning modes,",
     "ensemble size 1000, base seed 0"],
    code="scripts/_shared/cases/<case>.yaml", fill=INPUT_FILL, edge=ACCENT)

# Row 2: generation (GPU).
y2, h2 = 31, 17
box(9, y2, 82, h2, "Generate the ensembles (GPU)",
    ["one ensemble per conditioning mode: start, final, start + final",
     "member $k$ = 0, …, 999 is drawn one at a time with seed 0 + $k$",
     "18-level Heun sampler (SI Text S1); 12 frames, 6-hourly, $t_0$ to $t_0$ + 66 h"],
    code="dispatch_ensemble.py  →  ensemble_interpolation.py  →  merge_ensemble_zarrs.py")
for cx in (17, 50, 83):
    arrow((cx, y1 - 0.6), (cx, y2 + h2 + 0.6))
ax.text(18.2, (y1 + y2 + h2) / 2, "selected endpoints \u2192 HPX64", fontsize=6.6, color=ACCENT,
        style="italic", ha="left", va="center")

# Row 3: the ensemble itself.
y3, h3 = 15.5, 11
box(25, y3, 50, h3, "One Zarr store per case and mode",
    ["1000 members × 12 frames × 15 retained variables",
     "(of the 45 generated), HPX64, ≈ 26 GB"], fill=OUT_FILL)
arrow((50, y2 - 0.6), (50, y3 + h3 + 0.6))

# Row 4: analysis (CPU).
y4, h4 = 1.5, 11
box(25, y4, 50, h4, "Analysis and figures (CPU)",
    ["spread, RMSE and CRPS against ERA5; EOFs; TC tracks"], code="scripts/ensemble_analysis/",
    fill=OUT_FILL)
arrow((50, y3 - 0.6), (50, y4 + h4 + 0.6))

# ERA5's second use: the full window, kept on its own 0.25-degree grid, verifies the ensembles.
ax.plot([4.5, 4.5], [y1 - 0.6, y4 + h4 / 2], color=ACCENT, lw=1.0)
arrow((4.5, y4 + h4 / 2), (24.4, y4 + h4 / 2))
ax.text(14.5, y4 + h4 / 2 + 0.9, "12-frame ERA5 reference\n(0.25\u00b0 lat\u2013lon)", fontsize=6.6,
        color=ACCENT, style="italic", ha="center", va="bottom", linespacing=1.3)


out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=300)
print(f"wrote {out}")
