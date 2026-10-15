"""Overlay the makani vs cBottle synoptic-PCA results (per-ensemble, Text S2).

The EOFs are each ensemble's OWN basis, so "makani PC1" and "cBottle PC1" are
different axes and are NOT overlaid.  What IS comparable -- computed identically for
both from each npz's stored PC scores + impact target -- is:

* the **leading-8-PC -> impact-target regression R^2** (in-sample + 10-fold CV):
  the Figs S2-S5 headline, grid-robust, the paper's quoted number;
* the predicted-vs-ensemble-member scatter (the money figure), makani | cBottle
  side by side;
* the explained-variance spectrum, and ERA5's percentile along each leading PC.

Reads the makani npz (`synoptic_pca_makani.py`) and the cBottle npz (the paper's own
`compute_synoptic_pca.py` output, `synoptic_pca_end.npz`), normalising both schemas
into one shape.  Degrades to makani-only if the cBottle npz is absent.

**Track targets (Ian) come from the paper's track QC only.**  A cBottle npz baked
before `compute_synoptic_pca.py` used track QC holds path-0 targets
(`track_target_fe`, no `track_target_source`) that the paper's current regression code
does not read; for Ian, 19 cBottle members carry another storm there.  So a model's
track targets are recomputed from its tracks parquet (`--cbottle-parquet`,
`--makani-parquet`) through `case_track_qc`, and an npz whose targets did not come
from track QC is refused rather than plotted.

Figure conventions match the other compare figures: the models are displayed as
"Autoregressive" and "Video diffusion (cBottle)" (npz-internal keys stay
makani/cbottle), targets get the paper's spelled-out names, and the scatter's
vertical axis is labelled "ensemble member" -- each point is one member's own
free-end value, not an observation (ERA5 is the observation and is not drawn).
With >= 3 targets (the Ian track case) the scatter grid is one ROW per model and
one COLUMN per target, so the whole figure fits a single page.

Light: numpy + matplotlib + pca_stats (reads npz; no torch/zarr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _dispatch_lib as dl  # noqa: E402
from synoptic_pca_makani import regression_summary  # noqa: E402

# Display names, shared with the other compare figures (ageostrophic, TC tracks,
# free-end states).  Keys are the internal model keys; npz-stored names / file
# stems are only a fallback for unknown models.
_MODEL_LABELS = {"makani": "Autoregressive", "cbottle": "Video diffusion (cBottle)"}

# Spelled-out target names + units, matching the paper's track-regression figure
# (aggregate_synoptic_pca._TRACK_TARGETS) and its scalar precursor->impact panel.
_TARGET_LABELS = {
    "tc_lat": ("free-end latitude", "°N"),
    "tc_lon": ("free-end longitude", "°E"),
    # Newline keeps the widest column title inside its panel.
    "tc_dist_km": ("free-end distance\nfrom ERA5", "km"),
    "tc_msl_hpa": ("free-end\ncentral MSL", "hPa"),
    "t2m": ("land-mean t2m", "K"),
}
# Track target names, in `case_track_qc.TARGET_COLUMNS` order.
_TRACK_NAMES = ("tc_lat", "tc_lon", "tc_dist_km", "tc_msl_hpa")


def _model_label(mkey: str, m: Dict) -> str:
    return _MODEL_LABELS.get(mkey, str(m.get("model", mkey)))


def _target_label(nm: str) -> str:
    name, unit = _TARGET_LABELS.get(nm, (nm, ""))
    return f"{name} ({unit})" if unit else name


def _read_pca_npz(path: str) -> Dict:
    """Normalise a makani- or cBottle-schema PCA npz into a common dict.

    Returns ``{model, case, evr, pc_scores [N,d], era5_pc_percentile,
    targets: {name: y [N]}}`` with targets aligned to the PC-score rows.
    """
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    scores = z["pc_scores"]
    fe = int(z["free_end_frame"]) if "free_end_frame" in keys else 0
    out: Dict = {
        "model": str(z["model"]) if "model" in keys else Path(path).stem,
        "case": str(z["case_name"]) if "case_name" in keys else "",
        "evr": np.asarray(z["explained_variance_ratio"], float),
        "pc_scores": np.asarray(scores, float),
        "era5_pc_percentile": (np.asarray(z["era5_pc_percentile"], float)
                               if "era5_pc_percentile" in keys else None),
        "targets": {},
    }
    # Domain metadata for the title's subordinate lines.  The cBottle npz
    # stores the precursor (EOF) box explicitly; the makani npz stores the
    # used pixel indices + the full grid, so its box is recovered from those.
    out["precursor_box"] = None
    if {"lon_min", "lon_max", "lat_min", "lat_max"} <= keys:
        out["precursor_box"] = tuple(
            float(z[k]) for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
    elif {"domain_flat", "domain_lat", "domain_lon"} <= keys:
        flat = np.asarray(z["domain_flat"], np.int64)
        lat = np.asarray(z["domain_lat"], float)
        lon = np.asarray(z["domain_lon"], float)
        if flat.size and lat.size and lon.size:
            la = lat[flat // lon.size]
            lo = lon[flat % lon.size]
            lo = np.where(lo > 180.0, lo - 360.0, lo)  # 0..360 grid -> +-180
            out["precursor_box"] = (float(lo.min()), float(lo.max()),
                                    float(la.min()), float(la.max()))
    out["variables"] = ([str(v) for v in z["variables"]]
                        if "variables" in keys else [])
    for k in ("impact_bbox", "tc_domain"):  # cBottle npz only
        v = np.asarray(z[k], float).reshape(-1) if k in keys else np.zeros(0)
        out[k] = tuple(float(x) for x in v) if v.size == 4 else None
    out["impact_variable"] = (str(z["impact_variable"])
                              if "impact_variable" in keys else "")
    out["impact_mask_kind"] = (str(z["impact_mask_kind"])
                               if "impact_mask_kind" in keys else "")
    # PC-score rows -> member ids: makani stores its survivor mask; cBottle's rows
    # are all members.
    out["analyzable"] = (np.asarray(z["analyzable"], bool)
                         if "analyzable" in keys else None)
    # Where the track targets came from ("track_qc" or "path0"); None = scalar.
    out["track_source"] = None
    if "target_scalar" in keys:  # makani scalar
        nm = str(z["target_names"][0]) if "target_names" in keys else "impact"
        out["targets"][nm] = np.asarray(z["target_scalar"], float)
    elif "target_track_fe" in keys:  # makani track
        names = [str(x) for x in z["target_names"]]
        tgt = np.asarray(z["target_track_fe"], float)
        for j, nm in enumerate(names):
            out["targets"][nm] = tgt[:, j]
        out["track_source"] = (str(z["target_source"]) if "target_source" in keys
                               else "path0")
    elif "member_impact_scalar" in keys and np.asarray(z["member_impact_scalar"]).size:
        nm = str(z["impact_variable"]) if "impact_variable" in keys else "impact"
        out["targets"][nm] = np.asarray(z["member_impact_scalar"], float)[:, fe]
    elif "track_target_fe" in keys and np.asarray(z["track_target_fe"]).size:
        tgt = np.asarray(z["track_target_fe"], float)  # [N,4]
        for j, nm in enumerate(_TRACK_NAMES):
            out["targets"][nm] = tgt[:, j]
        # npz files baked before compute_synoptic_pca used track QC carry path-0
        # targets and no source key.
        source = (str(z["track_target_source"]) if "track_target_source" in keys
                  else "")
        out["track_source"] = source or "path0"
    return out


def _apply_track_qc(m: Dict, parquet: str, case: Dict) -> None:
    """Replace ``m``'s track targets with the paper's track-QC targets."""
    from case_track_qc import free_end_target_matrix, load_track_qc, qc_summary

    n_rows = m["pc_scores"].shape[0]
    rows = m["analyzable"]
    _ens, _era5, qc = load_track_qc(
        parquet, case, n_members=rows.size if rows is not None else n_rows)
    tgt = free_end_target_matrix(qc)
    if rows is not None:
        tgt = tgt[rows]
    if tgt.shape[0] != n_rows:
        raise SystemExit(f"[pca-cmp] {m['model']}: {tgt.shape[0]} track rows != "
                         f"{n_rows} PC-score rows")
    m["targets"] = {nm: tgt[:, j] for j, nm in enumerate(_TRACK_NAMES)}
    m["track_source"] = "track_qc"
    print(f"[pca-cmp] {m['model']} track QC from {parquet}: {qc_summary(qc)}")


def _r2_table(models: Dict[str, Dict], n_pc: int = 8) -> Dict:
    """Recompute in-sample + CV R^2 for every (model, target) the SAME way."""
    tbl: Dict = {}
    for mkey, m in models.items():
        for nm, y in m["targets"].items():
            r = regression_summary(m["pc_scores"], y, n_pc=n_pc)
            tbl[(mkey, nm)] = r
    return tbl


def _fit_predict(scores: np.ndarray, y: np.ndarray, n_pc: int = 8):
    """In-sample OLS prediction of y from the leading-n_pc PCs (finite y only)."""
    from pca_stats import _ols_fit

    finite = np.isfinite(y)
    P = scores[:, :n_pc]
    beta, intercept = _ols_fit(P[finite], y[finite])
    return y[finite], intercept + P[finite] @ beta


def _pvalue_text(p: float) -> str:
    """`` (p …)`` suffix for the stats box; empty when scipy was unavailable."""
    if not np.isfinite(p):
        return ""
    return " (p < 0.001)" if p < 1e-3 else f" (p = {p:.3f})"


def _scatter_panel(ax, m: Dict, mkey: str, nm: str, r: Dict, *, n_pc: int,
                   color: str, ps) -> None:
    """One predicted-vs-ensemble-member panel (square, 1:1 line, stats box)."""
    yobs, ypred = _fit_predict(m["pc_scores"], m["targets"][nm], n_pc)
    ax.scatter(ypred, yobs, s=8, alpha=0.35, color=color, edgecolors="none")
    lo = float(min(yobs.min(), ypred.min()))
    hi = float(max(yobs.max(), ypred.max()))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, alpha=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=ps.TICK_LABEL_SIZE_DENSE)
    r2_in = f"{r['r2_insample']:.2f}" if np.isfinite(r["r2_insample"]) else "n/a"
    r2_cv = f"{r['r2_cv']:.2f}" if np.isfinite(r["r2_cv"]) else "n/a"
    ax.text(0.97, 0.03,
            f"N = {r['n_used']}\n"
            f"$R^2$ in = {r2_in}{_pvalue_text(r.get('pvalue', float('nan')))}\n"
            f"$R^2$ CV = {r2_cv}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=ps.TICK_LABEL_SIZE_DENSE,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7",
                      boxstyle="round,pad=0.3"),
            zorder=6)


def _domain_lines(models: Dict[str, Dict], target_names: list) -> list:
    """Two subordinate title lines spelling out the precursor and impact domains.

    Reads whichever npz carries each piece (the cBottle npz stores the boxes
    explicitly; the makani precursor box is recovered from its pixel index);
    a line degrades to the domain's name alone when no box is available.
    """
    from plot_style import format_bbox  # matplotlib-importing

    def first(key: str):
        for m in models.values():
            if m.get(key):
                return m[key]
        return None

    pvars = first("variables")
    ptxt = " + ".join(pvars) if pvars else "precursor fields"
    pbox = first("precursor_box")
    pcoords = f" ({format_bbox(pbox)})" if pbox else ""
    lines = [f"precursor EOFs: free-end {ptxt} over the synoptic domain{pcoords}"]

    if "tc_lat" in target_names:  # track impact (Ian)
        tbox = first("tc_domain")
        tcoords = f" ({format_bbox(tbox)})" if tbox else ""
        lines.append("impact: free-end storm position + intensity from TC "
                     f"tracks in the track domain{tcoords}")
    else:  # scalar impact (PNW)
        iv = first("impact_variable") or (target_names[0] if target_names else "")
        mask = first("impact_mask_kind")
        mean = f"{mask}-mean" if mask in ("land", "sea") else "mean"
        ibox = first("impact_bbox")
        icoords = f" ({format_bbox(ibox)})" if ibox else ""
        lines.append(f"impact: free-end {mean} {iv} over the impact "
                     f"domain{icoords}")
    return lines


def _plot(out_png: Path, models: Dict[str, Dict], tbl: Dict, *, case: str, n_pc: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps  # matplotlib-importing, so after use("Agg")

    target_names = list(next(iter(models.values()))["targets"].keys())
    n_t = len(target_names)
    mkeys = list(models)
    colors = {"makani": "#1b6ca8", "cbottle": "#c1440e"}

    # Layout: one EVR row on top, then the scatter grid.  With >= 3 targets
    # (the Ian track case) the grid is one row per MODEL and one column per
    # TARGET, so the figure stays one page tall; with 1-2 targets (the PNW
    # scalar case) it is the transpose (rows = targets, columns = models).
    wide = n_t >= 3
    ncol = n_t if wide else max(len(mkeys), 1)
    n_scatter_rows = len(mkeys) if wide else n_t
    fig_w = 3.2 * ncol + 0.6
    fig_h = 2.6 + 3.2 * n_scatter_rows + 1.25  # headroom for the 3-line title
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1 + n_scatter_rows, ncol,
                          height_ratios=[2.6 / 3.2] + [1.0] * n_scatter_rows)

    # Row 0: EVR spectrum (spans up to two columns; the rest stays empty).
    ax0 = fig.add_subplot(gs[0, : min(2, ncol)])
    for mkey in mkeys:
        evr = models[mkey]["evr"][:10]
        ax0.plot(np.arange(1, evr.size + 1), 100 * evr, "o-",
                 color=colors.get(mkey, "#444"),
                 label=_model_label(mkey, models[mkey]))
    ax0.set_xlabel("EOF mode", fontsize=ps.AXIS_LABEL_SIZE_DENSE)
    ax0.set_ylabel("variance explained (%)", fontsize=ps.AXIS_LABEL_SIZE_DENSE)
    ax0.tick_params(axis="both", labelsize=ps.TICK_LABEL_SIZE_DENSE)
    ax0.set_title("EOF spectrum", fontsize=ps.TITLE_SIZE_DENSE)
    ax0.legend(fontsize=ps.LEGEND_SIZE_DENSE, frameon=False)
    ax0.grid(True, alpha=0.25)

    # Scatter grid: predicted vs ensemble member, one panel per (model, target).
    scatter_axes = []
    for ri in range(n_scatter_rows):
        for ci in range(ncol):
            mkey, nm = (
                (mkeys[ri], target_names[ci]) if wide
                else (mkeys[ci], target_names[ri])
            )
            ax = fig.add_subplot(gs[1 + ri, ci])
            m = models[mkey]
            if nm not in m["targets"]:
                ax.axis("off")
                continue
            scatter_axes.append(ax)
            _scatter_panel(ax, m, mkey, nm, tbl[(mkey, nm)], n_pc=n_pc,
                           color=colors.get(mkey, "#444"), ps=ps)
            mlabel = _model_label(mkey, m)
            if wide:
                # Column header = target; row identity = model (left edge).
                if ri == 0:
                    ax.set_title(_target_label(nm),
                                 fontsize=ps.AXIS_LABEL_SIZE_DENSE)
                if ci == 0:
                    ax.set_ylabel(f"{mlabel}\nensemble member",
                                  fontsize=ps.AXIS_LABEL_SIZE_DENSE)
            else:
                # Column header = model; row identity = target (left edge).
                if ri == 0:
                    ax.set_title(mlabel, fontsize=ps.AXIS_LABEL_SIZE_DENSE)
                if ci == 0:
                    ax.set_ylabel(f"ensemble-member\n{_target_label(nm)}",
                                  fontsize=ps.AXIS_LABEL_SIZE_DENSE)
            if ri == n_scatter_rows - 1:
                ax.set_xlabel(f"{n_pc}-PC prediction",
                              fontsize=ps.AXIS_LABEL_SIZE_DENSE)

    ps.add_subplot_labels([ax0, *scatter_axes],
                          fontsize=ps.SUBPLOT_LABEL_SIZE_DENSE)
    # Three-line title: the headline, then two subordinate gray lines that
    # spell out what the "synoptic" and "impact" domains are for this case.
    # Anchored in inches from the top so both figure heights read the same.
    fig.suptitle(f"{case}: synoptic-PCA precursor→impact regression "
                 f"(leading {n_pc} PCs)",
                 fontsize=ps.TITLE_SIZE, y=1.0 - 0.08 / fig_h, va="top")
    fig.text(0.5, 1.0 - 0.44 / fig_h,
             "\n".join(_domain_lines(models, target_names)),
             ha="center", va="top", fontsize=ps.AXIS_LABEL_SIZE_DENSE,
             color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.95 / fig_h))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-npz", required=True)
    p.add_argument("--cbottle-npz", default=None,
                   help="Paper's synoptic_pca_end.npz for cBottle (optional).")
    p.add_argument("--makani-parquet", default=None,
                   help="makani tracks parquet: recompute makani's track targets "
                   "through the paper's track QC (needed only for an npz written "
                   "before synoptic_pca_makani.py used track QC).")
    p.add_argument("--cbottle-parquet", default=None,
                   help="cBottle tracks parquet (the paper's tc_tracks_end.parquet); "
                   "required for a track case such as Ian.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--case", default="")
    p.add_argument("--n-pc", type=int, default=8)
    args = p.parse_args(argv)

    models: Dict[str, Dict] = {"makani": _read_pca_npz(args.makani_npz)}
    if args.cbottle_npz and Path(args.cbottle_npz).exists():
        models["cbottle"] = _read_pca_npz(args.cbottle_npz)
    else:
        print("[pca-cmp] WARNING: no cBottle npz; makani-only.", file=sys.stderr)

    # Track targets: only the paper's track QC (see the module docstring).
    parquets = {"makani": args.makani_parquet, "cbottle": args.cbottle_parquet}
    for mkey, m in models.items():
        if m["track_source"] is None:
            continue
        if parquets.get(mkey):
            if not args.case:
                raise SystemExit("[pca-cmp] --case is required with a tracks parquet.")
            _apply_track_qc(m, parquets[mkey], dl.load_case(args.case))
        elif m["track_source"] != "track_qc":
            raise SystemExit(
                f"[pca-cmp] {mkey} npz holds path-0 track targets, which can be "
                f"another storm; pass --{mkey}-parquet to recompute them through "
                f"the paper's track QC.")

    # ``--case`` (the slug, e.g. "ian") keys the output filenames; the figure
    # titles use the npz's stored display name (e.g. "Hurricane Ian") when one
    # exists, falling back to the slug.
    case = args.case or models["makani"]["case"]
    display = next((m["case"] for m in models.values() if m["case"]), case)
    tbl = _r2_table(models, n_pc=args.n_pc)

    print(f"[pca-cmp] {case}: leading-{args.n_pc}-PC impact regression R^2")
    print(f"  {'model':<8} {'target':<12} {'R2_insample':>12} {'R2_cv':>8} {'n':>6}")
    for (mkey, nm), r in tbl.items():
        print(f"  {mkey:<8} {nm:<12} {r['r2_insample']:>12.3f} "
              f"{r['r2_cv']:>8.3f} {r['n_used']:>6}")
    for mkey, m in models.items():
        if m["era5_pc_percentile"] is not None:
            pcts = [f"PC{j+1}={m['era5_pc_percentile'][j]:.0f}"
                    for j in range(min(args.n_pc, m["era5_pc_percentile"].size))]
            print(f"[pca-cmp] {mkey} ERA5 percentiles: {pcts}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"synoptic_pca_compare_{case or 'case'}.npz".replace(" ", "_"),
        r2_rows=np.array([f"{mk}:{nm}" for (mk, nm) in tbl]),
        r2_insample=np.array([tbl[k]["r2_insample"] for k in tbl], float),
        r2_cv=np.array([tbl[k]["r2_cv"] for k in tbl], float),
        n_used=np.array([tbl[k]["n_used"] for k in tbl], int),
    )
    _plot(out / f"synoptic_pca_compare_{case or 'case'}.png".replace(" ", "_"),
          models, tbl, case=display, n_pc=args.n_pc)
    print(f"[pca-cmp] wrote comparison to {out}")


if __name__ == "__main__":
    main()
