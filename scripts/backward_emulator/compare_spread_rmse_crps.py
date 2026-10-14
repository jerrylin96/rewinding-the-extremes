"""Spread / RMSE / CRPS vs lead for the backward-emulator comparison.

Runs the shared cos-lat reducer (:mod:`latlon_metrics`) over one variable and
region for one or more **field sources**, and writes an ``.npz`` of the curves
plus an overlay figure.  A *field source* is anything exposing ``lat``, ``lon``,
``lead_hours`` and ``load_variable(name) -> (ens (N,L,H,W), truth (L,H,W))`` on the
721x1440 lat/lon grid:

* :class:`makani_source.MakaniBackwardSource` -- native 0.25 deg, bundled ERA5
  truth (the fully-wired path).
* cBottle (added on the analysis HPC): a source that regrids the HPX ensemble
  ->lat/lon and carries the round-tripped ERA5 reference as its truth, so each
  model is scored against its own resolution-matched baseline while the reducer
  and region pixels stay identical.  See ``# HPC`` note in :func:`main`.

Because both sources yield ``(ens, truth)`` on the same grid and go through the
same reducer over the same region, any difference in the curves is a difference in
the models.  Region metadata (box, timezone, IC time) is read from the existing
cBottle case YAML so the two studies share definitions.

Example (makani side):
    python compare_spread_rmse_crps.py \\
        --makani-dir .../backward_emulator_data/2021_pnw_heatwave \\
        --case pnw_heatwave --variable z500 --region-name synoptic \\
        --out-dir ./out/pnw/spread_rmse_crps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SHARED))

from latlon_metrics import Region, spread_rmse_crps_from_source  # noqa: E402
from makani_source import MakaniBackwardSource, resolve_expected_anchor  # noqa: E402

import _dispatch_lib as dl  # noqa: E402
import var_metadata as vm  # noqa: E402

# The pinned frame is the window END (forward frame -1) for these backward,
# end-anchored ensembles -- excluded from the degenerate-pixel guard.
def _conditioning_frames(n_frames: int) -> np.ndarray:
    return np.array([n_frames - 1], dtype=int)


def run_source(source, variable: str, region: Optional[Region],
               weight_mask: Optional[np.ndarray] = None,
               alive: Optional[np.ndarray] = None) -> Dict:
    """Reduce one variable/region for a field source; add the spread/skill ratio.

    Uses the memory-safe region path (streams only the box's pixels), so it scales
    to the 1000-member case-study dumps.  ``alive`` (N, L) applies survivor
    censoring (makani); ``None`` reduces over all members (cBottle).
    """
    r = spread_rmse_crps_from_source(
        source, variable, region=region, weight_mask=weight_mask,
        conditioning_frames=_conditioning_frames(source.n_frames), alive=alive,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        r["ssr"] = r["spread"] / r["skill"]  # spread/skill ratio (~1 = calibrated)
    r["lead_hours"] = np.asarray(source.lead_hours, dtype=float)
    return r


def _save_npz(path: Path, results: Dict[str, Dict], *, variable: str,
              region: Optional[Region], region_name: str, case_meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "variable": variable,
        "region_name": region_name,
        "region": np.asarray(region if region is not None else [], dtype=float),
        "methods": np.array(list(results), dtype=object),
        **{k: np.array(v) for k, v in case_meta.items()},
    }
    for method, r in results.items():
        for key in ("spread", "skill", "member_rmse", "mean_error", "crps",
                    "ssr", "lead_hours", "survival"):
            if key in r:
                payload[f"{method}::{key}"] = np.asarray(r[key], dtype=float)
        payload[f"{method}::n_alive"] = np.asarray(r.get("n_alive", []), dtype=np.int64)
        payload[f"{method}::n_members"] = np.int64(r["n_members"])
        payload[f"{method}::n_pix"] = np.int64(r["n_pix"])
        payload[f"{method}::n_pix_degenerate"] = np.int64(r["n_pix_degenerate"])
        payload[f"{method}::censored"] = np.bool_(bool(r.get("censored", False)))
    np.savez(path, **payload)


_METHOD_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple"]


def _domain_label(region: Optional[Region]) -> str:
    """Exact analysis box for the figure title, in degree-direction form.

    A purpose-name like "synoptic" means nothing to an SI reader; the
    coordinates it resolves to are what make the statistics reproducible,
    so the title carries those (matching the SI caption convention, e.g.
    "170°W–100°W, 25°N–70°N").
    """
    if region is None:
        return "global"

    def _deg(v: float, pos: str, neg: str) -> str:
        if v == 0 or abs(v) == 180:
            return f"{abs(v):g}\N{DEGREE SIGN}"
        return f"{abs(v):g}\N{DEGREE SIGN}{pos if v > 0 else neg}"

    lon0, lon1, lat0, lat1 = region
    return (f"{_deg(lon0, 'E', 'W')}\N{EN DASH}{_deg(lon1, 'E', 'W')}, "
            f"{_deg(lat0, 'N', 'S')}\N{EN DASH}{_deg(lat1, 'N', 'S')}")


def _plot(path: Path, results: Dict[str, Dict], *, variable: str,
          region: Optional[Region], case_meta: Dict,
          land_masked: bool = False) -> None:
    """Five-panel spread/RMSE/CRPS comparison in the manuscript S6-S11 layout
    (SSR row, signed-error row, then spread/per-member-RMSE/CRPS), plus a survival
    row when any method is censored.  Series are cBottle vs makani (one colour
    each); values are in display units; the x-axis is local wall-clock time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from local_time_axis import format_local_time_axis  # noqa: E402
    import plot_style as ps  # noqa: E402

    start_iso = case_meta.get("start_time")
    tz = case_meta.get("timezone", "UTC")
    lead = np.asarray(next(iter(results.values()))["lead_hours"], float)
    colors = {m: _METHOD_COLORS[i % len(_METHOD_COLORS)]
              for i, m in enumerate(results)}
    has_surv = any("survival" in r and r.get("censored") for r in results.values())

    def disp(y):
        return np.asarray(vm.to_display_units(variable, np.asarray(y, float)))

    def draw(ax, key, *, convert=True, pinned=True):
        for method, r in results.items():
            y = disp(r[key]) if convert else np.asarray(r[key], float)
            ax.plot(lead, y, color=colors[method], lw=1.6, label=method)
            if pinned:  # enlarged marker at the pinned end (forward frame -1)
                ax.plot(lead[-1], y[-1], "o", color=colors[method], ms=8, zorder=5)
        ax.grid(True, alpha=0.3)
        ps.style_axes(ax)

    def time_axis(ax, *, labels: bool, thin: int = 1):
        """Local wall-clock x-axis; only the bottom row shows tick labels/title."""
        try:
            format_local_time_axis(ax, lead, start_iso, tz)
            if thin > 1:
                ax.set_xticks(ax.get_xticks()[::thin])
        except Exception:
            ax.set_xlabel("Lead into window (h)" if labels else "")
        if not labels:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    nrows = 4 if has_surv else 3
    fig = plt.figure(figsize=(10, 2.6 * nrows))
    gs = fig.add_gridspec(nrows, 3, hspace=0.35, wspace=0.38)
    ax_ssr = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, :])
    ax_sp = fig.add_subplot(gs[2, 0])
    ax_rm = fig.add_subplot(gs[2, 1])
    ax_cr = fig.add_subplot(gs[2, 2])
    ax_su = fig.add_subplot(gs[3, :]) if has_surv else None

    draw(ax_ssr, "ssr", convert=False)
    ax_ssr.axhline(1.0, color="0.5", lw=0.9, ls="--")  # consistency (Text S6)
    ax_ssr.set_ylabel("Spread / skill ratio", fontsize=ps.AXIS_LABEL_SIZE)
    ax_ssr.legend(fontsize=ps.LEGEND_SIZE)

    draw(ax_err, "mean_error")
    ax_err.axhline(0.0, color="0.5", lw=0.9, ls="--")
    ax_err.set_ylabel(vm.axis_label(variable, "mean error"), fontsize=ps.AXIS_LABEL_SIZE)

    draw(ax_sp, "spread")
    ax_sp.set_ylabel(vm.axis_label(variable, "spread"), fontsize=ps.AXIS_LABEL_SIZE)
    draw(ax_rm, "member_rmse")
    ax_rm.set_ylabel(vm.axis_label(variable, "per-member RMSE"), fontsize=ps.AXIS_LABEL_SIZE)
    draw(ax_cr, "crps")
    ax_cr.set_ylabel(vm.axis_label(variable, "CRPS"), fontsize=ps.AXIS_LABEL_SIZE)

    if ax_su is not None:
        # Every method that recorded a survival curve is drawn, not just the
        # censored one: the flat 100% line of the by-construction-stable
        # method (cBottle) is the stability contrast the panel exists to
        # show.  That flat line is dashed so it stays visible where the
        # censored curve saturates at 100% on top of it.
        for method, r in results.items():
            if "survival" not in r:
                continue
            ax_su.plot(lead, np.asarray(r["survival"], float) * 100.0,
                       color=colors[method], lw=1.6,
                       ls="-" if r.get("censored") else "--", label=method)
        ax_su.set_ylim(0, 105)
        ax_su.set_ylabel("Survivors (%)", fontsize=ps.AXIS_LABEL_SIZE)
        ax_su.grid(True, alpha=0.3)
        ax_su.legend(fontsize=ps.LEGEND_SIZE, loc="lower right")
        ps.style_axes(ax_su)

    # Only the bottom row of each column stack carries time-axis labels; the
    # narrow triptych thins its ticks so the two-line stamps don't collide.
    triptych_labels = not has_surv
    time_axis(ax_ssr, labels=False)
    time_axis(ax_err, labels=False)
    for ax in (ax_sp, ax_rm, ax_cr):
        time_axis(ax, labels=triptych_labels, thin=2)
    if triptych_labels:  # avoid the xlabel repeating under all three panels
        ax_sp.set_xlabel(""); ax_cr.set_xlabel("")
    if ax_su is not None:
        time_axis(ax_su, labels=True)

    ps.add_subplot_labels([a for a in (ax_ssr, ax_err, ax_sp, ax_rm, ax_cr, ax_su)
                           if a is not None])
    title = f"{vm.long_name(variable)} spread / skill / CRPS"
    if case_meta.get("display_name"):
        title = f"{case_meta['display_name']}: {title}"
    # The exact analysis box goes on its own line: appended to the case +
    # variable line it overflows the figure width, and the box is what the
    # region purpose-name never told the reader anyway.  A mask is named
    # too: a land-masked box mean is a different number from an unmasked
    # one, and the figure has to say which it is.
    title += f"\nDomain: {_domain_label(region)}"
    if land_masked:
        title += ", land only"
    fig.suptitle(title, fontsize=ps.SUPTITLE_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _resolve_region(case: Dict, region_name: str) -> Tuple[Optional[Region], str]:
    if region_name in ("global", "none"):
        return None, "global"
    box = dl.region_box(case, region_name)
    if box is None:
        raise SystemExit(
            f"region '{region_name}' not in case '{case['name']}'; "
            f"available: {sorted((case.get('regions') or {}))}"
        )
    return box, region_name


def _load_land_mask(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Load a boolean ``(H, W)`` land mask and check it matches the source grid.

    One shared mask (produced by ``make_land_mask.py`` from cBottle's ERA5 sst) is
    applied to BOTH models -- they are on the same 721x1440 grid, so the masked
    pixels are identical for makani and cBottle, and ``select_pixels`` intersects it
    with the region box.
    """
    m = np.load(path)
    H, W = int(lat.shape[0]), int(lon.shape[0])
    if m.shape != (H, W):
        raise SystemExit(
            f"land mask {path} shape {m.shape} != source grid ({H}, {W}); "
            f"regenerate it on the 721x1440 grid with make_land_mask.py."
        )
    return m.astype(bool)


def _load_or_compute_survival(
    makani, *, cache_path: Optional[Path], makani_analysis_dir: Optional[str]
) -> Dict:
    """Survivor mask for the makani source, from an npz cache when possible.

    The ``range`` detector streams t2m for all 1000 members over the whole globe
    (~50 GB) and its result depends ONLY on t2m -- it is identical for every
    ``(variable, region)`` in a case.  So a fan-out of jobs should compute it once
    and reuse it: pass ``--survival-npz PATH`` and the first run populates the
    cache while the rest load it (shape-checked against this source).  Schema is
    interchangeable with ``survival.py``'s standalone ``main()`` output.
    """
    from survival import range_alive_mask  # noqa: E402

    N, L = makani.n_members, makani.n_frames
    if cache_path is not None and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=False)
        alive = z["alive"]
        if alive.shape == (N, L):
            n_alive = z["n_alive"] if "n_alive" in z else alive.sum(axis=0)
            survival = (z["survival"] if "survival" in z
                        else n_alive.astype(np.float64) / float(N))
            print(f"[compare] survivor mask loaded from cache {cache_path} "
                  f"(N={N}, L={L})")
            return {
                "alive": alive, "survival": survival, "n_alive": n_alive,
                "detector": str(z["detector"]) if "detector" in z else "range",
                "channel": str(z["channel"]) if "channel" in z else "t2m",
                "bounds": tuple(z["bounds"]) if "bounds" in z else None,
                "n_members": N,
                "n_diverged": int((~alive.all(axis=1)).sum()),
                "cached": True,
            }
        print(f"[compare] cache {cache_path} shape {alive.shape} != ({N},{L}); "
              f"recomputing.", file=sys.stderr)

    surv = range_alive_mask(makani, makani_analysis_dir=makani_analysis_dir)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path, alive=surv["alive"], survival=surv["survival"],
            n_alive=surv["n_alive"], fail_leads=surv["fail_leads"],
            lead_hours=np.asarray(makani.lead_hours, float),
            channel=np.array(surv["channel"]), bounds=np.asarray(surv["bounds"]),
            detector=np.array(surv["detector"]),
        )
        print(f"[compare] survivor mask computed and cached to {cache_path}")
    return surv


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True,
                   help="Directory of trimmed/raw *fields*.h5 dumps for the case.")
    p.add_argument("--case", required=True,
                   help="cBottle case YAML name (for regions/timezone/IC), e.g. "
                   "pnw_heatwave or ian.")
    p.add_argument("--variable", required=True)
    p.add_argument("--region-name", default="synoptic")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--expected-anchor", default=None,
                   help="ISO-hour prefix the dumps' anchor valid-time must match "
                   "(default: the case's window end, start_time + 66 h; "
                   "'none' disables the check).")
    p.add_argument("--makani-label", default="Autoregressive (makani)")
    p.add_argument("--censor", dest="censor", action="store_true", default=True,
                   help="Apply makani survivor censoring (default on).")
    p.add_argument("--no-censor", dest="censor", action="store_false",
                   help="Reduce over all members (no censoring).")
    p.add_argument("--makani-analysis-dir", default=None,
                   help="Path to run-it-back/analysis for ensemble_horizon parity.")
    p.add_argument("--survival-npz", default=None,
                   help="Cache path for the survivor mask. If it exists it is "
                   "loaded (skips the ~50 GB t2m scan); if it does not, the mask "
                   "is computed and written there for reuse. The mask depends only "
                   "on t2m, so one file serves every (variable, region) in a case.")
    p.add_argument("--land-mask", default=None,
                   help="Path to a boolean (H, W) .npy land mask on the 721x1440 "
                   "grid (from make_land_mask.py). Intersected with the region and "
                   "applied identically to both models (e.g. PNW t2m over land).")
    p.add_argument("--cbottle", dest="cbottle", action="store_true", default=True,
                   help="Overlay the cBottle end-mode ensemble (default on).")
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false",
                   help="makani-only; skip the cBottle overlay.")
    p.add_argument("--cbottle-label", default="Video diffusion (cBottle)")
    p.add_argument("--cbottle-size", type=int, default=None,
                   help="cBottle ensemble size for the zarr path (default: case size).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Members per HPX->lat/lon regrid batch (cBottle).")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    region, region_name = _resolve_region(case, args.region_name)
    case_meta = {
        "display_name": case.get("display_name", args.case),
        "timezone": dl.resolve_timezone(case),
        "start_time": dl.start_time_iso_from_case(case),
    }

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames,
                                  expected_anchor_iso=resolve_expected_anchor(case, args.expected_anchor))

    land_mask = None
    if args.land_mask:
        land_mask = _load_land_mask(Path(args.land_mask), makani.lat, makani.lon)
        region_name = f"{region_name}_land"
        print(f"[compare] land mask {args.land_mask}: "
              f"{int(land_mask.sum())}/{land_mask.size} land pixels globally")

    alive = None
    surv_info = None
    if args.censor:
        cache = Path(args.survival_npz) if args.survival_npz else None
        surv_info = _load_or_compute_survival(
            makani, cache_path=cache, makani_analysis_dir=args.makani_analysis_dir)
        alive = surv_info["alive"]
        print(f"[compare] survivor censoring ({surv_info['detector']}, "
              f"{surv_info['channel']} {surv_info['bounds']}): "
              f"{surv_info['n_diverged']}/{surv_info['n_members']} members diverge; "
              f"survival at 66 h = {float(surv_info['survival'][0]):.3f}")

    m_res = run_source(makani, args.variable, region, weight_mask=land_mask,
                       alive=alive)
    m_res["censored"] = bool(args.censor)
    if surv_info is not None:
        m_res["survival"] = surv_info["survival"]  # divergence over the full grid
        m_res["n_alive"] = surv_info["n_alive"]
    results: Dict[str, Dict] = {args.makani_label: m_res}

    # cBottle overlay: regrid the HPX end-mode ensemble ->lat/lon and score it
    # against the round-tripped ERA5 truth, through the identical reducer/region.
    # No censoring (all end-conditioned members are valid). Heavy env only. A
    # failure here is fatal on purpose: a makani-only output written with exit 0
    # is byte-for-byte indistinguishable from an intentional --no-cbottle run.
    if args.cbottle:
        try:
            from cbottle_source import CBottleLatLonSource  # noqa: E402
            cb = CBottleLatLonSource(case, mode="end", size=args.cbottle_size,
                                     batch_size=args.batch_size)
            cb_res = run_source(cb, args.variable, region, weight_mask=land_mask,
                                alive=None)
            cb_res["censored"] = False
            # End-conditioned diffusion members cannot blow up the way
            # autoregressive rollouts do -- all members are valid -- so the
            # survival curve is identically 100%.  Record it explicitly so
            # the figure/npz state the stability contrast rather than
            # leaving cBottle absent from the survivors panel.
            n_lead = len(cb_res["lead_hours"])
            cb_res["survival"] = np.ones(n_lead)
            cb_res["n_alive"] = np.full(n_lead, int(cb_res["n_members"]),
                                        dtype=np.int64)
            results[args.cbottle_label] = cb_res
            print(f"[compare] cBottle overlay: N={cb_res['n_members']} "
                  f"pix={cb_res['n_pix']}")
        except Exception as exc:  # noqa: BLE001 - re-raised: never a silent makani-only output
            print(f"[compare] ERROR: cBottle overlay failed ({type(exc).__name__}: {exc}). "
                  "Nothing written. Pass --no-cbottle for an intentional makani-only run.",
                  file=sys.stderr)
            raise

    out = Path(args.out_dir)
    stem = f"spread_rmse_crps_{args.variable}_{region_name}"
    _save_npz(out / f"{stem}.npz", results, variable=args.variable,
              region=region, region_name=region_name, case_meta=case_meta)
    _plot(out / f"{stem}.png", results, variable=args.variable,
          region=region, case_meta=case_meta,
          land_masked=land_mask is not None)

    m = results[args.makani_label]
    print(f"[compare] {args.variable} / {region_name}: N={m['n_members']} "
          f"pix={m['n_pix']} (deg={m['n_pix_degenerate']})")
    print(f"[compare]   mean SSR={np.nanmean(m['ssr']):.3f}  "
          f"mean member_rmse={m['member_rmse'].mean():.4g}  "
          f"mean CRPS={m['crps'].mean():.4g}")
    print(f"[compare] wrote {out / (stem + '.npz')} and {stem}.png")


if __name__ == "__main__":
    main()
