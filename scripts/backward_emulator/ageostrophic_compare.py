"""Ageostrophic-imbalance comparison: makani (native) vs cBottle (regridded).

The paper's physical-consistency diagnostic (Text S5, Figs. S13-S15): the
``cos(lat)``-weighted domain mean of ``|V_a|/|V|`` -- the fraction of the 500 hPa
wind that is *not* in geostrophic balance -- and its component magnitudes
(``|V|``, ``|V_g|``, ``|V_a|``), per member, per lead.  A larger ageostrophic
fraction means the wind and mass fields are less mutually consistent.

This overlays the two rewind methods on that diagnostic, reusing the paper's exact
core (`compute_ageostrophic._spds_from_uvz` / `_weighted_domain_means` /
`_means_to_diag_dict` / `geostrophic_wind`) so any difference in the curves is a
difference in the models, not the analysis.  Following the locked design:

* **makani enters native 0.25 deg**, scored against its own native-ERA5 baseline;
  **cBottle** is regridded HPX->lat/lon and scored against the **round-tripped**
  ERA5 (era5_rt).  The ageostrophic fraction is resolution-dependent (and the
  HPX->lat/lon regrid can itself inflate cBottle's absolute fraction), so each model
  is measured against truth at its own resolution.  The comparison is then each
  model's DEPARTURE from its own baseline -- a fidelity result, not a sharpness
  artefact -- in whichever direction the data land.  We do NOT presuppose which
  model departs further (an early 2-member check hinted cBottle's raw fraction sat
  above makani's; the baseline-relative excess from the full run is what decides).
* **Domain: the paper's global 30-60 deg |lat| midlatitude band**, all longitudes --
  the broad "physical consistency" measure, identical for both models. Override
  with --lat-min/--lat-max/--lon-min/--lon-max.
* **makani survivor censoring**: a diverged member's wind field is garbage, so the
  makani ensemble summary is over survivors (shared survival cache); cBottle needs
  none.

Heavy env (torch + earth2grid via the cBottle regrid). makani-only degrades
gracefully if the cBottle stores / earth2grid are unavailable.

Usage:
    python ageostrophic_compare.py --makani-dir .../ian --case ian \\
        --out-dir ./out/ian/ageostrophic --survival-npz ./out/ian/survival_ian.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
AGEO_DIR = BW_DIR.parent / "ensemble_analysis" / "submission_scripts"
# Our own dirs first; the paper's submission_scripts dir (for compute_ageostrophic)
# is APPENDED so it can never shadow a same-named module of ours.
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(AGEO_DIR) not in sys.path:
    sys.path.append(str(AGEO_DIR))

import _dispatch_lib as dl  # noqa: E402
import plot_style  # noqa: E402
from local_time_axis import format_local_time_axis  # noqa: E402
from makani_source import MakaniBackwardSource  # noqa: E402

_KEYS = ("ratio", "mean_v", "mean_vg", "mean_va")
_ROW_LABEL = {
    "ratio": r"$|V_a|/|V|$", "mean_v": r"mean $|V|$ (m/s)",
    "mean_vg": r"mean $|V_g|$ (m/s)", "mean_va": r"mean $|V_a|$ (m/s)",
}


def _domain_setup(lat, lon, args):
    """(lat_idx, lon_idx, lat_weights torch, band_label) via the paper's helpers."""
    import torch

    import compute_ageostrophic as ageo  # heavy-env
    lat_idx, lon_idx = ageo._domain_indices(
        lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise SystemExit("configured band does not intersect the grid.")
    lat_weights = torch.from_numpy(
        np.cos(np.deg2rad(lat[lat_idx])).astype(np.float32))
    band = ageo._format_band(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    return lat_idx, lon_idx, lat_weights, band


def _diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_weights):
    """{ratio, mean_v, mean_vg, mean_va} for one (u,v,z) tensor triple."""
    import compute_ageostrophic as ageo
    spd = ageo._spds_from_uvz(u, v, z, lat, lon)
    return ageo._means_to_diag_dict(
        *ageo._weighted_domain_means(*spd, lat_idx, lon_idx, lat_weights))


def _makani_members(makani, names, lat, lon, lat_idx, lon_idx, lat_weights):
    """Per-member makani traces ``{key: (M, L)}`` + native-ERA5 baseline ``{key: (L,)}``."""
    import torch

    rows: Dict[str, List[np.ndarray]] = {k: [] for k in _KEYS}
    for _gid, fields in makani.iter_member_fields(names):
        u, v, z = (torch.from_numpy(fields[n]).float() for n in names)  # (nf,H,W)
        d = _diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_weights)
        for k in _KEYS:
            rows[k].append(d[k])
    ens = {k: np.stack(rows[k], axis=0) for k in _KEYS}  # (M, L)
    tf = makani.truth_fields(names)
    u, v, z = (torch.from_numpy(tf[n]).float() for n in names)
    base = _diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_weights)
    return ens, base


def _cbottle_members(cb, names, lat, lon, lat_idx, lon_idx, lat_weights, batch_size):
    """Per-member cBottle traces ``{key: (M, L)}`` + round-tripped-ERA5 baseline."""
    M, L = cb.n_members, cb.n_frames
    ens = {k: np.empty((M, L), np.float32) for k in _KEYS}
    for b0, fields in cb.iter_member_field_batches(names, batch_size=batch_size):
        u, v, z = (fields[n] for n in names)  # torch (bs,L,H,W)
        d = _diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_weights)
        bs = u.shape[0]
        for k in _KEYS:
            ens[k][b0:b0 + bs] = d[k]
    tf = cb.roundtrip_truth_fields(names)
    u, v, z = (tf[n] for n in names)
    base = _diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_weights)
    return ens, base


def _censored_summary(traces: np.ndarray, alive: Optional[np.ndarray]):
    """Per-lead (mean, std) over members, censoring dead member-leads when given."""
    x = np.asarray(traces, dtype=np.float64)
    keep = np.isfinite(x)
    if alive is not None:
        keep = keep & np.asarray(alive, dtype=bool)
    mean = np.full(x.shape[1], np.nan)
    std = np.full(x.shape[1], np.nan)
    for l in range(x.shape[1]):
        col = x[keep[:, l], l]
        if col.size:
            mean[l] = col.mean()
            std[l] = col.std(ddof=1) if col.size > 1 else 0.0
    return mean, std


def _plot(path: Path, models: Dict[str, Dict], *, lead_hours, band_label,
          pinned_frame, case_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(_KEYS), 1, figsize=(7.2, 9.2), sharex=True)
    colors = {"makani": "#1b6ca8", "cbottle": "#c1440e"}
    lead = np.asarray(lead_hours, float)

    for row, key in enumerate(_KEYS):
        ax = axes[row]
        for mkey, m in models.items():
            c = colors.get(mkey, "#444")
            mean, std = m["summary"][key]
            ax.fill_between(lead, mean - std, mean + std, color=c, alpha=0.18, lw=0)
            ax.plot(lead, mean, color=c, lw=2.0, label=m["label"])
            base = m["baseline"][key]
            ax.plot(lead, base, color=c, lw=1.1, ls="--", alpha=0.9,
                    label=f"{m['baseline_label']}")
        ax.set_ylabel(_ROW_LABEL[key], fontsize=plot_style.AXIS_LABEL_SIZE)
        if pinned_frame is not None and 0 <= pinned_frame < lead.size:
            ax.axvline(lead[pinned_frame], color="0.5", ls=":", lw=1.0)
        plot_style.style_axes(ax)
        if row == 0:
            ax.legend(fontsize=8, ncol=2, frameon=False, loc="best")

    format_local_time_axis(
        axes[-1], lead, case_meta.get("start_time", ""),
        case_meta.get("timezone", "UTC"))
    # Two lines (headline, then the band detail bare -- no wrapping parens),
    # padded upward so neither line collides with the outside "a.)" label.
    axes[0].set_title(
        f"{case_meta.get('display_name','')}: ageostrophic imbalance\n{band_label}",
        fontsize=plot_style.AXIS_LABEL_SIZE, pad=30)
    plot_style.add_subplot_labels(axes)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_npz(path: Path, models, *, lead_hours, band_label, level, pinned_frame,
              case_meta, survival=None):
    payload = {
        "lead_hours": np.asarray(lead_hours, float),
        "band_label": np.array(band_label), "level": np.int32(level),
        "conditioning_frames": np.asarray([pinned_frame], np.int32),
        "case_name": np.array(case_meta.get("display_name", "")),
        "start_time": np.array(case_meta.get("start_time", "")),
        "timezone": np.array(case_meta.get("timezone", "UTC")),
    }
    for mkey, m in models.items():
        for k in _KEYS:
            payload[f"{mkey}::{k}_ens"] = m["ens"][k]
            payload[f"{mkey}::{k}_base"] = m["baseline"][k]
    if survival is not None:
        payload["makani::survival"] = np.asarray(survival, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--level", type=int, default=500)
    p.add_argument("--lat-min", type=float, default=30.0)
    p.add_argument("--lat-max", type=float, default=60.0)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--censor", dest="censor", action="store_true", default=True)
    p.add_argument("--no-censor", dest="censor", action="store_false")
    p.add_argument("--survival-npz", default=None)
    p.add_argument("--makani-analysis-dir", default=None)
    p.add_argument("--cbottle", dest="cbottle", action="store_true", default=True)
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false")
    p.add_argument("--cbottle-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--makani-label", default="Autoregressive")
    p.add_argument("--cbottle-label", default="Video diffusion (cBottle)")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    case_meta = {
        "display_name": case.get("display_name", args.case),
        "timezone": dl.resolve_timezone(case),
        "start_time": dl.start_time_iso_from_case(case),
    }
    names = [f"u{args.level}", f"v{args.level}", f"z{args.level}"]

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames)
    lat, lon = makani.lat, makani.lon
    lat_idx, lon_idx, lat_weights, band = _domain_setup(lat, lon, args)
    pinned = args.n_frames - 1
    print(f"[ageo-cmp] {args.case} level={args.level} band={band} "
          f"({lat_idx.size} lat x {lon_idx.size} lon)")

    # makani (native).
    print("[ageo-cmp] makani per-member (native lat/lon) ...", flush=True)
    m_ens, m_base = _makani_members(makani, names, lat, lon,
                                    lat_idx, lon_idx, lat_weights)

    alive = None
    survival = None
    if args.censor:
        from compare_spread_rmse_crps import _load_or_compute_survival
        cache = Path(args.survival_npz) if args.survival_npz else None
        surv = _load_or_compute_survival(
            makani, cache_path=cache, makani_analysis_dir=args.makani_analysis_dir)
        alive = surv["alive"]
        survival = surv["survival"]
        print(f"[ageo-cmp] survivor censoring: survival@66h="
              f"{float(np.asarray(survival)[-1]):.3f}")

    models: Dict[str, Dict] = {
        "makani": {
            "label": args.makani_label, "ens": m_ens, "baseline": m_base,
            "baseline_label": "ERA5 (native)",
            "summary": {k: _censored_summary(m_ens[k], alive) for k in _KEYS},
        }
    }

    if args.cbottle:
        try:
            from cbottle_source import CBottleLatLonSource
            cb = CBottleLatLonSource(case, mode="end", size=args.cbottle_size,
                                     batch_size=args.batch_size)
            if not np.allclose(np.asarray(cb.lat), np.asarray(lat)):
                print("[ageo-cmp] WARNING: cBottle lat grid differs from makani; "
                      "domain indices assume a shared 721x1440 grid.", file=sys.stderr)
            print("[ageo-cmp] cBottle per-member (HPX->lat/lon regrid) ...", flush=True)
            c_ens, c_base = _cbottle_members(cb, names, lat, lon, lat_idx, lon_idx,
                                             lat_weights, args.batch_size)
            models["cbottle"] = {
                "label": args.cbottle_label, "ens": c_ens, "baseline": c_base,
                "baseline_label": "ERA5 (round-trip)",
                "summary": {k: _censored_summary(c_ens[k], None) for k in _KEYS},
            }
            print(f"[ageo-cmp] cBottle overlay: N={cb.n_members}")
        except Exception as exc:  # noqa: BLE001 - degrade to makani-only
            print(f"[ageo-cmp] WARNING: cBottle overlay skipped "
                  f"({type(exc).__name__}: {exc}); makani-only.", file=sys.stderr)

    out = Path(args.out_dir)
    stem = f"ageostrophic_compare_{args.case}_L{args.level}"
    _save_npz(out / f"{stem}.npz", models, lead_hours=makani.lead_hours,
              band_label=band, level=args.level, pinned_frame=pinned,
              case_meta=case_meta, survival=survival)
    _plot(out / f"{stem}.png", models, lead_hours=makani.lead_hours,
          band_label=band, pinned_frame=pinned, case_meta=case_meta)

    r = models["makani"]["summary"]["ratio"][0]
    print(f"[ageo-cmp] makani mean |V_a|/|V| over leads = {np.nanmean(r):.4f}")
    if "cbottle" in models:
        rc = models["cbottle"]["summary"]["ratio"][0]
        print(f"[ageo-cmp] cBottle mean |V_a|/|V| over leads = {np.nanmean(rc):.4f}")
    print(f"[ageo-cmp] wrote {out / (stem + '.npz')} and {stem}.png")


if __name__ == "__main__":
    main()
