"""Ageostrophic-imbalance comparison: makani (native) vs cBottle (regridded).

The paper's physical-consistency diagnostic (Text S5, Figs. S15-S17): the
``cos(lat)``-weighted domain mean of ``|V_a|/|V|`` -- the fraction of the 500 hPa
wind that is *not* in geostrophic balance -- and its component magnitudes
(``|V|``, ``|V_g|``, ``|V_a|``), per member, per lead.  A larger ageostrophic
fraction means the wind and mass fields are less mutually consistent.

This computes that diagnostic for both rewind methods, reusing the paper's exact
core (`compute_ageostrophic._spds_from_uvz` / `_weighted_domain_means` /
`_means_to_diag_dict` / `geostrophic_wind`) so any difference in the curves is a
difference in the models, not the analysis.  Following the locked design:

* **makani enters native 0.25 deg**, scored against its own native-ERA5 baseline;
  **cBottle** is regridded HPX->lat/lon and scored against the **round-tripped**
  ERA5 (era5_rt).  The ageostrophic fraction is resolution-dependent, so each model
  is measured against truth at its own resolution.  The comparison is then each
  model's DEPARTURE from its own baseline -- a fidelity result, not a sharpness
  artefact -- in whichever direction the data land.  We do NOT presuppose which
  model departs further (an early 2-member check hinted cBottle's raw fraction sat
  above makani's; the baseline-relative excess from the full run is what decides).
* **Domain: the paper's global 30-60 deg |lat| midlatitude band**, all longitudes --
  the broad "physical consistency" measure, identical for both models. Override
  with --lat-min/--lat-max/--lon-min/--lon-max.
* **makani survivor censoring**: a diverged member's wind field is garbage, so the
  npz carries makani's survivor mask (``makani::alive``, from the shared survival
  cache) and every makani statistic is taken over survivors; cBottle needs none.

This script computes; ``plot_ageostrophic_compare.py`` draws the figure from the npz
it writes, so restyling the figure never needs this job again.

Heavy env (torch + earth2grid via the cBottle regrid). A failing cBottle overlay
raises and writes nothing; ``--no-cbottle`` requests a makani-only run.

Usage:
    python ageostrophic_compare.py --makani-dir .../ian --case ian \\
        --out-dir ./out/ian/ageostrophic --survival-npz ./out/ian/survival_ian.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
from makani_source import MakaniBackwardSource, resolve_expected_anchor  # noqa: E402

_KEYS = ("ratio", "mean_v", "mean_vg", "mean_va")


def npz_name(case: str, level: int) -> str:
    """File name of one case's output; ``plot_ageostrophic_compare.py`` reads it."""
    return f"ageostrophic_compare_{case}_L{level}.npz"


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


def _save_npz(path: Path, models, *, lead_hours, band_label, level, pinned_frame,
              case_meta, alive, survival=None):
    """Write one case's traces.  ``alive`` is makani's (members, frames) survivor
    mask; ``None`` (an explicit ``--no-censor`` run) is stored as all-alive with
    ``makani::censored`` False, so the reader can say which it is plotting."""
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
    m_shape = models["makani"]["ens"]["ratio"].shape
    payload["makani::censored"] = np.bool_(alive is not None)
    payload["makani::alive"] = (np.ones(m_shape, bool) if alive is None
                                else np.asarray(alive, bool))
    if payload["makani::alive"].shape != m_shape:
        raise ValueError(f"survivor mask {payload['makani::alive'].shape} != "
                         f"makani traces {m_shape}")
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
    p.add_argument("--expected-anchor", default=None,
                   help="ISO-hour prefix the dumps' anchor valid-time must match "
                   "(default: the case's window end, start_time + 66 h; "
                   "'none' disables the check).")
    p.add_argument("--censor", dest="censor", action="store_true", default=True)
    p.add_argument("--no-censor", dest="censor", action="store_false")
    p.add_argument("--survival-npz", default=None)
    p.add_argument("--makani-analysis-dir", default=None)
    p.add_argument("--cbottle", dest="cbottle", action="store_true", default=True)
    p.add_argument("--no-cbottle", dest="cbottle", action="store_false")
    p.add_argument("--cbottle-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    case_meta = {
        "display_name": case.get("display_name", args.case),
        "timezone": dl.resolve_timezone(case),
        "start_time": dl.start_time_iso_from_case(case),
    }
    names = [f"u{args.level}", f"v{args.level}", f"z{args.level}"]

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames,
                                  expected_anchor_iso=resolve_expected_anchor(case, args.expected_anchor))
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
        # Forward frame 0 is the window start, 66 h back; frame -1 is the
        # pinned present (always 1.0).
        print(f"[ageo-cmp] survivor censoring: survival@66h="
              f"{float(np.asarray(survival)[0]):.3f}")

    models: Dict[str, Dict] = {"makani": {"ens": m_ens, "baseline": m_base}}

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
            models["cbottle"] = {"ens": c_ens, "baseline": c_base}
            print(f"[ageo-cmp] cBottle overlay: N={cb.n_members}")
        except Exception as exc:  # noqa: BLE001 - re-raised: never a silent makani-only output
            print(f"[ageo-cmp] ERROR: cBottle overlay failed ({type(exc).__name__}: {exc}). "
                  "Nothing written. Pass --no-cbottle for an intentional makani-only run.",
                  file=sys.stderr)
            raise

    out = Path(args.out_dir) / npz_name(args.case, args.level)
    _save_npz(out, models, lead_hours=makani.lead_hours,
              band_label=band, level=args.level, pinned_frame=pinned,
              case_meta=case_meta, alive=alive, survival=survival)
    print(f"[ageo-cmp] wrote {out}")


if __name__ == "__main__":
    main()
