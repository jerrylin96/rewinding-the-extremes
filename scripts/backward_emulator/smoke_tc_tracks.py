"""Quick interactive gate for the makani TC-tracking glue.

Tracks a couple of makani members + the ERA5 reference through the real
DetectNodes/StitchNodes (the paper's `track_member_with_tempest`) on a login node,
so a wiring bug -- or a missing TempestExtremes binary, or an uncached WB2
orography fetch -- surfaces in a minute, not hours into the full 1000-member job.

Usage (in e2s-custom):
    python smoke_tc_tracks.py --case ian
    python smoke_tc_tracks.py --case ian --no-orography   # skip the WB2 fetch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
EA_DIR = BW_DIR.parent / "ensemble_analysis"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(EA_DIR) not in sys.path:  # appended: never shadow our own modules
    sys.path.append(str(EA_DIR))

import _dispatch_lib as dl  # noqa: E402
import tc_tracks_compare as tt  # noqa: E402
from makani_source import MakaniBackwardSource  # noqa: E402


def _check(cond, msg):
    if not cond:
        print(f"[smoke-tc] FAIL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[smoke-tc]   ok: {msg}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", default="ian")
    p.add_argument("--makani-dir", default=None)
    p.add_argument("--n-members", type=int, default=2)
    p.add_argument("--no-orography", dest="orography", action="store_false", default=True)
    p.add_argument("--data-root",
                   default="/projectnb/eb-general/jlin404/scratch/backward_emulator_data")
    args = p.parse_args(argv)

    # Fail early + clearly if the binaries aren't on PATH.
    from tempest_extremes_runner import _check_binaries
    _check_binaries()
    print("[smoke-tc]   ok: DetectNodes/StitchNodes on PATH")

    case = dl.load_case(args.case)
    warm_core = bool((case.get("tc_tracks") or {}).get("warm_core", False))
    variables = tt._tempest_variables(warm_core)
    subdir = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}.get(args.case, args.case)
    makani_dir = args.makani_dir or f"{args.data_root}/{subdir}"

    makani = MakaniBackwardSource(makani_dir, n_frames=12)
    lats, lons = makani.lat, makani.lon
    step_h = float(abs(makani.step_hours))
    _check(all(v in makani.channels for v in variables),
           f"tracker channels present {variables}")
    print(f"[smoke-tc] case={args.case} warm_core={warm_core} step={step_h}h "
          f"grid={lats.shape[0]}x{lons.shape[0]}")

    orography = None
    if args.orography:
        from tempest_extremes_runner import fetch_surface_orography
        print("[smoke-tc] fetching WB2 orography (checks network/cache) ...", flush=True)
        orography = fetch_surface_orography(np.asarray(lats), np.asarray(lons))
        _check(orography.shape == (lats.shape[0], lons.shape[0]),
               f"orography grid matches ({orography.shape})")

    work = BW_DIR / "_smoke_tc_work"
    work.mkdir(exist_ok=True)

    n = 0
    for gid, fields in makani.iter_member_fields(variables):
        slab = tt.build_member_slab(fields, variables)
        _check(slab.shape == (12, len(variables), lats.shape[0], lons.shape[0]),
               f"member {gid} slab shape {slab.shape}")
        tracks = tt._track_one(slab, variables, lats, lons, step_h, gid, work,
                               orography, warm_core)
        _check(tracks.ndim == 3 and tracks.shape[-1] == 4,
               f"member {gid} tracks shape {tracks.shape} == [P,T,4]")
        n += 1
        if n >= args.n_members:
            break

    print("[smoke-tc] tracking ERA5 reference (makani truth) ...", flush=True)
    tf = makani.truth_fields(variables)
    era5 = tt._track_one(tt.build_member_slab(tf, variables), variables, lats, lons,
                         step_h, -1, work, orography, warm_core)
    n_fix = int(np.isfinite(era5[..., 0]).sum())
    _check(era5.ndim == 3 and era5.shape[-1] == 4, f"ERA5 track shape {era5.shape}")
    print(f"[smoke-tc]   ERA5 reference finite fixes: {n_fix} "
          f"(a real storm in {args.case} should give some)")

    print(f"[smoke-tc] PASS: makani TC-tracking glue works for {args.case} "
          f"(DetectNodes+StitchNodes on native fields).")


if __name__ == "__main__":
    main()
