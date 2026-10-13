"""Quick interactive gate for the ageostrophic comparison glue.

Runs the paper's ageostrophic core through both sources on a handful of members --
makani via ``iter_member_fields`` (native lat/lon) and cBottle via
``iter_member_field_batches`` (HPX->lat/lon regrid) -- and checks the domain-mean
``|V_a|/|V|`` is finite and in ``[0, 1]`` for each.  Seconds on a login node, so a
wiring bug in the new full-field iterators surfaces here, not hours into a job.

Usage (in e2s-custom):
    python smoke_ageostrophic.py --case pnw_heatwave
    python smoke_ageostrophic.py --case ian --n-members 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _dispatch_lib as dl  # noqa: E402
import ageostrophic_compare as ac  # noqa: E402
from makani_source import MakaniBackwardSource  # noqa: E402


def _check(cond, msg):
    if not cond:
        print(f"[smoke-ageo] FAIL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[smoke-ageo]   ok: {msg}")


class _Args:
    lat_min, lat_max, lon_min, lon_max = 30.0, 60.0, None, None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case", required=True)
    p.add_argument("--makani-dir", default=None,
                   help="makani case dir (default: data-root/<case-subdir>).")
    p.add_argument("--level", type=int, default=500)
    p.add_argument("--n-members", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--data-root",
                   default="/projectnb/eb-general/jlin404/scratch/backward_emulator_data")
    args = p.parse_args(argv)

    import torch  # noqa: F401  (fail early if the heavy env is missing)

    case = dl.load_case(args.case)
    names = [f"u{args.level}", f"v{args.level}", f"z{args.level}"]
    subdir = {"pnw_heatwave": "2021_pnw_heatwave", "ian": "ian"}.get(args.case, args.case)
    makani_dir = args.makani_dir or f"{args.data_root}/{subdir}"

    makani = MakaniBackwardSource(makani_dir, n_frames=12)
    lat, lon = makani.lat, makani.lon
    lat_idx, lon_idx, lat_w, band = ac._domain_setup(lat, lon, _Args)
    print(f"[smoke-ageo] case={args.case} level={args.level} band={band}")

    # makani: first few members via the full-field iterator.
    ratios = []
    for i, (_gid, fields) in enumerate(makani.iter_member_fields(names)):
        u, v, z = (torch.from_numpy(fields[n]).float() for n in names)
        d = ac._diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_w)
        ratios.append(d["ratio"])
        if i + 1 >= args.n_members:
            break
    r = np.stack(ratios)  # (n, nf)
    _check(np.isfinite(r).all(), "makani |V_a|/|V| finite")
    _check((r >= 0).all() and (r <= 1).all(), f"makani |V_a|/|V| in [0,1] (max {r.max():.3f})")
    print(f"[smoke-ageo]   makani mean ratio over leads = {r.mean():.4f}")

    # cBottle: a small member batch via the regrid iterator.
    from cbottle_source import CBottleLatLonSource
    cb = CBottleLatLonSource(case, mode="end", n_members=args.n_members,
                             batch_size=args.batch_size)
    _check(np.allclose(np.asarray(cb.lat), np.asarray(lat)),
           "cBottle lat grid matches makani (shared 721x1440)")
    b0, fields = next(iter(cb.iter_member_field_batches(names, batch_size=args.batch_size)))
    u, v, z = (fields[n] for n in names)
    d = ac._diag_from_uvz(u, v, z, lat, lon, lat_idx, lon_idx, lat_w)
    rc = np.asarray(d["ratio"])
    _check(np.isfinite(rc).all(), "cBottle |V_a|/|V| finite")
    _check((rc >= 0).all() and (rc <= 1).all(),
           f"cBottle |V_a|/|V| in [0,1] (max {rc.max():.3f})")
    print(f"[smoke-ageo]   cBottle mean ratio over leads = {rc.mean():.4f}")

    print(f"[smoke-ageo] PASS: ageostrophic glue works for {args.case} "
          f"(makani native + cBottle regrid).")


if __name__ == "__main__":
    main()
