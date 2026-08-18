"""Compute ensemble spread, RMSE, bias, and CRPS vs lead time for one mode.

Batch-form computation for the full 1000-member ensemble, suitable for
queue submission via submit_spread_rmse_crps.sh.

This script writes the numerics for ONE conditioning mode. The aggregated
plot across all three modes is produced separately by
``aggregate_spread_rmse_crps.py`` after all three per-mode jobs finish.

Outputs (in --output-dir):
    spread_rmse_crps_<mode>.npz
        spread              : (n_leads,)  float32   # sqrt(mean-pix variance)
        skill               : (n_leads,)  float32   # RMSE of ensemble mean
                                                    # vs ERA5; denominator of
                                                    # the spread/skill ratio
                                                    # (not plotted directly)
        member_rmse         : (n_leads,)  float32   # joint RMS over
                                                    # (member, pixel)
        mean_error          : (n_leads,)  float32   # signed mean error of
                                                    # the ensemble mean
                                                    # vs ERA5
        crps                : (n_leads,)  float32
        lead_hours          : (n_leads,)  float64
        n_members           : ()          int
        conditioning_frames : object      (list)
        var_name            : ()          str
        case_name           : ()          str
        landfall_frame      : ()          int
        mode                : ()          str
        region              : (4,)        float64  # (lon_min, lon_max,
                                                   # lat_min, lat_max), or an
                                                   # empty array when global
        region_name         : ()          str      # label for the figure
        mask_kind           : ()          str      # 'none' / 'land' / 'sea'
        n_pix_used          : ()          int
        n_pix_degenerate    : ()          int      # pixels with ~zero
                                                   # across-member variance at
                                                   # every free lead

Domain.  Without ``--region`` every metric reduces over the whole globe,
which is what this script did before the flag existed.  With it, the
reduction is restricted to the case's synoptic box -- the same pixels the
EOF pipeline uses -- so the free-end spread ratios this battery implies are
comparable with the synoptic-PCA numbers the paper quotes.  Global runs
write ``spread_rmse_crps_<mode>.npz``; regional runs write
``spread_rmse_crps_<mode>_<region_name>.npz`` so the two never collide.

Mask.  ``--region`` selects a lat/lon box; ``--mask land|sea`` further
restricts it, using the same ERA5-SST-derived mask
``compute_free_end_states.py`` and ``compute_synoptic_pca.py`` apply.  A
surface variable whose paper-facing average is masked needs this to be
comparable at all: `pnw_heatwave.yaml` averages t2m over land only, and an
unmasked run mixes in ocean pixels where t2m is essentially the prescribed
member-identical SST -- those contribute no spread but full weight to the
pixel mean, so they drag the spread/skill ratio toward zero.  The npz
records ``n_pix_degenerate`` and the run warns when it is non-zero, so this
failure mode is visible in the job log rather than silent in the figure.
Masked runs write ``spread_rmse_crps_<mode>_<region_name>_<mask>.npz``.

Normally you do not pass ``--region`` / ``--mask`` by hand:
``dispatch_spread_rmse_crps.py`` reads a per-variable domain map from
``diagnostics.spread_rmse_crps.domains`` in the case YAML and submits one
job per distinct ``(region, mask)`` pair, so z500 lands on the synoptic box
while the case's impact variable lands on the impact box with whatever mask
it needs.  The flags here are the mechanism, and remain useful for one-off
runs.

Usage:
    python compute_spread_rmse_crps.py \\
        --ensemble-zarr /path/to/ensemble_f11_n1000.zarr \\
        --era5-zarr     /path/to/era5_reference.zarr \\
        --output-dir    /path/to/diagnostics/spread_rmse_crps/<variable> \\
        --mode          end \\
        --region        -170 -100 25 70 --region-name synoptic

    # Surface variable over a masked impact box:
    python compute_spread_rmse_crps.py ... --var-name t2m \\
        --region -126 -110 38 55 --region-name impact --mask land
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import lead_hours  # noqa: E402

from custom_utils.diagnostics import diagnostics_from_zarr  # noqa: E402


def _to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensemble-zarr", required=True)
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end", "both"],
        help="Conditioning mode for this ensemble; encoded in the output "
        "filename so the three modes can coexist in --output-dir.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="msl",
        help="Variable to analyse (default: msl).",
    )
    parser.add_argument("--landfall-frame", type=int, default=11)
    parser.add_argument("--case-name", type=str, default="Hurricane Sandy")
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Ensemble IC as an ISO 8601 UTC string (e.g. 2012-10-27T06:00:00Z); "
        "the dispatcher forwards this from ensemble.start_time in the case YAML.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone for the lead-axis (forwarded to the aggregator).",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=None,
        help="Number of members to process (default: all in the zarr).",
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Restrict every reduction to this lat/lon box, in degrees with "
        "lon in [-180, 180). LON_MIN > LON_MAX spans the dateline. Default "
        "is a global reduction.",
    )
    parser.add_argument(
        "--region-name",
        type=str,
        default=None,
        help="Short label for --region (e.g. 'synoptic'); used in the output "
        "filename and the figure subtitle. Required when --region is given.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="none",
        choices=["none", "land", "sea"],
        help="Restrict the reduction to land or sea pixels, using the same "
        "ERA5-SST-derived mask compute_free_end_states.py and "
        "compute_synoptic_pca.py apply (default: none). Required to make a "
        "surface variable comparable with a masked paper number -- e.g. "
        "pnw_heatwave averages t2m over land only, because ocean t2m is "
        "essentially the prescribed member-identical SST and contributes no "
        "spread. Encoded in the output filename so masked and unmasked runs "
        "of the same region cannot collide.",
    )
    args = parser.parse_args()

    if args.region is not None and not args.region_name:
        raise SystemExit(
            "ERROR: --region-name is required when --region is given, so the "
            "regional npz cannot silently overwrite the global one."
        )

    region = tuple(args.region) if args.region is not None else None
    region_name = args.region_name if region is not None else "global"
    suffix = f"_{args.region_name}" if region is not None else ""
    if args.mask != "none":
        suffix += f"_{args.mask}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spread_rmse_crps] case        : {args.case_name}")
    print(f"[spread_rmse_crps] mode        : {args.mode}")
    print(f"[spread_rmse_crps] ensemble    : {args.ensemble_zarr}")
    print(f"[spread_rmse_crps] era5        : {args.era5_zarr}")
    print(f"[spread_rmse_crps] output dir  : {output_dir}")
    print(f"[spread_rmse_crps] var_name    : {args.var_name}")
    print(f"[spread_rmse_crps] landfall    : frame {args.landfall_frame}")
    print(f"[spread_rmse_crps] members     : {args.n_members or 'all'}")
    print(
        f"[spread_rmse_crps] domain      : "
        f"{region_name if region is None else f'{region_name} {region}'}"
    )
    print(f"[spread_rmse_crps] mask        : {args.mask}")

    if not os.path.isdir(args.ensemble_zarr):
        raise SystemExit(f"ERROR: Ensemble zarr does not exist: {args.ensemble_zarr}")
    if not os.path.isdir(args.era5_zarr):
        raise SystemExit(f"ERROR: ERA5 zarr does not exist: {args.era5_zarr}")

    print("[spread_rmse_crps] running diagnostics_from_zarr ...", flush=True)
    t0 = time.time()
    diag = diagnostics_from_zarr(
        ensemble_zarr_path=args.ensemble_zarr,
        era5_zarr_path=args.era5_zarr,
        var_name=args.var_name,
        n_members=args.n_members,
        region=region,
        mask=args.mask,
    )
    elapsed = time.time() - t0
    print(f"[spread_rmse_crps] done in {elapsed:.1f}s")
    print(f"[spread_rmse_crps] pixels used          : {diag['n_pix_used']}")
    print(f"[spread_rmse_crps] degenerate pixels    : {diag['n_pix_degenerate']}")

    hours = lead_hours(diag["lead_times"])
    N = diag["n_members"]
    print(f"[spread_rmse_crps] n_members            : {N}")
    print(f"[spread_rmse_crps] lead_h               : {hours.tolist()}")
    print(
        f"[spread_rmse_crps] mean spread          : {float(_to_np(diag['spread']).mean()):.4g}"
    )
    print(
        f"[spread_rmse_crps] mean skill (ens-mean): {float(_to_np(diag['skill']).mean()):.4g}"
    )
    print(
        f"[spread_rmse_crps] mean member RMSE     : {float(_to_np(diag['member_rmse']).mean()):.4g}"
    )
    print(
        f"[spread_rmse_crps] mean signed error    : {float(_to_np(diag['mean_error']).mean()):+.4g}"
    )
    print(
        f"[spread_rmse_crps] mean CRPS            : {float(_to_np(diag['crps']).mean()):.4g}"
    )

    npz_path = output_dir / f"spread_rmse_crps_{args.mode}{suffix}.npz"
    cond_frames = diag["conditioning_frames"] or []
    np.savez(
        npz_path,
        spread=_to_np(diag["spread"]).astype(np.float32),
        skill=_to_np(diag["skill"]).astype(np.float32),
        member_rmse=_to_np(diag["member_rmse"]).astype(np.float32),
        mean_error=_to_np(diag["mean_error"]).astype(np.float32),
        bias=_to_np(diag["mean_error"]).astype(
            np.float32
        ),  # Deprecated alias for backward compatibility
        crps=_to_np(diag["crps"]).astype(np.float32),
        lead_hours=np.asarray(hours, dtype=np.float64),
        n_members=np.int64(N),
        conditioning_frames=np.array(cond_frames, dtype=object),
        var_name=np.array(args.var_name),
        case_name=np.array(args.case_name),
        landfall_frame=np.int32(args.landfall_frame),
        mode=np.array(args.mode),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
        region=np.asarray(region if region is not None else [], dtype=np.float64),
        region_name=np.array(region_name),
        mask_kind=np.array(args.mask),
        n_pix_used=np.int64(diag["n_pix_used"]),
        n_pix_degenerate=np.int64(diag["n_pix_degenerate"]),
    )
    print(f"[spread_rmse_crps] wrote {npz_path}")
    print("[spread_rmse_crps] done.")


if __name__ == "__main__":
    main()
