#!/usr/bin/env python3
"""
Animate Ensemble Output

Reads zarr stores produced by ensemble_interpolation.py and renders
MP4 animations. No GPU required — runs entirely on CPU.

Examples:
    # Animate a single variable from ensemble output
    python animate_ensemble.py \
        --ensemble-zarr output/ensemble_both_n16.zarr \
        --era5-zarr output/era5_reference.zarr \
        --variable t2m --vmin 280 --vmax 320 --cmap cmr.ember \
        --output-dir output/animations

    # Animate with geographic extent and specific members
    python animate_ensemble.py \
        --ensemble-zarr output/ensemble_both_n16.zarr \
        --variable msl --vmin 96000 --vmax 101000 \
        --members 0 1 2 \
        --extent -100 -50 10 45 \
        --output-dir output/animations
"""

import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
from custom_utils.ensemble_loading import inspect_zarr  # noqa: E402
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402

import torch


def setup_args():
    parser = argparse.ArgumentParser(
        description='Animate ensemble interpolation output from zarr stores'
    )

    parser.add_argument('--ensemble-zarr', type=str, required=True,
                        help='Path to ensemble zarr store from ensemble_interpolation.py')
    parser.add_argument('--era5-zarr', type=str, default=None,
                        help='Path to era5_reference.zarr for ERA5 animation. '
                             'If not provided, only ensemble animations are generated.')
    parser.add_argument('--variable', type=str, required=True,
                        help='Variable to animate (e.g. t2m, t850, msl)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Colorbar minimum')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Colorbar maximum')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap name (default: viridis)')
    parser.add_argument('--extent', type=float, nargs=4, default=None,
                        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
                        help='Geographic extent for zoomed animation')
    parser.add_argument('--members', type=int, nargs='+', default=None,
                        help='Ensemble member indices to animate (default: all)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--interp-factor', type=int, default=5,
                        help='Temporal interpolation factor for smoother playback (default: 5)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for output MP4 files')

    return parser.parse_args()


def _latlon_coords(root, info, lat, lon, *, include_lead_time=True):
    """Build a lat/lon coord dict for ``CBottleUtils.animate_latlon``.

    ``info`` provides variables / lead_times; ``lat`` / ``lon`` are passed
    explicitly so the same builder works for ERA5 (native lat/lon) and the
    ensemble (regridded HPX -> lat/lon at animation time).
    """
    coords = {"time": np.array(root["time"][:])}
    if include_lead_time:
        coords["lead_time"] = info.lead_times
    coords["variable"] = np.array(info.variables)
    coords["lat"] = lat
    coords["lon"] = lon
    return coords


def _round_trip_era5_member(era5_data, era5_root, era5_info, hpx_level):
    """Project ERA5 onto the HPX-resolvable subspace, then back to lat/lon.

    Visual side-by-side with cBottle output is only fair if both fields
    carry HPX-resolvable content; ERA5 is otherwise visibly sharper than
    anything cBottle can produce.  Returns ``(rt_data, rt_lat, rt_lon)``.
    """
    coords = {
        "time": np.array(era5_root["time"][:]),
        "lead_time": era5_info.lead_times,
        "variable": np.array(era5_info.variables),
        "lat": era5_info.lat,
        "lon": era5_info.lon,
    }
    hpx_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
        era5_data, coords, hpx_level=hpx_level,
    )
    rt_data, rt_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
        hpx_data, hpx_coords, hpx_level=hpx_level,
    )
    return rt_data, rt_coords["lat"], rt_coords["lon"]


def _hpx_level_from_info(ens_info):
    if ens_info.hpx_level is not None:
        return int(ens_info.hpx_level)
    return int(np.log2(ens_info.hpx.shape[0] // 12) / 2)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    var = args.variable

    # The ensemble lives on HPX; learn the level so we can regrid both
    # ensemble members and (round-tripped) ERA5 onto matching lat/lon axes.
    ens_info = inspect_zarr(args.ensemble_zarr)
    ens_root = zarr.open_group(args.ensemble_zarr, mode='r')
    if ens_info.grid != "hpx":
        raise SystemExit(
            f"animate_ensemble.py requires HEALPix ensemble zarrs; got "
            f"grid={ens_info.grid} for {args.ensemble_zarr}"
        )
    hpx_level = _hpx_level_from_info(ens_info)

    # ERA5 animation (round-tripped for visual parity with the cBottle frames)
    if args.era5_zarr:
        print(f"--- ERA5 animation for '{var}' (HPX round-trip) ---")
        era5_info = inspect_zarr(args.era5_zarr)
        era5_root = zarr.open_group(args.era5_zarr, mode='r')
        era5_data = torch.from_numpy(np.array(era5_root["data"][:]))
        era5_rt_data, rt_lat, rt_lon = _round_trip_era5_member(
            era5_data, era5_root, era5_info, hpx_level,
        )
        era5_coords = _latlon_coords(
            era5_root, era5_info, rt_lat, rt_lon, include_lead_time=False,
        )

        era5_var_dict = {
            "data": era5_rt_data.squeeze(1),  # remove lead_time dim
            "coords": era5_coords,
            "var_name": var,
            "colormap": args.cmap,
            "vmin": args.vmin,
            "vmax": args.vmax,
        }
        era5_path = os.path.join(args.output_dir, f"era5_{var}.mp4")
        print(f"  Saving: {era5_path}")
        CBottleUtils.animate_latlon(
            era5_var_dict, interpolated=False,
            output_filename=era5_path, extent=args.extent,
            fps=args.fps, interp_factor=args.interp_factor,
        )
        del era5_data, era5_rt_data

    # Ensemble animations (regrid HPX -> lat/lon per member)
    print(f"\n--- Ensemble animations for '{var}' ---")
    total_members = ens_info.n_members

    members = args.members if args.members is not None else list(range(total_members))
    for m in members:
        if m >= total_members:
            print(f"  Skipping member {m} (only {total_members} members available)")
            continue

        # Member shape: [1, n_leads, n_vars, n_pix] -> regrid to [1, n_leads, n_vars, n_lat, n_lon]
        member_hpx = torch.from_numpy(np.array(ens_root["data"][m]))
        member_hpx_coords = {
            "time": np.array(ens_root["time"][:]),
            "lead_time": ens_info.lead_times,
            "variable": np.array(ens_info.variables),
            "hpx": ens_info.hpx,
        }
        member_latlon, member_latlon_coords = CBottleUtils.regrid_healpix_nest_to_latlon(
            member_hpx, member_hpx_coords, hpx_level=hpx_level,
        )
        ens_coords = _latlon_coords(
            ens_root, ens_info,
            member_latlon_coords["lat"], member_latlon_coords["lon"],
        )
        del member_hpx
        interp_var_dict = {
            "data": member_latlon,
            "coords": ens_coords,
            "var_name": var,
            "colormap": args.cmap,
            "vmin": args.vmin,
            "vmax": args.vmax,
        }
        anim_path = os.path.join(args.output_dir, f"ensemble_{var}_member_{m}.mp4")
        print(f"  Member {m}: {anim_path}")
        CBottleUtils.animate_latlon(
            interp_var_dict, interpolated=True,
            output_filename=anim_path, extent=args.extent,
            fps=args.fps, interp_factor=args.interp_factor,
        )
        del member_latlon

    print(f"\nAll animations saved to {args.output_dir}")


if __name__ == "__main__":
    main(setup_args())
