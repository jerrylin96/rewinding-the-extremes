import os
import sys
# 1. Force the script to only "see" the GPU SGE gave it
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    # If running locally/interactively without SGE, this prevents crashes
    print("Warning: CUDA_VISIBLE_DEVICES not set.")
import argparse
import calendar
import gc
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import zarr
from datetime import datetime, timedelta
import logging
from loguru import logger
from numcodecs import Blosc

# Earth2Studio imports
from earth2studio.data import WB2ERA5
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.data.utils import fetch_data

def setup_args():
    parser = argparse.ArgumentParser(
        description="Fetch raw ERA5 data and save to zarr format"
    )
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        "--chunk_time",
        type=int,
        default=1,
        help="Chunk size for time dimension (default: 1)",
    )
    parser.add_argument(
        "--compression_level",
        type=int,
        default=2,
        help="Blosc compression level 0-9 (default: 2)",
    )
    return parser.parse_args()

def create_datetime_array(year, month):
    _, num_days = calendar.monthrange(year, month)
    start_time = datetime(year=year, month=month, day=1)
    return np.array([start_time + timedelta(hours=6*i) for i in range(num_days * 4)], dtype="datetime64[ns]")

def log_nans(data, var_names):
    """
    Scans the tensor for NaNs and prints a detailed report of which variables/times are affected.
    Assumes data shape is [time, lead_time, var, lat, lon].
    """
    if not torch.isnan(data).any():
        return # Clean data
    
    print(f"Diagnostic Report:")
    
    # Iterate over variables (Dimension 1)
    for i, var_name in enumerate(var_names):
        # Slice: All times, specific variable, all lat/lon
        # Dimensions: time, lead_time (size 1), var, lat, lon
        var_slice = data[:, 0, i, :, :]
        nan_count = torch.isnan(var_slice).sum().item()
        
        if nan_count > 0:
            print(f"  - Variable '{var_name}' (Index {i}): {nan_count} NaNs found.")
    

def get_variables():
    """Get all variables available from WB2ERA5."""
    wb2_vars = set(WB2Lexicon.VOCAB.keys())
    cbottle_vars = set(CBottleLexicon.VOCAB.keys())
    variables = sorted(list(cbottle_vars.intersection(wb2_vars)))
    return variables


def create_zarr_store(zarr_path, year, month, total_timesteps, variables, args):
    """Create and initialize the zarr store structure."""

    # Remove existing store if it exists
    if zarr_path.exists():
        print(f"  [WARNING] Removing existing zarr store: {zarr_path}")
        shutil.rmtree(zarr_path)

    # Define dimensions
    n_var = len(variables)
    n_lat = 721
    n_lon = 1440

    # Coordinate arrays
    coords_template = {
        'variable': np.array(variables, dtype=str),
        'lat': np.arange(90, -90.25, -0.25),
        'lon': np.arange(0, 360, 0.25),
    }

    print(f"  Variables ({n_var}): {variables}")
    print(f"  Lat: {coords_template['lat'].min():.2f} to {coords_template['lat'].max():.2f} ({n_lat} points)")
    print(f"  Lon: {coords_template['lon'].min():.2f} to {coords_template['lon'].max():.2f} ({n_lon} points)")
    print(f"  Total timesteps: {total_timesteps}")

    # Set up compression
    compressors = zarr.codecs.BloscCodec(
        cname='zstd',
        clevel=args.compression_level,
        shuffle=zarr.codecs.BloscShuffle.bitshuffle
    )

    # Define chunks: chunk by time, keep full spatial extent
    chunks = (args.chunk_time, n_var, n_lat, n_lon)
    print(f"  Chunk shape: {chunks}")

    # Create zarr store
    root = zarr.open_group(str(zarr_path), mode='w')

    # Create the main data array
    # Shape is 4D: [time, variable, lat, lon]
    # We consciously drop lead_time (dim 1) because for ERA5 reanalysis, lead_time is always 0.
    full_shape = (total_timesteps, n_var, n_lat, n_lon)
    data_array = root.create_array(
        "data",
        shape=full_shape,
        chunks=chunks,
        dtype=np.float32,
        compressors=compressors,
    )

    # Create coordinate arrays
    time_arr = root.create_array("time", shape=(total_timesteps,), dtype='datetime64[ns]')
    var_arr = root.create_array("variable", shape=(n_var,), dtype=str)
    lat_arr = root.create_array("lat", shape=(n_lat,), dtype=np.float64)
    lon_arr = root.create_array("lon", shape=(n_lon,), dtype=np.float64)

    # Write static coordinates
    var_arr[:] = coords_template['variable']
    lat_arr[:] = coords_template['lat']
    lon_arr[:] = coords_template['lon']

    # Add metadata
    root.attrs["year"] = year
    root.attrs["month"] = month
    root.attrs["description"] = "Raw ERA5 data from WeatherBench2 (no infilling, no regridding)"
    root.attrs["source"] = "WB2ERA5"
    root.attrs["dimensions"] = ["time", "variable", "lat", "lon"]
    root.attrs["chunk_time"] = args.chunk_time
    root.attrs["compression"] = f"blosc-zstd-level{args.compression_level}"
    root.attrs["grid_type"] = "lat_lon"
    root.attrs["n_variables"] = n_var

    return root, data_array


def main(args):
    # --- DIAGNOSTICS START ---
    print(f"DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEBUG: NVIDIA_VISIBLE_DEVICES = {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    print(f"DEBUG: CUDA_DEVICE_ORDER = {os.environ.get('CUDA_DEVICE_ORDER')}")
    print(f"DEBUG: torch.version.cuda = {torch.version.cuda}")
    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEBUG: Device Count = {torch.cuda.device_count()}")
        print(f"DEBUG: Current Device Index = {torch.cuda.current_device()}")
        print(f"DEBUG: Current Device Name = {torch.cuda.get_device_name(0)}")
    else:
        print("DEBUG: No CUDA devices available to PyTorch.")
    # --- DIAGNOSTICS END ---
    # 1. Silence Logs
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logging.getLogger('earth2studio').setLevel(logging.WARNING)

    year = args.year
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Processing for Year {year} ---")

    if not torch.cuda.is_available():
        print("Warning: GPU not available. Using CPU.")
        # raise RuntimeError("This script requires a GPU for efficient fetching and downloading.")

    torch.cuda.set_device(0)
    device_gpu = torch.device("cuda:0")
    device_cpu = torch.device("cpu")
    
    variables = get_variables()
    print(f"\nVariables to fetch: {len(variables)}")
    print(f"  {variables}")

    # Define dimensions directly based on fetched variables and standard ERA5 grid
    n_var = len(variables)
    n_lat = 721
    n_lon = 1440
    n_lead = 1

    print(f"  Variables ({n_var}) configured.")
    print(f"  Grid: {n_lat}x{n_lon}")

    # Set up compression
    compressors = zarr.codecs.BloscCodec(cname = 'zstd', clevel = args.compression_level, shuffle = zarr.codecs.BloscShuffle.bitshuffle)

    # Define chunks: chunk by time, keep full spatial extent
    # This is optimal for "give me a month of data for all locations" access pattern
    chunks = (args.chunk_time, n_lead, n_var, n_lat, n_lon)
    print(f"  Chunk shape: {chunks}")


    for month in range(1, 13):

        times = create_datetime_array(year, month)
        total_timesteps = len(times)

        # Setup zarr store

        print(f"\nCreating zarr store for {year}-{month:02d}...")

        zarr_path = output_path / str(year) / f"era5_raw_{year}_{month:02d}.zarr"
        zarr_path.parent.mkdir(parents=True, exist_ok=True)  # ← Create year directory
        root, data_array = create_zarr_store(
            zarr_path, year, month, total_timesteps, variables, args
        )
        
        # [FIX] Removed erroneous code block that re-opened the Zarr in 'w' mode (wiping it)
        # and attempted to create a 5D array. We want the 4D array created above.

        print(f"\nProcessing {year}-{month:02d}...")

        # Fetch Data (CPU)

        ds = WB2ERA5()
        
        max_retries = 5
        fetch_success = False
        data = None
        coords = None

        for attempt in range(max_retries):
            try:
                data, coords = fetch_data(ds, times, variables, device=device_cpu)
                fetch_success = True
                break
            except Exception as e:
                wait_time = 15 * (attempt + 1)
                print(f"[WARNING] Fetch attempt {attempt+1}/{max_retries} failed. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    print(f"CRITICAL FAILURE for {year}-{month:02d}:")
                    traceback.print_exc()
                time.sleep(wait_time)

        if not fetch_success:
            exit(1)

        try:
            # Check for NaNs first
            var_names = coords.get('variable', variables)
            log_nans(data, var_names)

            print(f"  Writing to zarr {total_timesteps} timesteps for Year {year} Month {month}...")
            t0 = time.time()

            # Write data slice
            # data has shape [time, lead_time, var, lat, lon] (5D)
            # data_array has shape [time, var, lat, lon] (4D)
            # We squeeze out dim 1 (lead_time) to match the Zarr target.
            data_array[:] = data.squeeze(1).cpu().numpy()

            # Write time coordinate
            root["time"][:total_timesteps] = coords["time"]

            elapsed = time.time() - t0
            # -----------------------
            
            zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
            print(f"Zarr store size: {zarr_size / 1e9:.2f} GB")
            print(f"Output: {zarr_path}")
            print(f"year: {year}, month: {month}, total_timesteps: {total_timesteps}")
            
            del data, coords
            torch.cuda.empty_cache()
            gc.collect()
            print(f"✓ Saved {year}-{month:02d}")
        except Exception as e:
            print(f"Error during Saving {year}-{month:02d}:")
            # If it's our specific NaN error, print the message, otherwise traceback
            traceback.print_exc()
            exit(1)

    print(f"--- Year {year} Completed Successfully ---")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main(setup_args())
