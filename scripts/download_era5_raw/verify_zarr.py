
import zarr
import numpy as np
import os
import sys

def verify_zarr(zarr_path):
    print(f"Checking Zarr store at: {zarr_path}")
    
    if not os.path.exists(zarr_path):
        print("ERROR: Path does not exist!")
        return

    try:
        # Open the group
        root = zarr.open_group(zarr_path, mode='r')
        
        print("\n1. Structure Check:")
        print(f"   Keys found: {list(root.keys())}")
        
        required_keys = ['data', 'lat', 'lon', 'time', 'variable']
        missing = [k for k in required_keys if k not in root]
        if missing:
            print(f"   WARNING: Missing keys: {missing}")
        else:
            print("   Pass: All required keys present.")

        # Check Dimensions
        data = root['data']
        time = root['time']
        lat = root['lat']
        lon = root['lon']
        vars_arr = root['variable']
        
        print("\n2. Dimension Check:")
        print(f"   Data Shape: {data.shape}")
        print(f"   Time: {time.shape[0]} steps")
        print(f"   Vars: {vars_arr.shape[0]} variables")
        print(f"   Lat:  {lat.shape[0]} points")
        print(f"   Lon:  {lon.shape[0]} points")
        
        if data.shape != (time.shape[0], vars_arr.shape[0], lat.shape[0], lon.shape[0]):
             print("   WARNING: Data shape does not match coordinate shapes!")
        else:
             print("   Pass: Dimensions align.")

        # Check for NaNs
        print("\n3. content Integrity (NaN Check):")
        # We'll check a subset if it's huge, or use a streaming check
        # Checking the first timestep for all variables
        
        sample_data = data[0, :, :, :] 
        nan_count = np.isnan(sample_data).sum()
        
        if nan_count > 0:
            print(f"{nan_count} NaNs found in the first timestep. NaNs are expected in SSTs for land mask")
            # Identify which variables have NaNs
            vars_list = vars_arr[:]
            for i, var_name in enumerate(vars_list):
                 v_nans = np.isnan(sample_data[i]).sum()
                 if v_nans > 0:
                     print(f"     - {var_name}: {v_nans} NaNs")

        # Basic Stats
        print("\n4. Sample Statistics (First Timestep, First Variable):")
        var0 = vars_arr[0]
        v0_data = sample_data[0]
        print(f"   Variable: {var0}")
        print(f"   Min: {v0_data.min()}")
        print(f"   Max: {v0_data.max()}")
        print(f"   Mean: {v0_data.mean()}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    path = "/projectnb/eb-general/shared_data/data/raw/era5_raw_zarr/1959/era5_raw_1959_01.zarr"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    verify_zarr(path)
