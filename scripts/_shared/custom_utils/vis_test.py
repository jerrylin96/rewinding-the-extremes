import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from interpolation_utils import CBottleUtils
import earth2grid

# --- Setup ---
output_dir = "/projectnb/eb-general/jlin404/scratch/utils_testing/vis_test_output"
os.makedirs(output_dir, exist_ok=True)

# Define dummy coords
n_times = 5
n_lat = 721
n_lon = 1440
hpx_level = 6
n_pixels = 12 * 4**hpx_level

time_coords = np.arange(n_times)
lead_coords = np.array([0])
var_name = "test_var"
variable_coords = np.array([var_name])

latlon_coords = OrderedDict({
    "time": time_coords,
    "lead_time": lead_coords,
    "variable": variable_coords,
    "lat": np.linspace(90, -90, n_lat),
    "lon": np.linspace(0, 360, n_lon, endpoint=False)
})

hpx_coords = OrderedDict({
    "time": time_coords,
    "lead_time": lead_coords,
    "variable": variable_coords,
    "hpx": np.arange(n_pixels)
})

# Define dummy data (moving wave)
# LatLon: [time, lead, var, lat, lon]
print("Generating LatLon sine wave data...")
latlon_data = torch.zeros((n_times, 1, 1, n_lat, n_lon))
X, Y = np.meshgrid(np.linspace(0, 2*np.pi, n_lon), np.linspace(0, np.pi, n_lat))
for t in range(n_times):
    latlon_data[t, 0, 0, :, :] = torch.from_numpy(np.sin(X + t * 0.5) * np.sin(Y))

# HPX: Create by regridding LatLon -> Healpix NEST to ensure it looks like the wave
print("Regridding LatLon to HEALPix (NEST) for consistent test data...")
hpx_grid = earth2grid.healpix.Grid(hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST)
latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(n_lat, n_lon, includes_south_pole=True)
regridder = earth2grid.get_regridder(latlon_grid, hpx_grid)

# Flatten dimensions for regridder: [..., lat, lon]
# latlon_data is [time, lead, var, lat, lon] -> flatten first 3 dims
in_flat = latlon_data.view(-1, n_lat, n_lon).double()
out_flat = regridder(in_flat).float()
hpx_data = out_flat.view(n_times, 1, 1, n_pixels)

print(f"hpx_data shape: {hpx_data.shape}")

# Build variable dicts with data and coords included
latlon_variable_dict = {
    "data": latlon_data,
    "coords": latlon_coords,
    "var_name": var_name,
    "colormap": "viridis",
    "vmin": -1.0,
    "vmax": 1.0,
}

hpx_variable_dict = {
    "data": hpx_data,
    "coords": hpx_coords,
    "var_name": var_name,
    "colormap": "viridis",
    "vmin": -1.0,
    "vmax": 1.0,
}

# --- Test 1: plot_latlon ---
print("Testing plot_latlon...")
try:
    fig, ax = CBottleUtils.plot_latlon(latlon_variable_dict)
    fig.savefig(os.path.join(output_dir, "plot_latlon.png"))
    print("plot_latlon success.")
except Exception as e:
    print(f"plot_latlon failed: {e}")

# --- Test 2: plot_healpix_nest ---
print("\nTesting plot_healpix_nest...")
try:
    fig, ax = CBottleUtils.plot_healpix_nest(hpx_variable_dict, hpx_level)
    fig.savefig(os.path.join(output_dir, "plot_healpix.png"))
    print("plot_healpix_nest success.")
except Exception as e:
    print(f"plot_healpix_nest failed: {e}")

# --- Test 3: animate_latlon ---
print("\nTesting animate_latlon...")
try:
    anim = CBottleUtils.animate_latlon(
        latlon_variable_dict,
        interpolated=False,
        output_filename=os.path.join(output_dir, "anim_latlon.mp4"),
        fps=5,
    )
    print("animate_latlon success.")
except Exception as e:
    print(f"animate_latlon failed: {e}")

# --- Test 4: animate_healpix_nest ---
print("\nTesting animate_healpix_nest...")
try:
    anim = CBottleUtils.animate_healpix_nest(
        hpx_variable_dict,
        interpolated=False,
        hpx_level=hpx_level,
        output_filename=os.path.join(output_dir, "anim_healpix.mp4"),
        fps=5,
    )
    print("animate_healpix_nest success.")
except Exception as e:
    print(f"animate_healpix_nest failed: {e}")

print(f"\nCheck output in {output_dir}")
