#!/usr/bin/env python3
"""Run a TC tracker on an ensemble zarr store and serialize the tracks.

Runs the configured TC tracker on every ensemble member and writes:
  1. ``tc_tracks_<mode>.parquet``   (always; long-format track table)
  2. ``tc_tracks_<mode>.mp4``       (when --animate; per-mode animation)

The cross-mode static PNG (3-panel side-by-side) is produced separately
by ``aggregate_tc_tracks.py`` once the three per-mode parquets exist.

Supported trackers (--tracker flag):
  wuduan        – Wu & Duan 2023 vorticity-based tracker
  minmsl        – Simple minimum MSL pressure tracker (default)
  tempest       – Real TempestExtremes binaries via the wrapper in
                  tempest_extremes_runner.py; requires DetectNodes and
                  StitchNodes on PATH (e2s-custom conda env supplies them)

Usage:
    python plot_ensemble_tc_tracks.py \
        --ensemble-zarr /path/to/ensemble.zarr \
        --output-dir /path/to/diagnostics/tc_tracks/<tracker> \
        --mode end \
        --tracker wuduan \
        --lon-min -105 --lon-max -60 --lat-min 10 --lat-max 40
"""

import argparse
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from collections import OrderedDict

try:
    import cmasher  # noqa: F401 — registers "cmr.*" colormap names with matplotlib

    MSL_CMAP = "cmr.sapphire_r"
except ImportError:
    MSL_CMAP = "Blues_r"

# Repo root for earth2studio; _shared/ for _dispatch_lib; parent dir for tc_tracks_io.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SHARED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, SHARED_DIR)
sys.path.insert(0, PARENT_DIR)
from _dispatch_lib import (  # noqa: E402
    mode_frames,
    wrap_lon_0_360,
    zarr_variable_index,
)
from plot_style import LEGEND_SIZE, TITLE_SIZE, draw_domain_box  # noqa: E402
from tc_tracks_io import (  # noqa: E402
    ensemble_mean_track,
    save_tracks_parquet,
)
from track_qc import ANCHOR_RADIUS_KM, compute_track_qc_arrays  # noqa: E402
from tempest_extremes_runner import (  # noqa: E402
    fetch_surface_orography,
    infer_timestep_hours,
    pad_and_stack_tracks,
    track_member_with_tempest,
    track_with_tempest,
)
from earth2studio.models.dx import (  # noqa: E402
    TCTrackerMinMSL,
    TCTrackerWuDuan,
)
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402
from earth2studio.models.dx.tc_tracking import (
    VARIABLES_TCMSL,
    VARIABLES_TCWD,
    VARIABLES_TEMPEST_BASE,
    VARIABLES_TEMPEST_WARM_CORE_EXTRAS,
)  # noqa: E402

# 'tempest' uses the real TempestExtremes binaries via
# tempest_extremes_runner.track_with_tempest, so it has no torch class
# (cls=None sentinel marks the dispatch branch in main()) and its
# variable list depends on the --warm-core flag.  NEVER index
# ``TRACKER_CHOICES["tempest"]["variables"]`` directly — call
# ``_tempest_tracker_variables(warm_core=...)`` instead.  The ``None``
# here is a tripwire: any code that bypasses the helper will crash
# loudly downstream rather than silently track on the wrong variable set.
TRACKER_CHOICES = {
    "wuduan": {"cls": TCTrackerWuDuan, "variables": VARIABLES_TCWD},
    "minmsl": {"cls": TCTrackerMinMSL, "variables": VARIABLES_TCMSL},
    "tempest": {"cls": None, "variables": None},
}


def _tempest_tracker_variables(warm_core: bool = False) -> list[str]:
    """Variable list the tempest tracker needs to read out of the zarr.

    Tempest is the only tracker whose variable list isn't fixed: the
    Z&U 2017 thickness contour needs z300/z500 on top of the base
    msl/u10m/v10m, so the list grows with ``warm_core``.  All other
    trackers have a static list in :data:`TRACKER_CHOICES` and don't
    need this helper.
    """
    if warm_core:
        return VARIABLES_TEMPEST_BASE + VARIABLES_TEMPEST_WARM_CORE_EXTRAS
    return list(VARIABLES_TEMPEST_BASE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _hpx_level_from_root(root):
    if "hpx" not in root.array_keys():
        return None
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    return int(np.log2(root["hpx"].shape[0] // 12) / 2)


# Standard regridder lat/lon target (matches ``regrid_healpix_nest_to_latlon``
# defaults).  cBottle outputs land here after the HPX -> lat/lon regrid.
_LATLON_TARGET_NLAT = 721
_LATLON_TARGET_NLON = 1440
_LATLON_TARGET_LAT = np.linspace(90, -90, _LATLON_TARGET_NLAT)
_LATLON_TARGET_LON = np.linspace(0, 360, _LATLON_TARGET_NLON, endpoint=False)


def _regrid_hpx_batch_to_latlon(arr_hpx, hpx_level):
    """Regrid a HEALPix tensor to lat/lon, preserving leading dims.

    ``arr_hpx`` must end in the HPX pixel dim.  Returns a torch.Tensor
    on lat/lon with shape ``(*leading, _LATLON_TARGET_NLAT, _LATLON_TARGET_NLON)``.
    """
    if not isinstance(arr_hpx, torch.Tensor):
        arr_hpx = torch.from_numpy(np.asarray(arr_hpx)).float()
    leading = arr_hpx.shape[:-1]
    n_pix = arr_hpx.shape[-1]
    # The regridder contract wants ``time``/``lead_time``/``variable`` keys.
    # We collapse ``arr_hpx`` to ``[B, 1, 1, n_pix]`` to satisfy it, then
    # reshape back to ``(*leading, n_lat, n_lon)``.
    flat = arr_hpx.reshape(-1, 1, 1, n_pix)
    coords = {
        "time": np.arange(flat.shape[0]),
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": np.array(["_"]),
        "hpx": np.arange(n_pix),
    }
    latlon, _ = CBottleUtils.regrid_healpix_nest_to_latlon(
        flat, coords, hpx_level=hpx_level,
    )
    out = latlon[:, 0, 0, :, :]
    return out.reshape(*leading, _LATLON_TARGET_NLAT, _LATLON_TARGET_NLON)


def _round_trip_era5_field(arr_latlon, hpx_level):
    """Round-trip an ERA5 array (``[*leading, n_lat, n_lon]``) through HPX.

    Returns a torch tensor on lat/lon with the same leading shape.
    """
    if not isinstance(arr_latlon, torch.Tensor):
        arr_latlon = torch.from_numpy(np.asarray(arr_latlon)).float()
    leading = arr_latlon.shape[:-2]
    n_lat, n_lon = arr_latlon.shape[-2], arr_latlon.shape[-1]
    flat = arr_latlon.reshape(-1, 1, 1, n_lat, n_lon)
    coords = {
        "time": np.arange(flat.shape[0]),
        "lead_time": np.array([np.timedelta64(0, "h")]),
        "variable": np.array(["_"]),
        "lat": np.linspace(90, -90, n_lat),
        "lon": np.linspace(0, 360, n_lon, endpoint=False),
    }
    hpx_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
        flat, coords, hpx_level=hpx_level,
    )
    rt, _ = CBottleUtils.regrid_healpix_nest_to_latlon(
        hpx_data, hpx_coords, hpx_level=hpx_level,
    )
    out = rt[:, 0, 0, :, :]
    return out.reshape(*leading, _LATLON_TARGET_NLAT, _LATLON_TARGET_NLON)


def load_zarr_data(zarr_path):
    """Open a zarr group and return (root, coords, time_steps).

    Coords always carry ``lat`` / ``lon`` arrays (the regridder target
    grid for HPX stores) so downstream tracker / plotting code doesn't
    need to know the storage format.  HPX-only stores additionally
    include ``hpx_level`` for the regrid step.
    """
    print(f"Loading {zarr_path}...")
    root = zarr.open_group(zarr_path, mode="r")
    hpx_level = _hpx_level_from_root(root)

    if hpx_level is not None:
        lat = _LATLON_TARGET_LAT
        lon = _LATLON_TARGET_LON
    else:
        lat = np.array(root["lat"][:])
        lon = np.array(root["lon"][:])

    coords = OrderedDict(
        {
            "time": np.array(root["time"][:]),
            "lead_time": np.array(root["lead_time"][:]),
            "variable": np.array(root["variable"][:]),
            "lat": lat,
            "lon": lon,
        }
    )
    if hpx_level is not None:
        coords["hpx_level"] = hpx_level

    if root["data"].shape[1] == 1:
        time_steps = coords["lead_time"]
    elif root["data"].shape[2] == 1:
        time_steps = coords["time"]
    else:
        time_steps = coords["time"]

    return root, coords, time_steps


def load_era5_replacement(
    era5_path, ens_coords, tracker_variables, conditioned_frames, hpx_level,
):
    """Pre-load ERA5 data for conditioned frames, round-tripped through HPX.

    Conditioned frames are swapped in for ensemble data inside the tracker
    loop.  When the ensemble is on HEALPix, the ERA5 replacements need to
    pass through the same HPX projection so the tracker doesn't see a
    field with finer-than-HPX structure on the conditioning frames.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping from frame index to ``[V_tracker, H, W]`` tensors on the
        regridder's lat/lon target grid.
    """
    print(f"Loading ERA5 reference from {era5_path} for frames {conditioned_frames}...")
    era5_root = zarr.open_group(era5_path, mode="r")
    era5_var_indices = [zarr_variable_index(era5_root, v) for v in tracker_variables]

    replacements = {}
    for frame_idx in conditioned_frames:
        # Stack tracker variables -> [V, H, W] on raw ERA5 lat/lon.
        slices = [
            torch.from_numpy(
                np.array(era5_root["data"][frame_idx, 0, era5_v_idx, :, :])
            )
            for era5_v_idx in era5_var_indices
        ]
        raw = torch.stack(slices, dim=0).float()  # [V, H, W]
        if hpx_level is not None:
            raw = _round_trip_era5_field(raw, hpx_level)
        replacements[frame_idx] = raw
        print(
            f"  Frame {frame_idx}: loaded ERA5 replacement {raw.shape}"
        )

    return replacements


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def run_era5_tracking(era5_path, device, tracker_name="minmsl", hpx_level=None):
    """Run TC tracker on the ERA5 reference zarr (single "member").

    When ``hpx_level`` is given (the standard path now that the ensemble
    is on HEALPix), ERA5 is round-tripped through HPX before tracking so
    the ERA5 reference fixes share the cBottle ensemble's effective
    resolution — otherwise raw ERA5's finer-than-HPX structure would
    bias the intensity-CI comparisons.

    Returns
    -------
    np.ndarray
        Track array of shape ``[1, n_paths, n_steps, 4]`` (same format as
        the ensemble track output from ``run_tracking``).
    """
    print(f"Running {tracker_name} tracker on ERA5 reference"
          + (" (HPX round-trip)" if hpx_level is not None else "")
          + "...")
    era5_root = zarr.open_group(era5_path, mode="r")

    tracker_info = TRACKER_CHOICES[tracker_name]
    tracker_cls = tracker_info["cls"]
    tracker_variables = tracker_info["variables"]

    tracker_var_indices = [zarr_variable_index(era5_root, v) for v in tracker_variables]
    n_timesteps = era5_root["data"].shape[0]

    # ERA5 shape: [time, lead_time=1, variable, lat, lon] -> [time, variable, lat, lon]
    era5_data = torch.from_numpy(np.array(era5_root["data"][:, 0, :, :, :])).float()
    if hpx_level is not None:
        era5_data = _round_trip_era5_field(era5_data, hpx_level)

    tracker = tracker_cls().to(device)
    tracker_input_coords = tracker.input_coords()

    for t in range(n_timesteps):
        x_t = era5_data[t, tracker_var_indices, :, :].unsqueeze(0).to(device)
        batch_coords = OrderedDict(
            {
                "batch": np.array(["era5"]),
                "variable": np.array(tracker_variables),
                "lat": tracker_input_coords["lat"],
                "lon": tracker_input_coords["lon"],
            }
        )
        output, output_coords = tracker(x_t, batch_coords)

    era5_tracks = tracker.path_buffer.cpu()
    era5_tracks_np = torch.where(
        era5_tracks == tracker.PATH_FILL_VALUE, torch.nan, era5_tracks
    ).numpy()
    print(f"ERA5 tracks shape: {era5_tracks_np.shape}")
    return era5_tracks_np


def run_tracking(
    root,
    coords,
    time_steps,
    device,
    tracker_name="minmsl",
    batch_size=8,
    era5_replacements=None,
    hpx_level=None,
):
    """Run TC tracker over all ensemble members in batches.

    When ``hpx_level`` is set the ensemble is on HEALPix; each batch is
    regridded HPX -> lat/lon before tracking, since the trackers consume
    fields on a regular lat/lon grid.

    Parameters
    ----------
    tracker_name : str
        One of ``"wuduan"``, ``"minmsl"``, or ``"tempest"``.
    era5_replacements : dict[int, torch.Tensor] or None
        If provided, a mapping from frame index to ERA5 tensor ``[V, H, W]``
        that will be used in place of ensemble data at those frames.
    hpx_level : int or None
        HEALPix level of the ensemble store (None for legacy lat/lon).
    """
    tracker_info = TRACKER_CHOICES[tracker_name]
    tracker_cls = tracker_info["cls"]
    tracker_variables = tracker_info["variables"]

    print(f"Running {tracker_name} tracking in batches of {batch_size}"
          + (f" (regrid HPX -> lat/lon, level={hpx_level})" if hpx_level is not None else "")
          + "...")
    if era5_replacements:
        print(
            f"  ERA5 replacement active for frames: {sorted(era5_replacements.keys())}"
        )

    tracker_var_indices = [zarr_variable_index(root, v) for v in tracker_variables]

    n_members = root["data"].shape[0]
    n_timesteps = len(time_steps)
    print(f"Total Members: {n_members}, Timesteps: {n_timesteps}")

    all_tc_tracks = []

    for start_idx in range(0, n_members, batch_size):
        end_idx = min(start_idx + batch_size, n_members)
        batch_size_actual = end_idx - start_idx
        print(f"Processing members {start_idx} to {end_idx - 1}...")

        batch_data = torch.from_numpy(np.array(root["data"][start_idx:end_idx])).float()

        if batch_data.shape[1] == 1:
            batch_data = batch_data.squeeze(1)
        elif batch_data.shape[2] == 1:
            batch_data = batch_data.squeeze(2)
        else:
            batch_data = batch_data.squeeze(2)

        # batch_data is now [bs, n_t, V, ...spatial...].  For HPX storage
        # the spatial dim is a single ``hpx`` axis; regrid to lat/lon so
        # the rest of this loop sees the same shape it always did.
        if hpx_level is not None:
            batch_data = _regrid_hpx_batch_to_latlon(batch_data, hpx_level)

        tracker = tracker_cls().to(device)
        tracker_input_coords = tracker.input_coords()

        for t in range(n_timesteps):
            x_t = batch_data[:, t, tracker_var_indices, :, :].to(device)

            # Swap in ERA5 data for conditioned frames
            if era5_replacements and t in era5_replacements:
                era5_frame = era5_replacements[t].to(device)
                # Broadcast [V, H, W] -> [batch, V, H, W]
                x_t = era5_frame.unsqueeze(0).expand(batch_size_actual, -1, -1, -1)

            batch_coords = OrderedDict(
                {
                    "batch": np.array(
                        [f"member_{i}" for i in range(start_idx, end_idx)]
                    ),
                    "variable": np.array(tracker_variables),
                    "lat": tracker_input_coords["lat"],
                    "lon": tracker_input_coords["lon"],
                }
            )

            output, output_coords = tracker(x_t, batch_coords)

        batch_tc_tracks = tracker.path_buffer.cpu()
        batch_tc_tracks_np = torch.where(
            batch_tc_tracks == tracker.PATH_FILL_VALUE, torch.nan, batch_tc_tracks
        ).numpy()
        all_tc_tracks.append(batch_tc_tracks_np)

    # Pad all batches to the same n_paths (dimension 1) before concatenating,
    # since each batch's tracker may discover a different number of paths.
    max_paths = max(a.shape[1] for a in all_tc_tracks)
    padded = []
    for a in all_tc_tracks:
        if a.shape[1] < max_paths:
            pad_shape = (a.shape[0], max_paths - a.shape[1], a.shape[2], a.shape[3])
            a = np.concatenate([a, np.full(pad_shape, np.nan)], axis=1)
        padded.append(a)

    tc_tracks_np = np.concatenate(padded, axis=0)
    print(f"Final TC tracks shape: {tc_tracks_np.shape}")
    return tc_tracks_np


# ---------------------------------------------------------------------------
# Real TempestExtremes path
#
# The streaming-tracker interface above doesn't fit DetectNodes /
# StitchNodes, which operate on the whole time series at once. For
# `--tracker tempest` each ensemble member is read from the zarr store
# in a worker process and handed to tempest_extremes_runner, which
# shells out to the canonical TempestExtremes binaries. Streaming +
# multiprocessing keeps peak memory bounded at
# ``num_workers * per_member_size`` instead of the full ensemble slab,
# which would be ~650 GiB at 1000 members x 12 steps x 14 vars x
# 721 x 1440 fp32.
# ---------------------------------------------------------------------------


def _tempest_worker_init():
    """ProcessPoolExecutor initializer: pin each worker to a single OMP thread.

    The SGE script exports ``OMP_NUM_THREADS=$NSLOTS`` so the *main* process
    can use all allocated cores.  When the executor forks N workers each
    one inherits that setting, so torch/MKL try to spawn N threads per
    worker — i.e. N*N threads competing for N physical cores.  At 4
    workers x 4 cores that's a 4x oversubscription, which made the
    per-member HPX -> lat/lon regrid orders-of-magnitude slower and let
    1000-member tempest jobs walk off the 5h h_rt budget without ever
    printing a progress line.
    """
    torch.set_num_threads(1)


def _tempest_member_worker(
    member_idx,
    zarr_path,
    tracker_var_indices,
    raw_layout,
    variables,
    lats,
    lons,
    timestep_hours,
    era5_replacements_np,
    work_dir,
    hpx_level,
    orography,
    warm_core,
):
    """ProcessPool worker: read one member from zarr, run TempestExtremes.

    Defined at module scope so it pickles cleanly. Each call reopens the
    zarr group (cheap; metadata is a single JSON read) and pulls just
    the slab for ``member_idx``.  HEALPix-stored members are regridded
    to lat/lon in-process before being handed to TempestExtremes.

    ``orography`` is a static ``[H, W]`` surface-elevation array fetched
    once in the parent process and pickled to each worker; passing
    ``None`` skips the Mu-Ting ``zs <= 150 m`` stitch threshold.
    """
    root = zarr.open_group(zarr_path, mode="r")
    raw = root["data"]
    if hpx_level is not None:
        # HPX layout: [M, 1, T, V, n_pix] or [M, T, 1, V, n_pix]
        if raw_layout == "lead_time_first":
            member_data = np.array(raw[member_idx, 0, :, :, :])
        else:
            member_data = np.array(raw[member_idx, :, 0, :, :])
        member_data = member_data[:, tracker_var_indices, :]
        # Regrid HPX -> lat/lon: shape becomes [T, V, n_lat, n_lon].
        member_t = torch.from_numpy(member_data).float()
        member_data = _regrid_hpx_batch_to_latlon(
            member_t, hpx_level,
        ).cpu().numpy()
    else:
        # Lat/lon layout: [M, 1, T, V, H, W] or [M, T, 1, V, H, W]
        if raw_layout == "lead_time_first":
            member_data = np.array(raw[member_idx, 0, :, :, :, :])
        else:
            member_data = np.array(raw[member_idx, :, 0, :, :, :])
        member_data = member_data[:, tracker_var_indices, :, :]

    if era5_replacements_np:
        for t, era5_np in era5_replacements_np.items():
            member_data[t] = era5_np

    return track_member_with_tempest(
        member_data,
        variables=variables,
        lats=lats,
        lons=lons,
        timestep_hours=timestep_hours,
        member_idx=member_idx,
        work_dir=Path(work_dir),
        orography=orography,
        warm_core=warm_core,
    )


def run_tracking_tempest(
    root,
    coords,
    time_steps,
    zarr_path,
    era5_replacements=None,
    num_workers=None,
    hpx_level=None,
    use_orography=True,
    warm_core=False,
):
    """Run real TempestExtremes per ensemble member, streaming from zarr.

    Each member's ``[T, V, H, W]`` slab (~700 MB at 12x14x721x1440 fp32)
    is read in a worker process and handed to TempestExtremes; peak
    memory is bounded by ``num_workers * per_member_size`` instead of
    the full ``[M, T, V, H, W]`` slab.  When the ensemble is on HEALPix
    (``hpx_level`` set), each worker regrids its member HPX -> lat/lon
    before TempestExtremes runs.  Returns a ``[M, n_paths, n_steps, 4]``
    array matching the layout of :func:`run_tracking` so the result is
    interchangeable downstream.

    When ``use_orography`` is True (default), the static WB2 ERA5
    surface elevation is fetched once in the parent process and
    broadcast to every worker, enabling the Mu-Ting ``zs <= 150 m``
    stitch threshold.  When ``warm_core`` is True, the Z&U 2017
    _DIFF(z300,z500) thickness contour is added to ``closedcontourcmd``;
    this also requires z300/z500 in the input zarr.
    """
    variables = _tempest_tracker_variables(warm_core=warm_core)
    tracker_var_indices = [zarr_variable_index(root, v) for v in variables]

    # Detect which of (lead_time, time) is the singleton axis, mirroring
    # the squeeze logic in run_tracking. raw shape: [M, ?, ?, V, H, W].
    raw = root["data"]
    if raw.shape[1] == 1:
        raw_layout = "lead_time_first"
        n_steps = raw.shape[2]
    else:
        # raw.shape[2] == 1 is the canonical case; fall through here too
        # for the legacy fallback that the old code used.
        raw_layout = "time_first"
        n_steps = raw.shape[1]
    n_members = raw.shape[0]

    timestep_hours = infer_timestep_hours(time_steps)

    era5_replacements_np = None
    if era5_replacements:
        era5_replacements_np = {
            t: (frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame))
            for t, frame in era5_replacements.items()
        }

    if num_workers is None:
        num_workers = int(os.environ.get("NSLOTS") or os.cpu_count() or 1)
    num_workers = max(1, min(num_workers, n_members))

    # SGE NetCDF staging + Tempest binaries write per-member files into
    # work_dir; track_member_with_tempest cleans them up after each
    # member finishes, so disk usage stays bounded at
    # num_workers * ~100 MB.
    work_dir = Path(tempfile.mkdtemp(prefix="tempest_extremes_"))
    print(
        f"Running tempest tracking on {n_members} members, {n_steps} steps, "
        f"dt={timestep_hours}h, {num_workers} workers (scratch: {work_dir})..."
    )

    lats = np.asarray(coords["lat"])
    lons = np.asarray(coords["lon"])

    # Fetch the static surface orography once in the parent and pickle
    # it to each worker.  Doing it here (instead of inside the worker)
    # avoids fan-out of N redundant WB2 fetches and surfaces any fetch
    # failure before we burn time spinning up the pool.
    orography = None
    if use_orography:
        print("Fetching static surface orography from WB2 ERA5...")
        orography = fetch_surface_orography(lats, lons)
        print(
            f"  Orography ready: shape {orography.shape}, "
            f"min {orography.min():.1f} m, max {orography.max():.1f} m"
        )

    try:
        per_member: list = [None] * n_members
        # 1% increments — earlier visibility means stuck/slow workers
        # surface within a few members instead of after >5% (silent for
        # ~hours at 1000 members).
        progress_step = max(1, n_members // 100)
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_tempest_worker_init,
        ) as ex:
            futures = {
                ex.submit(
                    _tempest_member_worker,
                    m,
                    zarr_path,
                    tracker_var_indices,
                    raw_layout,
                    variables,
                    lats,
                    lons,
                    timestep_hours,
                    era5_replacements_np,
                    str(work_dir),
                    hpx_level,
                    orography,
                    warm_core,
                ): m
                for m in range(n_members)
            }
            completed = 0
            for fut in as_completed(futures):
                m = futures[fut]
                per_member[m] = fut.result()
                completed += 1
                if completed % progress_step == 0 or completed == n_members:
                    print(f"  Tracked {completed}/{n_members} members")
        return pad_and_stack_tracks(per_member, n_members, n_steps)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_era5_tracking_tempest(
    era5_path, work_dir=None, hpx_level=None, warm_core=False,
):
    """Run real TempestExtremes on the ERA5 reference zarr (single member).

    When ``hpx_level`` is given, ERA5 is round-tripped through HPX so the
    reference fixes share the cBottle ensemble's effective resolution.
    When ``warm_core`` is True, the Z&U 2017 warm-core thickness contour
    is added (requires z300/z500 in the ERA5 reference).

    Returns ``[1, n_paths, n_steps, 4]`` matching :func:`run_era5_tracking`.
    """
    print(f"Running tempest tracker on ERA5 reference {era5_path}"
          + (" (HPX round-trip)" if hpx_level is not None else "")
          + "...")
    era5_root = zarr.open_group(era5_path, mode="r")

    variables = _tempest_tracker_variables(warm_core=warm_core)
    tracker_var_indices = [zarr_variable_index(era5_root, v) for v in variables]

    # ERA5 layout: [time, lead_time=1, V, H, W]; squeeze lead_time.
    era5_data = np.array(era5_root["data"][:, 0, :, :, :])
    era5_data = era5_data[:, tracker_var_indices, :, :]  # [T, V, H, W]

    if hpx_level is not None:
        era5_t = torch.from_numpy(era5_data).float()
        era5_data = _round_trip_era5_field(era5_t, hpx_level).cpu().numpy()
        lats = _LATLON_TARGET_LAT
        lons = _LATLON_TARGET_LON
    else:
        lats = np.array(era5_root["lat"][:])
        lons = np.array(era5_root["lon"][:])

    era5_data = era5_data[None, ...]  # add member axis -> [1, T, V, H, W]

    times = np.array(era5_root["time"][:])
    timestep_hours = infer_timestep_hours(times)

    return track_with_tempest(
        era5_data,
        variables=variables,
        lats=lats,
        lons=lons,
        timestep_hours=timestep_hours,
        work_dir=work_dir,
        warm_core=warm_core,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def animate_tracks(
    tc_tracks_np,
    root,
    coords,
    time_steps,
    out_path,
    title,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    era5_tracks_np=None,
    mode=None,
    landfall_frame=-1,
    anchor_radius_km=ANCHOR_RADIUS_KM,
):
    """Create an MP4 animation of ensemble TC tracks evolving over time.

    Overlays the path-0 ensemble-mean track (circular longitude mean so
    it stays well-behaved across the dateline) on top of the per-member
    spaghetti, with the ERA5 reference drawn last when present.

    Track drawing inherits the round-5 resolution rule: ``track_qc``
    resolves each member's storm of interest anchor-first (the tracker path
    reaching the ERA5 fix at the pinned end — the landfall frame under
    end-conditioning, frame 0 otherwise) and the animation draws that one
    path per member, so unrelated systems can't bias the mean or clutter
    the map.  Members with no anchor-reaching storm are dropped from the
    animation.  The parquet on disk keeps every original track.  When ERA5
    is unavailable (no anchor reference) the raw primary path is drawn.

    When ``mode`` is ``"end"`` or ``"start"`` a per-member dot is drawn
    at the *free* end of each member's track — the end that wasn't
    anchored to ERA5 by the conditioning — so the spread of free-end
    landing/starting positions is visible from the first frame.  The
    dots sit on the full-track endpoint, not the leading-edge step, so
    for start-conditioning they preview future endpoints as the
    spaghetti grows toward them.
    """
    print(f"Animating tracks to {out_path}...")
    n_total = tc_tracks_np.shape[0]
    if era5_tracks_np is not None:
        n_leads = tc_tracks_np.shape[2]
        anchor_frame = (
            0
            if mode in ("start", "both")
            else (landfall_frame if 0 <= landfall_frame < n_leads else n_leads - 1)
        )
        qc = compute_track_qc_arrays(
            tc_tracks_np,
            era5_tracks_np,
            track_box=(lon_min, lon_max, lat_min, lat_max),
            anchor_frame=anchor_frame,
            free_end_frame=0,
            anchor_radius_km=anchor_radius_km,
        )
        tc_tracks_np = qc.storm_of_interest_array(tc_tracks_np)
        # ERA5's storm of interest is its path 0 (defines the anchor).
        era5_tracks_np = era5_tracks_np[:, 0:1]
        counts = qc.counts()
        print(
            f"  Anchor-first resolution (frame {anchor_frame}): "
            f"{counts['analyzable']}/{n_total} members analyzable "
            f"({counts['excluded']} excluded)."
        )
    else:
        # No ERA5 reference => no anchor; fall back to the raw primary path.
        tc_tracks_np = tc_tracks_np[:, 0:1]
        print(
            "  No ERA5 reference; drawing the raw primary path per member "
            "(no anchor-first resolution)."
        )
    fig = plt.figure(figsize=(14, 8))
    writer = animation.FFMpegWriter(
        fps=2, metadata=dict(artist="earth2studio"), bitrate=1800
    )

    _lon_min = wrap_lon_0_360(lon_min)
    _lon_max = wrap_lon_0_360(lon_max)
    msl_idx = zarr_variable_index(root, "msl")
    lats, lons = coords["lat"], coords["lon"]
    hpx_level = coords.get("hpx_level")
    n_timesteps = len(time_steps)
    cmap = plt.cm.jet

    # Pre-compute the path-0 ensemble-mean track once; the animation just
    # slices it up to the current step inside the frame loop.
    mean_lats, mean_lons = ensemble_mean_track(tc_tracks_np, path_id=0)

    # Per-member free-end fixes (the end that wasn't anchored to ERA5).
    # Pre-computed once so the frame loop just scatters a fixed set of
    # (lat, lon) points each frame.  Empty list for mode == "both" or
    # mode is None — neither end is "free" under both-end conditioning
    # so there is nothing distinctive to mark.
    free_end_fixes: list = []
    free_end_pick = {"end": "first", "start": "last"}.get(mode or "")
    if free_end_pick is not None:
        for path_id in range(tc_tracks_np.shape[1]):
            for member_idx in range(tc_tracks_np.shape[0]):
                m_lats = tc_tracks_np[member_idx, path_id, :, 0]
                m_lons = tc_tracks_np[member_idx, path_id, :, 1]
                m_valid = ~np.isnan(m_lats) & ~np.isnan(m_lons)
                if m_valid.sum() < 2:
                    continue
                idx = 0 if free_end_pick == "first" else -1
                color = cmap(member_idx / max(1, tc_tracks_np.shape[0] - 1))
                free_end_fixes.append(
                    (m_lons[m_valid][idx], m_lats[m_valid][idx], color)
                )

    with writer.saving(fig, out_path, dpi=120):
        for t in range(n_timesteps):
            fig.clf()
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            if hpx_level is not None:
                # HPX layout: [N, 1, n_leads, V, n_pix]
                if root["data"].shape[1] == 1:
                    msl_bg_t = root["data"][:, 0, t, msl_idx, :]
                else:
                    msl_bg_t = root["data"][:, t, 0, msl_idx, :]
                msl_hpx = np.mean(msl_bg_t, axis=0)
                msl_bg = _regrid_hpx_batch_to_latlon(
                    torch.from_numpy(msl_hpx).float(), hpx_level,
                ).cpu().numpy()
            else:
                if root["data"].shape[1] == 1:
                    msl_bg_t = root["data"][:, 0, t, msl_idx, :, :]
                else:
                    msl_bg_t = root["data"][:, t, 0, msl_idx, :, :]
                msl_bg = np.mean(msl_bg_t, axis=0)

            im = ax.pcolormesh(
                lons,
                lats,
                msl_bg / 100,
                cmap=MSL_CMAP,
                vmin=920,
                vmax=1020,
                transform=ccrs.PlateCarree(),
                alpha=0.5,
            )
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cb.set_label("Mean Sea Level Pressure (hPa) [Ensemble Mean]")

            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax.gridlines(
                draw_labels=True,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            ax.set_extent(
                [_lon_min, _lon_max, lat_min, lat_max], crs=ccrs.PlateCarree()
            )
            # Outline the tracking domain (tracks are restricted to this box).
            draw_domain_box(ax, [_lon_min, _lon_max, lat_min, lat_max])

            for path_id in range(tc_tracks_np.shape[1]):
                for member_idx in range(tc_tracks_np.shape[0]):
                    tc_lats = tc_tracks_np[member_idx, path_id, : t + 1, 0]
                    tc_lons = tc_tracks_np[member_idx, path_id, : t + 1, 1]

                    valid = ~np.isnan(tc_lats) & ~np.isnan(tc_lons)
                    if valid.sum() < 2:
                        continue

                    color = cmap(member_idx / max(1, tc_tracks_np.shape[0] - 1))
                    ax.plot(
                        tc_lons[valid],
                        tc_lats[valid],
                        color=color,
                        linewidth=1.5,
                        alpha=0.5,
                        marker="o",
                        markersize=3,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.plot(
                        tc_lons[valid][-1],
                        tc_lats[valid][-1],
                        "*",
                        color=color,
                        markersize=10,
                        alpha=0.5,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(),
                    )

            # Free-end dots: one per member, sized so the cloud reads
            # at ensemble scale without overwhelming the leading-edge
            # markers.  Plotted before the ensemble-mean overlay so the
            # mean's own marker stays on top.
            for fe_lon, fe_lat, fe_color in free_end_fixes:
                ax.plot(
                    fe_lon,
                    fe_lat,
                    "o",
                    color=fe_color,
                    markersize=5,
                    alpha=0.75,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )

            # Ensemble-mean track up to this step (circular mean of
            # path-0 lon/lat).  Sits above the spaghetti but below the
            # ERA5 reference so the eye reads ERA5 as the truth line and
            # the mean as the model's central tendency.
            legend_drawn = False
            if mean_lats.size:
                m_lats_t = mean_lats[: t + 1]
                m_lons_t = mean_lons[: t + 1]
                m_valid = ~np.isnan(m_lats_t) & ~np.isnan(m_lons_t)
                if m_valid.sum() >= 2:
                    ax.plot(
                        m_lons_t[m_valid],
                        m_lats_t[m_valid],
                        color="crimson",
                        linewidth=2.5,
                        alpha=0.95,
                        marker="o",
                        markersize=4,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        transform=ccrs.PlateCarree(),
                        label="Ensemble mean track",
                        zorder=9,
                    )
                    legend_drawn = True

            # ERA5 reference track up to this time step
            if era5_tracks_np is not None:
                n_era5_plotted = 0
                for path_id in range(era5_tracks_np.shape[1]):
                    e_lats = era5_tracks_np[0, path_id, : t + 1, 0]
                    e_lons = era5_tracks_np[0, path_id, : t + 1, 1]
                    ev = ~np.isnan(e_lats) & ~np.isnan(e_lons)
                    if ev.sum() < 2:
                        continue
                    label = "ERA5 track" if n_era5_plotted == 0 else None
                    ax.plot(
                        e_lons[ev],
                        e_lats[ev],
                        color="black",
                        linewidth=2.5,
                        alpha=0.9,
                        marker="D",
                        markersize=5,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(),
                        label=label,
                        zorder=10,
                    )
                    n_era5_plotted += 1
                if n_era5_plotted > 0:
                    legend_drawn = True
            if legend_drawn:
                ax.legend(
                    loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.8
                )

            try:
                time_str = str(time_steps[t])[:16]
            except Exception:
                time_str = f"Step {t}"

            ax.set_title(
                f"{title} - Step {t + 1}/{n_timesteps} ({time_str})",
                fontsize=TITLE_SIZE,
            )
            writer.grab_frame()
    plt.close()
    print("Animation saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot ensemble TC tracks from a zarr store.",
    )
    parser.add_argument(
        "--ensemble-zarr", required=True, help="Path to ensemble zarr store."
    )
    parser.add_argument(
        "--era5-zarr",
        default=None,
        help="Path to ERA5 reference zarr. When provided together with "
        "--conditioning, conditioned frames use ERA5 data instead of "
        "ensemble data for tracking.",
    )
    parser.add_argument(
        "--conditioning",
        default=None,
        choices=["start", "end", "both"],
        help="Conditioning mode. Determines which frames are replaced with "
        "ERA5 data: start=frame 0, end=frame 11, both=frames 0 & 11.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output artifacts."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end", "both"],
        help="Conditioning mode for this ensemble; encoded in the parquet "
        "(and animation, if --animate) filename so all three modes can "
        "coexist in --output-dir.",
    )
    parser.add_argument("--title", default="Ensemble TC Tracks", help="Animation title.")
    parser.add_argument(
        "--tracker",
        choices=list(TRACKER_CHOICES.keys()),
        default="minmsl",
        help="TC tracking algorithm: 'minmsl' (raw minimum MSL pressure, "
        "default), 'tempest' (real TempestExtremes via DetectNodes/StitchNodes; "
        "needs the binaries on PATH from the e2s-custom conda env), or "
        "'wuduan' (vorticity-based, tropical only).",
    )
    parser.add_argument("--lon-min", type=float, default=-105.0)
    parser.add_argument("--lon-max", type=float, default=-60.0)
    parser.add_argument("--lat-min", type=float, default=10.0)
    parser.add_argument("--lat-max", type=float, default=40.0)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Ensemble members per batch."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker processes for the tempest tracker. Defaults to "
        "$NSLOTS (the SGE allocation) or os.cpu_count().",
    )
    parser.add_argument(
        "--animate", action="store_true", help="Also produce an MP4 animation."
    )
    parser.add_argument(
        "--warm-core",
        action="store_true",
        help="Tempest tracker only: enable the Z&U 2017 _DIFF(z300,z500) "
        "warm-core thickness contour.  Requires z300/z500 in the input "
        "zarr.  Off by default so archived ensemble runs without z300 "
        "can still be tracked (and so storms like Hurricane Sandy keep "
        "their post-tropical landfall phase, which the canonical setup "
        "truncates at extratropical transition).",
    )
    parser.add_argument(
        "--landfall-frame",
        type=int,
        default=-1,
        help="Pinned-end anchor frame (the case YAML's tc_tracks.landfall_"
        "frame; last lead when unset) for the --animate track_qc storm-of-"
        "interest resolution under end-conditioning.",
    )
    args = parser.parse_args()

    if args.warm_core and args.tracker != "tempest":
        print(
            f"NOTE: --warm-core has no effect for tracker '{args.tracker}'; "
            "ignored."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root, coords, time_steps = load_zarr_data(args.ensemble_zarr)
    hpx_level = coords.get("hpx_level")

    # Load ERA5 replacements for conditioned frames
    era5_replacements = None
    if args.era5_zarr and args.conditioning:
        if args.tracker == "tempest":
            variables = _tempest_tracker_variables(warm_core=args.warm_core)
        else:
            variables = TRACKER_CHOICES[args.tracker]["variables"]
        conditioned_frames = mode_frames(args.conditioning)
        era5_replacements = load_era5_replacement(
            args.era5_zarr,
            coords,
            variables,
            conditioned_frames,
            hpx_level,
        )
    elif args.era5_zarr and not args.conditioning:
        print(
            "NOTE: --era5-zarr provided without --conditioning; "
            "ERA5 frame replacement disabled (ERA5 reference track "
            "will still be plotted)."
        )
    elif args.conditioning and not args.era5_zarr:
        print(
            "WARNING: --conditioning provided without --era5-zarr; "
            "ERA5 replacement disabled."
        )

    if args.tracker == "tempest":
        tc_tracks_np = run_tracking_tempest(
            root,
            coords,
            time_steps,
            args.ensemble_zarr,
            era5_replacements=era5_replacements,
            num_workers=args.num_workers,
            hpx_level=hpx_level,
            warm_core=args.warm_core,
        )
    else:
        tc_tracks_np = run_tracking(
            root,
            coords,
            time_steps,
            device,
            args.tracker,
            args.batch_size,
            era5_replacements=era5_replacements,
            hpx_level=hpx_level,
        )

    # Run the same tracker on ERA5 to get the reference track (round-tripped
    # through HPX so it shares the cBottle ensemble's effective resolution).
    era5_tracks_np = None
    if args.era5_zarr:
        if args.tracker == "tempest":
            era5_tracks_np = run_era5_tracking_tempest(
                args.era5_zarr,
                hpx_level=hpx_level,
                warm_core=args.warm_core,
            )
        else:
            era5_tracks_np = run_era5_tracking(
                args.era5_zarr,
                device,
                tracker_name=args.tracker,
                hpx_level=hpx_level,
            )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.animate:
        anim_path = os.path.join(args.output_dir, f"tc_tracks_{args.mode}.mp4")
        animate_tracks(
            tc_tracks_np,
            root,
            coords,
            time_steps,
            anim_path,
            args.title,
            args.lon_min,
            args.lon_max,
            args.lat_min,
            args.lat_max,
            era5_tracks_np=era5_tracks_np,
            mode=args.mode,
            landfall_frame=args.landfall_frame,
        )

    # Save raw tracks for downstream use (long-format parquet, schema in
    # tc_tracks_io.py — keeps tc_msl + tc_w10m alongside lat/lon for the
    # TC intensity / trajectory analysis notebooks).
    tracks_path = os.path.join(args.output_dir, f"tc_tracks_{args.mode}.parquet")
    save_tracks_parquet(tracks_path, tc_tracks_np, era5_arr=era5_tracks_np)
    print(f"Saved raw tracks to {tracks_path}")


if __name__ == "__main__":
    main()
