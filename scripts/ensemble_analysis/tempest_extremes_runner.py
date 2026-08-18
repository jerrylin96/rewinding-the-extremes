"""Run real TempestExtremes DetectNodes/StitchNodes on Earth2Studio ensemble outputs.

Wraps the canonical TempestExtremes binaries
(https://github.com/ClimateGlobalChange/tempestextremes) so the ensemble
TC-tracking pipeline can use them directly instead of an in-Python port.
``DetectNodes`` and ``StitchNodes`` must be on ``PATH``; the ``e2s-custom``
conda environment installs them via the ``tempest-extremes`` package on
conda-forge.

The output array shape ``(n_members, n_paths, n_steps, 4)`` with trailing
axis ``(tc_lat, tc_lon, tc_msl, tc_w10m)`` matches the in-Python trackers
in :mod:`earth2studio.models.dx.tc_tracking`, so the result drops into
:func:`tc_tracks_io.save_tracks_parquet` without further changes.

DetectNodes defaults follow Zarzycki and Ullrich (2017,
https://doi.org/10.1002/2016GL071606):

- ``--searchbymin msl`` with ``--mergedist 6.0`` (degrees)
- ``--closedcontourcmd "msl,200.0,5.5,0"``, optionally appending
  ``";_DIFF(z300,z500),-58.8,6.5,1.0"`` when ``warm_core=True``.

  The optional second contour is the canonical warm-core test: the
  300-500 hPa geopotential thickness must drop by at least 58.8 m^2/s^2
  (= 6 m of geopotential height) within 6.5 deg of a local thickness
  maximum found within 1.0 deg of the candidate.  Delta is in m^2/s^2
  because CBottleVideo / WB2-ERA5 store geopotential in raw units; see
  ``custom_utils/check_geopotential_units.ipynb`` for verification.
  Defaults to off so archived ensemble runs that didn't persist z300
  can still be tracked; with warm-core off the tracker also captures
  the post-tropical landfall phase of storms like Hurricane Sandy that
  the canonical setup truncates at extratropical transition.

- ``--outputcmd "msl,min,0;_VECMAG(u10m,v10m),max,2.0"``; when
  orography is enabled, ``";zs,min,0"`` is appended.

StitchNodes defaults adopt the configuration used by Mu-Ting Chien for
ACE2 TC tracking
(https://github.com/muting-chien/TC_ACE2_2020/blob/main/TC/track_TC/StichNodes/SN.ACE2_TC_Bellgroup.csh):

- ``--range 8.0`` deg
- ``--mintime 54h``, ``--maxgap 24h``
- ``--threshold "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<=,150.0,10"``
  (the ``zs`` clause is added only when orography is enabled)

These defaults assume the input is a forecast long enough to support
``mintime 54h`` and a ten-step persistence requirement.  CBottleVideo's
default 12-step (72 h) window is short relative to these filters and
will reject TCs that don't reach mature intensity within the window;
that is the documented behaviour of the Mu-Ting recipe.

Surface elevation (``zs``) is a static field, so when
``use_orography=True`` (the default) we fetch it once from
``earth2studio.data.WB2ERA5`` via the ``z`` (geopotential_at_surface)
lexicon key, divide by ``g0 = 9.80665`` to get meters, and broadcast
it across every member's staged NetCDF.
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DETECT_NODES_BIN = "DetectNodes"
STITCH_NODES_BIN = "StitchNodes"

DEFAULT_DETECT_FLAGS: dict[str, str] = {
    "searchbymin": "msl",
    "mergedist": "6.0",
    "closedcontourcmd": "msl,200.0,5.5,0",
    "outputcmd": "msl,min,0;_VECMAG(u10m,v10m),max,2.0",
}

# Z&U 2017 warm-core thickness contour, appended to closedcontourcmd
# when ``warm_core=True``.
WARM_CORE_CONTOUR = "_DIFF(z300,z500),-58.8,6.5,1.0"

# StitchNodes' --in_fmt declares the column names of the candidate file
# in order, after the leading i_lon, i_lat columns DetectNodes always
# emits. Must stay consistent with --outputcmd above.
DEFAULT_STITCH_FLAGS: dict[str, str] = {
    "in_fmt": "lon,lat,msl,wind",
    "range": "8.0",
}

# Mu-Ting Chien's stitch filters: each clause must hold for at least
# 10 timesteps along a candidate track for it to survive stitching.
# Wind and zs values come from --outputcmd, lat is computed internally.
DEFAULT_STITCH_THRESHOLD_NO_OROG = (
    "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10"
)
DEFAULT_STITCH_THRESHOLD_WITH_OROG = (
    f"{DEFAULT_STITCH_THRESHOLD_NO_OROG};zs,<=,150.0,10"
)

DEFAULT_MINTIME_HOURS = 54
DEFAULT_MAXGAP_HOURS = 24

# Gravitational constant for geopotential -> geometric height conversion.
# Matches the ERA5 / WB2 convention.
_G0 = 9.80665


def _check_binaries() -> None:
    """Raise a helpful error if DetectNodes/StitchNodes are not on PATH."""
    missing = [
        b for b in (DETECT_NODES_BIN, STITCH_NODES_BIN) if shutil.which(b) is None
    ]
    if missing:
        raise RuntimeError(
            f"TempestExtremes binaries not found on PATH: {missing}. "
            "Activate the e2s-custom conda environment (which installs the "
            "tempest-extremes package from conda-forge) before running the "
            "tempest tracker."
        )


def fetch_surface_orography(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Fetch the static surface elevation (meters) from WB2 ERA5.

    Surface elevation doesn't vary in time, so this is a one-shot call
    shared across every ensemble member; the caller is expected to fetch
    once at the top of :func:`track_with_tempest` and pass the resulting
    ``[H, W]`` array down to each member.

    Fails hard if the fetch fails or the returned grid doesn't match the
    tracker's ``lats`` / ``lons`` exactly.  Compute-node deployments
    without network access should pre-fetch this field once on a login
    node so the ``EARTH2STUDIO_CACHE`` directory carries it before the
    job starts.

    Parameters
    ----------
    lats, lons : np.ndarray
        1-D lat/lon arrays the staged NetCDFs will be written on.  The
        WB2 ERA5 grid must match these exactly.

    Returns
    -------
    np.ndarray
        ``[H, W]`` float32 array of surface elevation in meters.
    """
    import torch

    from earth2studio.data import WB2ERA5
    from earth2studio.data.utils import fetch_data

    # Any valid ERA5 timestamp works; the field is constant in time.
    # Pick 2020-01-01 to avoid any first-of-record edge cases.
    sample_time = datetime.datetime(2020, 1, 1)
    ds = WB2ERA5()
    data, coords = fetch_data(
        ds, [sample_time], ["z"], device=torch.device("cpu")
    )
    # fetch_data shape: [time=1, lead_time=1, var=1, H, W]
    geopotential = data[0, 0, 0].numpy()
    elevation = (geopotential / _G0).astype(np.float32)

    wb2_lats = np.asarray(coords["lat"])
    wb2_lons = np.asarray(coords["lon"])
    if elevation.shape != (len(lats), len(lons)):
        raise RuntimeError(
            f"WB2 surface geopotential has shape {elevation.shape}, "
            f"but tracker grid is {(len(lats), len(lons))}.  The staged "
            "ensemble data must be on the standard ERA5 721x1440 grid."
        )
    if not np.allclose(wb2_lats, lats) or not np.allclose(wb2_lons, lons):
        raise RuntimeError(
            "WB2 surface geopotential lat/lon do not match the tracker "
            "grid.  Make sure the staged data uses the same WB2-ERA5 "
            "lat/lon convention (90 -> -90, 0 -> 359.75)."
        )
    return elevation


def _compose_default_flags(
    use_orography: bool,
    warm_core: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (detect_flags, stitch_flags) with orography clauses applied.

    Keeps :data:`DEFAULT_DETECT_FLAGS` / :data:`DEFAULT_STITCH_FLAGS`
    free of orography-specific content so callers that override flags
    explicitly aren't surprised by hidden ``zs`` columns.  When
    ``warm_core`` is True the Z&U 2017 _DIFF(z300,z500) thickness
    contour is appended to ``closedcontourcmd``.
    """
    detect = dict(DEFAULT_DETECT_FLAGS)
    stitch = dict(DEFAULT_STITCH_FLAGS)
    if warm_core:
        detect["closedcontourcmd"] = (
            detect["closedcontourcmd"] + ";" + WARM_CORE_CONTOUR
        )
    if use_orography:
        detect["outputcmd"] = detect["outputcmd"] + ";zs,min,0"
        stitch["in_fmt"] = stitch["in_fmt"] + ",zs"
        stitch["threshold"] = DEFAULT_STITCH_THRESHOLD_WITH_OROG
    else:
        stitch["threshold"] = DEFAULT_STITCH_THRESHOLD_NO_OROG
    return detect, stitch


def _stage_member_netcdf(
    data: np.ndarray,
    variables: Sequence[str],
    lats: np.ndarray,
    lons: np.ndarray,
    timestep_hours: float,
    out_path: Path,
    orography: Optional[np.ndarray] = None,
) -> None:
    """Write one member's ``[T, V, H, W]`` slab to a CF-compliant NetCDF.

    The time axis is synthetic: step ``i`` is recorded as ``i *
    timestep_hours`` hours since 1970-01-01. Real timestamps from the
    source zarr are not used because TempestExtremes only needs a
    monotonic time axis, and a synthetic axis makes mapping each track
    point back to a step index trivial in :func:`_parse_track_file`.

    When ``orography`` is supplied it is added as a time-invariant
    ``zs`` variable (broadcast along the time axis so TempestExtremes
    sees it at every step).
    """
    import xarray as xr

    n_t = data.shape[0]
    time_axis = np.arange(n_t, dtype=np.float64) * float(timestep_hours)

    data_vars: dict[str, tuple] = {}
    for v_idx, v_name in enumerate(variables):
        data_vars[v_name] = (
            ("time", "lat", "lon"),
            data[:, v_idx, :, :].astype(np.float32),
        )
    if orography is not None:
        # np.broadcast_to is read-only; xarray reads it as-is without
        # materialising a [T, H, W] copy, keeping memory bounded.
        zs = np.broadcast_to(orography.astype(np.float32), (n_t, *orography.shape))
        data_vars["zs"] = (("time", "lat", "lon"), zs)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (
                "time",
                time_axis,
                {
                    "units": "hours since 1970-01-01 00:00:00",
                    "calendar": "standard",
                },
            ),
            "lat": ("lat", lats.astype(np.float64), {"units": "degrees_north"}),
            "lon": ("lon", lons.astype(np.float64), {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(out_path, unlimited_dims=["time"])


def _run_detect_nodes(nc_path: Path, out_path: Path, flags: dict[str, str]) -> None:
    cmd = [DETECT_NODES_BIN, "--in_data", str(nc_path), "--out", str(out_path)]
    for k, v in flags.items():
        cmd.extend([f"--{k}", str(v)])
    # Binary name is a fixed constant; flag values default to the
    # Zarzycki & Ullrich 2017 set or are supplied by the in-process
    # caller, not by user input. S603 doesn't apply.
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"DetectNodes failed (exit {result.returncode}).\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


def _run_stitch_nodes(
    in_path: Path,
    out_path: Path,
    flags: dict[str, str],
    mintime_hours: int,
    maxgap_hours: int,
) -> None:
    cmd = [
        STITCH_NODES_BIN,
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--mintime",
        f"{mintime_hours}h",
        "--maxgap",
        f"{maxgap_hours}h",
    ]
    for k, v in flags.items():
        cmd.extend([f"--{k}", str(v)])
    # See note in _run_detect_nodes; same rationale applies.
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"StitchNodes failed (exit {result.returncode}).\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


def _parse_track_file(
    track_path: Path,
    n_steps: int,
    timestep_hours: float,
    has_orography_col: bool = False,
) -> np.ndarray:
    """Parse a StitchNodes output file into ``[n_paths, n_steps, 4]``.

    Trailing axis is ``(tc_lat, tc_lon, tc_msl, tc_w10m)``. Cells with no
    fix are NaN.

    StitchNodes output format (without orography)::

        start <num_pts> <yyyy> <mm> <dd> <hh>
            <i_lon> <i_lat> <lon> <lat> <msl> <wind> <yyyy> <mm> <dd> <hh>
            ...

    With orography enabled there is an extra ``zs`` column between
    ``wind`` and ``yyyy``; the time fields shift by one position.

    The step index is reconstructed from the ``yyyy mm dd hh`` columns by
    measuring the offset from the synthetic 1970-01-01 base time
    :func:`_stage_member_netcdf` writes into the input NetCDF.
    """
    base = datetime.datetime(1970, 1, 1)
    tracks: list[np.ndarray] = []
    cur: Optional[np.ndarray] = None

    time_col_start = 7 if has_orography_col else 6
    min_cols = time_col_start + 4  # need yyyy, mm, dd, hh

    if not track_path.exists() or track_path.stat().st_size == 0:
        return np.zeros((0, n_steps, 4), dtype=np.float64)

    with open(track_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("start"):
                if cur is not None:
                    tracks.append(cur)
                cur = np.full((n_steps, 4), np.nan, dtype=np.float64)
                continue
            if cur is None:
                continue
            parts = line.split()
            if len(parts) < min_cols:
                continue
            lon = float(parts[2])
            lat = float(parts[3])
            msl = float(parts[4])
            wind = float(parts[5])
            year, month, day, hour = (
                int(parts[i]) for i in range(time_col_start, time_col_start + 4)
            )
            offset_h = (
                datetime.datetime(year, month, day, hour) - base
            ).total_seconds() / 3600.0
            step = int(round(offset_h / float(timestep_hours)))
            if 0 <= step < n_steps:
                cur[step, 0] = lat
                cur[step, 1] = lon
                cur[step, 2] = msl
                cur[step, 3] = wind
        if cur is not None:
            tracks.append(cur)

    if not tracks:
        return np.zeros((0, n_steps, 4), dtype=np.float64)
    return np.stack(tracks, axis=0)


def parse_candidates_file(
    cand_path: Path,
    n_steps: int,
    timestep_hours: float,
) -> list[np.ndarray]:
    """Parse a DetectNodes candidate file into per-step candidate arrays.

    Returns a list of length ``n_steps``; entry ``t`` is a ``[n_cand, 3]``
    array of ``(lat, lon, msl)`` for the candidates DetectNodes admitted at
    step ``t`` (empty ``[0, 3]`` where none).  Used by the anchor-diagnostics
    animations to show what the detector saw *before* stitching, so a member
    whose storm never makes it into a stitched track can be split into
    "DetectNodes rejected the low" vs "StitchNodes failed to stitch it".

    DetectNodes output format (one block per time step)::

        <yyyy>  <mm>  <dd>  <n_candidates>  <hh>
            <i_lon>  <i_lat>  <lon>  <lat>  <outputcmd columns...>

    Only the first ``outputcmd`` column (``msl`` under the default flags) is
    kept.  The step index is reconstructed from the ``yyyy mm dd hh`` header
    exactly like :func:`_parse_track_file` -- via the synthetic 1970-01-01
    time base :func:`_stage_member_netcdf` writes into the staged NetCDF.
    """
    base = datetime.datetime(1970, 1, 1)
    per_step: list[list[list[float]]] = [[] for _ in range(n_steps)]
    step: Optional[int] = None

    if not cand_path.exists() or cand_path.stat().st_size == 0:
        return [np.zeros((0, 3), dtype=np.float64) for _ in range(n_steps)]

    with open(cand_path) as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            # Header line: exactly 5 integer tokens (yyyy mm dd count hh).
            if len(parts) == 5 and all(p.lstrip("-").isdigit() for p in parts):
                year, month, day, _, hour = (int(p) for p in parts)
                offset_h = (
                    datetime.datetime(year, month, day, hour) - base
                ).total_seconds() / 3600.0
                idx = int(round(offset_h / float(timestep_hours)))
                step = idx if 0 <= idx < n_steps else None
                continue
            if step is None or len(parts) < 5:
                continue
            lon, lat, msl = float(parts[2]), float(parts[3]), float(parts[4])
            per_step[step].append([lat, lon, msl])

    return [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in per_step]


def infer_timestep_hours(time_steps: np.ndarray) -> float:
    """Infer dt (hours) from a 1-D ``time_steps`` array.

    Accepts ``datetime64`` or numeric (interpreted as hours). Falls back
    to 1 h when the array is too short to compute a difference.
    """
    if len(time_steps) < 2:
        return 1.0
    diff = time_steps[1] - time_steps[0]
    if isinstance(diff, np.timedelta64):
        return float(diff / np.timedelta64(1, "h"))
    return float(diff)


def track_member_with_tempest(
    member_data: np.ndarray,
    variables: Sequence[str],
    lats: np.ndarray,
    lons: np.ndarray,
    timestep_hours: float,
    member_idx: int,
    work_dir: Path,
    detect_flags: Optional[dict[str, str]] = None,
    stitch_flags: Optional[dict[str, str]] = None,
    mintime_hours: int = DEFAULT_MINTIME_HOURS,
    maxgap_hours: int = DEFAULT_MAXGAP_HOURS,
    orography: Optional[np.ndarray] = None,
    warm_core: bool = False,
) -> np.ndarray:
    """Run DetectNodes + StitchNodes on one member's ``[T, V, H, W]`` slab.

    Returns ``[n_paths, n_steps, 4]`` with trailing axis ``(tc_lat,
    tc_lon, tc_msl, tc_w10m)``. NaN where the tracker has no fix.

    The per-member intermediates are written to and cleaned up from
    ``work_dir`` (the directory itself is left in place so the caller
    can run multiple members in parallel against the same scratch dir).
    Used directly by per-member parallel callers in
    ``plot_ensemble_tc_tracks.py`` that stream members from a zarr store
    to keep peak memory bounded.

    Parameters
    ----------
    member_data : np.ndarray
        ``[T, V, H, W]`` slab for one ensemble member.
    variables : Sequence[str]
        Variable names along axis ``V``. Must include ``msl``, ``u10m``,
        ``v10m``; when ``warm_core=True`` must also include ``z300`` and
        ``z500``.
    orography : np.ndarray, optional
        Static surface elevation in meters, shape ``[H, W]``.  When
        supplied, added as the ``zs`` variable in the staged NetCDF and
        the orography-aware Mu-Ting threshold is applied during
        stitching.  The runner does not fetch this itself: callers that
        want orography should pull it once via
        :func:`fetch_surface_orography` and pass it in.
    warm_core : bool
        When True, append the Z&U 2017 _DIFF(z300,z500) warm-core
        thickness contour to the default ``closedcontourcmd``.  Default
        False so archived ensemble runs without z300 can still be
        tracked.  If ``detect_flags`` overrides ``closedcontourcmd``,
        ``warm_core`` has no effect on the override — the explicit
        value wins.  Include the ``_DIFF(z300,z500),-58.8,6.5,1.0``
        clause manually in the override if you want warm-core there.
    """
    use_orography = orography is not None
    default_detect, default_stitch = _compose_default_flags(
        use_orography,
        warm_core=warm_core,
    )
    detect = dict(default_detect)
    if detect_flags:
        detect.update(detect_flags)
    stitch = dict(default_stitch)
    if stitch_flags:
        stitch.update(stitch_flags)

    if member_data.ndim != 4:
        raise ValueError(
            f"member_data must have shape [T, V, H, W]; got {member_data.shape}"
        )
    n_steps, n_vars, _, _ = member_data.shape
    if n_vars != len(variables):
        raise ValueError(
            f"variables ({len(variables)}) doesn't match V axis ({n_vars})"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    nc_path = work_dir / f"member_{member_idx:04d}.nc"
    cand_path = work_dir / f"member_{member_idx:04d}.candidates.txt"
    track_path = work_dir / f"member_{member_idx:04d}.tracks.txt"
    try:
        _stage_member_netcdf(
            member_data,
            variables,
            lats,
            lons,
            timestep_hours,
            nc_path,
            orography=orography,
        )
        _run_detect_nodes(nc_path, cand_path, detect)
        _run_stitch_nodes(cand_path, track_path, stitch, mintime_hours, maxgap_hours)
        return _parse_track_file(
            track_path,
            n_steps,
            timestep_hours,
            has_orography_col=use_orography,
        )
    finally:
        # Per-member intermediates are deleted so disk usage stays
        # bounded for large ensembles (~100 MB per member-NetCDF at
        # 721x1440x12).
        nc_path.unlink(missing_ok=True)
        cand_path.unlink(missing_ok=True)
        track_path.unlink(missing_ok=True)


def pad_and_stack_tracks(
    per_member: list[np.ndarray], n_members: int, n_steps: int
) -> np.ndarray:
    """Pad ``[n_paths_m, n_steps, 4]`` arrays to common ``n_paths`` and stack."""
    max_paths = max((t.shape[0] for t in per_member), default=0)
    if max_paths == 0:
        return np.full((n_members, 0, n_steps, 4), np.nan, dtype=np.float64)
    padded: list[np.ndarray] = []
    for t in per_member:
        pad = max_paths - t.shape[0]
        if pad > 0:
            t = np.concatenate(
                [t, np.full((pad, n_steps, 4), np.nan, dtype=t.dtype)],
                axis=0,
            )
        padded.append(t)
    return np.stack(padded, axis=0)


def track_with_tempest(
    data: np.ndarray,
    variables: Sequence[str],
    lats: np.ndarray,
    lons: np.ndarray,
    timestep_hours: float,
    work_dir: Optional[Path] = None,
    detect_flags: Optional[dict[str, str]] = None,
    stitch_flags: Optional[dict[str, str]] = None,
    mintime_hours: int = DEFAULT_MINTIME_HOURS,
    maxgap_hours: int = DEFAULT_MAXGAP_HOURS,
    use_orography: bool = True,
    warm_core: bool = False,
) -> np.ndarray:
    """Run DetectNodes + StitchNodes per member and return ``[M, P, T, 4]``.

    Parameters
    ----------
    data : np.ndarray
        ``[M, T, V, H, W]`` slab. ``M`` is the member axis (use 1 for an
        ERA5-style single-trace input), ``T`` the time steps, ``V`` the
        per-tracker variable count, ``H``/``W`` the lat/lon grid.
    variables : Sequence[str]
        Variable names along axis ``V``, in order. Must include any
        names referenced by the detect/stitch flags.  Always needs
        ``msl``, ``u10m``, ``v10m``; with ``warm_core=True`` also
        ``z300`` and ``z500``.
    lats, lons : np.ndarray
        1-D coordinate arrays.
    timestep_hours : float
        Hours between consecutive time steps. Assumed uniform.
    work_dir : Path, optional
        Directory for staged NetCDFs and TempestExtremes scratch files.
        If None, a temporary directory is created and removed on exit.
    detect_flags, stitch_flags : dict, optional
        Override the default Zarzycki & Ullrich (2017) detect / Mu-Ting
        Chien stitch flags.
    mintime_hours, maxgap_hours : int
        StitchNodes ``--mintime`` and ``--maxgap`` in hours.  Defaults
        (54 h, 24 h) follow Mu-Ting's ACE2 configuration.
    use_orography : bool
        When ``True`` (default), fetch the static WB2 ERA5 surface
        elevation once and stage it as a ``zs`` variable in every
        member's NetCDF.  This enables the canonical Mu-Ting
        ``zs <= 150 m`` stitch threshold that suppresses spurious
        tracks over high terrain.  Fails hard if the WB2 fetch fails;
        pass ``False`` to skip the orography filter entirely.
    warm_core : bool
        When True, append the Z&U 2017 _DIFF(z300,z500) warm-core
        thickness contour to the default ``closedcontourcmd``.  Default
        False so archived ensemble runs without z300 can still be
        tracked (and so storms like Hurricane Sandy keep their
        post-tropical landfall phase, which the canonical setup
        truncates at extratropical transition).  If ``detect_flags``
        overrides ``closedcontourcmd``, ``warm_core`` has no effect on
        the override — the explicit value wins.  Include the
        ``_DIFF(z300,z500),-58.8,6.5,1.0`` clause manually in the
        override if you want warm-core there.

    Returns
    -------
    np.ndarray
        ``[M, P, T, 4]`` with trailing axis ``(tc_lat, tc_lon, tc_msl,
        tc_w10m)``. NaN where the tracker has no fix.
    """
    _check_binaries()

    if data.ndim != 5:
        raise ValueError(f"data must have shape [M, T, V, H, W]; got {data.shape}")
    n_members, n_steps, _, _, _ = data.shape

    # Fetch the static orography once before forking per-member work so
    # we don't pay the WB2 round-trip M times (and so a fetch failure
    # surfaces before any TempestExtremes work starts).
    orography: Optional[np.ndarray] = None
    if use_orography:
        print("Fetching static surface orography from WB2 ERA5...")
        orography = fetch_surface_orography(lats, lons)
        print(
            f"  Orography ready: shape {orography.shape}, "
            f"min {orography.min():.1f} m, max {orography.max():.1f} m"
        )

    cleanup = work_dir is None
    if cleanup:
        work_dir = Path(tempfile.mkdtemp(prefix="tempest_extremes_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        per_member: list[np.ndarray] = []
        for m in range(n_members):
            print(f"  Member {m + 1}/{n_members}: running DetectNodes + StitchNodes...")
            per_member.append(
                track_member_with_tempest(
                    data[m],
                    variables=variables,
                    lats=lats,
                    lons=lons,
                    timestep_hours=timestep_hours,
                    member_idx=m,
                    work_dir=work_dir,
                    detect_flags=detect_flags,
                    stitch_flags=stitch_flags,
                    mintime_hours=mintime_hours,
                    maxgap_hours=maxgap_hours,
                    orography=orography,
                    warm_core=warm_core,
                )
            )
        return pad_and_stack_tracks(per_member, n_members, n_steps)
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
