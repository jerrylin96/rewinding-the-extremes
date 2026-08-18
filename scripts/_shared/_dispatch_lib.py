"""Shared helpers for the generic dispatch_* drivers.

Centralizes: case-YAML loading + validation, canonical path conventions,
`qsub` submission (with optional `-hold_jid` chaining), job-id parsing,
and resolution of palette variable sets / variable defaults.

Keep this module free of heavy imports so it's cheap to import from every
driver script.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CASES_DIR = SCRIPT_DIR / "cases"
PALETTES_PATH = CASES_DIR / "_palettes.yaml"


# ----------------------------------------------------------------------
# Canonical conditioning-mode conventions
# ----------------------------------------------------------------------

# Each conditioning mode maps to a fixed list of frame indices and a
# corresponding suffix in the output zarr filename.
MODE_FRAMES: Dict[str, List[int]] = {
    "start": [0],
    "end": [11],
    "both": [0, 11],
}

# Default matplotlib tab colours per mode — shared by plot scripts that
# need consistent coloring across figures (NOT used by the track-on-map
# plots, which define their own muted palette).
MODE_COLORS_TAB: Dict[str, str] = {
    "start": "tab:blue",
    "end": "tab:orange",
    "both": "tab:green",
}

# Short display labels.  Scripts that want e.g. "Start Conditioning" can
# format on top of these.
MODE_LABELS_SHORT: Dict[str, str] = {
    "start": "Start",
    "end": "End",
    "both": "Both",
}


def mode_frames(mode: str) -> List[int]:
    """Return the frame indices for a conditioning mode."""
    if mode not in MODE_FRAMES:
        raise ValueError(
            f"Unknown conditioning mode '{mode}'. "
            f"Valid modes: {sorted(MODE_FRAMES)}"
        )
    return MODE_FRAMES[mode]


def frames_suffix(frames: Sequence[int]) -> str:
    """Filename suffix for a frame list: [0] -> 'f0', [0,11] -> 'f0_11'."""
    return "f" + "_".join(str(f) for f in frames)


def ensemble_zarr_path(base: str, mode: str, size: int) -> str:
    """Canonical ensemble zarr path: ``<base>/<mode>/ensemble_<fsuffix>_n<N>.zarr``."""
    return f"{base}/{mode}/ensemble_{frames_suffix(mode_frames(mode))}_n{size}.zarr"


def era5_zarr_path(base: str, mode: str) -> str:
    """Canonical ERA5 reference zarr path: ``<base>/<mode>/era5_reference.zarr``."""
    return f"{base}/{mode}/era5_reference.zarr"


DEFAULT_TRACKER = "minmsl"


def tc_tracks_dir(diag_root: str, tracker: str) -> str:
    """Canonical tc_tracks output directory for one tracker.

    Layout: ``<diag_root>/tc_tracks/<tracker>`` where ``diag_root`` is
    typically ``<case_base>/diagnostics``. The three conditioning modes
    share this directory; per-mode artifacts are distinguished by a
    ``_<mode>`` filename suffix and the aggregated plot has no suffix.
    """
    return f"{diag_root}/tc_tracks/{tracker}"


def tc_tracks_parquet_path(diag_root: str, mode: str, tracker: str) -> str:
    """Canonical per-mode tc_tracks parquet path.

    Layout: ``<diag_root>/tc_tracks/<tracker>/tc_tracks_<mode>.parquet``.
    """
    return f"{tc_tracks_dir(diag_root, tracker)}/tc_tracks_{mode}.parquet"


def trackers_from_case(
    case: Dict[str, Any], override: Optional[Sequence[str]] = None
) -> List[str]:
    """Resolve the TC tracker list for a case.

    Reads ``tc_tracks.trackers`` from the case YAML (list of names).  If
    ``override`` is supplied (typically from a repeatable ``--tracker``
    CLI flag), it restricts the list to that subset; an unknown override
    name raises.  If no list is configured, falls back to a single-item
    list of ``DEFAULT_TRACKER`` so old-style cases with only a tracker
    domain configured still produce something.
    """
    tc_block = case.get("tc_tracks") or {}
    configured = tc_block.get("trackers")
    if configured is None:
        configured = [DEFAULT_TRACKER]
    if not isinstance(configured, (list, tuple)) or not configured:
        raise SystemExit(
            f"ERROR: case '{case['name']}' tc_tracks.trackers must be a "
            f"non-empty list of tracker names; got {configured!r}."
        )
    configured = [str(t) for t in configured]
    if override:
        unknown = [t for t in override if t not in configured]
        if unknown:
            raise SystemExit(
                f"ERROR: case '{case['name']}' does not configure tracker(s) "
                f"{unknown}; available: {configured}"
            )
        return list(override)
    return configured


def validate_zarrs_exist(case: Dict[str, Any], modes: Sequence[str], size: int) -> None:
    """Exit with a friendly error if upstream zarrs are missing.

    Pre-flight check shared by every ``dispatch_*.py`` driver: confirms
    the canonical ensemble + ERA5 zarr stores exist for each requested
    mode before any qsub job is submitted.
    """
    base = case["base"]
    missing: List[str] = []
    for mode in modes:
        missing.extend(
            f"  {case['name']}/{mode}: {path}"
            for path in (
                ensemble_zarr_path(base, mode, size),
                era5_zarr_path(base, mode),
            )
            if not Path(path).exists()
        )
    if missing:
        print(
            "ERROR: the following upstream zarr stores do not exist "
            "(run dispatch_ensemble.py first):",
            file=sys.stderr,
        )
        print("\n".join(missing), file=sys.stderr)
        raise SystemExit(1)


def resolve_landfall_frame(case: Dict[str, Any]) -> int:
    """Pick a landfall reference frame for plot annotations.

    Reads ``tc_tracks.landfall_frame`` from the case YAML when present;
    returns ``-1`` (no landfall line) otherwise.  The last-frame
    convention used by current TC case YAMLs (frame 11 for a 66h window)
    is not assumed — it must be declared explicitly so cases with
    non-standard windows or post-landfall extensions can opt into a
    different frame without misleading plot scripts.
    """
    tc = case.get("tc_tracks") or {}
    landfall = tc.get("landfall_frame")
    return int(landfall) if landfall is not None else -1


def warm_core_from_case(case: Dict[str, Any]) -> bool:
    """Per-case default for the tempest warm-core thickness filter.

    Reads ``tc_tracks.warm_core`` from the case YAML (bool); ``False`` when
    absent.  When True, ``dispatch_tc_tracks`` forwards ``--warm-core`` to the
    tempest worker so DetectNodes also requires the Zarzycki & Ullrich 2017
    ``_DIFF(z300,z500)`` thickness contour -- a warm-core test that rejects
    cold-core extratropical lows.  Requires z300/z500 in the input zarr; the
    minmsl and wuduan trackers ignore it.  An explicit ``--warm-core`` /
    ``--no-warm-core`` CLI flag overrides this default for a single run.
    """
    tc = case.get("tc_tracks") or {}
    return bool(tc.get("warm_core", False))


# ----------------------------------------------------------------------
# Small helpers shared across plot scripts
# ----------------------------------------------------------------------


def resolve_timezone(case: Dict[str, Any]) -> str:
    """Return the case's IANA timezone name, falling back to ``UTC``.

    Cases declare ``timezone: <IANA name>`` at the top level of the YAML
    so figures with a lead-time axis can show wall-clock time at the
    event location (Sandy in EDT, the PNW heat dome in PDT, etc.).
    Cases that omit the field still render — the helpers in
    ``local_time_axis.py`` fall back to UTC ticks rather than raising.
    """
    tz = case.get("timezone")
    return str(tz) if tz else "UTC"


def start_time_iso_from_case(case: Dict[str, Any]) -> str:
    """Render ``ensemble.start_time`` as an ISO 8601 UTC string.

    The case YAML stores the IC as a ``{year, month, day, hour}`` dict;
    the compute scripts want a single CLI-friendly string they can parse
    with ``datetime.fromisoformat``.  The trailing ``Z`` makes the UTC
    intent explicit at a glance.
    """
    start = case["ensemble"]["start_time"]
    return (
        f"{int(start['year']):04d}-{int(start['month']):02d}-"
        f"{int(start['day']):02d}T{int(start['hour']):02d}:00:00Z"
    )


def lead_hours(lead_times: Any) -> Any:
    """Convert a ``lead_time`` coordinate array to float hours.

    Handles ``timedelta64`` (common in zarr stores) and plain numeric arrays.
    """
    import numpy as np  # local import — keeps dispatch scripts numpy-free

    lt = np.asarray(lead_times)
    if np.issubdtype(lt.dtype, np.timedelta64):
        return (lt / np.timedelta64(1, "h")).astype(float)
    return lt.astype(float)


def wrap_lon_0_360(lon: float) -> float:
    """Wrap a longitude into the ``[0, 360)`` range."""
    return lon if lon >= 0 else lon + 360


def zarr_variable_index(root: Any, var_name: str) -> int:
    """Look up the index of ``var_name`` in a zarr store's ``variable`` coord.

    Raises ``ValueError`` with the available variables if missing.
    """
    import numpy as np  # local import — keeps dispatch scripts numpy-free

    var_list = list(np.array(root["variable"][:]))
    if var_name not in var_list:
        raise ValueError(
            f"Variable '{var_name}' not found in zarr store. " f"Available: {var_list}"
        )
    return var_list.index(var_name)


def parse_mode_list(csv: Optional[str], default: Sequence[str]) -> List[str]:
    """Parse a comma-separated ``--modes`` CLI value, falling back to ``default``."""
    if csv is None:
        return list(default)
    return [m.strip() for m in csv.split(",") if m.strip()]


def parse_member_spec(spec: str, n_members: int) -> List[int]:
    """Expand a ``--members`` CLI value into a sorted list of member indices.

    Accepts ``all``, a range (``0-99``), an explicit list (``5,6,7``), or a
    mix of the two (``0-3,10,20-25``).  Out-of-range and malformed entries
    raise ``ValueError`` here rather than surfacing as an IndexError after a
    caller has already spent time on the members that did parse.
    """
    if spec.strip().lower() == "all":
        return list(range(n_members))

    members: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo_s, _, hi_s = part.partition("-")
                bounds = (int(lo_s), int(hi_s))
            else:
                bounds = (int(part), int(part))
        except ValueError as exc:
            raise ValueError(
                f"member spec '{spec}': cannot parse entry '{part}' "
                f"(expected 'N' or 'LO-HI', both non-negative)"
            ) from exc

        lo, hi = bounds
        if lo > hi:
            raise ValueError(
                f"member spec '{spec}': range '{part}' is inverted "
                f"(expected LO-HI with LO <= HI)"
            )
        members.update(range(lo, hi + 1))

    if not members:
        raise ValueError(f"member spec '{spec}' selected no members")
    out_of_range = sorted(m for m in members if m < 0 or m >= n_members)
    if out_of_range:
        shown = out_of_range[:5]
        suffix = "" if len(out_of_range) == len(shown) else ", ..."
        raise ValueError(
            f"member spec '{spec}' includes {len(out_of_range)} out-of-range "
            f"member(s) {shown}{suffix} (ensemble has {n_members} members)"
        )
    return sorted(members)


# ----------------------------------------------------------------------
# Case + palette loading
# ----------------------------------------------------------------------


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_palettes() -> Dict[str, Any]:
    """Load ``cases/_palettes.yaml`` (shared variable sets + plot defaults)."""
    if not PALETTES_PATH.exists():
        raise FileNotFoundError(f"Palettes file not found: {PALETTES_PATH}")
    return _read_yaml(PALETTES_PATH)


def list_cases() -> List[str]:
    """List available case names (yaml files under ``cases/`` excluding _palettes)."""
    return sorted(
        p.stem for p in CASES_DIR.glob("*.yaml") if not p.stem.startswith("_")
    )


def load_case(name_or_path: str) -> Dict[str, Any]:
    """Load a case YAML by name (e.g. ``katrina``) or explicit path.

    Injects ``name`` into the returned dict (from the filename stem).
    """
    p = Path(name_or_path)
    if not p.is_file():
        candidate = CASES_DIR / f"{name_or_path}.yaml"
        if not candidate.is_file():
            available = ", ".join(list_cases())
            raise FileNotFoundError(
                f"No case YAML found for '{name_or_path}'. " f"Available: {available}"
            )
        p = candidate

    data = _read_yaml(p)
    data["name"] = p.stem
    _validate_case(data)
    return data


def _validate_case(case: Dict[str, Any]) -> None:
    """Lightweight schema validation on a case dict."""
    name = case.get("name", "<unknown>")
    required_top = ["display_name", "base", "ensemble"]
    for k in required_top:
        if k not in case:
            raise ValueError(f"case '{name}': missing required field '{k}'")

    ens = case["ensemble"]
    for k in ["start_time", "size", "modes"]:
        if k not in ens:
            raise ValueError(f"case '{name}': ensemble.{k} is required")

    start = ens["start_time"]
    for k in ["year", "month", "day", "hour"]:
        if k not in start:
            raise ValueError(f"case '{name}': ensemble.start_time.{k} is required")

    for mode in ens["modes"]:
        if mode not in MODE_FRAMES:
            raise ValueError(
                f"case '{name}': unknown conditioning mode '{mode}' "
                f"(valid: {sorted(MODE_FRAMES)})"
            )

    regions = case.get("regions")
    if regions is not None:
        if not isinstance(regions, dict):
            raise ValueError(
                f"case '{name}': 'regions' must be a mapping of name -> "
                f"{{lon: [min, max], lat: [min, max]}}"
            )
        for rname, box in regions.items():
            lon = (box or {}).get("lon")
            lat = (box or {}).get("lat")
            if not lon or not lat or len(lon) != 2 or len(lat) != 2:
                raise ValueError(
                    f"case '{name}': regions.{rname} must set lon: [min, max] "
                    f"and lat: [min, max]; got {box!r}"
                )

    # TC-ness keys off the presence of a ``tc_tracks`` block, but the detection
    # box now lives in ``regions.track``.  Keep the two coupled so they cannot
    # drift: a stray ``tc_tracks`` without a box dies late in dispatch, and a
    # stray ``regions.track`` without a block silently disagrees with the
    # synoptic-PCA scalar/track split.  Enforce the biconditional up front.
    has_tc_block = bool(case.get("tc_tracks"))
    has_track_region = bool((regions or {}).get("track"))
    if has_tc_block and not has_track_region:
        raise ValueError(
            f"case '{name}': has a 'tc_tracks' block but no 'regions.track' box; "
            f"TC diagnostics need the detection domain.  Add regions.track (the "
            f"TC-track detection box) or remove the tc_tracks block."
        )
    if has_track_region and not has_tc_block:
        raise ValueError(
            f"case '{name}': defines 'regions.track' but has no 'tc_tracks' block; "
            f"the track region is only consumed by TC diagnostics, so it would be "
            f"silently ignored.  Add a tc_tracks block or remove regions.track."
        )


def resolve_variable_set(
    case: Dict[str, Any], palettes: Optional[Dict] = None
) -> List[str]:
    """Resolve the diagnostic variable list for a case.

    Priority: ``diagnostics.variables`` (inline list, per-case override) >
    the palette's ``standard`` variable set (shared across every case).
    """
    diag = case.get("diagnostics", {}) or {}
    if "variables" in diag:
        return list(diag["variables"])

    palettes = palettes or load_palettes()
    sets = palettes.get("variable_sets", {})
    if "standard" not in sets:
        raise ValueError(
            "Palette file is missing 'variable_sets/standard'. "
            f"Available sets: {sorted(sets)}"
        )
    return list(sets["standard"])


# ----------------------------------------------------------------------
# Spatial regions (top-level ``regions:`` block)
# ----------------------------------------------------------------------
#
# Each case declares purpose-named lat/lon boxes under a top-level
# ``regions:`` key, e.g.::
#
#     regions:
#       synoptic: { lon: [-170, -100], lat: [25, 70] }   # cause / precursor
#       impact:   { lon: [-125, -116], lat: [42, 52] }   # effect (averaging)
#       track:    { lon: [-85, -60],   lat: [25, 45] }   # TC detection (TC only)
#
# One box, one purpose -- the figure that analyses a region draws exactly it:
#   * synoptic -- synoptic-PCA precursor PCA domain + its map view.
#   * impact   -- the single effect region.  free_end_states reduces/ranks/maps
#                 over it AND synoptic-PCA averages its scalar impact (e.g. t2m)
#                 over it and draws it as the dashed annotation box.  Both share
#                 this box and the same mask, so there is exactly one impact
#                 averaging, never two divergent ones.
#   * track    -- TC-track detection domain + track-map view + the synoptic-PCA
#                 track-impact annotation.
#
# The member-plot viewport is presentational, not an analysis region: it reads
# ``plots.view`` (a region name or a custom extent) and defaults to ``impact``.


def region_box(
    case: Dict[str, Any], name: str
) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(lon_min, lon_max, lat_min, lat_max)`` for ``regions.<name>``.

    Returns ``None`` when the ``regions`` block or the named region is absent,
    so callers can apply their own default or raise a domain-specific error.
    Raises ``SystemExit`` when the region is present but malformed.
    """
    regions = case.get("regions") or {}
    box = regions.get(name)
    if not box:
        return None
    lon = box.get("lon")
    lat = box.get("lat")
    if not lon or not lat or len(lon) != 2 or len(lat) != 2:
        raise SystemExit(
            f"case '{case.get('name', '<unknown>')}': regions.{name} must set "
            f"lon: [min, max] and lat: [min, max]; got {box!r}."
        )
    return (float(lon[0]), float(lon[1]), float(lat[0]), float(lat[1]))


def require_region(
    case: Dict[str, Any], name: str
) -> Tuple[float, float, float, float]:
    """Like :func:`region_box` but raises a clear error when the region is absent."""
    box = region_box(case, name)
    if box is None:
        raise SystemExit(
            f"case '{case.get('name', '<unknown>')}': regions.{name} is required "
            f"for this diagnostic but is not defined in the case YAML."
        )
    return box


def resolve_view(
    case: Dict[str, Any], default: str = "impact"
) -> Optional[Tuple[float, float, float, float]]:
    """Resolve the member-plot viewport to ``(lon_min, lon_max, lat_min, lat_max)``.

    ``plots.view`` may be a region name (resolved via :func:`region_box`) or a
    custom ``[lon_min, lon_max, lat_min, lat_max]`` extent.  When absent it
    falls back to the ``default`` region (``impact``).  Returns ``None`` only
    when the resolved region is itself absent, so plot code can skip
    ``set_extent`` and fall back to a global view.
    """
    plots = case.get("plots") or {}
    view = plots.get("view")
    if view is None:
        return region_box(case, default)
    if isinstance(view, str):
        box = region_box(case, view)
        if box is None:
            raise SystemExit(
                f"case '{case.get('name', '<unknown>')}': plots.view = '{view}' "
                f"but regions.{view} is not defined."
            )
        return box
    if isinstance(view, (list, tuple)) and len(view) == 4:
        return (float(view[0]), float(view[1]), float(view[2]), float(view[3]))
    raise SystemExit(
        f"case '{case.get('name', '<unknown>')}': plots.view must be a region "
        f"name or a [lon_min, lon_max, lat_min, lat_max] list; got {view!r}."
    )


# ----------------------------------------------------------------------
# qsub helpers
# ----------------------------------------------------------------------

_JOBID_RE = re.compile(r"Your job\s+(\d+)")


def parse_job_id(qsub_stdout: str) -> Optional[str]:
    """Extract the numeric job id from ``qsub`` stdout.

    SGE format: ``Your job 123456 ("name") has been submitted``.
    """
    m = _JOBID_RE.search(qsub_stdout)
    return m.group(1) if m else None


def build_qsub_cmd(
    *,
    script: str,
    job_name: str,
    h_rt: str,
    script_args: Sequence[str],
    hold_jids: Iterable[str] = (),
    env_vars: Optional[Dict[str, str]] = None,
    extra_resource_flags: Sequence[str] = (),
) -> List[str]:
    """Build a ``qsub`` command list."""
    cmd: List[str] = ["qsub", "-N", job_name, "-l", f"h_rt={h_rt}"]

    hold_jids = [j for j in hold_jids if j]
    if hold_jids:
        cmd += ["-hold_jid", ",".join(hold_jids)]

    if env_vars:
        cmd += ["-v", ",".join(f"{k}={v}" for k, v in env_vars.items())]

    cmd += list(extra_resource_flags)
    cmd.append(script)
    cmd += list(script_args)
    return cmd


def submit_qsub(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    cwd: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Optional[str]:
    """Submit a qsub command.  Returns the SGE job id, or None in dry-run / failure.

    * ``dry_run=True``  -> print and return None (caller should supply a
      fake placeholder jid to keep -hold_jid wiring consistent).
    * ``log_dir``       -> created before submission if provided.
    """
    cwd = cwd or str(SCRIPT_DIR)

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return None

    if log_dir:
        Path(log_dir).mkdir(exist_ok=True, parents=True)

    print(f"[SUBMIT] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"  -> FAILED: {result.stderr.strip()}", file=sys.stderr)
        return None

    stdout = result.stdout.strip()
    print(f"  -> {stdout}")
    return parse_job_id(stdout)


# ----------------------------------------------------------------------
# Convenience: iterate conditioning modes of a case
# ----------------------------------------------------------------------


def iter_modes(case: Dict[str, Any]) -> Iterable[Tuple[str, List[int]]]:
    """Yield (mode, frames) pairs for every conditioning mode configured."""
    for mode in case["ensemble"]["modes"]:
        yield mode, mode_frames(mode)


def case_ensemble_size(case: Dict[str, Any], override: Optional[int] = None) -> int:
    """Resolve the ensemble size for a case, with optional CLI override."""
    return override if override is not None else int(case["ensemble"]["size"])
