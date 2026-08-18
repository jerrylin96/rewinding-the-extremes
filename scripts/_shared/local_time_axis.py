"""Lead-time -> local-datetime axis helpers for the compute_* plot scripts.

The ensemble pipeline writes ``lead_time`` (a timedelta from the ensemble
initialization time) as the only temporal coordinate on the zarr stores.
For figures whose audience is regional weather impacts (Sandy in the NYC
metro, the PNW heat dome over Portland, etc.), it is more useful to label
the lead-axis ticks with the actual wall-clock time at the event location
than with raw "Lead time (h)" since IC.

Each compute_*.py script that draws a lead-time axis takes a ``--start-time``
(ISO 8601 UTC string for the ensemble IC) and a ``--timezone`` (IANA name
like ``America/New_York``) on the CLI; the dispatcher forwards both from
the case YAML.  When ``--timezone`` is omitted the helpers fall back to
UTC, which still produces a wall-clock axis but skips the regional cue.

This module is matplotlib-aware on purpose — it sets ticks, labels, and
axis text — so it stays out of ``_dispatch_lib.py`` (which the dispatcher
imports without pulling in matplotlib).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np


def _parse_start_time(start_time_iso: str) -> datetime:
    """Parse an ISO 8601 UTC string into a tz-aware UTC ``datetime``.

    Accepts both ``...Z`` (Zulu) and ``...+00:00`` forms.  Anything else
    is treated as naive UTC.
    """
    s = start_time_iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _zone(tz_name: str | None) -> ZoneInfo:
    """Resolve an IANA timezone name; falls back to UTC when missing/empty."""
    if not tz_name:
        return ZoneInfo("UTC")
    return ZoneInfo(tz_name)


def local_datetimes(
    start_time_iso: str,
    lead_h: Sequence[float] | np.ndarray,
    tz_name: str | None,
) -> list[datetime]:
    """Return tz-aware local datetimes for each lead-hour offset from IC."""
    ic = _parse_start_time(start_time_iso)
    tz = _zone(tz_name)
    return [(ic + timedelta(hours=float(h))).astimezone(tz) for h in np.asarray(lead_h)]


def _axis_label(local_times: Sequence[datetime], tz_name: str | None) -> str:
    """Compose a 'Local time (Zone; ABBR)' label, abbrev pulled from the first tick."""
    # IANA names use '_' for word separators (e.g. ``America/New_York``);
    # the underscore reads as a typo on a finished plot.  The actual
    # ZoneInfo lookup still uses ``tz_name`` unchanged elsewhere.
    zone = (tz_name or "UTC").replace("_", " ")
    abbrev = local_times[0].strftime("%Z") if local_times else ""
    if abbrev and abbrev != zone:
        return f"Local time ({zone}; {abbrev})"
    return f"Local time ({zone})"


def _pick_tick_indices(n: int) -> list[int]:
    """Subsample tick indices so the ``MMM DD\\nHH:MM`` labels do not overlap.

    The compute scripts emit 12 leads (66 h on a 6 h grid) by default, but
    nothing guarantees that — keep the rule simple and bounded.
    """
    if n <= 0:
        return []
    if n <= 8:
        step = 1
    elif n <= 16:
        step = 2
    else:
        step = 3
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        # The endpoint always wants to be shown, but appending it
        # blindly crowds the previous tick when the natural step does
        # not divide ``n - 1`` (e.g. ``n=12, step=2`` lands at index 10
        # with the endpoint one step short).  Replace the last regular
        # tick rather than appending if the gap would be sub-step.
        if (n - 1) - idx[-1] < step:
            idx[-1] = n - 1
        else:
            idx.append(n - 1)
    return idx


def format_local_time_axis(
    ax,
    lead_h: Sequence[float] | np.ndarray,
    start_time_iso: str,
    tz_name: str | None,
) -> None:
    """Replace a lead-hour x-axis with local wall-clock ticks (``MMM DD\\nHH:MM``)."""
    lead_arr = np.asarray(lead_h, dtype=float)
    local_times = local_datetimes(start_time_iso, lead_arr, tz_name)
    tick_idx = _pick_tick_indices(len(lead_arr))
    tick_pos = [float(lead_arr[i]) for i in tick_idx]
    tick_labels = [local_times[i].strftime("%b %d\n%H:%M") for i in tick_idx]
    label = _axis_label(local_times, tz_name)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(label)


def format_local_time_title(
    start_time_iso: str,
    lead_h_value: float,
    tz_name: str | None,
) -> str:
    """Single-line local-time stamp for figure/panel titles.

    Used by scripts that previously wrote ``"t + N h"`` or
    ``"t+N h"`` inside titles rather than on an axis.
    """
    dt = local_datetimes(start_time_iso, [lead_h_value], tz_name)[0]
    abbrev = dt.strftime("%Z")
    suffix = f" {abbrev}" if abbrev else ""
    return dt.strftime("%b %d %H:%M") + suffix
