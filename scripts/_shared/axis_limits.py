"""Shared axis-limit lookup for cross-comparable diagnostic plots.

The compute_* scripts in ``scripts/ensemble_analysis/submission_scripts/``
each plot a single (mode, case) run; comparison across runs happens by
laying figures out side-by-side. Uniform axis ranges make those visual
comparisons honest. This module reads ``axis_limits.yaml`` and exposes a
small helper that the scripts call when setting up each panel.

Design points:

* Lookup is keyed by ``(analysis, variable, panel)``. Missing keys return
  ``None`` so matplotlib's autoscale stays in charge -- the failure mode
  is visible, not silent.
* When a configured limit is in force, callers pass the actual data so
  the helper can warn if values fall outside the configured range. That
  catches "Hurricane Florence is hotter than the table thinks" before the
  reader is.
* Edit the YAML to widen a range; no code change needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

_YAML_PATH = Path(__file__).resolve().parent / "axis_limits.yaml"
_CACHE: dict | None = None

# The storm-intensity axis (central MSLP vs time) shared by both papers'
# hurricane figures: aggregate_tc_tracks.py and tc_landfall_split.py
# (Extremes on Rewind) and backward_emulator/plot_tc_tracks_compare.py
# (Run it Back).  The limits key and the label are defined once here, so the
# three scripts cannot pin different ranges or name the axis differently.
TC_INTENSITY = ("tc_tracks", "tc_msl", "trajectory")
TC_INTENSITY_LABEL = "Storm central MSLP (hPa)"


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        if not _YAML_PATH.exists():
            _CACHE = {}
        else:
            with open(_YAML_PATH) as f:
                _CACHE = yaml.safe_load(f) or {}
    return _CACHE


def get_limits(analysis: str, variable: str, panel: str) -> dict | None:
    """Return the ``{"ylim": [...], "xlim": [...]}`` entry, or ``None``.

    Any of ``analysis`` / ``variable`` / ``panel`` may be missing from the
    YAML, in which case ``None`` is returned and the caller should let
    matplotlib autoscale.
    """
    cfg = _load()
    entry = cfg.get(analysis, {}).get(variable, {}).get(panel)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ValueError(
            f"axis_limits.yaml: {analysis}.{variable}.{panel} must be a "
            f"mapping with 'ylim' and/or 'xlim' keys, got {entry!r}"
        )
    return entry


def apply_limits(
    ax,
    analysis: str,
    variable: str,
    panel: str,
    *,
    y_data: Iterable[float] | None = None,
    x_data: Iterable[float] | None = None,
    log_tag: str = "axis_limits",
) -> None:
    """Apply configured limits to ``ax``; warn if data exceeds them.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to constrain.
    analysis, variable, panel : str
        YAML lookup keys.
    y_data, x_data : iterable, optional
        Actual data being plotted on each axis. When provided and a limit
        is configured, the helper checks min/max against the limit and
        prints a warning if data falls outside the configured range.
    log_tag : str
        Prefix used for warning lines, matching the per-script ``[tag]``
        convention.
    """
    entry = get_limits(analysis, variable, panel)
    if entry is None:
        return

    ylim = entry.get("ylim")
    if ylim is not None:
        ax.set_ylim(*ylim)
        if y_data is not None:
            _warn_if_outside(
                y_data, ylim, log_tag, f"{analysis}/{variable}/{panel}", axis="y"
            )

    xlim = entry.get("xlim")
    if xlim is not None:
        ax.set_xlim(*xlim)
        if x_data is not None:
            _warn_if_outside(
                x_data, xlim, log_tag, f"{analysis}/{variable}/{panel}", axis="x"
            )


def pin_y_to_drawn(
    ax,
    analysis: str,
    variable: str,
    panel: str,
    *,
    log_tag: str = "axis_limits",
) -> None:
    """Apply the configured y-limits to a finished panel and log its y range.

    The range is matplotlib's own ``ax.dataLim``, the extent of every data
    artist on the panel (lines, fills, line collections, scatters; never
    text), so the clipping warning and the logged range cannot drift from
    the drawing code.  Call it after the panel's last data artist.

    The range is logged even when no limit is configured: that log line is
    how the pinned limits in ``axis_limits.yaml`` are derived and re-derived.
    """
    lo, hi = (float(v) for v in ax.dataLim.intervaly)
    key = f"{analysis}/{variable}/{panel}"
    if np.isfinite(lo) and np.isfinite(hi):
        print(f"[{log_tag}] {key}: drawn y range {lo:.6g} to {hi:.6g}")
    else:
        print(f"[{log_tag}] {key}: nothing drawn")
    apply_limits(ax, analysis, variable, panel, y_data=(lo, hi), log_tag=log_tag)


def _warn_if_outside(
    data: Iterable[float],
    lim: list[float],
    log_tag: str,
    key: str,
    axis: str,
) -> None:
    arr = np.asarray(list(data), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    dmin, dmax = float(finite.min()), float(finite.max())
    lo, hi = float(lim[0]), float(lim[1])
    if dmin < lo:
        print(
            f"[{log_tag}] {key}: data min {dmin:.4g} below configured "
            f"{axis}min {lo:.4g} (clipping)"
        )
    if dmax > hi:
        print(
            f"[{log_tag}] {key}: data max {dmax:.4g} above configured "
            f"{axis}max {hi:.4g} (clipping)"
        )
