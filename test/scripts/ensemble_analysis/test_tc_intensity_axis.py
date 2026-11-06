# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One pinned storm-intensity axis for every hurricane panel in both papers.

``axis_limits.pin_y_to_drawn`` logs the y range a panel actually draws (the
numbers the pinned limit is derived from) and applies the configured limit.
The Extremes on Rewind panels -- every conditioning row of the track figure
and the landfall split -- must read the one ``TC_INTENSITY`` key and label;
Run it Back's track figure is checked in
``scripts/backward_emulator/tests/test_case_track_qc.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import axis_limits  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PINNED = (930.0, 1020.0)
START, TZ = "2022-09-28T00:00:00Z", "America/New_York"


@pytest.fixture()
def pinned(monkeypatch):
    """Configure the intensity key as if axis_limits.yaml pinned it."""
    analysis, variable, panel = axis_limits.TC_INTENSITY
    cfg = {analysis: {variable: {panel: {"ylim": list(PINNED)}}}}
    monkeypatch.setattr(axis_limits, "_CACHE", cfg)


@pytest.fixture()
def unpinned(monkeypatch):
    monkeypatch.setattr(axis_limits, "_CACHE", {})


@pytest.fixture()
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_drawn_range_covers_lines_fills_and_collections_only(ax, unpinned, capsys):
    """Lines, fills and vlines count; an axvline (axes-fraction y) and text don't."""
    ax.plot([0, 1], [990.0, 1000.0])
    ax.fill_between([0, 1], [985.0, 985.0], [1002.0, 1002.0])
    ax.vlines(0.5, 980.0, 1003.0)
    ax.axvline(0.2)  # y in axes fraction 0..1: not data
    ax.annotate("ERA5 p50", xy=(0.5, 1000.0), xytext=(0.5, 2000.0))
    autoscaled = ax.get_ylim()
    axis_limits.pin_y_to_drawn(ax, *axis_limits.TC_INTENSITY, log_tag="t")
    out = capsys.readouterr().out
    assert "tc_tracks/tc_msl/trajectory: drawn y range 980 to 1003" in out
    assert ax.get_ylim() == autoscaled  # nothing configured -> untouched


def test_pinned_limit_applies_and_clipping_is_loud(ax, pinned, capsys):
    ax.plot([0, 1], [925.0, 1000.0])  # below the pinned floor
    axis_limits.pin_y_to_drawn(ax, *axis_limits.TC_INTENSITY, log_tag="t")
    assert ax.get_ylim() == PINNED
    assert (
        "data min 925 below configured ymin 930 (clipping)" in capsys.readouterr().out
    )


def test_empty_panel_logs_nothing_drawn(ax, pinned, capsys):
    axis_limits.pin_y_to_drawn(ax, *axis_limits.TC_INTENSITY, log_tag="t")
    out = capsys.readouterr().out
    assert "nothing drawn" in out and "clipping" not in out
    assert ax.get_ylim() == PINNED


def _tracks(n=5, t=12, deep=950.0, weak=1005.0):
    """``[n, 1, t, 4]`` member tracks and ``[1, 1, t, 4]`` ERA5, MSL in Pa."""
    ens = np.full((n, 1, t, 4), np.nan)
    for i in range(n):
        ens[i, 0, :, 2] = np.linspace(deep + 5 * i, weak - 5 * i, t) * 100.0
    era5 = ens[:1].copy()
    era5[0, 0, :, 2] = np.linspace(960.0, 995.0, t) * 100.0
    return ens, era5


@pytest.mark.parametrize("mode", ["start", "end", "both"])
def test_track_figure_rows_share_the_pinned_axis(ax, pinned, capsys, mode):
    pytest.importorskip("cartopy")
    import aggregate_tc_tracks as atc

    ens, era5 = _tracks()
    ok = atc._draw_tc_msl_trajectory(
        ax,
        ens,
        era5,
        lead_hours=np.arange(12) * 6.0,
        start_time_iso=START,
        timezone_name=TZ,
        mode=mode,
        era5_pct=50.0,
        era5_pct_n=5,
    )
    assert ok
    assert ax.get_ylim() == PINNED
    assert ax.get_ylabel() == axis_limits.TC_INTENSITY_LABEL
    out = capsys.readouterr().out
    assert f"[aggregate_tc_tracks {mode}] tc_tracks/tc_msl/trajectory" in out
    assert "drawn y range 950 to 1005" in out


def test_landfall_split_uses_the_pinned_axis(ax, pinned, capsys):
    pytest.importorskip("cartopy")
    import tc_landfall_split as tls

    ens, era5 = _tracks()
    tls._draw_group_trajectory(
        ax,
        ens,
        np.array([0, 1, 2]),
        np.array([3, 4]),
        era5,
        lead_hours=np.arange(12) * 6.0,
        start_time_iso=START,
        timezone_name=TZ,
        fl_label="FL",
        sc_label="SC",
        fl_pct=60.0,
        bypass_pct=40.0,
    )
    assert ax.get_ylim() == PINNED
    assert ax.get_ylabel() == axis_limits.TC_INTENSITY_LABEL
    out = capsys.readouterr().out
    assert (
        "[tc_landfall_split] tc_tracks/tc_msl/trajectory: drawn y range 950 to 1005"
        in out
    )
