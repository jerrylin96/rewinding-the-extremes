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

"""Tests for the TC track-regression figure in ``aggregate_synoptic_pca``: the
panels' targets and ERA5 values, the per-cell member counts that color the
dots, and the ERA5 star's filled/hollow state.  A stand-in for ``TrackQC``
supplies the targets, so no parquet is read.

Skipped where the plotting stack (matplotlib + cartopy) is absent, as for
``test_synoptic_pca_regression``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))

from aggregate_synoptic_pca import (  # noqa: E402
    _cell_counts,
    _era5_range,
    _era5_star_style,
    _star_under_stats_box,
    _track_regression_panels,
    plot_track_regression,
)

N_MEMBERS, MAX_D = 400, 10
ERA5_FREE_END = {"lat": 30.2, "lon": -74.8, "msl_hpa": 958.0}


def _diag(era5_pc_score: np.ndarray) -> dict:
    """A TC-mode npz dict whose targets depend on the leading PCs."""
    rng = np.random.default_rng(0)
    pcs = rng.standard_normal((N_MEMBERS, MAX_D))
    analyzable = rng.random(N_MEMBERS) < 0.8
    values = {
        "fe_lat_deg": 31 + 0.5 * pcs[:, 0] + rng.normal(0, 1, N_MEMBERS),
        "fe_lon_deg": -73 + 0.5 * pcs[:, 1] + rng.normal(0, 1, N_MEMBERS),
        "fe_dist_era5_km": 400 + 50 * pcs[:, 2] + rng.normal(0, 90, N_MEMBERS),
        "fe_msl_hpa": 980 + 2 * pcs[:, 3] + rng.normal(0, 4, N_MEMBERS),
    }
    targets = {k: np.where(analyzable, v, np.nan) for k, v in values.items()}
    qc = types.SimpleNamespace(  # the parts of TrackQC the figure reads
        free_end_targets=lambda: dict(targets),
        era5_free_end=dict(ERA5_FREE_END),
        counts=lambda: {"excluded": int((~analyzable).sum())},
    )
    return {
        "impact_kind": np.array("track"),
        "pc_scores": pcs,
        "era5_pc_score": era5_pc_score,
        "case_name": np.array("Test storm"),
        "track_qc": qc,
    }


def test_panels_are_lat_lon_pressure_with_era5_looked_up_by_name() -> None:
    """Bug 1: the central-pressure panel's ERA5 value is ERA5's central
    pressure (it was NaN while distance sat between longitude and pressure)."""
    panels = _track_regression_panels(_diag(np.zeros(MAX_D)), n_pc=8)
    assert [p.ylabel for p in panels] == [
        "free-end latitude",
        "free-end longitude",
        "free-end central MSL",
    ]
    assert [p.era5_obs for p in panels] == [30.2, -74.8, 958.0]
    assert all(np.isfinite(p.era5_pred) for p in panels)


def test_no_era5_prediction_without_scores_for_every_pc() -> None:
    """An npz with ERA5 scores for only the 6 shown EOFs (Ian, bug 2)."""
    diag = _diag(np.zeros(6))
    assert _era5_range(diag, 8) is None
    assert all(np.isnan(p.era5_pred) for p in _track_regression_panels(diag, 8))


def test_cell_counts() -> None:
    edges = np.linspace(0.0, 1.0, 3)  # 2 x 2 cells
    x = np.array([0.1, 0.2, 0.9, 1.0])  # the last point sits on the top edge
    y = np.array([0.1, 0.3, 0.9, 1.0])
    assert _cell_counts(x, y, edges).tolist() == [2, 2, 2, 2]
    assert _cell_counts(x[:3], y[:3], edges).tolist() == [2, 2, 1]


def test_star_filled_inside_hollow_outside() -> None:
    diag = _diag(np.zeros(MAX_D))
    inside = _era5_range(diag, 8)
    diag["era5_pc_score"] = np.full(MAX_D, 10.0)
    outside = _era5_range(diag, 8)
    assert inside.inside and not outside.inside
    assert _era5_star_style(inside)["facecolor"] == "black"
    assert _era5_star_style(outside)["facecolor"] == "none"


@pytest.mark.parametrize("era5_scale, state", [(0.0, "inside"), (10.0, "outside")])
def test_figure_renders(
    tmp_path: Path, capsys: pytest.CaptureFixture, era5_scale: float, state: str
) -> None:
    out = tmp_path / "track_regression.png"
    plot_track_regression(_diag(np.full(MAX_D, era5_scale)), "end", out)
    assert out.stat().st_size > 0
    assert f"ERA5 {state} (" in capsys.readouterr().out


def test_stats_box_moves_only_for_a_star_under_it() -> None:
    span = (950.0, 1000.0)
    assert _star_under_stats_box((990.0, 951.0), span)  # bottom-right
    assert not _star_under_stats_box((960.0, 951.0), span)  # bottom-left
    assert not _star_under_stats_box((990.0, 990.0), span)  # top-right
