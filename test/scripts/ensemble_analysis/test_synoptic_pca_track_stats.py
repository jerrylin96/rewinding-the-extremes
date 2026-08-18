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

"""Unit tests for the paper-quotable summary statistics in
``synoptic_pca_track_stats``.

Covers the target distribution summary (min/max bracketing the percentile
columns, NaN handling) and the ERA5-relative free-end fix fractions
(east/north splits, longitude wraparound, empty-input degradation).
Pure-NumPy/pandas -- no zarr, parquet, or plotting I/O is exercised.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("yaml")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from synoptic_pca_track_stats import (  # noqa: E402
    SUMMARY_PCTS,
    era5_relative_fractions,
    target_summary,
)


class TestTargetSummary:
    def test_min_max_bracket_percentiles(self) -> None:
        vals = np.array([990.0, 950.0, 970.0, 981.5])
        df = target_summary({"fe_msl_hpa": vals})
        row = df.iloc[0]
        assert row["n"] == 4
        assert row["min"] == pytest.approx(950.0)
        assert row["max"] == pytest.approx(990.0)
        # Column order keeps min/max bracketing the percentile columns so
        # the report table reads as a monotone distribution summary.
        cols = df.columns.tolist()
        pct_cols = [f"p{p:g}" for p in SUMMARY_PCTS]
        assert cols == ["target", "n", "mean", "min", *pct_cols, "max"]

    def test_nan_members_dropped_per_target(self) -> None:
        df = target_summary({"fe_lat_deg": np.array([np.nan, 30.0, 32.0])})
        row = df.iloc[0]
        assert row["n"] == 2
        assert row["min"] == pytest.approx(30.0)
        assert row["max"] == pytest.approx(32.0)

    def test_all_nan_target_keeps_n_zero_row(self) -> None:
        df = target_summary({"fe_msl_hpa": np.array([np.nan, np.nan])})
        assert df.iloc[0]["n"] == 0
        assert "min" not in df.columns


class TestEra5RelativeFractions:
    ERA5 = {"lat": 39.5, "lon": -74.5, "msl_hpa": 950.7}

    def test_east_north_split(self) -> None:
        targets = {
            "fe_lat_deg": np.array([40.0, 39.0, 41.0, np.nan]),
            "fe_lon_deg": np.array([-70.0, -76.0, -74.5, -60.0]),
        }
        rel = era5_relative_fractions(targets, self.ERA5)
        # The NaN-latitude member is excluded from n entirely; a fix at
        # exactly the ERA5 longitude does not count as east.
        assert rel == {"n": 3, "east": 1, "north": 2}

    def test_longitude_wraparound(self) -> None:
        era5 = {"lat": 0.0, "lon": 179.0, "msl_hpa": 1000.0}
        targets = {
            "fe_lat_deg": np.array([1.0, -1.0]),
            "fe_lon_deg": np.array([-179.0, 170.0]),
        }
        rel = era5_relative_fractions(targets, era5)
        # -179 E is 2 deg east of 179 E across the dateline; 170 E is west.
        assert rel == {"n": 2, "east": 1, "north": 1}

    def test_tracker_0360_convention(self) -> None:
        # TempestExtremes fixes on [0, 360) against an ERA5 ref on
        # [-180, 180]: 290 E == -70 E is east of -74.5 E.
        targets = {
            "fe_lat_deg": np.array([39.0]),
            "fe_lon_deg": np.array([290.0]),
        }
        rel = era5_relative_fractions(targets, self.ERA5)
        assert rel == {"n": 1, "east": 1, "north": 0}

    def test_empty_and_missing_targets(self) -> None:
        assert era5_relative_fractions({}, self.ERA5) == {}
        all_nan = {
            "fe_lat_deg": np.array([np.nan]),
            "fe_lon_deg": np.array([np.nan]),
        }
        assert era5_relative_fractions(all_nan, self.ERA5) == {}
