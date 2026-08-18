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

"""Unit tests for the pinned-end anchor diagnostics helpers.

Covers the parquet-only classification path of
``tc_track_anchor_diagnostics.py`` (category derivation, per-member table,
representative-member selection), its consistency with the canonical
``tc_track_targets._anchor_check`` counts the paper quotes, and the
dispatcher's anchor-frame resolution.  No matplotlib / cartopy / zarr /
TempestExtremes I/O is exercised.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from tc_track_anchor_diagnostics import (  # noqa: E402
    classify_members,
    load_all_paths,
    member_table,
    pick_spread,
)
from tc_track_targets import _anchor_check, anchor_min_distances  # noqa: E402
from tc_tracks_io import save_tracks_parquet  # noqa: E402

ANCHOR_FRAME = 11
N_STEPS = 12
# ERA5 anchor fix (lon on the tracker's [0, 360) convention -> -70 E).
ANCHOR_LAT, ANCHOR_LON_0360 = 41.0, 290.0


def _era5_array() -> np.ndarray:
    """ERA5 track [1, 1, 12, 4]: northward along 290E, ending at the anchor."""
    arr = np.full((1, 1, N_STEPS, 4), np.nan)
    for t in range(N_STEPS):
        arr[0, 0, t] = [30.0 + t, ANCHOR_LON_0360, 100000.0, 20.0]
    return arr


def _ensemble_array() -> np.ndarray:
    """Four members covering the three categories.

    m0 anchored  : path 0 ends on the anchor fix.
    m1 alt_path  : path 0 tracks a system thousands of km away (valid fixes
                   inside the anchor window), path 1 ends near the anchor.
    m2 lost      : path 0 exists but stops at step 6 (no fix inside the
                   window), path 1 empty.
    m3 lost      : no valid fixes on any path.
    """
    arr = np.full((4, 2, N_STEPS, 4), np.nan)
    for t in range(N_STEPS):
        arr[0, 0, t] = [30.0 + t, ANCHOR_LON_0360, 99000.0, 25.0]
        arr[1, 0, t] = [10.0, 40.0, 100500.0, 15.0]  # far-away system
        arr[1, 1, t] = [30.5 + t, ANCHOR_LON_0360 - 0.5, 99500.0, 22.0]
    for t in range(7):
        arr[2, 0, t] = [25.0 + t, ANCHOR_LON_0360 + 3.0, 100200.0, 18.0]
    return arr


@pytest.fixture
def parquet_path(tmp_path):
    path = str(tmp_path / "tc_tracks_end.parquet")
    save_tracks_parquet(path, _ensemble_array(), era5_arr=_era5_array())
    return path


def test_classify_members_categories(parquet_path):
    ens, era5_track = load_all_paths(parquet_path)
    category, dmin, anchor_fix, fell_back = classify_members(
        ens, era5_track, ANCHOR_FRAME, 500.0
    )
    assert list(category) == ["anchored", "alt_path", "lost", "lost"]
    assert not fell_back
    assert anchor_fix[0] == pytest.approx(ANCHOR_LAT)
    assert anchor_fix[1] == pytest.approx(ANCHOR_LON_0360 - 360.0)
    # m0 path 0 sits exactly on the anchor; m1's secondary path is close.
    assert dmin[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert dmin[1, 1] < 500.0 < dmin[1, 0]
    # m2 has fixes, but none inside the +-2-frame anchor window.
    assert np.isinf(dmin[2]).all()


def test_classify_matches_anchor_check_counts(parquet_path):
    ens, era5_track = load_all_paths(parquet_path)
    category, _, anchor_fix, _ = classify_members(ens, era5_track, ANCHOR_FRAME, 500.0)
    _, counts = _anchor_check(
        ens, float(anchor_fix[0]), float(anchor_fix[1]), ANCHOR_FRAME, 500.0
    )
    assert counts["anchored"] == int((category == "anchored").sum())
    assert counts["alt_path"] == int((category == "alt_path").sum())
    assert counts["lost"] == int((category == "lost").sum())


def test_member_table_contents(parquet_path):
    ens, era5_track = load_all_paths(parquet_path)
    category, dmin, _, _ = classify_members(ens, era5_track, ANCHOR_FRAME, 500.0)
    df = member_table(ens, category, dmin)

    m0, m1, m2, m3 = (df.loc[df["member"] == m].iloc[0] for m in range(4))
    assert m0["best_path_id"] == 0 and m0["n_paths"] == 1
    assert m1["best_path_id"] == 1 and m1["n_paths"] == 2
    assert m1["best_path_min_anchor_km"] < 500.0
    # m2: a real path that ends early -- distances are inf -> NaN in the CSV.
    assert m2["n_paths"] == 1 and m2["best_path_id"] == -1
    assert m2["path0_last_step"] == 6
    assert np.isnan(m2["path0_min_anchor_km"])
    # m3: no stitched path at all.
    assert m3["n_paths"] == 0 and m3["best_path_id"] == -1
    assert m3["path0_n_fixes"] == 0 and m3["path0_first_step"] == -1


def test_anchor_min_distances_window_tolerance():
    # A single fix at frame 9 lies inside the +-2 window of frame 11; the
    # same fix at frame 8 does not.
    ens = np.full((2, 1, N_STEPS, 4), np.nan)
    ens[0, 0, 9, :2] = [ANCHOR_LAT, -70.0]
    ens[1, 0, 8, :2] = [ANCHOR_LAT, -70.0]
    dmin = anchor_min_distances(ens, ANCHOR_LAT, -70.0, ANCHOR_FRAME)
    assert dmin[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.isinf(dmin[1, 0])


def test_anchor_min_distances_undefined_beyond_lead_axis():
    ens = np.zeros((1, 1, N_STEPS, 4))
    assert anchor_min_distances(ens, 0.0, 0.0, N_STEPS + 5) is None


def test_pick_spread_basic():
    members = np.array([10, 20, 30])
    key = np.array([500.0, 100.0, np.inf])  # sorted order: 20, 10, 30
    assert pick_spread(members, key, 0) == []
    # Both extremes of the key range are always represented (inf = the most
    # extreme failure, e.g. no fix in the anchor window at all).
    assert pick_spread(members, key, 2) == [20, 30]
    # n >= len returns everything, ordered by key.
    assert pick_spread(members, key, 10) == [20, 10, 30]


def test_pick_spread_includes_extremes_and_is_deterministic():
    members = np.arange(100)
    key = np.linspace(600.0, 5000.0, 100)
    picked = pick_spread(members, key, 8)
    assert len(picked) == 8
    assert picked[0] == 0 and picked[-1] == 99
    assert picked == pick_spread(members, key, 8)


def test_dispatcher_anchor_frame_resolution():
    from dispatch_tc_track_anchor_diagnostics import anchor_frame_for_mode

    case_with_landfall = {"tc_tracks": {"landfall_frame": 11}}
    case_without = {"tc_tracks": {}}
    assert anchor_frame_for_mode(case_with_landfall, "start") == 0
    assert anchor_frame_for_mode(case_with_landfall, "end") == 11
    # No declared landfall frame -> the conditioned end frame itself.
    assert anchor_frame_for_mode(case_without, "end") == 11
