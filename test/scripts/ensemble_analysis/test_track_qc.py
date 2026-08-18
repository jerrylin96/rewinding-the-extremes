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

"""Unit tests for the round-5 unified track QC (``track_qc``).

Exercises anchor-first storm-of-interest resolution and the exclusion
taxonomy against synthetic parquets, including the three cases the round-5
spec calls out explicitly: an anchor-reaching path that is not the stored
path 0, a 2-fix path touching the anchor window (``insufficient_fixes``, not
analyzable), and a coherent path that never approaches the anchor
(``unanchored``).  Also covers the exact-frame subset / free-end MSL
percentile and the storm-of-interest array reduction the figure scripts
draw.  Pure NumPy / pandas / pyarrow -- no zarr, matplotlib, or cartopy.
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

from tc_tracks_io import save_tracks_parquet, tracks_in_domain  # noqa: E402
from track_qc import (  # noqa: E402
    compute_track_qc,
    compute_track_qc_arrays,
    resolve_storm_of_interest,
)

T = 12
ANCHOR_FRAME = 11  # end-mode: the pinned landfall frame
FREE_END_FRAME = 0  # end-mode: the free (IC) end
TRACK_BOX = (-85.0, -60.0, 25.0, 45.0)  # lon_min, lon_max, lat_min, lat_max


def _lon360(x: float) -> float:
    return x % 360.0


def _era5() -> np.ndarray:
    """ERA5 track [1,1,12,4]: NE-bound, free end (26N,-78E), anchor (40N,-74E)."""
    arr = np.full((1, 1, T, 4), np.nan)
    for t in range(T):
        lat = 26.0 + t * (40.0 - 26.0) / 11.0
        lon = _lon360(-78.0 + t * (-74.0 - -78.0) / 11.0)
        # free-end (frame 0) central MSL = 980 hPa
        arr[0, 0, t] = [lat, lon, 98000.0, 30.0]
    return arr


def _storm(msl_pa: float = 99000.0, skip_frame0: bool = False, frames=range(T)):
    p = np.full((T, 4), np.nan)
    for t in frames:
        if skip_frame0 and t == 0:
            continue
        lat = 26.0 + t * (40.0 - 26.0) / 11.0
        lon = _lon360(-78.0 + t * (-74.0 - -78.0) / 11.0)
        p[t] = [lat, lon, msl_pa, 30.0]
    return p


def _ensemble() -> np.ndarray:
    """Six members, one per outcome the taxonomy must separate."""
    arr = np.full((6, 3, T, 4), np.nan)
    # m0 analyzable, storm on path 0, exact free-end fix (990 hPa @ frame 0)
    arr[0, 0] = _storm(msl_pa=99000.0)
    # m1 analyzable but the storm lives on path 1 (path 0 is a far system)
    for t in range(T):
        arr[1, 0, t] = [10.0, 40.0, 100500.0, 15.0]
    arr[1, 1] = _storm(msl_pa=97000.0)  # 970 hPa @ frame 0
    # m2 insufficient_fixes: 2 in-domain fixes at the anchor window
    arr[2, 0] = _storm(frames=[10, 11])
    # m3 unanchored: coherent storm near (30,-80), never near the anchor
    for t in range(T):
        arr[3, 0, t] = [30.0, _lon360(-80.0), 100000.0, 20.0]
    # m4 no_track: only a far-away system, no in-domain fix
    for t in range(T):
        arr[4, 0, t] = [10.0, 40.0, 100500.0, 15.0]
    # m5 analyzable with a nearest-frame free-end fallback (no frame-0 fix)
    arr[5, 0] = _storm(msl_pa=99500.0, skip_frame0=True)
    return arr


@pytest.fixture
def qc(tmp_path):
    pq = str(tmp_path / "tc_tracks_end.parquet")
    save_tracks_parquet(pq, _ensemble(), era5_arr=_era5())
    return compute_track_qc(
        pq,
        track_box=TRACK_BOX,
        anchor_frame=ANCHOR_FRAME,
        free_end_frame=FREE_END_FRAME,
    )


class TestTaxonomy:
    def test_exclusion_labels(self, qc):
        assert qc.table["exclusion"].tolist() == [
            "",
            "",
            "insufficient_fixes",
            "unanchored",
            "no_track",
            "",
        ]

    def test_analyzable_and_anchored_masks(self, qc):
        assert qc.analyzable.tolist() == [True, True, False, False, False, True]
        # insufficient_fixes reached the anchor -> anchored True but not analyzable
        assert qc.anchored.tolist() == [True, True, True, False, False, True]

    def test_storm_on_non_path0_is_resolved(self, qc):
        # m1's storm is stored on path 1; resolution recovers it.
        assert qc.resolved_path_id.tolist() == [0, 1, 0, -1, -1, 0]

    def test_resolved_domain_fix_counts(self, qc):
        # Resolved-path in-domain fix count (0 where nothing resolved).
        assert qc.table["n_domain_fixes"].tolist() == [12, 12, 2, 0, 0, 11]

    def test_two_fix_anchor_touch_is_insufficient(self, qc):
        row = qc.table.iloc[2]
        assert row["anchored"] and not row["analyzable"]
        assert row["exclusion"] == "insufficient_fixes"
        assert row["n_domain_fixes"] == 2

    def test_counts_partition(self, qc):
        c = qc.counts()
        assert c["total"] == 6
        assert c["analyzable"] == 3
        assert c["excluded"] == 3
        assert c["no_track"] == 1
        assert c["unanchored"] == 1
        assert c["insufficient_fixes"] == 1
        # excluded == sum of the three reasons
        assert c["excluded"] == c["no_track"] + c["unanchored"] + c["insufficient_fixes"]


class TestFreeEnd:
    def test_exact_frame_excludes_fallback(self, qc):
        # m0, m1 have exact frame-0 fixes; m5 fell back -> analyzable but not exact.
        assert qc.exact_frame.tolist() == [True, True, False, False, False, False]
        assert qc.fe_fallback.tolist() == [False, False, True, False, False, True]

    def test_free_end_targets_nan_for_excluded(self, qc):
        fet = qc.free_end_targets()  # analyzable-only by default
        lat = fet["fe_lat_deg"]
        assert np.isfinite(lat[[0, 1, 5]]).all()
        assert np.isnan(lat[[2, 3, 4]]).all()
        # distance from ERA5 is finite exactly where the fix is.
        assert np.isfinite(fet["fe_dist_era5_km"][0])

    def test_era5_msl_pct_over_exact_frame(self, qc):
        # exact-frame members m0 (990 hPa) and m1 (970) vs ERA5 980 -> 1/2 below.
        pct, n = qc.era5_free_end_msl_pct()
        assert n == 2
        assert pct == pytest.approx(50.0)


class TestStormOfInterestArray:
    def test_reduces_to_resolved_path(self):
        ens, era5 = _ensemble(), _era5()
        reordered, qc = resolve_storm_of_interest(
            ens,
            era5,
            track_box=TRACK_BOX,
            anchor_frame=ANCHOR_FRAME,
            free_end_frame=FREE_END_FRAME,
        )
        assert reordered.shape == (6, 1, T, 4)
        # m1's resolved storm (original path 1) is now at index 0.
        assert np.allclose(reordered[1, 0], ens[1, 1], equal_nan=True)
        # unanchored (m3) and no_track (m4) come back all-NaN (skipped in drawing).
        assert np.isnan(reordered[3, 0]).all()
        assert np.isnan(reordered[4, 0]).all()
        # insufficient_fixes (m2) keeps its short resolved path.
        assert np.isfinite(reordered[2, 0]).any()

    def test_reordered_domain_membership(self):
        ens, era5 = _ensemble(), _era5()
        reordered, _ = resolve_storm_of_interest(
            ens,
            era5,
            track_box=TRACK_BOX,
            anchor_frame=ANCHOR_FRAME,
            free_end_frame=FREE_END_FRAME,
        )
        in_dom = tracks_in_domain(reordered, *TRACK_BOX)[:, 0]
        # Every resolved storm (m0, m1, m2, m5) touches the track box.
        assert in_dom[[0, 1, 2, 5]].all()
        assert not in_dom[[3, 4]].any()


class TestTaxonomyOrdering:
    def test_no_track_beats_out_of_domain_anchor_fix(self, tmp_path):
        # A path can reach the anchor via an out-of-domain window fix while
        # having no in-domain fix at all.  The spec's order makes that
        # no_track (checked first), not insufficient_fixes.
        box = (-80.0, -75.0, 38.0, 42.0)  # tight box; anchor sits near the edge
        era5 = np.full((1, 1, T, 4), np.nan)
        for t in range(T):
            era5[0, 0, t] = [38.0 + t * 4.0 / 11.0, _lon360(-77.5), 98000.0, 30.0]
        # anchor (frame 11) = (42, -77.5), free end (frame 0) = (38, -77.5).
        ens = np.full((2, 1, T, 4), np.nan)
        # m0: single fix ~330 km from the anchor but out of the box (lon -73.5).
        ens[0, 0, 11] = [42.0, _lon360(-73.5), 99000.0, 30.0]
        # m1: a normal in-box analyzable storm.
        for t in range(T):
            ens[1, 0, t] = [38.0 + t * 4.0 / 11.0, _lon360(-77.5), 99000.0, 30.0]
        pq = str(tmp_path / "tc_tracks_end.parquet")
        save_tracks_parquet(pq, ens, era5_arr=era5)
        qc = compute_track_qc(
            pq, track_box=box, anchor_frame=ANCHOR_FRAME, free_end_frame=FREE_END_FRAME
        )
        assert qc.table["exclusion"].tolist() == ["no_track", ""]
        assert qc.resolved_path_id.tolist() == [-1, 0]


class TestDegradation:
    def test_undefined_anchor_falls_back_to_domain_rule(self, tmp_path):
        # Anchor frame beyond the lead axis -> anchor undefined; resolution
        # degrades to "most in-domain fixes", no member is 'unanchored'.
        pq = str(tmp_path / "tc_tracks_end.parquet")
        save_tracks_parquet(pq, _ensemble(), era5_arr=_era5())
        qc = compute_track_qc_arrays(
            _ensemble(),
            _era5(),
            track_box=TRACK_BOX,
            anchor_frame=999,
            free_end_frame=FREE_END_FRAME,
        )
        assert qc.era5_anchor is None
        assert qc.config["anchor_defined"] is False
        assert int((qc.table["exclusion"] == "unanchored").sum()) == 0

    def test_n_members_padding(self, tmp_path):
        # Requesting more members than the parquet has -> extras are no_track.
        pq = str(tmp_path / "tc_tracks_end.parquet")
        save_tracks_parquet(pq, _ensemble(), era5_arr=_era5())
        qc = compute_track_qc(
            pq,
            track_box=TRACK_BOX,
            anchor_frame=ANCHOR_FRAME,
            free_end_frame=FREE_END_FRAME,
            n_members=8,
        )
        assert qc.n_members == 8
        assert qc.table["exclusion"].tolist()[6:] == ["no_track", "no_track"]
