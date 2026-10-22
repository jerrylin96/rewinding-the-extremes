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

"""Unit tests for the unified track QC (``track_qc``).

Exercises anchor-first storm-of-interest resolution and the exclusion
taxonomy against synthetic parquets: an anchor-reaching path that is not the
stored path 0, a 2-fix path touching the anchor window (``insufficient_fixes``),
a coherent path that never approaches the anchor (``unanchored``), and a storm
with no fix at the free-end frame (``no_free_end_fix``: never replaced by a fix
from another frame).  Also covers the ERA5 reference checks that raise instead
of degrading, the per-mode frames, the free-end targets and percentile, the
storm-of-interest array the figures draw, and the rotated-frame targets.  Pure
NumPy / pandas / pyarrow -- no zarr, matplotlib, or cartopy.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from tc_track_targets import rotated_track_targets  # noqa: E402
from tc_tracks_io import save_tracks_parquet, tracks_in_domain  # noqa: E402
from track_qc import compute_track_qc, compute_track_qc_arrays  # noqa: E402

T = 12
TRACK_BOX = (-85.0, -60.0, 25.0, 45.0)  # lon_min, lon_max, lat_min, lat_max


def _lon360(x: float) -> float:
    return x % 360.0


def _pos(t: int):
    """ERA5's storm at frame t: NE-bound from (26N,-78E) to (40N,-74E)."""
    return 26.0 + t * 14.0 / 11.0, -78.0 + t * 4.0 / 11.0


def _era5(frames=range(T)) -> np.ndarray:
    """ERA5 track [1,1,12,4]; free end (frame 0) MSL 980 hPa."""
    arr = np.full((1, 1, T, 4), np.nan)
    for t in frames:
        lat, lon = _pos(t)
        arr[0, 0, t] = [lat, _lon360(lon), 98000.0, 30.0]
    return arr


def _storm(msl_pa: float = 99000.0, frames=range(T)) -> np.ndarray:
    p = np.full((T, 4), np.nan)
    for t in frames:
        lat, lon = _pos(t)
        p[t] = [lat, _lon360(lon), msl_pa, 30.0]
    return p


def _ensemble() -> np.ndarray:
    """Six members, one per outcome the taxonomy must separate."""
    arr = np.full((6, 3, T, 4), np.nan)
    # m0 analyzable, storm on path 0 (990 hPa at the free end)
    arr[0, 0] = _storm(msl_pa=99000.0)
    # m1 analyzable but the storm lives on path 1 (path 0 is a far system)
    for t in range(T):
        arr[1, 0, t] = [10.0, 40.0, 100500.0, 15.0]
    arr[1, 1] = _storm(msl_pa=97000.0)
    # m2 insufficient_fixes: 2 in-domain fixes at the anchor window (and no
    # free-end fix either: the earlier reason wins)
    arr[2, 0] = _storm(frames=[10, 11])
    # m3 unanchored: coherent storm near (30,-80), never near the anchor
    for t in range(T):
        arr[3, 0, t] = [30.0, _lon360(-80.0), 100000.0, 20.0]
    # m4 no_track: only a far-away system, no in-domain fix
    for t in range(T):
        arr[4, 0, t] = [10.0, 40.0, 100500.0, 15.0]
    # m5 no_free_end_fix: the storm forms one frame after the free end
    arr[5, 0] = _storm(msl_pa=99500.0, frames=range(1, T))
    return arr


@pytest.fixture
def qc(tmp_path):
    pq = str(tmp_path / "tc_tracks_end.parquet")
    save_tracks_parquet(pq, _ensemble(), era5_arr=_era5())
    _ens, _era5_arr, out = compute_track_qc(pq, track_box=TRACK_BOX, mode="end")
    return out


class TestTaxonomy:
    def test_exclusion_labels(self, qc):
        assert qc.table["exclusion"].tolist() == [
            "",
            "",
            "insufficient_fixes",
            "unanchored",
            "no_track",
            "no_free_end_fix",
        ]

    def test_analyzable_and_anchored_masks(self, qc):
        assert qc.analyzable.tolist() == [True, True, False, False, False, False]
        assert qc.anchored.tolist() == [True, True, True, False, False, True]

    def test_storm_on_non_path0_is_resolved(self, qc):
        assert qc.resolved_path_id.tolist() == [0, 1, 0, -1, -1, 0]

    def test_resolved_domain_fix_counts(self, qc):
        assert qc.table["n_domain_fixes"].tolist() == [12, 12, 2, 0, 0, 11]

    def test_counts_partition(self, qc):
        c = qc.counts()
        assert (c["total"], c["analyzable"], c["excluded"]) == (6, 2, 4)
        for reason in ("no_track", "unanchored", "insufficient_fixes", "no_free_end_fix"):
            assert c[reason] == 1
        assert qc.summary() == (
            "2/6 analyzable (excluded: no_track=1 unanchored=1 "
            "insufficient_fixes=1 no_free_end_fix=1)"
        )

    def test_frames_come_from_the_mode(self, qc):
        assert (qc.config["anchor_frame"], qc.config["free_end_frame"]) == (11, 0)


class TestFreeEnd:
    def test_targets_are_exact_frame_and_nan_for_excluded(self, qc):
        fet = qc.free_end_targets()
        np.testing.assert_allclose(fet["fe_lat_deg"][[0, 1]], 26.0)
        np.testing.assert_allclose(fet["fe_msl_hpa"][[0, 1]], [990.0, 970.0])
        # m5's frame-1 fix is never used as its free end.
        assert np.isnan(fet["fe_lat_deg"][[2, 3, 4, 5]]).all()
        assert np.isfinite(fet["fe_dist_era5_km"][[0, 1]]).all()

    def test_era5_msl_pct_over_analyzable(self, qc):
        # m0 (990 hPa) and m1 (970) vs ERA5 980 -> 1/2 at or below.
        pct, n = qc.era5_free_end_msl_pct()
        assert n == 2
        assert pct == pytest.approx(50.0)


class TestModes:
    def test_start_mode_needs_the_last_frame(self):
        # Start-conditioning: anchor at frame 0, free end at the last frame.
        ens = np.full((2, 1, T, 4), np.nan)
        ens[0, 0] = _storm()
        ens[1, 0] = _storm(frames=range(0, T - 1))  # lost one frame early
        qc = compute_track_qc_arrays(ens, _era5(), track_box=TRACK_BOX, mode="start")
        assert (qc.config["anchor_frame"], qc.config["free_end_frame"]) == (0, 11)
        assert qc.table["exclusion"].tolist() == ["", "no_free_end_fix"]
        assert qc.era5_free_end["lat"] == pytest.approx(40.0)

    def test_both_mode_has_no_free_end(self):
        ens = np.full((1, 1, T, 4), np.nan)
        ens[0, 0] = _storm(frames=range(0, T - 1))
        qc = compute_track_qc_arrays(ens, _era5(), track_box=TRACK_BOX, mode="both")
        assert qc.era5_free_end is None
        assert qc.analyzable.tolist() == [True]
        with pytest.raises(ValueError, match="no free end"):
            qc.free_end_targets()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            compute_track_qc_arrays(
                _ensemble(), _era5(), track_box=TRACK_BOX, mode="forward"
            )


class TestEra5Reference:
    def test_missing_anchor_fix_raises(self):
        with pytest.raises(ValueError, match="anchor frame 11"):
            compute_track_qc_arrays(
                _ensemble(), _era5(frames=range(0, T - 1)), track_box=TRACK_BOX, mode="end"
            )

    def test_missing_free_end_fix_raises(self):
        with pytest.raises(ValueError, match="free-end frame 0"):
            compute_track_qc_arrays(
                _ensemble(), _era5(frames=range(1, T)), track_box=TRACK_BOX, mode="end"
            )

    def test_path0_outside_the_track_box_raises(self):
        # ERA5 path 0 is some other system: its anchor fix is not in the box.
        era5 = _era5()
        era5[0, 0, :, :2] = [10.0, 40.0]
        with pytest.raises(ValueError, match="outside the track box"):
            compute_track_qc_arrays(_ensemble(), era5, track_box=TRACK_BOX, mode="end")

    def test_era5_required(self):
        with pytest.raises(ValueError, match="ERA5"):
            compute_track_qc_arrays(_ensemble(), None, track_box=TRACK_BOX, mode="end")


class TestMemberAxis:
    def test_n_members_padding(self, tmp_path):
        # The parquet has no rows for a member the tracker found nothing in, so
        # trailing members are no_track.
        pq = str(tmp_path / "tc_tracks_end.parquet")
        save_tracks_parquet(pq, _ensemble(), era5_arr=_era5())
        _e, _r, qc = compute_track_qc(pq, track_box=TRACK_BOX, mode="end", n_members=8)
        assert qc.n_members == 8
        assert qc.table["exclusion"].tolist()[6:] == ["no_track", "no_track"]

    def test_fewer_members_than_tracks_raises(self):
        with pytest.raises(ValueError, match="n_members=4"):
            compute_track_qc_arrays(
                _ensemble(), _era5(), track_box=TRACK_BOX, mode="end", n_members=4
            )

    def test_anchor_window_past_the_array_is_unanchored(self):
        # No member has a fix at frames 9-11, so the tracker array stops at
        # frame 8: nobody reaches the anchor (no domain-only stand-in).
        ens = _storm(frames=range(0, 9))[None, None, :9]
        qc = compute_track_qc_arrays(ens, _era5(), track_box=TRACK_BOX, mode="end")
        assert qc.table["exclusion"].tolist() == ["unanchored"]


class TestTaxonomyOrdering:
    def test_no_track_beats_out_of_domain_anchor_fix(self):
        # A path can reach the anchor via an out-of-domain window fix while
        # having no in-domain fix at all: that is no_track, checked first.
        box = (-80.0, -75.0, 38.0, 42.0)  # tight box; anchor sits near the edge
        era5 = np.full((1, 1, T, 4), np.nan)
        for t in range(T):
            era5[0, 0, t] = [38.0 + t * 4.0 / 11.0, _lon360(-77.5), 98000.0, 30.0]
        ens = np.full((2, 1, T, 4), np.nan)
        # m0: single fix ~330 km from the anchor but out of the box (lon -73.5).
        ens[0, 0, 11] = [42.0, _lon360(-73.5), 99000.0, 30.0]
        # m1: a normal in-box analyzable storm.
        for t in range(T):
            ens[1, 0, t] = [38.0 + t * 4.0 / 11.0, _lon360(-77.5), 99000.0, 30.0]
        qc = compute_track_qc_arrays(ens, era5, track_box=box, mode="end")
        assert qc.table["exclusion"].tolist() == ["no_track", ""]
        assert qc.resolved_path_id.tolist() == [-1, 0]


class TestStormOfInterestArray:
    def test_draws_only_the_analyzable_storms(self, qc):
        ens = _ensemble()
        soi = qc.storm_of_interest_array(ens)
        assert soi.shape == (6, 1, T, 4)
        # m1's storm (original path 1) is now at index 0.
        assert np.allclose(soi[1, 0], ens[1, 1], equal_nan=True)
        # Every excluded member, including a full storm with no free-end fix
        # (m5), is all-NaN: the figures draw the statistics' population.
        assert np.isnan(soi[[2, 3, 4, 5]]).all()
        in_dom = tracks_in_domain(soi, *TRACK_BOX)[:, 0]
        assert in_dom.tolist() == [True, True, False, False, False, False]

    def test_single_member_figures_keep_a_storm_lacking_only_the_free_end_fix(self, qc):
        ens = _ensemble()
        soi = qc.storm_of_interest_array(ens, require_free_end_fix=False)
        assert np.allclose(soi[[0, 1, 5], 0], ens[[0, 1, 5], [0, 1, 0]], equal_nan=True)
        # Too few fixes, unanchored and no track stay excluded.
        assert np.isnan(soi[[2, 3, 4]]).all()


class TestSynopticPcaFigureTracks:
    """Which storms ``aggregate_synoptic_pca`` hands its panels, and how it marks them."""

    @pytest.fixture
    def agg(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("cartopy")
        import aggregate_synoptic_pca

        return aggregate_synoptic_pca

    def test_member_rows_keep_a_storm_lacking_only_the_free_end_fix(
        self, agg, tmp_path
    ):
        pq = tmp_path / "tc_tracks" / "tempest" / "tc_tracks_end.parquet"
        pq.parent.mkdir(parents=True)
        save_tracks_parquet(str(pq), _ensemble(), era5_arr=_era5())
        diag = {
            "tc_domain": np.array(TRACK_BOX),
            "pc_scores": np.zeros((6, 2)),
            "free_end_frame": np.array(0),
            "lead_hours": np.arange(T) * 6.0,
        }
        agg._attach_track_qc(diag, tmp_path / "synoptic_pca", "end")
        population = diag["member_track_latlon"]
        single = diag["single_member_track_latlon"]
        assert np.isfinite(population[[0, 1], 0]).all()
        assert np.isnan(population[[2, 3, 4, 5]]).all()
        # m5's storm (no fix at the free end) is drawn in a single-member row.
        assert np.allclose(single[5], _ensemble()[5, 0, :, :2], equal_nan=True)
        assert np.isnan(single[[2, 3, 4]]).all()

    def test_position_marker_needs_a_fix_at_that_frame(self, agg):
        import matplotlib.pyplot as plt

        track = np.full((T, 2), np.nan)
        for t in range(10):  # the storm is last detected at frame 9
            track[t] = _pos(t)
        ax = plt.axes(projection=agg.ccrs.PlateCarree())
        agg._draw_progressive_track(ax, track, 9, "k", "-")
        assert len(ax.lines) == 2  # track + marker at frame 9
        ax.cla()
        agg._draw_progressive_track(ax, track, 11, "k", "-")
        assert len(ax.lines) == 1  # track only: no fix at frame 11
        plt.close("all")


class TestRotatedTargets:
    def test_along_track_frame(self, qc):
        era5 = _era5()
        targets = qc.free_end_targets()
        targets["fe_lat_deg"][1] += 1.0  # one member a degree north of ERA5
        targets["fe_lat_deg"][2] = 26.0  # a third fix for the cloud frame
        targets["fe_lon_deg"][2] = -78.5
        rotated, info = rotated_track_targets(era5, targets, qc.era5_free_end, 0)
        assert rotated["fe_along_km"][0] == pytest.approx(0.0, abs=1e-6)
        assert rotated["fe_along_km"][1] > 0  # north is along ERA5's NE motion
        assert 0.0 < info["era5_motion_bearing_deg"] < 90.0

    def test_missing_neighbour_fix_raises(self, qc):
        era5 = _era5()
        era5[0, 0, 1] = np.nan
        with pytest.raises(ValueError, match="frames 0, 1"):
            rotated_track_targets(era5, qc.free_end_targets(), qc.era5_free_end, 0)

    def test_too_few_fixes_raises(self, qc):
        with pytest.raises(ValueError, match="need >= 3"):
            rotated_track_targets(_era5(), qc.free_end_targets(), qc.era5_free_end, 0)
