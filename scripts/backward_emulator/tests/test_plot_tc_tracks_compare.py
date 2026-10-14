"""Light-env tests for the TC-track figure's numpy reductions.

The parquet read + cartopy map are heavy-env; here we cover the MSLP unit
detection, the central-pressure band, and the great-circle track error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_tc_tracks_compare as tp  # noqa: E402


def test_to_hpa_detects_pa_and_passes_hpa():
    pa = np.array([100500.0, 99000.0, 101000.0])
    np.testing.assert_allclose(tp._to_hpa(pa), pa / 100.0)
    hpa = np.array([1005.0, 990.0, 1010.0])
    np.testing.assert_allclose(tp._to_hpa(hpa), hpa)  # already hPa, unchanged


def test_haversine_one_degree_lat_is_about_111km():
    d = tp._haversine_km(0.0, 0.0, 1.0, 0.0)
    assert 110.0 < float(d) < 112.0


def test_primary_msl_band_orders_and_shapes():
    # [N, P, T, 4]; path 0 msl (index 2) varies across members.
    N, P, T = 50, 2, 12
    ens = np.full((N, P, T, 4), np.nan)
    # give path 0 a spread of central pressures in Pa
    ens[:, 0, :, 2] = (np.linspace(96000, 101000, N)[:, None]
                       * np.ones((1, T)))
    lo, mid, hi = tp.primary_msl_band(ens, qlo=10, qhi=90)
    assert lo.shape == (T,)
    assert np.all(lo <= mid) and np.all(mid <= hi)
    assert 950 < mid[0] < 1020  # converted Pa -> hPa


def test_position_error_zero_when_on_reference():
    N, P, T = 10, 1, 6
    ens = np.zeros((N, P, T, 4))
    era5 = np.zeros((1, P, T, 4))
    # put both at the same (lat, lon)
    ens[:, 0, :, 0] = 25.0; ens[:, 0, :, 1] = -80.0
    era5[0, 0, :, 0] = 25.0; era5[0, 0, :, 1] = -80.0
    err = tp.position_error_km(ens, era5)
    np.testing.assert_allclose(err, 0.0, atol=1e-6)
    # shift members 1 deg north -> ~111 km
    ens[:, 0, :, 0] = 26.0
    err2 = tp.position_error_km(ens, era5)
    assert np.all(err2 > 100.0) and np.all(err2 < 120.0)


# --- closest-to-ERA5 member (min-MSLP) ------------------------------------- #
def test_closest_member_index_picks_min_rmse():
    era5 = np.array([1000.0, 990.0, 980.0])
    members = np.array(
        [
            [1000.0, 990.0, 980.0],  # exact match -> RMSE 0 (closest)
            [1005.0, 995.0, 985.0],  # +5 everywhere
            [900.0, 900.0, 900.0],  # far off
        ]
    )
    assert tp.closest_member_index(members, era5) == 0


def test_closest_member_index_scores_on_overlap_and_respects_eligibility():
    era5 = np.array([1000.0, 990.0, 980.0])
    members = np.array(
        [
            [np.nan, np.nan, np.nan],  # no overlap -> never chosen
            [1001.0, 991.0, 981.0],  # +1 everywhere
            [1000.0, 990.0, 980.0],  # exact
        ]
    )
    assert tp.closest_member_index(members, era5) == 2  # exact wins
    # excluding the exact member falls back to the +1 member, not the NaN one
    elig = np.array([True, True, False])
    assert tp.closest_member_index(members, era5, eligible=elig) == 1


def test_closest_member_index_none_when_no_overlap_or_shape_mismatch():
    era5 = np.array([1000.0, 990.0])
    assert tp.closest_member_index(np.array([[np.nan, np.nan]]), era5) is None
    # T mismatch between members and reference -> None (skip the highlight)
    assert (
        tp.closest_member_index(np.array([[1.0, 2.0]]), np.array([1.0, 2.0, 3.0]))
        is None
    )


# --- PC-percentile colouring helpers --------------------------------------- #
def test_score_percentile_monotone_endpoints():
    pct = tp.score_percentile(np.array([5.0, 1.0, 3.0, 2.0, 4.0]))
    # smallest -> 0, largest -> 100; ranks preserve order
    assert pct[np.argmin([5, 1, 3, 2, 4])] == 0.0
    assert pct[np.argmax([5, 1, 3, 2, 4])] == 100.0
    assert np.all((pct >= 0) & (pct <= 100))
    # single element -> midpoint (avoids divide-by-zero)
    np.testing.assert_allclose(tp.score_percentile(np.array([7.0])), [50.0])


def test_choose_pc_picks_most_correlated_with_tc_lon():
    rng = np.random.default_rng(0)
    N = 200
    tc_lon = rng.normal(size=N)
    # PC0 noise, PC1 strongly correlated with tc_lon, PC2 weakly
    scores = np.column_stack([rng.normal(size=N),
                              2.0 * tc_lon + 0.1 * rng.normal(size=N),
                              0.3 * tc_lon + rng.normal(size=N)])
    assert tp.choose_pc(scores, tc_lon, n_pc=3) == 1
    # no target -> PC0
    assert tp.choose_pc(scores, None, n_pc=3) == 0


def test_pc_percentile_full_maps_back_through_analyzable_mask():
    # Na=3 analyzable rows out of N=5, in original-member order.
    n_members = 5
    analyzable = np.array([True, False, True, False, True])
    pc_scores = np.array([[10.0], [20.0], [30.0]])  # ascending -> pct 0,50,100
    full = tp.pc_percentile_full(pc_scores, analyzable, n_members, jstar=0)
    assert full.shape == (n_members,)
    assert np.isnan(full[1]) and np.isnan(full[3])          # off-analyzable -> NaN
    np.testing.assert_allclose(full[[0, 2, 4]], [0.0, 50.0, 100.0])


def test_pc_percentile_full_identity_when_no_mask():
    pc_scores = np.array([[1.0], [2.0], [3.0], [4.0]])
    full = tp.pc_percentile_full(pc_scores, None, 4, jstar=0)
    np.testing.assert_allclose(full, [0.0, 100.0 / 3, 200.0 / 3, 100.0])


def test_pc_percentile_full_rejects_length_mismatch():
    # analyzable says 2 rows, pc_scores has 3 -> guard raises rather than mis-align.
    analyzable = np.array([True, True, False])
    pc_scores = np.array([[1.0], [2.0], [3.0]])
    try:
        tp.pc_percentile_full(pc_scores, analyzable, 3, jstar=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on analyzable/scores mismatch")


def test_resolve_color_tokens_auto_and_indices():
    pca = {"pc_scores": np.zeros((10, 4)), "tc_lon": None, "n_pc": 4}
    out = tp.resolve_color_tokens(["auto", "1", "3", "9"], pca)
    labels = [lab for lab, _ in out]
    js = [j for _, j in out]
    assert labels[0].startswith("auto")   # auto resolves (PC0 with no target)
    assert (0 in js) and (2 in js)         # 1-based 1->0, 3->2
    assert all(j < 4 for _, j in out)      # PC 9 (out of range) dropped


def test_orient_by_target_flips_negative_correlation():
    y = np.linspace(-1, 1, 50)
    # score anti-correlated with tc_lon -> sign should be -1 (flip so yellow==east)
    assert tp.orient_by_target(-3.0 * y, y) == -1.0
    assert tp.orient_by_target(+3.0 * y, y) == 1.0
    assert tp.orient_by_target(y, None) == 1.0            # no target -> no flip
    assert tp.orient_by_target(np.zeros_like(y), y) == 1.0  # constant -> no flip


def test_pc_percentile_full_sign_reverses_ranking():
    pc = np.array([[10.0], [20.0], [30.0]])
    up = tp.pc_percentile_full(pc, None, 3, jstar=0, sign=1.0)
    dn = tp.pc_percentile_full(pc, None, 3, jstar=0, sign=-1.0)
    np.testing.assert_allclose(up, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(dn, [100.0, 50.0, 0.0])     # flipped colour direction


def test_resolve_var_index_fuzzy_match():
    assert tp.resolve_var_index(["t2m", "z500", "msl"], "z500") == 1
    assert tp.resolve_var_index(["t2m", "z_500"], "z500") == 1   # underscore-insensitive
    assert tp.resolve_var_index(["t2m", "msl"], "z500") == 0     # fallback to 0


def test_scatter_to_box_reshapes_rectangular_subgrid():
    # full grid 4 lat x 5 lon; a 2x3 synoptic box sub-grid.
    full_lat = np.array([10.0, 20.0, 30.0, 40.0])
    full_lon = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    rows = np.array([1, 1, 1, 2, 2, 2])       # lat idx 1,2
    cols = np.array([2, 3, 4, 2, 3, 4])       # lon idx 2,3,4
    vals = np.arange(6, dtype=float)
    field, lat1d, lon1d = tp._scatter_to_box(vals, rows, cols, full_lat, full_lon)
    assert field.shape == (2, 3)
    np.testing.assert_allclose(lat1d, [20.0, 30.0])
    np.testing.assert_allclose(lon1d, [20.0, 30.0, 40.0])
    np.testing.assert_allclose(field, [[0, 1, 2], [3, 4, 5]])
