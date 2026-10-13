"""Light-env tests for the makani free-end-states pure reductions.

The makani/h5 read is heavy-env; here we cover the grid reshape, the land-weighted
mean, and the signed-bias tail ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import free_end_states_makani as fes  # noqa: E402


def test_grid_axes_and_scatter_roundtrip_rectangular_box():
    # full grid H=4 x W=5; a 2x3 box (lat idx 1,2 ; lon idx 2,3,4).
    W = 5
    flat = np.array([1 * W + 2, 1 * W + 3, 1 * W + 4,
                     2 * W + 2, 2 * W + 3, 2 * W + 4])
    ur, uc, ri, ci = fes.grid_axes(flat, W)
    np.testing.assert_array_equal(ur, [1, 2])
    np.testing.assert_array_equal(uc, [2, 3, 4])
    vals = np.arange(6, dtype=float)
    grid = fes.scatter_to_grid(vals, ri, ci, ur.size, uc.size)
    assert grid.shape == (2, 3)
    np.testing.assert_allclose(grid, [[0, 1, 2], [3, 4, 5]])


def test_scatter_to_grid_keeps_leading_dims_and_nan_holes():
    # one pixel missing -> that cell stays NaN; leading (member) dim preserved.
    W = 3
    flat = np.array([0, 1, 3])          # (r0c0), (r0c1), (r1c0) -- (r1c1) absent
    ur, uc, ri, ci = fes.grid_axes(flat, W)
    vals = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])  # [2 members, P=3]
    grid = fes.scatter_to_grid(vals, ri, ci, ur.size, uc.size)
    assert grid.shape == (2, 2, 2)
    np.testing.assert_allclose(grid[:, 0, 0], [10.0, 20.0])
    assert np.isnan(grid[0, 1, 1]) and np.isnan(grid[1, 1, 1])


def test_weighted_mean_last_matches_manual():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])  # [N=2, P=3]
    w = np.array([0.5, 0.25, 0.25])
    np.testing.assert_allclose(fes.weighted_mean_last(arr, w), [1.75, 5.5])


def test_signed_bias_rank_low_are_coolest_high_are_warmest():
    bias = np.array([0.0, -3.0, 2.0, -1.0, 5.0])
    low, high = fes.signed_bias_rank(bias, k=2)
    # coolest two = indices of -3.0 then -1.0; warmest two = 5.0 then 2.0
    np.testing.assert_array_equal(low, [1, 3])
    np.testing.assert_array_equal(high, [4, 2])


def test_signed_bias_rank_caps_k_at_population():
    low, high = fes.signed_bias_rank(np.array([1.0, 2.0]), k=5)
    assert low.size == 2 and high.size == 2


def test_metric_and_excluded_splits_view_pixels():
    # view = pixels {0,1,2,3,4}; impact box = {1,2,3}; of those, land = {1,3}.
    flat_view = np.array([0, 1, 2, 3, 4])
    flat_impact_all = np.array([1, 2, 3])
    flat_impact_land = np.array([1, 3])
    metric, excluded = fes.metric_and_excluded(
        flat_view, flat_impact_land, flat_impact_all)
    np.testing.assert_array_equal(metric, [False, True, False, True, False])
    # pixel 2 is inside the box but not land -> excluded (hatched); outside box -> not
    np.testing.assert_array_equal(excluded, [False, False, True, False, False])
