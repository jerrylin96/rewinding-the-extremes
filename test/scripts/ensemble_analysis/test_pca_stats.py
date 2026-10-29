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

"""Unit tests for ``pca_stats.ellipsoid_test``: is ERA5's PC state within the
members' range?  Pure NumPy; checked against its definition and against OLS
leverage computed directly from the hat matrix.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "_shared")
    ),
)

from pca_stats import ellipsoid_test  # noqa: E402

N_MEMBERS, N_PCS = 200, 8


@pytest.fixture
def scores() -> np.ndarray:
    """Members with unequal PC variances, like real PC scores."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_MEMBERS, N_PCS)) * np.linspace(5.0, 1.0, N_PCS)


def _leverage(scores: np.ndarray, point: np.ndarray) -> tuple:
    """OLS leverage (with an intercept) of each member and of ``point``, from
    the hat matrix directly -- independent of ``ellipsoid_test``'s algebra."""
    design = np.column_stack([np.ones(len(scores)), scores])
    gram_inv = np.linalg.inv(design.T @ design)
    h_members = np.einsum("ij,jk,ik->i", design, gram_inv, design)
    point_row = np.r_[1.0, point]
    return h_members, float(point_row @ gram_inv @ point_row)


def _farthest(scores: np.ndarray) -> np.ndarray:
    """The member with the highest leverage, i.e. farthest from the center."""
    h_members, _ = _leverage(scores, scores[0])
    return scores[int(np.argmax(h_members))]


def test_center_is_inside(scores: np.ndarray) -> None:
    result = ellipsoid_test(scores, scores.mean(axis=0))
    assert result.d2_point == pytest.approx(0.0, abs=1e-12)
    assert result.m == N_MEMBERS
    assert result.inside
    assert result.p_value == pytest.approx(1.0)


def test_just_beyond_farthest_member_is_outside(scores: np.ndarray) -> None:
    mean = scores.mean(axis=0)
    direction = _farthest(scores) - mean
    inside = ellipsoid_test(scores, mean + 0.999 * direction)
    outside = ellipsoid_test(scores, mean + 1.001 * direction)
    assert inside.inside
    assert not outside.inside and outside.m == 0
    assert outside.d2_point > outside.d2_max
    assert outside.p_value == pytest.approx(1.0 / (N_MEMBERS + 1))


def test_unchanged_when_pcs_are_rescaled(scores: np.ndarray) -> None:
    point = 1.2 * scores[0]  # an arbitrary point with some members beyond it
    scale = np.logspace(-3, 3, N_PCS)
    raw = ellipsoid_test(scores, point)
    rescaled = ellipsoid_test(scores * scale, point * scale)
    assert 0 < raw.m < N_MEMBERS
    assert rescaled.m == raw.m
    assert rescaled.d2_point == pytest.approx(raw.d2_point, rel=1e-9)
    assert rescaled.d2_max == pytest.approx(raw.d2_max, rel=1e-9)


@pytest.mark.parametrize("stretch", [0.5, 1.5])
def test_agrees_with_hat_matrix_leverage(scores: np.ndarray, stretch: float) -> None:
    """With an intercept, leverage is ``1/n + d^2/(n-1)``, so inside means the
    point's leverage is at most the largest member's."""
    point = scores.mean(axis=0) + stretch * (_farthest(scores) - scores.mean(axis=0))
    h_members, h_point = _leverage(scores, point)
    result = ellipsoid_test(scores, point)
    n = N_MEMBERS
    assert h_point == pytest.approx(1 / n + result.d2_point / (n - 1), rel=1e-9)
    assert h_members.max() == pytest.approx(1 / n + result.d2_max / (n - 1), rel=1e-9)
    assert result.m == int((h_members >= h_point).sum())
    assert result.inside == (h_point <= h_members.max())


def test_uses_full_covariance() -> None:
    """Two correlated PCs: a point inside each PC's own range but off the line
    the members lie along is outside.  A variance-only distance would call it
    inside."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500)
    two = np.column_stack([a, a + 0.1 * rng.standard_normal(500)])
    point = np.array([1.0, -1.0])  # within ~1 sd on each axis
    assert not ellipsoid_test(two, point).inside


@pytest.mark.parametrize(
    "scores, point",
    [
        (np.ones((10, 2)) * np.nan, np.zeros(2)),  # non-finite members
        (np.random.default_rng(0).standard_normal((10, 2)), np.array([np.nan, 0.0])),
        (np.random.default_rng(0).standard_normal((10, 2)), np.zeros(3)),  # shape
        (np.random.default_rng(0).standard_normal((2, 2)), np.zeros(2)),  # n <= k
    ],
)
def test_rejects_bad_input(scores: np.ndarray, point: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ellipsoid_test(scores, point)
