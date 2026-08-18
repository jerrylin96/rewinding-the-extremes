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

"""Unit tests for the precursor -> impact regression helpers in
``aggregate_synoptic_pca``.

These are the numerics behind the combined ``plot_precursor_impact``
figure: the OLS fit of the scalar impact on the leading PCs (the predicted-
impact index), its cross-validated skill, and the field-on-impact weights
that build the per-unit regression-composite map.  Pure-NumPy, so they are
checked here against closed-form expectations without any zarr I/O or
plotting.

Importing the aggregator pulls in matplotlib + cartopy; the test is skipped
(rather than erroring) where those are unavailable, so it runs in the
analysis environment that renders the figures but does not break a base
test env that omits the plotting stack.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
# The aggregator imports matplotlib + cartopy at module load; skip cleanly
# when the plotting stack is absent rather than failing collection.
pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

SUBMIT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "scripts",
        "ensemble_analysis",
        "submission_scripts",
    )
)
sys.path.insert(0, SUBMIT_DIR)

from aggregate_synoptic_pca import (  # noqa: E402
    _field_impact_regression_weights,
    _kfold_r2,
    _ols_fit,
    _r2_score,
    _regression_f_pvalue,
    _to_display_units,
)


def _orthogonal_zero_mean_columns(n_rows: int, n_cols: int) -> np.ndarray:
    """``[n_rows, n_cols]`` columns that are mutually orthogonal and zero-mean.

    QR of a zero-mean random matrix gives orthonormal columns; re-centering is
    a no-op up to round-off but is applied so the additivity identity below
    holds to tight tolerance.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_rows, n_cols))
    a = a - a.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(a)
    return q - q.mean(axis=0, keepdims=True)


def test_ols_fit_recovers_known_coefficients():
    # Noiseless linear target -> the fit must return the exact coefficients.
    rng = np.random.default_rng(1)
    pcs = rng.standard_normal((500, 3))
    beta_true = np.array([2.0, -3.0, 0.5])
    intercept_true = 5.0
    y = intercept_true + pcs @ beta_true
    beta, intercept = _ols_fit(pcs, y)
    np.testing.assert_allclose(beta, beta_true, atol=1e-9)
    assert intercept == pytest.approx(intercept_true, abs=1e-9)


def test_r2_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r2_score(y, y) == pytest.approx(1.0)
    # Predicting the mean everywhere gives R^2 == 0 by construction.
    assert _r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_orthogonal_predictors_r2_is_sum_of_squared_correlations():
    # The headline statistical claim: with orthogonal predictors the multiple
    # R^2 equals the sum of squared marginal correlations -- so weak single-mode
    # correlations legitimately *add*.  Verify on exactly-orthogonal columns.
    n = 400
    pcs = _orthogonal_zero_mean_columns(n, 4)
    rng = np.random.default_rng(2)
    y = 1.5 * pcs[:, 0] - 0.8 * pcs[:, 2] + 0.3 * rng.standard_normal(n)
    beta, intercept = _ols_fit(pcs, y)
    yhat = intercept + pcs @ beta
    r2 = _r2_score(y, yhat)
    sum_sq_corr = sum(
        float(np.corrcoef(pcs[:, k], y)[0, 1]) ** 2 for k in range(pcs.shape[1])
    )
    assert r2 == pytest.approx(sum_sq_corr, abs=1e-6)


def test_kfold_r2_high_for_strong_signal_low_for_noise():
    rng = np.random.default_rng(3)
    pcs = rng.standard_normal((600, 3))
    # Strong, learnable linear signal -> CV R^2 close to in-sample, both high.
    y_signal = pcs @ np.array([2.0, -1.0, 0.5]) + 0.2 * rng.standard_normal(600)
    beta, intercept = _ols_fit(pcs, y_signal)
    r2_in = _r2_score(y_signal, intercept + pcs @ beta)
    r2_cv = _kfold_r2(pcs, y_signal, n_folds=10)
    assert r2_cv > 0.9
    assert r2_cv <= r2_in + 1e-9
    # Pure-noise target uncorrelated with the predictors -> CV R^2 must not be
    # fooled by in-sample overfitting; it should land near or below zero.
    y_noise = rng.standard_normal(600)
    assert _kfold_r2(pcs, y_noise, n_folds=10) < 0.1


def test_kfold_r2_nan_when_too_few_members():
    pcs = np.zeros((5, 3))
    y = np.arange(5.0)
    assert np.isnan(_kfold_r2(pcs, y, n_folds=10))


def test_field_impact_weights_match_grid_regression():
    # The composite map sum_k c_k EOF_k must equal the per-grid-point regression
    # of the reconstructed field on the impact.  Build a field from known PCs +
    # EOFs, regress each pixel on y directly, and compare to the weighted sum.
    rng = np.random.default_rng(4)
    n, k, npix = 300, 4, 12
    pcs = rng.standard_normal((n, k))
    eofs = rng.standard_normal((k, npix))  # [K, pixels]
    field = pcs @ eofs  # [N, pixels]  (anomaly field, mean handled below)
    y = pcs @ np.array([1.0, 0.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(n)

    c = _field_impact_regression_weights(pcs, y)
    composite = c @ eofs  # [pixels]

    var_y = y.var()
    y_dev = y - y.mean()
    direct = np.array(
        [
            float(((field[:, p] - field[:, p].mean()) * y_dev).mean() / var_y)
            for p in range(npix)
        ]
    )
    np.testing.assert_allclose(composite, direct, atol=1e-9)


def test_field_impact_weights_zero_when_impact_constant():
    pcs = np.random.default_rng(5).standard_normal((50, 3))
    y = np.full(50, 7.0)
    np.testing.assert_array_equal(_field_impact_regression_weights(pcs, y), np.zeros(3))


def test_regression_f_pvalue_properties():
    # Larger R^2 (same n, k) -> smaller p (monotone).
    assert _regression_f_pvalue(0.10, 1000, 8) < _regression_f_pvalue(0.02, 1000, 8)
    # The headline regime (R^2 ~ 0.09, N=1000, k=8) is highly significant: a
    # small effect size is not the same as a null relationship.
    assert _regression_f_pvalue(0.09, 1000, 8) < 1e-3
    # A near-zero R^2 is not significant.
    assert _regression_f_pvalue(0.001, 1000, 8) > 0.1
    # Degenerate inputs -> NaN.
    assert np.isnan(_regression_f_pvalue(0.0, 1000, 8))
    assert np.isnan(_regression_f_pvalue(0.5, 8, 8))  # n <= k + 1


def test_to_display_units_skips_empty_optional_fields():
    # The decile-composite fields are empty on a non-scalar (TC) impact case;
    # _to_display_units must not try to slice their (absent) variable axis when
    # the precursor is geopotential -- otherwise it IndexErrors the whole load.
    from var_metadata import DISPLAY_GRAVITY  # noqa: E402

    era5 = np.full((2, 2, 3, 4), DISPLAY_GRAVITY)  # [n_leads, n_vars, lat, lon]
    diag = {
        "variables": np.array(["z500", "t2m"]),
        "era5_trajectory_latlon": era5.copy(),
        "impact_decile_warm_latlon": np.zeros((0, 0, 0), dtype=np.float32),
    }
    _to_display_units(diag)  # must not raise
    # z500 (var 0) is converted m^2/s^2 -> m (divide by g); t2m (var 1) untouched.
    np.testing.assert_allclose(diag["era5_trajectory_latlon"][:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        diag["era5_trajectory_latlon"][:, 1], DISPLAY_GRAVITY, atol=1e-6
    )
    assert diag["impact_decile_warm_latlon"].size == 0
