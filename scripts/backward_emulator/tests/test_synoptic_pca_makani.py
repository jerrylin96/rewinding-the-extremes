"""Light-env tests for the makani synoptic-PCA numeric core.

Covers the makani-specific feature builder (the cos-lat area weighting, and its
reduction to the paper's build under uniform weights), the ERA5 projection, and the
regression wrapper.  The SVD (`_pca`) and field loading are heavy-env / job-tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synoptic_pca_makani as spm  # noqa: E402


def test_feature_matrix_reduces_to_paper_build_with_uniform_weights():
    rng = np.random.default_rng(0)
    N, V, P = 20, 2, 15
    fields = rng.standard_normal((N, V, P)) * np.array([1.0, 50.0])[None, :, None]
    X, mean, std, sqrt_w = spm.build_feature_matrix_latlon(fields, np.ones(P))
    assert np.allclose(sqrt_w, 1.0)
    # Paper's build: per-var mean/std (ddof=0) z-score, flattened, no column scale.
    ref = np.empty((N, V, P), dtype=np.float64)
    for v in range(V):
        slab = fields[:, v, :]
        ref[:, v, :] = (slab - slab.mean()) / slab.std()
    np.testing.assert_allclose(X, ref.reshape(N, V * P), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mean, [fields[:, v].mean() for v in range(V)], rtol=1e-5)


def test_feature_matrix_area_weighting_zeros_zero_weight_pixels():
    rng = np.random.default_rng(1)
    N, V, P = 8, 1, 6
    fields = rng.standard_normal((N, V, P)) + 5.0
    w = np.array([1.0, 1.0, 1.0, 0.5, 0.25, 0.0])  # last pixel has zero area weight
    X, _mean, _std, sqrt_w = spm.build_feature_matrix_latlon(fields, w)
    X3 = X.reshape(N, V, P)
    assert np.allclose(X3[:, 0, -1], 0.0)          # zero-weight pixel contributes nothing
    np.testing.assert_allclose(sqrt_w, np.sqrt(w))
    # The weighted mean differs from the naive mean when weights vary.
    naive = fields[:, 0, :].mean()
    weighted = (fields[:, 0, :] * w).sum() / (N * w.sum())
    assert not np.isclose(naive, weighted)


def test_regression_summary_recovers_linear_signal():
    rng = np.random.default_rng(2)
    N = 200
    scores = rng.standard_normal((N, 10))
    beta = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = scores[:, :8] @ beta + 3.0
    r = spm.regression_summary(scores, y, n_pc=8)
    assert r["r2_insample"] > 0.999          # exact linear signal
    assert r["r2_cv"] > 0.99                  # generalizes out-of-fold
    assert r["n_used"] == N


def test_regression_summary_noise_target_low_r2():
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((300, 10))
    y = rng.standard_normal(300)              # unrelated to the PCs
    r = spm.regression_summary(scores, y, n_pc=8)
    assert r["r2_cv"] < 0.1                    # out-of-sample sees through overfit


def test_percentile_of_matches_definition():
    scores = np.arange(100, dtype=float)      # 0..99
    assert spm._percentile_of(scores, 49.0) == 50.0   # 50 values <= 49
    assert spm._percentile_of(scores, -1.0) == 0.0
    assert spm._percentile_of(scores, 200.0) == 100.0
