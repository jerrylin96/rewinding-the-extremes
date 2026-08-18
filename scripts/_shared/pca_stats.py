"""Dependency-light regression / rank-correlation helpers for PCA diagnostics.

Shared by the synoptic-PCA compute + aggregation + track-stats scripts so the
precursor -> impact figure, the TC track-regression figure, and the track-stats
report all quote the *same* statistics from one definition. Pure NumPy except
for the F-test p-value (scipy, a declared earth2studio dependency, imported
lazily); no cartopy / matplotlib, so a stats-only login-node run stays cheap
to import.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays (dependency-free).

    Ranks both inputs (``argsort``-``argsort``) and Pearson-correlates the
    ranks, so no scipy is needed.  Returns NaN when fewer than three finite
    pairs remain or either ranking is constant (undefined correlation).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 3:
        return float("nan")
    af = a[finite]
    bf = b[finite]
    # A constant input has no variance, so the correlation is undefined --
    # guard on the raw values (``argsort``-``argsort`` would otherwise
    # manufacture spurious distinct ranks for it and return a fake +-1).
    if af.std() == 0.0 or bf.std() == 0.0:
        return float("nan")
    ra = np.argsort(np.argsort(af)).astype(np.float64)
    rb = np.argsort(np.argsort(bf)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _ols_fit(predictors: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS fit ``target ~= intercept + predictors @ beta``.

    ``predictors`` is ``[N, K]``, ``target`` is ``[N]``.  Returns
    ``(beta [K], intercept)``.  ``lstsq`` is used for the intercept and to
    stay robust if a near-degenerate mode appears.
    """
    design = np.column_stack([np.ones(predictors.shape[0]), predictors])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef[1:].astype(np.float64), float(coef[0])


def _r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot`` (about the mean)."""
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_res = float(np.sum((target - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _kfold_r2(predictors: np.ndarray, target: np.ndarray, n_folds: int = 10) -> float:
    """Out-of-sample R^2 of the ``target ~ predictors`` OLS fit (k-fold CV).

    Folds are assigned deterministically (member ``i`` -> fold ``i % n_folds``)
    so the score is reproducible run-to-run without seeding an RNG.  Each
    fold's held-out members are predicted from a fit on the rest; the R^2 is
    computed once on the stacked held-out predictions (about the global mean).
    Returns NaN when there are too few members to fit a fold.
    """
    n = int(target.shape[0])
    k = int(predictors.shape[1])
    if n < 2 * n_folds or n <= k + 1:
        return float("nan")
    fold = np.arange(n) % n_folds
    pred = np.empty(n, dtype=np.float64)
    for f in range(n_folds):
        test = fold == f
        train = ~test
        if int(train.sum()) <= k + 1:
            return float("nan")
        beta, intercept = _ols_fit(predictors[train], target[train])
        pred[test] = intercept + predictors[test] @ beta
    return _r2_score(target, pred)


def _regression_f_pvalue(r2: float, n: int, k: int) -> float:
    """F-test p-value for an OLS regression (H0: all ``k`` slopes are zero).

    ``r2`` is the in-sample R^2, ``n`` the sample size, ``k`` the number of
    predictors (excluding the intercept).  Returns the probability of an R^2
    this large under no relationship.  Uses scipy (a project dependency);
    NaN when the statistic is undefined.
    """
    if not 0.0 < r2 < 1.0 or n <= k + 1 or k < 1:
        return float("nan")
    from scipy import stats  # scipy is a declared earth2studio dependency

    df_resid = n - k - 1
    f_stat = (r2 / k) / ((1.0 - r2) / df_resid)
    return float(stats.f.sf(f_stat, k, df_resid))
