"""Dependency-light regression / rank-correlation helpers for PCA diagnostics.

Shared by the synoptic-PCA compute + aggregation + track-stats scripts so the
precursor -> impact figure, the TC track-regression figure, and the track-stats
report all quote the *same* statistics from one definition. Pure NumPy except
for the F-test p-value (scipy, a declared earth2studio dependency, imported
lazily); no cartopy / matplotlib, so a stats-only login-node run stays cheap
to import.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

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


class EllipsoidTest(NamedTuple):
    """Where a point sits relative to the members' ellipsoid (``ellipsoid_test``).

    ``n`` members; ``d2_point`` / ``d2_max`` are the point's and the farthest
    member's squared Mahalanobis distances from the members' mean; ``m`` is the
    number of members at least as far as the point.
    """

    n: int
    d2_point: float
    d2_max: float
    m: int

    @property
    def inside(self) -> bool:
        """At least one member is at least as far from the center."""
        return self.m >= 1

    @property
    def p_value(self) -> float:
        """``(m+1)/(n+1)``: if the point were one more draw like the members, the
        chance it would rank at least this far out among all ``n+1``."""
        return (self.m + 1) / (self.n + 1)


def ellipsoid_test(scores: np.ndarray, point: np.ndarray) -> EllipsoidTest:
    """Is ``point`` within the members' range, measured by Mahalanobis distance?

    ``scores`` is ``[N, K]`` (one row per member, e.g. 8 PC scores); ``point``
    is ``[K]`` (e.g. ERA5's scores).  Each row's squared distance is
    ``d^2 = (x - xbar)^T S^-1 (x - xbar)`` with the rows' mean and sample
    covariance (full, not just the variances: PCs are uncorrelated over the
    population they were fit on, only approximately over a subset of it).
    The point is inside iff some member is at least as far out.

    This is the regression extrapolation check: with an intercept, OLS
    leverage is ``h = 1/N + d^2/(N-1)``, so inside means the point's leverage
    is at most the largest member's.  The convex hull is not used: in 8-D
    many ordinary members fall outside the hull of the rest.

    The caller chooses the rows; non-finite input raises rather than being
    dropped, so ``N`` is always the population the caller meant.
    """
    x = np.asarray(scores, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    if x.ndim != 2 or p.shape != (x.shape[1],):
        raise ValueError(f"scores {x.shape} and point {p.shape} do not match")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(p))):
        raise ValueError("scores and point must be finite")
    n, k = x.shape
    if n <= k:
        raise ValueError(f"need more members than dimensions, got {n} <= {k}")
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False).reshape(k, k)
    centered = np.vstack([x, p]) - mean
    d2 = np.einsum("ij,ji->i", centered, np.linalg.solve(cov, centered.T))
    d2_members, d2_point = d2[:-1], float(d2[-1])
    return EllipsoidTest(
        n=n,
        d2_point=d2_point,
        d2_max=float(d2_members.max()),
        m=int((d2_members >= d2_point).sum()),
    )
