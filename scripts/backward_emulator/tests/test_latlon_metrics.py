"""Tests for the shared lat/lon spread/RMSE/CRPS reducer.

The load-bearing checks are analytic:
* the exact bias-variance identity member_rmse**2 == ((N-1)/N) spread**2 + skill**2,
* CRPS against a brute-force O(N^2) reference,
* a hand-computable constant-error / zero-spread case,
* cos(lat) weighting and region selection behaving as specified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latlon_metrics import (  # noqa: E402
    _empirical_crps,
    cos_lat_weights,
    reduce_selected,
    select_pixels,
    spread_rmse_crps_latlon,
)


def _grid(H=6, W=8):
    lat = np.linspace(85, -85, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    return lat, lon


def test_bias_variance_identity():
    rng = np.random.default_rng(0)
    N, L, H, W = 7, 4, 6, 8
    lat, lon = _grid(H, W)
    ens = rng.standard_normal((N, L, H, W)) * 3 + 280
    truth = rng.standard_normal((L, H, W)) * 3 + 280
    r = spread_rmse_crps_latlon(ens, truth, lat, lon)
    lhs = r["member_rmse"] ** 2
    rhs = ((N - 1) / N) * r["spread"] ** 2 + r["skill"] ** 2
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_crps_matches_bruteforce():
    rng = np.random.default_rng(1)
    N, L, P = 9, 3, 5
    ens = rng.standard_normal((N, L, P))
    truth = rng.standard_normal((L, P))
    got = _empirical_crps(ens, truth)
    # brute force: mean_i|x_i-y| - 1/(2N^2) sum_{i,j}|x_i-x_j|
    term1 = np.abs(ens - truth[None]).mean(axis=0)
    pair = np.abs(ens[:, None] - ens[None, :]).sum(axis=(0, 1)) / (2 * N * N)
    np.testing.assert_allclose(got, term1 - pair, rtol=1e-12, atol=1e-12)


def test_constant_error_zero_spread():
    # Every member identical, offset +5 from truth: spread 0, skill 5,
    # member_rmse 5, mean_error +5, and CRPS = |offset| = 5.
    N, L, H, W = 4, 2, 6, 8
    lat, lon = _grid(H, W)
    truth = np.full((L, H, W), 100.0)
    ens = np.full((N, L, H, W), 105.0)
    r = spread_rmse_crps_latlon(ens, truth, lat, lon)
    np.testing.assert_allclose(r["spread"], 0.0, atol=1e-12)
    np.testing.assert_allclose(r["skill"], 5.0, atol=1e-10)
    np.testing.assert_allclose(r["member_rmse"], 5.0, atol=1e-10)
    np.testing.assert_allclose(r["mean_error"], 5.0, atol=1e-10)
    np.testing.assert_allclose(r["crps"], 5.0, atol=1e-10)
    assert r["n_pix"] == H * W


def test_cos_lat_weights_and_normalization():
    lat = np.array([0.0, 60.0])
    w = cos_lat_weights(lat)
    np.testing.assert_allclose(w, [1.0, 0.5], atol=1e-12)
    # select_pixels normalizes to sum 1 over kept pixels.
    lon = np.array([0.0, 180.0])
    _, wsel = select_pixels(lat, lon)
    np.testing.assert_allclose(wsel.sum(), 1.0, atol=1e-12)


def test_region_selection_and_dateline_wrap():
    lat = np.linspace(80, -80, 9)     # includes 0
    lon = np.linspace(0, 360, 12, endpoint=False)  # 0,30,...,330
    # Non-wrapped box: western hemisphere, tropics.
    flat, w = select_pixels(lat, lon, region=(-120, -60, -20, 20))
    assert flat.size > 0 and abs(w.sum() - 1.0) < 1e-12
    # Wrapped box across the dateline picks the +/-180 neighborhood.
    flatw, _ = select_pixels(lat, lon, region=(150, -150, -20, 20))
    # 150E and 210(=-150) columns exist -> at least the tropics rows selected.
    assert flatw.size > 0


def test_weight_mask_restricts_domain():
    lat, lon = _grid(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0] = True  # keep only the top row
    flat, w = select_pixels(lat, lon, weight_mask=mask)
    assert flat.size == 4
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_alive_all_true_matches_no_mask():
    rng = np.random.default_rng(5)
    N, L, P = 8, 4, 6
    ens = rng.standard_normal((N, L, P)) * 2 + 5
    truth = rng.standard_normal((L, P)) * 2 + 5
    w = np.full(P, 1.0 / P)
    r_none = reduce_selected(ens, truth, w)
    r_alive = reduce_selected(ens, truth, w, alive=np.ones((N, L), bool))
    for k in ("spread", "skill", "member_rmse", "mean_error", "crps"):
        np.testing.assert_allclose(r_none[k], r_alive[k], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(r_alive["survival"], np.ones(L))
    np.testing.assert_array_equal(r_alive["n_alive"], np.full(L, N))


def test_censoring_reduces_over_survivors_only():
    # Two members; member 1 is "dead" at lead 0 and huge there. With censoring,
    # lead 0 must reduce over member 0 alone (spread NaN, skill from member 0).
    N, L, P = 2, 2, 3
    ens = np.zeros((N, L, P))
    ens[0] = 10.0
    ens[1] = 10.0
    ens[1, 0] = 1e6  # member 1 blows up at lead 0
    truth = np.full((L, P), 9.0)
    w = np.full(P, 1.0 / P)
    alive = np.ones((N, L), bool)
    alive[1, 0] = False  # censor member 1 at lead 0
    r = reduce_selected(ens, truth, w, alive=alive)
    # lead 0: only member 0 (value 10, truth 9) -> skill 1, member_rmse 1,
    #         mean_error +1, spread NaN (n<2), survival 0.5.
    assert np.isnan(r["spread"][0])
    np.testing.assert_allclose(r["skill"][0], 1.0)
    np.testing.assert_allclose(r["member_rmse"][0], 1.0)
    np.testing.assert_allclose(r["mean_error"][0], 1.0)
    np.testing.assert_allclose(r["crps"][0], 1.0)  # single member -> |x-y|
    assert r["survival"][0] == 0.5 and r["n_alive"][0] == 1
    # lead 1: both alive, both 10 vs truth 9 -> spread 0, skill 1.
    np.testing.assert_allclose(r["spread"][1], 0.0)
    np.testing.assert_allclose(r["skill"][1], 1.0)
    assert r["survival"][1] == 1.0


def test_bias_variance_identity_under_censoring():
    rng = np.random.default_rng(6)
    N, L, P = 9, 3, 5
    ens = rng.standard_normal((N, L, P)) * 3 + 100
    truth = rng.standard_normal((L, P)) * 3 + 100
    w = np.full(P, 1.0 / P)
    alive = rng.random((N, L)) > 0.3  # random survivors, >=2 per lead likely
    # ensure >=2 alive per lead for a defined spread
    for l in range(L):
        if alive[:, l].sum() < 2:
            alive[:2, l] = True
    r = reduce_selected(ens, truth, w, alive=alive)
    for l in range(L):
        nl = int(alive[:, l].sum())
        lhs = r["member_rmse"][l] ** 2
        rhs = ((nl - 1) / nl) * r["spread"][l] ** 2 + r["skill"][l] ** 2
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)


def test_degenerate_pixel_count_excludes_conditioning():
    # One pixel flat across members at all leads -> degenerate, unless that lead
    # set is entirely conditioning frames.
    N, L, H, W = 5, 3, 2, 2
    lat, lon = _grid(H, W)
    rng = np.random.default_rng(3)
    ens = rng.standard_normal((N, L, H, W))
    ens[:, :, 0, 0] = 7.0  # pixel (0,0) identical across members at every lead
    truth = np.zeros((L, H, W))
    r = spread_rmse_crps_latlon(ens, truth, lat, lon)
    assert r["n_pix_degenerate"] == 1
    # If every free lead is a conditioning frame, there are no free leads to judge.
    r2 = spread_rmse_crps_latlon(
        ens, truth, lat, lon, conditioning_frames=np.arange(L)
    )
    assert r2["n_pix_degenerate"] == 0
