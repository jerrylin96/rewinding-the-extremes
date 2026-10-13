"""Light-env tests for the ageostrophic comparison driver's pure-numpy parts.

The numeric core (geostrophic wind, regrid) is heavy-env (torch/earth2grid) and is
exercised by the smoke test / the job; here we cover the survivor-censoring summary
that turns per-member traces into the plotted ensemble mean/spread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ageostrophic_compare as ac  # noqa: E402


def test_censored_summary_ignores_dead_member_leads():
    # 3 members, 2 leads. Member 2 is dead at lead 1 (alive=False there).
    traces = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 100.0]], dtype=float)  # 100 = diverged garbage
    alive = np.array([[True, True],
                      [True, True],
                      [True, False]], dtype=bool)
    mean, std = ac._censored_summary(traces, alive)
    # lead 0: mean of {1,3,5}=3; lead 1: mean of {2,4}=3 (100 censored).
    np.testing.assert_allclose(mean, [3.0, 3.0])
    assert std[1] < 2.0  # without censoring the 100 would blow the spread up


def test_censored_summary_none_uses_all_members():
    traces = np.array([[1.0], [3.0], [5.0]])
    mean, std = ac._censored_summary(traces, None)
    np.testing.assert_allclose(mean, [3.0])
    np.testing.assert_allclose(std, [np.std([1, 3, 5], ddof=1)])


def test_censored_summary_all_dead_lead_is_nan():
    traces = np.array([[1.0, 2.0], [3.0, 4.0]])
    alive = np.array([[True, False], [True, False]])
    mean, _ = ac._censored_summary(traces, alive)
    assert np.isfinite(mean[0]) and np.isnan(mean[1])
