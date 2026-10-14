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


def test_ageostrophic_refuses_shifted_dump_and_fails_loud_without_cbottle(tmp_path, monkeypatch):
    """Same two guards as the spread/RMSE/CRPS driver: a dump anchored on the wrong
    date is refused before any output, and a failing cBottle overlay raises."""
    import builtins
    import sys
    from pathlib import Path

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_compare_spread_rmse_crps import _write_case
    import ageostrophic_compare as ac

    shifted = tmp_path / "pnw_shifted"
    _write_case(shifted, anchor_ts=b"2021-06-29T12")
    with pytest.raises(ValueError, match="window end"):
        ac.main(["--makani-dir", str(shifted), "--case", "pnw_heatwave",
                 "--out-dir", str(tmp_path / "o1"), "--no-cbottle", "--no-censor"])
    assert not (tmp_path / "o1").exists()

    good = tmp_path / "pnw"
    _write_case(good)
    # The makani side needs the heavy stack (torch + the paper's compute module,
    # which both the domain setup and the per-member loop import); stub both so
    # this test is about the overlay guard alone.
    import numpy as np

    def fake_domain(lat, lon, args):
        return np.arange(lat.size), np.arange(lon.size), np.ones(lat.size), "test band"

    def fake_members(makani, names, lat, lon, lat_idx, lon_idx, lat_weights):
        ens = {k: np.ones((makani.n_members, makani.n_frames)) for k in ac._KEYS}
        base = {k: np.ones(makani.n_frames) for k in ac._KEYS}
        return ens, base

    monkeypatch.setattr(ac, "_domain_setup", fake_domain)
    monkeypatch.setattr(ac, "_makani_members", fake_members)
    real_import = builtins.__import__

    def no_cbottle(name, *a, **k):
        if name == "cbottle_source":
            raise ImportError("heavy env missing (simulated)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cbottle)
    with pytest.raises(ImportError, match="simulated"):
        ac.main(["--makani-dir", str(good), "--case", "pnw_heatwave",
                 "--out-dir", str(tmp_path / "o2"), "--no-censor"])
    assert not (tmp_path / "o2").exists()
