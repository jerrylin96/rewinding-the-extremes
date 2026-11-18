"""Light-env tests for the ageostrophic comparison driver's pure-numpy parts.

The numeric core (geostrophic wind, regrid) is heavy-env (torch/earth2grid) and is
exercised by the smoke test / the job; here we cover what the npz records (the
survivor mask the figure censors with) and the two fail-loud guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ageostrophic_compare as ac  # noqa: E402


def _models(m=3, n=4, cbottle=True):
    ens = {k: np.full((m, n), 0.2) for k in ac._KEYS}
    base = {k: np.full(n, 0.2) for k in ac._KEYS}
    models = {"makani": {"ens": ens, "baseline": base}}
    if cbottle:
        models["cbottle"] = {"ens": ens, "baseline": base}
    return models


def _save(path, alive, **kw):
    ac._save_npz(path, _models(**kw), lead_hours=np.arange(4) * 6.0, band_label="b",
                 level=500, pinned_frame=3, case_meta={"display_name": "Case"},
                 alive=alive)


def test_npz_carries_the_survivor_mask(tmp_path):
    alive = np.array([[True] * 4, [True] * 4, [False, False, True, True]])
    _save(tmp_path / "a.npz", alive)
    z = np.load(tmp_path / "a.npz")
    np.testing.assert_array_equal(z["makani::alive"], alive)
    assert bool(z["makani::censored"])


def test_uncensored_run_is_recorded_as_such(tmp_path):
    _save(tmp_path / "a.npz", None)
    z = np.load(tmp_path / "a.npz")
    assert z["makani::alive"].all() and not bool(z["makani::censored"])


def test_mask_of_the_wrong_shape_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="survivor mask"):
        _save(tmp_path / "a.npz", np.ones((2, 4), bool))
    assert not (tmp_path / "a.npz").exists()


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
