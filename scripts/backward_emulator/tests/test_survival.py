"""Tests for survivor censoring: the range detector, monotone masking, and the
forward/rollout ordering (the subtle bit -- the pinned present must stay alive
while divergence grows into the past).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from makani_source import MakaniBackwardSource  # noqa: E402
from survival import RANGE_DEFAULT, range_alive_mask  # noqa: E402

_CH = ["t2m", "z500", "msl", "u500", "v500", "z300", "u10m", "v10m"]


def _write_source(dirpath: Path, *, blowups, M, L=6, H=3, W=4):
    """One dump in makani (dump) order. ``blowups[m]`` = first makani-order lead
    (0=present, increasing into past) at which member m's t2m jumps to 400 K
    (> the 350 K ceiling); None = never diverges. truth t2m is a physical 290 K.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    C = len(_CH)
    truth = np.full((1, L, C, H, W), 290.0, np.float32)
    pred = np.full((1, M, L, C, H, W), 290.0, np.float32)
    ti = _CH.index("t2m")
    for m in range(M):
        r = blowups.get(m)
        if r is not None:
            pred[0, m, r:, ti] = 400.0  # blown-up from makani-lead r into the past
    pred[0, :, 0] = truth[0, 0]  # IC: present frame pred == truth (in bounds)
    with h5py.File(dirpath / "backward_fields_000.h5", "w") as f:
        f.create_dataset("truth", data=truth)
        f.create_dataset("pred", data=pred)
        f.create_dataset("channel", data=np.array([c.encode() for c in _CH], dtype="S64"))
        f.create_dataset("lead_hours_into_past", data=np.arange(L, dtype=np.int64) * 6)
        f.create_dataset("lat", data=np.linspace(80, -80, H))
        f.create_dataset("lon", data=np.linspace(0, 360, W, endpoint=False))
        f.create_dataset("timestamps", data=np.array([[b"2022-09-30T18"] * L], dtype="S64"))
        f.create_dataset("anchor_index", data=np.array([0]))
        f.create_dataset("member_index", data=np.arange(M))
        f.attrs["step_hours"] = -6
        f.attrs["direction"] = "backward"


def test_range_default_matches_makani():
    assert RANGE_DEFAULT["t2m"] == (150.0, 350.0)


def test_clean_ensemble_all_survive(tmp_path):
    _write_source(tmp_path / "c", blowups={0: None, 1: None, 2: None}, M=3)
    src = MakaniBackwardSource(tmp_path / "c", n_frames=6)
    res = range_alive_mask(src, channel="t2m")
    assert res["n_diverged"] == 0
    assert np.all(res["alive"])
    np.testing.assert_array_equal(res["survival"], np.ones(6))


def test_divergence_censors_past_not_present(tmp_path):
    # member 1 blows up from makani-lead 3 into the past; member 0 clean.
    _write_source(tmp_path / "c", blowups={0: None, 1: 3}, M=2, L=6)
    src = MakaniBackwardSource(tmp_path / "c", n_frames=6)
    res = range_alive_mask(src, channel="t2m")

    # forward order: makani-lead r maps to forward frame L-1-r. Fail at r=3 ->
    # dead at makani leads 3,4,5 -> dead at forward frames 2,1,0; alive 3,4,5.
    expected_alive_m1 = np.array([False, False, False, True, True, True])
    np.testing.assert_array_equal(res["alive"][0], np.ones(6, bool))   # clean member
    np.testing.assert_array_equal(res["alive"][1], expected_alive_m1)
    # The pinned present (forward frame -1) is alive for BOTH members.
    assert res["alive"][:, -1].all()
    # fail_leads recorded in leads-into-past.
    assert res["fail_leads"][0] == -1 and res["fail_leads"][1] == 3
    # survival fraction: forward frames 0-2 have 1/2 alive, 3-5 have 2/2.
    np.testing.assert_array_equal(res["survival"], [0.5, 0.5, 0.5, 1.0, 1.0, 1.0])


def test_nonfinite_also_censors(tmp_path):
    d = tmp_path / "c"
    _write_source(d, blowups={0: None, 1: None}, M=2, L=6)
    # inject a NaN into member 1 at makani-lead 4 (into the past).
    with h5py.File(d / "backward_fields_000.h5", "r+") as f:
        ti = _CH.index("t2m")
        f["pred"][0, 1, 4, ti] = np.nan
    src = MakaniBackwardSource(d, n_frames=6)
    res = range_alive_mask(src, channel="t2m")
    # dead from makani-lead 4 -> forward frames 0,1 dead; 2..5 alive.
    np.testing.assert_array_equal(res["alive"][1],
                                  [False, False, True, True, True, True])


def test_main_n_frames_zero_scans_all_leads(tmp_path, capsys):
    from survival import main

    _write_source(tmp_path / "c", blowups={0: None, 1: 3}, M=2, L=6)
    out = tmp_path / "surv.npz"
    main(["--makani-dir", str(tmp_path / "c"), "--out", str(out),
          "--n-frames", "0"])
    with np.load(out) as d:
        assert d["survival"].size == 6
        assert d["lead_hours"].size == 6
    logged = capsys.readouterr().out
    assert "scanning all 6 leads" in logged
    assert "at the window start (30 h before the anchor)" in logged
