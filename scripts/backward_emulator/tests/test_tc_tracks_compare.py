"""Light-env tests for the makani TC-tracking driver's pure-numpy seam.

The tracking itself is heavy-env (TempestExtremes + earth2studio) and is covered by
`smoke_tc_tracks.py` / the job; here we cover the slab builder (the makani-source ->
tracker contract) and the dispatcher's dry-run wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tc_tracks_compare as tt  # noqa: E402
import dispatch_tc_tracks_compare as dtc  # noqa: E402


def test_build_member_slab_orders_and_shapes():
    T, H, W = 12, 4, 5
    fields = {
        "msl": np.full((T, H, W), 1.0),
        "u10m": np.full((T, H, W), 2.0),
        "v10m": np.full((T, H, W), 3.0),
        "z300": np.full((T, H, W), 4.0),
        "z500": np.full((T, H, W), 5.0),
    }
    variables = ["msl", "u10m", "v10m", "z300", "z500"]
    slab = tt.build_member_slab(fields, variables)
    assert slab.shape == (T, len(variables), H, W)  # [T, V, H, W]
    # Variable order is preserved along axis 1.
    for v_idx, name in enumerate(variables):
        assert np.all(slab[:, v_idx] == fields[name][:, 0, 0].reshape(-1)[0])


def test_build_member_slab_missing_channel_raises():
    fields = {"msl": np.zeros((2, 3, 3)), "u10m": np.zeros((2, 3, 3))}
    with pytest.raises(KeyError):
        tt.build_member_slab(fields, ["msl", "u10m", "v10m"])


def test_dispatch_tc_defaults_to_ian(tmp_path):
    rec = dtc.dispatch("ian", out_root=str(tmp_path), dry_run=True)
    cmd = " ".join(rec["cmd"])
    assert "--case ian" in cmd
    assert "tc_tracks" in cmd
    assert "--makani-dir" in rec["cmd"] and "/ian" in cmd
