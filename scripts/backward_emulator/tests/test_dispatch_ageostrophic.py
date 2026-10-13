"""Dry-run test for the ageostrophic dispatcher."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dispatch_ageostrophic as da  # noqa: E402


def test_ageostrophic_job_loads_shared_survival_cache(tmp_path):
    rec = da.dispatch("ian", out_root=str(tmp_path), dry_run=True)
    cmd = " ".join(rec["cmd"])
    assert "--case ian" in cmd
    # Reuses the same survival cache the spread dispatcher writes.
    assert f"--survival-npz {tmp_path}/ian/survival_ian.npz" in cmd
    assert "--out-dir" in rec["cmd"] and "ageostrophic" in cmd
    assert "--no-cbottle" not in rec["cmd"]


def test_no_cbottle_flag(tmp_path):
    rec = da.dispatch("pnw_heatwave", out_root=str(tmp_path), cbottle=False,
                      dry_run=True)
    assert "--no-cbottle" in rec["cmd"]
