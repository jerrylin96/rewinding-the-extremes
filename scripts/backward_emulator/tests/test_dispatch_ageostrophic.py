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


def test_submits_its_own_survivor_job_and_holds_on_it(tmp_path, capsys):
    rec = da.dispatch("sandy", out_root=str(tmp_path), dry_run=True)
    assert rec["survival_jid"] == "<Surv_sandy>"
    assert "-hold_jid" in rec["cmd"] and "<Surv_sandy>" in rec["cmd"]
    assert "-N Surv_sandy" in capsys.readouterr().out


def test_external_survivor_job_is_held_on_and_none_submitted(tmp_path, capsys):
    rec = da.dispatch("ian", out_root=str(tmp_path), survival_hold_jid="123",
                      dry_run=True)
    assert rec["cmd"][rec["cmd"].index("-hold_jid") + 1] == "123"
    assert "Surv_ian" not in capsys.readouterr().out


def test_failed_survivor_submit_queues_nothing(tmp_path, monkeypatch):
    import dispatch_compare as dc
    import pytest

    submitted = []

    def fake_submit(cmd, **kw):
        submitted.append(cmd)
        return None  # qsub refused

    monkeypatch.setattr(dc, "submit_qsub", fake_submit)
    monkeypatch.setattr(da, "submit_qsub", fake_submit)
    with pytest.raises(SystemExit, match="survivor-mask job for ian"):
        da.dispatch("ian", out_root=str(tmp_path), dry_run=False)
    assert len(submitted) == 1  # the survivor job only; no Ageo job behind it
