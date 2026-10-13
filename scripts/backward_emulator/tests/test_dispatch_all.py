"""Dry-run test for the one-shot overnight orchestrator.

Checks the dependency wiring: one survivor root per case, and spread/ageo/PCA all
holding on it; Ian additionally chains its PCA on the TC-tracks job.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dispatch_all as da  # noqa: E402


def _held_on(rec, jid_placeholder):
    cmd = " ".join(rec["cmd"])
    return "-hold_jid" in rec["cmd"] and jid_placeholder in cmd


def test_shared_survivor_root_and_holds(tmp_path):
    summary = da.run(["pnw_heatwave", "ian"], out_root=str(tmp_path), dry_run=True)

    # PNW: one survivor root; spread/ageo/pca hold on it; no TC.
    pnw = summary["pnw_heatwave"]
    root = pnw["survival_jid"]
    assert root == "<Surv_pnw_heatwave>"
    assert pnw["tc"] is None
    assert all(_held_on(r, root) for r in pnw["spread"] if r["role"] == "compare")
    assert _held_on(pnw["ageo"], root)
    assert _held_on(pnw["pca"], root)

    # Ian: TC job present; PCA holds on BOTH the survivor root and the TC job.
    ian = summary["ian"]
    root_i = ian["survival_jid"]
    assert ian["tc"] is not None
    tc_jid = ian["tc"]["jid"] or "<TC_ian>"
    pca_cmd = " ".join(ian["pca"]["cmd"])
    assert root_i in pca_cmd and tc_jid in pca_cmd
    assert "--impact-kind track" in pca_cmd


def test_preflight_refuses_when_land_mask_missing(tmp_path):
    # Real submit (dry_run=False) must abort BEFORE any qsub when PNW's land mask
    # is absent, rather than orphan a survivor root.
    import pytest
    with pytest.raises(SystemExit, match="land mask"):
        da.run(["pnw_heatwave"], out_root=str(tmp_path), dry_run=False)


def test_only_one_survivor_root_per_case(tmp_path):
    # The per-case sub-dispatchers must NOT submit their own survivor root when a
    # shared one is threaded in (else we'd double the 50 GB scan).
    summary = da.run(["pnw_heatwave"], out_root=str(tmp_path), dry_run=True)
    roles = [r["role"] for r in summary["pnw_heatwave"]["spread"]]
    assert roles.count("survival") == 0  # dispatch_all owns the root, not dispatch_compare
