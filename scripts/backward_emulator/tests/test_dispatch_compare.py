"""Dry-run tests for the comparison dispatcher.

Exercises the job matrix off the analysis HPC: reads the real pnw_heatwave / ian
case YAMLs, resolves each variable's (region, mask) from
``diagnostics.spread_rmse_crps.domains``, and checks the emitted qsub commands
without submitting anything (dry_run skips the zarr/land-mask pre-flight).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dispatch_compare as dc  # noqa: E402


def _by_role(recs, role):
    return [r for r in recs if r["role"] == role]


def _cmd_str(rec):
    return " ".join(rec["cmd"])


def test_pnw_matrix_has_survival_root_and_land_masked_t2m(tmp_path):
    recs = dc.dispatch("pnw_heatwave", out_root=str(tmp_path), dry_run=True)

    surv = _by_role(recs, "survival")
    assert len(surv) == 1
    root = surv[0]
    assert root["jid"] == "<Surv_pnw_heatwave>"
    assert "--out" in root["cmd"]
    surv_npz = root["cmd"][root["cmd"].index("--out") + 1]

    comps = _by_role(recs, "compare")
    assert {c["variable"] for c in comps} == {"z500", "msl", "t2m"}

    for c in comps:
        s = _cmd_str(c)
        # Every compare job loads the cached mask and holds on the root.
        assert surv_npz in c["cmd"]
        assert "--survival-npz" in c["cmd"]
        assert "-hold_jid" in c["cmd"]
        assert "<Surv_pnw_heatwave>" in c["cmd"]

    t2m = next(c for c in comps if c["variable"] == "t2m")
    assert t2m["region"] == "impact" and t2m["mask"] == "land"
    assert "--land-mask" in t2m["cmd"]
    assert "land_mask_pnw_heatwave_721x1440.npy" in _cmd_str(t2m)

    z500 = next(c for c in comps if c["variable"] == "z500")
    assert z500["region"] == "synoptic" and z500["mask"] == "none"
    assert "--land-mask" not in z500["cmd"]


def test_no_cbottle_flag_propagates(tmp_path):
    recs = dc.dispatch("ian", out_root=str(tmp_path), cbottle=False, dry_run=True)
    for c in _by_role(recs, "compare"):
        assert "--no-cbottle" in c["cmd"]


def test_variable_override_limits_matrix(tmp_path):
    recs = dc.dispatch("pnw_heatwave", out_root=str(tmp_path),
                       variables=["z500"], dry_run=True)
    comps = _by_role(recs, "compare")
    assert [c["variable"] for c in comps] == ["z500"]
