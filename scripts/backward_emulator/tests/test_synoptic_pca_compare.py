"""Light-env tests for the PCA compare reader + dispatcher.

Covers the two-schema npz normaliser (makani vs the paper's cBottle layout) and the
R^2 recompute, plus the dispatcher's per-case wiring (scalar+land-mask for PNW,
track+parquet for Ian, and the TC hold).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synoptic_pca_compare as spc  # noqa: E402
import dispatch_synoptic_pca_makani as dpca  # noqa: E402


def _makani_npz(path, N=200):
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((N, 10))
    y = scores[:, :8] @ np.array([2.0, -1, .5, 0, 0, 0, 0, 0]) + 1.0
    np.savez(path, model="makani", case_name="Test", free_end_frame=0,
             impact_kind="scalar",
             pc_scores=scores.astype(np.float32),
             explained_variance_ratio=np.linspace(0.4, 0.0, 10).astype(np.float32),
             era5_pc_percentile=np.full(8, 50.0, np.float32),
             target_scalar=y.astype(np.float32), target_names=np.array(["t2m"]))


def _cbottle_npz(path, N=200, L=12, case_name="Test"):
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((N, 10))
    mis = rng.standard_normal((N, L)).astype(np.float32)
    mis[:, 0] = scores[:, :8] @ np.array([1.0, 1, 0, 0, 0, 0, 0, 0])  # free-end col
    np.savez(path, model="cbottle", case_name=case_name, free_end_frame=0,
             impact_kind="scalar",
             pc_scores=scores.astype(np.float32),
             explained_variance_ratio=np.linspace(0.3, 0.0, 10).astype(np.float32),
             era5_pc_percentile=np.full(8, 40.0, np.float32),
             member_impact_scalar=mis, impact_variable="t2m")


def test_read_normalises_both_schemas(tmp_path):
    mk = tmp_path / "mk.npz"; cb = tmp_path / "cb.npz"
    _makani_npz(mk); _cbottle_npz(cb)
    m = spc._read_pca_npz(str(mk))
    c = spc._read_pca_npz(str(cb))
    assert m["model"] == "makani" and "t2m" in m["targets"]
    assert m["pc_scores"].shape == (200, 10)
    # cBottle target is member_impact_scalar at the free-end frame.
    assert "t2m" in c["targets"] and c["targets"]["t2m"].shape == (200,)


def test_r2_table_both_models(tmp_path):
    mk = tmp_path / "mk.npz"; cb = tmp_path / "cb.npz"
    _makani_npz(mk); _cbottle_npz(cb)
    models = {"makani": spc._read_pca_npz(str(mk)),
              "cbottle": spc._read_pca_npz(str(cb))}
    tbl = spc._r2_table(models, n_pc=8)
    assert ("makani", "t2m") in tbl and ("cbottle", "t2m") in tbl
    assert tbl[("makani", "t2m")]["r2_insample"] > 0.99  # linear signal
    assert np.isfinite(tbl[("cbottle", "t2m")]["r2_cv"])


def test_compare_without_case_refuses_npz_from_two_cases(tmp_path):
    mk, cb = tmp_path / "mk.npz", tmp_path / "cb.npz"
    _makani_npz(mk)
    _cbottle_npz(cb, case_name="Other")
    with pytest.raises(SystemExit, match="case 'Other', but .*mk.npz has 'Test'"):
        spc.main(["--makani-npz", str(mk), "--cbottle-npz", str(cb),
                  "--out-dir", str(tmp_path / "out")])


@pytest.mark.parametrize("cbottle", ["typo.npz", ""])  # "" = an unset "$VAR"
def test_compare_refuses_a_missing_cbottle_npz(tmp_path, cbottle):
    mk = tmp_path / "mk.npz"
    _makani_npz(mk)
    path = str(tmp_path / cbottle) if cbottle else ""
    with pytest.raises(FileNotFoundError):  # not a makani-only run with exit 0
        spc.main(["--makani-npz", str(mk), "--cbottle-npz", path,
                  "--out-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_dispatch_pnw_scalar_with_land_mask(tmp_path):
    rec = dpca.dispatch("pnw_heatwave", out_root=str(tmp_path), dry_run=True)
    cmd = " ".join(rec["cmd"])
    assert "--impact-kind scalar" in cmd and "--land-mask" in cmd
    assert "land_mask_pnw_heatwave_721x1440.npy" in cmd
    assert "--survival-npz" in cmd


def test_dispatch_ian_track_holds_on_tc(tmp_path):
    rec = dpca.dispatch("ian", out_root=str(tmp_path), tc_hold_jid="12345",
                        dry_run=True)
    cmd = " ".join(rec["cmd"])
    assert "--impact-kind track" in cmd and "--tc-parquet" in cmd
    assert "tc_tracks_makani_ian_end.parquet" in cmd
    assert "-hold_jid" in rec["cmd"] and "12345" in cmd
