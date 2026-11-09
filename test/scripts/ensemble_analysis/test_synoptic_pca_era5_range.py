# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ERA5-vs-members range table (``synoptic_pca_era5_range``) on
synthetic ``synoptic_pca_<mode>.npz`` files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("yaml")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

import synoptic_pca_era5_range as era5_range  # noqa: E402

N_MEMBERS, MAX_D = 300, 10


def _write_npz(path: Path, era5_scale: float, n_era5: int = MAX_D) -> None:
    """Members ~ N(0, 1) in every PC; ERA5 at ``era5_scale`` on each of its PCs."""
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pc_scores=rng.standard_normal((N_MEMBERS, MAX_D)).astype(np.float32),
        era5_pc_score=np.full(n_era5, era5_scale, dtype=np.float32),
        variables=np.array(["z500"]),
    )


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(era5_range, "load_case", lambda name: {"base": tmp_path / name})
    return tmp_path


def test_inside_and_outside_rows(case_dir: Path) -> None:
    npz_dir = case_dir / "storm" / "diagnostics" / "synoptic_pca"
    _write_npz(npz_dir / "synoptic_pca_end.npz", era5_scale=0.1)
    _write_npz(npz_dir / "synoptic_pca_start.npz", era5_scale=5.0)
    table = era5_range.range_table(["storm"], ["end", "start"], n_pc=8)
    end, start = table.iloc[0], table.iloc[1]
    assert end["case"] == "storm" and end["mode"] == "end"
    assert end["N"] == N_MEMBERS and end["vars"] == "z500"
    assert end["ERA5"] == "inside" and end["p"] > 0.5
    assert start["ERA5"] == "outside"
    assert start["m"] == 0
    assert start["p"] == pytest.approx(1 / (N_MEMBERS + 1))
    assert start["d2_era5"] > start["d2_max"]


def test_too_few_era5_pcs_says_rerun(case_dir: Path) -> None:
    """Ian's npz from before the fix saved ERA5 for its 6 shown EOFs only."""
    npz = case_dir / "ian" / "diagnostics" / "synoptic_pca" / "synoptic_pca_end.npz"
    _write_npz(npz, era5_scale=0.1, n_era5=6)
    with pytest.raises(SystemExit, match="Rerun compute_synoptic_pca"):
        era5_range.range_table(["ian"], ["end"], n_pc=8)


def test_missing_npz_is_an_error(case_dir: Path) -> None:
    with pytest.raises(SystemExit, match="missing"):
        era5_range.range_table(["storm"], ["end"], n_pc=8)
