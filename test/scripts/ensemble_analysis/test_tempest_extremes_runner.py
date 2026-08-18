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

"""Unit tests for the pure-Python helpers in tempest_extremes_runner.

Covers the flag-composition logic that picks up the Zarzycki & Ullrich
warm-core contour and the orography clauses, plus the variable-list
helper in plot_ensemble_tc_tracks.  No TempestExtremes binaries or
zarr I/O are exercised.
"""

from __future__ import annotations

import os
import sys

import pytest

# The runner imports numpy at module load; bail early in envs without it.
pytest.importorskip("numpy")

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis", "submission_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from tempest_extremes_runner import (  # noqa: E402
    DEFAULT_DETECT_FLAGS,
    WARM_CORE_CONTOUR,
    _compose_default_flags,
)

# ---------------------------------------------------------------------------
# _compose_default_flags
# ---------------------------------------------------------------------------


def test_compose_default_flags_no_orog_no_warm_core():
    """Without warm-core, closedcontourcmd is just the msl clause."""
    detect, _ = _compose_default_flags(use_orography=False, warm_core=False)
    assert detect["closedcontourcmd"] == "msl,200.0,5.5,0"
    # No zs in outputcmd either.
    assert "zs" not in detect["outputcmd"]


def test_compose_default_flags_no_orog_warm_core():
    """With warm-core, the Z&U 2017 thickness contour is appended."""
    detect, _ = _compose_default_flags(use_orography=False, warm_core=True)
    assert detect["closedcontourcmd"] == (
        "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0"
    )


def test_compose_default_flags_orog_and_warm_core():
    """Orography + warm-core compose without interfering with each other."""
    detect, stitch = _compose_default_flags(use_orography=True, warm_core=True)
    assert detect["closedcontourcmd"] == (
        "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0"
    )
    assert detect["outputcmd"].endswith(";zs,min,0")
    assert "zs" in stitch["in_fmt"]
    assert "zs,<=,150.0,10" in stitch["threshold"]


def test_compose_default_flags_does_not_mutate_module_default():
    """The default-flag dict must not be mutated by composition."""
    original = DEFAULT_DETECT_FLAGS["closedcontourcmd"]
    _compose_default_flags(use_orography=True, warm_core=True)
    assert DEFAULT_DETECT_FLAGS["closedcontourcmd"] == original


def test_warm_core_contour_constant():
    """Sanity check on the constant the runner uses for the appended clause."""
    assert WARM_CORE_CONTOUR == "_DIFF(z300,z500),-58.8,6.5,1.0"


# ---------------------------------------------------------------------------
# _tempest_tracker_variables
#
# Lives in plot_ensemble_tc_tracks.py which pulls in matplotlib/cartopy at
# import time; skip cleanly if those aren't installed in the test env.
# ---------------------------------------------------------------------------


def _import_helper():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from plot_ensemble_tc_tracks import _tempest_tracker_variables

    return _tempest_tracker_variables


def test_tempest_tracker_variables_no_warm_core():
    helper = _import_helper()
    assert helper(warm_core=False) == ["u10m", "v10m", "msl"]


def test_tempest_tracker_variables_warm_core():
    helper = _import_helper()
    assert helper(warm_core=True) == ["u10m", "v10m", "msl", "z300", "z500"]


def test_tempest_tracker_variables_default_is_no_warm_core():
    """Default argument keeps warm-core off (archived-runs-safe)."""
    helper = _import_helper()
    assert helper() == ["u10m", "v10m", "msl"]


# ---------------------------------------------------------------------------
# parse_candidates_file
#
# DetectNodes candidate output: one header line per time step
# (yyyy mm dd n_candidates hh, dates on the synthetic 1970-01-01 base
# _stage_member_netcdf writes), then one row per candidate
# (i_lon i_lat lon lat <outputcmd columns...>).
# ---------------------------------------------------------------------------


def _write_candidates(tmp_path):
    from pathlib import Path

    text = "\n".join(
        [
            "1970\t1\t1\t2\t0",
            "\t500\t200\t310.5\t25.0\t100500.0\t15.2",
            "\t600\t300\t320.0\t30.0\t100800.0\t12.0",
            "1970\t1\t1\t0\t6",
            "1970\t1\t2\t1\t12",
            "\t100\t100\t290.0\t40.0\t99000.0\t20.0",
        ]
    )
    path = Path(tmp_path) / "member_0000.candidates.txt"
    path.write_text(text + "\n")
    return path


def test_parse_candidates_file(tmp_path):
    """Candidates land on the right steps as (lat, lon, msl) rows."""
    import numpy as np
    from tempest_extremes_runner import parse_candidates_file

    per_step = parse_candidates_file(_write_candidates(tmp_path), 12, 6.0)
    assert len(per_step) == 12
    # Step 0 (hour 0): two candidates.
    assert per_step[0].shape == (2, 3)
    np.testing.assert_allclose(per_step[0][0], [25.0, 310.5, 100500.0])
    # Step 1 (hour 6): header present but zero candidates.
    assert per_step[1].shape == (0, 3)
    # Step 6 (1970-01-02 12:00 = 36 h at dt=6h): one candidate.
    assert per_step[6].shape == (1, 3)
    np.testing.assert_allclose(per_step[6][0], [40.0, 290.0, 99000.0])
    # Steps with no header stay empty.
    assert all(per_step[t].shape == (0, 3) for t in (2, 3, 4, 5, 7, 11))


def test_parse_candidates_file_missing_or_empty(tmp_path):
    """A missing or empty file parses to all-empty steps, not an error."""
    from pathlib import Path

    from tempest_extremes_runner import parse_candidates_file

    missing = Path(tmp_path) / "does_not_exist.txt"
    per_step = parse_candidates_file(missing, 4, 6.0)
    assert len(per_step) == 4
    assert all(c.shape == (0, 3) for c in per_step)

    empty = Path(tmp_path) / "empty.txt"
    empty.write_text("")
    per_step = parse_candidates_file(empty, 4, 6.0)
    assert all(c.shape == (0, 3) for c in per_step)
