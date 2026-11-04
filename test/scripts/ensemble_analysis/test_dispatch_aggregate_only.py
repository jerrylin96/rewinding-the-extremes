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

"""``--aggregate-only`` submits the figure job and nothing else.

Re-rendering a figure after a label or style change must not resubmit the
compute jobs (hours of queue time, and they overwrite the saved outputs the
figure is drawn from).  Each dispatcher is driven through its own
``dispatch()`` with ``submit_qsub`` recorded instead of called; the control
(``aggregate_only=False``) shows the compute jobs are really there to skip.
"""

from __future__ import annotations

import importlib
import os
import sys

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))

import pytest  # noqa: E402

DISPATCHERS = [
    "dispatch_free_end_states",
    "dispatch_synoptic_pca",
    "dispatch_tc_tracks",
    "dispatch_spread_rmse_crps",
    "dispatch_ageostrophic",
]


def _submitted_scripts(monkeypatch, name, **kwargs):
    mod = importlib.import_module(name)
    scripts = []

    def record(cmd, *args, **kw):
        scripts.append(next(str(c) for c in cmd if str(c).endswith(".sh")))
        return None

    monkeypatch.setattr(mod, "submit_qsub", record)
    monkeypatch.setattr(mod, "validate_zarrs_exist", lambda *a, **k: None)
    mod.dispatch("ian", dry_run=True, **kwargs)
    return mod, scripts


@pytest.mark.parametrize("name", DISPATCHERS)
def test_aggregate_only_submits_only_the_figure_job(monkeypatch, name):
    mod, scripts = _submitted_scripts(monkeypatch, name, aggregate_only=True)
    assert scripts, "no figure job submitted"
    assert set(scripts) == {mod.AGG_SCRIPT}


@pytest.mark.parametrize("name", DISPATCHERS)
def test_default_also_submits_compute_jobs(monkeypatch, name):
    mod, scripts = _submitted_scripts(monkeypatch, name)
    assert mod.AGG_SCRIPT in scripts
    assert set(scripts) - {mod.AGG_SCRIPT}, "control found no compute jobs"
