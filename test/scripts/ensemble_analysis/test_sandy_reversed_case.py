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

"""Check the endpoint swap against the ordinary Sandy experiment."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS / "_shared"))
sys.path.insert(0, str(SCRIPTS / "ensemble_run"))

from _dispatch_lib import (  # noqa: E402
    iter_modes,
    load_case,
    require_region,
    resolve_view,
)
from dispatch_ensemble import build_script_args  # noqa: E402


def test_reversed_endpoints_start_the_model_clock_at_landfall() -> None:
    control = load_case("sandy")
    case = load_case("sandy_reversed")
    ordinary = control["ensemble"]
    reversed_ensemble = case["ensemble"]
    assert reversed_ensemble["random_seed"] == ordinary["random_seed"]
    assert case["base"] != control["base"]
    assert list(iter_modes(case)) == [("both", [0, 11])]

    start = datetime(**ordinary["start_time"])
    end = start + timedelta(hours=66)
    model_start = datetime(**reversed_ensemble["start_time"])
    sources = {0: model_start}
    sources.update(
        {
            spec["frame"]: datetime(**{k: v for k, v in spec.items() if k != "frame"})
            for spec in reversed_ensemble["custom_frames"]
        }
    )
    assert model_start == end
    assert sources == {0: end, 11: start}
    assert sources[11] - sources[0] == -timedelta(hours=66)

    model_end = model_start + timedelta(hours=66)
    assert model_end == datetime(2012, 11, 1, 18)
    assert model_end - sources[11] == timedelta(hours=132)
    assert (model_end.hour - sources[11].hour) % 24 == 12


def test_dispatch_uses_landfall_start_and_offshore_final_pin() -> None:
    case = load_case("sandy_reversed")
    args = build_script_args(case, "both", [0, 11], 8, 1)
    custom = [
        args[i + 1 : i + 6] for i, arg in enumerate(args) if arg == "--custom-frame"
    ]
    assert custom == [
        ["11", "2012", "10", "27", "6"],
    ]
    assert args[args.index("--start-day") + 1] == "30"
    assert args[args.index("--start-hour") + 1] == "0"
    assert args[args.index("--conditioning-frames") + 1] == "0,11"
    assert args[args.index("--output-dir") + 1] == f"{case['base']}/both"
    assert "--tc-frame" not in args


def test_stress_test_has_a_view_without_case_study_regions() -> None:
    case = load_case("sandy_reversed")
    assert resolve_view(case) == resolve_view(load_case("sandy"))
    assert "tc_tracks" not in case
    for region in ("impact", "synoptic", "track"):
        with pytest.raises(SystemExit):
            require_region(case, region)
