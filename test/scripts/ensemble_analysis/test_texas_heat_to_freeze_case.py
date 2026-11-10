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

"""Check the summer-to-winter stress test pins what its YAML header says."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS / "_shared"))
sys.path.insert(0, str(SCRIPTS / "ensemble_run"))

from _dispatch_lib import iter_modes, load_case, require_region  # noqa: E402
from dispatch_ensemble import build_script_args  # noqa: E402

CASE = "texas_heat_to_freeze"


def test_only_the_season_contradicts_the_slot() -> None:
    ens = load_case(CASE)["ensemble"]
    start = datetime(**ens["start_time"])
    (pin_spec,) = ens["custom_frames"]
    assert pin_spec["frame"] == 11
    pin = datetime(**{k: v for k, v in pin_spec.items() if k != "frame"})

    slot = start + timedelta(hours=66)
    assert start == datetime(2011, 8, 28, 18)
    assert pin == datetime(2021, 2, 15, 12)
    assert slot == datetime(2011, 8, 31, 12)
    # Same time of day at both ends, opposite seasons.
    assert slot.hour == pin.hour
    assert (slot.month, pin.month) == (8, 2)


def test_dispatch_pins_frame_11_to_the_february_state() -> None:
    case = load_case(CASE)
    assert list(iter_modes(case)) == [("both", [0, 11])]
    args = build_script_args(case, "both", [0, 11], 8, 1)
    custom = [
        args[i + 1 : i + 6] for i, arg in enumerate(args) if arg == "--custom-frame"
    ]
    assert custom == [["11", "2021", "2", "15", "12"]]
    start = [
        args[args.index(f"--start-{k}") + 1] for k in ("year", "month", "day", "hour")
    ]
    assert start == ["2011", "8", "28", "18"]
    assert args[args.index("--conditioning-frames") + 1] == "0,11"
    assert args[args.index("--output-dir") + 1] == f"{case['base']}/both"


def test_stress_test_has_no_verification_regions() -> None:
    case = load_case(CASE)
    assert "tc_tracks" not in case
    for region in ("impact", "synoptic", "track"):
        with pytest.raises(SystemExit):
            require_region(case, region)
