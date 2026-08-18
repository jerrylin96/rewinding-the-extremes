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

"""Unit tests for the equator-crossing impossible hurricane case.

Validates schema constraints, custom frame formatting, in-month ERA5 indexing,
verification lockouts, and script CLI argument generation for:
- `equator_indian_ocean` (Cyclone Agni 2004 -> TS Abaimba 2003)
"""

from __future__ import annotations

import calendar
import os
import sys
from datetime import datetime, timedelta

import pytest

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_run"))

from _dispatch_lib import (  # noqa: E402
    load_case,
    require_region,
    resolve_view,
)
from dispatch_ensemble import build_script_args  # noqa: E402

CASE_NAME = "equator_indian_ocean"
EXPECTED_METADATA = {
    "year": 2004,
    "display_name": "Equator crossing: Cyclone Agni (2004) to TS Abaimba (2003)",
    "start_time": {"year": 2004, "month": 11, "day": 27, "hour": 12},
    "custom_frames": [{"frame": 11, "year": 2003, "month": 10, "day": 2, "hour": 12}],
    "start_era5_idx": 106,
    "custom_era5_idx": 6,
    "view": (58.0, 78.0, -10.0, 5.0),
}


def test_equator_case_loadable_and_valid():
    """Case YAML must load cleanly and pass schema validation."""
    case = load_case(CASE_NAME)
    assert case["name"] == CASE_NAME
    assert case["display_name"] == EXPECTED_METADATA["display_name"]
    assert case["year"] == EXPECTED_METADATA["year"]
    assert case["base"] == f"/projectnb/eb-general/jlin404/scratch/{CASE_NAME}"
    assert case["timezone"] == "UTC"
    assert "ensemble" in case
    assert case.get("plots", {}).get("members") == [0, 1, 2, 3]
    # Assert on variable NAMES only.  vmin/vmax are rendering parameters that
    # get retuned against the data, and pinning whole entries here turned every
    # colorbar tweak into a test failure.  The names themselves are worth
    # pinning: dropping vort850 would silently remove the only diagnostic that
    # can separate a spin-preserving equator crossing from a spin-flipping one.
    names = [v["name"] for v in (case.get("plots", {}).get("variables") or [])]
    assert names == ["msl", "wspd850", "vort850"]


def test_equator_case_conditioning_and_ensemble_size():
    """Synthetic case must specify both-end conditioning, 100 members, seed 0."""
    case = load_case(CASE_NAME)
    ens = case["ensemble"]
    assert ens["modes"] == ["both"]
    assert ens["size"] == 100
    assert ens["random_seed"] == 0
    assert "custom_frames" in ens
    assert len(ens["custom_frames"]) == 1
    assert ens["custom_frames"][0]["frame"] == 11
    assert "tc_frames" not in ens


def test_equator_case_start_and_custom_frames():
    """Verify exact timestamps for start_time and custom_frames."""
    case = load_case(CASE_NAME)
    assert case["ensemble"]["start_time"] == EXPECTED_METADATA["start_time"]
    assert case["ensemble"]["custom_frames"] == EXPECTED_METADATA["custom_frames"]


def test_equator_case_non_tautological_contradiction_invariant():
    """Verify that frame 11 is NOT equal to natural start_time + 66h window."""
    case = load_case(CASE_NAME)
    st = case["ensemble"]["start_time"]
    cf = case["ensemble"]["custom_frames"][0]

    start_dt = datetime(st["year"], st["month"], st["day"], st["hour"])
    natural_end_dt = start_dt + timedelta(hours=66)
    pinned_cf_dt = datetime(cf["year"], cf["month"], cf["day"], cf["hour"])

    # Must be a true synthetic contradiction, not a no-op hindcast
    assert pinned_cf_dt != natural_end_dt
    assert (
        pinned_cf_dt.year != natural_end_dt.year
        or pinned_cf_dt.month != natural_end_dt.month
        or pinned_cf_dt.day != natural_end_dt.day
    )


def test_equator_case_synoptic_hours_and_in_month_indices():
    """Hours must be synoptic {0, 6, 12, 18} and in-month index must match expected."""
    case = load_case(CASE_NAME)
    ens = case["ensemble"]

    st = ens["start_time"]
    assert st["hour"] in {0, 6, 12, 18}
    _, days_in_st_month = calendar.monthrange(st["year"], st["month"])
    st_idx = (st["day"] - 1) * 4 + st["hour"] // 6
    assert 0 <= st_idx < days_in_st_month * 4
    assert st_idx == EXPECTED_METADATA["start_era5_idx"]

    for cf in ens["custom_frames"]:
        assert cf["hour"] in {0, 6, 12, 18}
        assert cf["frame"] == 11
        _, days_in_cf_month = calendar.monthrange(cf["year"], cf["month"])
        cf_idx = (cf["day"] - 1) * 4 + cf["hour"] // 6
        assert 0 <= cf_idx < days_in_cf_month * 4
        assert cf_idx == EXPECTED_METADATA["custom_era5_idx"]


def test_equator_case_verification_lockout():
    """Synthetic case must omit regions and tc_tracks to prevent unphysical evaluation."""
    case = load_case(CASE_NAME)
    assert "regions" not in case
    assert "tc_tracks" not in case

    for region_name in ("synoptic", "impact", "track"):
        with pytest.raises(SystemExit):
            require_region(case, region_name)


def test_equator_case_viewports():
    """Viewport must be defined explicitly and span both sides of the equator."""
    case = load_case(CASE_NAME)
    view = resolve_view(case)
    assert view == EXPECTED_METADATA["view"]
    lon_min, lon_max, lat_min, lat_max = view
    assert -180.0 <= lon_min < lon_max <= 180.0
    assert -90.0 <= lat_min < 0.0 < lat_max <= 90.0


def test_equator_case_build_script_args():
    """Verify build_script_args generates proper conditioning, custom-frame, and seed flags."""
    case = load_case(CASE_NAME)
    args = build_script_args(case, "both", [0, 11], 100, 1)

    st = EXPECTED_METADATA["start_time"]
    assert "--start-year" in args
    assert args[args.index("--start-year") + 1] == str(st["year"])
    assert "--start-month" in args
    assert args[args.index("--start-month") + 1] == str(st["month"])
    assert "--start-day" in args
    assert args[args.index("--start-day") + 1] == str(st["day"])
    assert "--start-hour" in args
    assert args[args.index("--start-hour") + 1] == str(st["hour"])

    assert "--conditioning-frames" in args
    cond_idx = args.index("--conditioning-frames")
    assert args[cond_idx + 1] == "0,11"

    assert "--custom-frame" in args
    cf_idx = args.index("--custom-frame")
    assert args[cf_idx + 1] == "11"
    cf = EXPECTED_METADATA["custom_frames"][0]
    assert args[cf_idx + 2 : cf_idx + 6] == [
        str(cf["year"]),
        str(cf["month"]),
        str(cf["day"]),
        str(cf["hour"]),
    ]

    assert "--ensemble-size" in args
    assert args[args.index("--ensemble-size") + 1] == "100"
    assert "--batch-size" in args
    assert args[args.index("--batch-size") + 1] == "1"

    assert "--random-seed" in args
    assert args[args.index("--random-seed") + 1] == "0"

    assert "--output-dir" in args
    assert args[args.index("--output-dir") + 1] == f"{case['base']}/both"
