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

"""Unit tests for the pure-Python case helpers in _dispatch_lib.

Covers ``warm_core_from_case``, which resolves the per-case default for
the tempest warm-core thickness filter from ``tc_tracks.warm_core`` in
the case YAML.  No zarr I/O or qsub submission is exercised.
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from _dispatch_lib import warm_core_from_case  # noqa: E402


def test_warm_core_true_when_set():
    assert warm_core_from_case({"tc_tracks": {"warm_core": True}}) is True


def test_warm_core_false_when_set_false():
    assert warm_core_from_case({"tc_tracks": {"warm_core": False}}) is False


def test_warm_core_defaults_false_when_key_absent():
    # A tc_tracks block without the key falls back to off.
    assert warm_core_from_case({"tc_tracks": {"trackers": ["tempest"]}}) is False


def test_warm_core_false_when_no_tc_block():
    assert warm_core_from_case({}) is False
