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

"""Unit tests for `scripts/_shared/derived_fields.py`.

The rigid-rotation case is the load-bearing one: a wind field rotating with the
planet has relative vorticity exactly equal to the Coriolis parameter, which
pins the magnitude, the spherical metric terms, and — for the equator-crossing
cases — the hemisphere sign convention all at once.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))

from derived_fields import (  # noqa: E402
    EARTH_RADIUS_M,
    OMEGA,
    coriolis_parameter,
    relative_vorticity,
)

# ERA5's latitude convention: descending, 90 -> -90.
LAT = np.linspace(90.0, -90.0, 181)
LON = np.linspace(0.0, 360.0, 360, endpoint=False)

# np.gradient falls back to one-sided differences at the array edges, and the
# 1/cos(phi) metric blows up at the poles, so accuracy is asserted over the
# interior band only.  Every case view box is well inside it.
INTERIOR = slice(20, -20)


def _rigid_rotation(omega):
    """Wind field rotating with the planet at rate ``omega``: u = omega*a*cos(phi)."""
    cos_phi = np.cos(np.radians(LAT))[None, :, None]
    u = omega * EARTH_RADIUS_M * cos_phi * np.ones((1, LAT.size, LON.size))
    v = np.zeros_like(u)
    return u, v


def test_coriolis_parameter_signs_and_equator():
    """f is positive in the NH, negative in the SH, and exactly zero at 0 deg."""
    assert coriolis_parameter(0.0) == 0.0
    assert coriolis_parameter(10.0) > 0.0
    assert coriolis_parameter(-10.0) < 0.0
    assert coriolis_parameter(90.0) == pytest.approx(2.0 * OMEGA)


def test_rigid_rotation_reproduces_coriolis_parameter():
    """zeta of a planet-locked wind field equals f = 2*Omega*sin(phi)."""
    u, v = _rigid_rotation(OMEGA)
    zeta = relative_vorticity(u, v, LAT, LON)

    expected = coriolis_parameter(LAT)[None, :, None] * np.ones_like(zeta)
    np.testing.assert_allclose(
        zeta[:, INTERIOR, :], expected[:, INTERIOR, :], rtol=2e-3, atol=1e-9
    )


def test_rigid_rotation_sign_flips_across_the_equator():
    """The hemisphere sign convention the equator-crossing cases test for."""
    u, v = _rigid_rotation(OMEGA)
    zeta = relative_vorticity(u, v, LAT, LON)

    north = zeta[0, np.argmin(np.abs(LAT - 20.0)), :]
    south = zeta[0, np.argmin(np.abs(LAT + 20.0)), :]
    assert np.all(north > 0.0)
    assert np.all(south < 0.0)
    # Antisymmetric about the equator.
    np.testing.assert_allclose(north, -south, rtol=2e-3)


def test_counter_rotation_negates_vorticity():
    """zeta is linear in the rotation rate, so reversing the spin negates it."""
    u_pos, v_pos = _rigid_rotation(OMEGA)
    u_neg, v_neg = _rigid_rotation(-OMEGA)

    zeta_pos = relative_vorticity(u_pos, v_pos, LAT, LON)
    zeta_neg = relative_vorticity(u_neg, v_neg, LAT, LON)
    np.testing.assert_allclose(
        zeta_pos[:, INTERIOR, :], -zeta_neg[:, INTERIOR, :], rtol=1e-10, atol=1e-12
    )


def test_quiescent_field_has_zero_vorticity():
    """A field at rest has no relative vorticity anywhere, poles included."""
    zeros = np.zeros((2, LAT.size, LON.size))
    zeta = relative_vorticity(zeros, zeros, LAT, LON)
    assert zeta.shape == zeros.shape
    np.testing.assert_allclose(zeta, 0.0, atol=1e-18)


def test_cyclonic_vortex_is_positive_in_the_northern_hemisphere():
    """A counter-clockwise vortex near 10N has zeta > 0 at its core."""
    lat0, lon0 = 10.0, 70.0
    lat2d, lon2d = np.meshgrid(LAT, LON, indexing="ij")
    # Local tangent-plane offsets (m) from the vortex centre.
    x = np.radians(lon2d - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat2d - lat0) * EARTH_RADIUS_M

    k = 1.0e-5  # solid-body core; zeta = 2k analytically
    u = (-k * y)[None, :, :]
    v = (k * x)[None, :, :]

    zeta = relative_vorticity(u, v, LAT, LON)
    core = zeta[0, np.argmin(np.abs(LAT - lat0)), np.argmin(np.abs(LON - lon0))]
    assert core > 0.0
    assert core == pytest.approx(2.0 * k, rel=0.05)


def test_frames_are_independent():
    """Each frame's vorticity depends only on that frame's wind field."""
    u_pos, v_pos = _rigid_rotation(OMEGA)
    u_neg, v_neg = _rigid_rotation(-OMEGA)
    u = np.concatenate([u_pos, u_neg], axis=0)
    v = np.concatenate([v_pos, v_neg], axis=0)

    zeta = relative_vorticity(u, v, LAT, LON)
    np.testing.assert_allclose(
        zeta[0, INTERIOR, :], -zeta[1, INTERIOR, :], rtol=1e-10, atol=1e-12
    )
