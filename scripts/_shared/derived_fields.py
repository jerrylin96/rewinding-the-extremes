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

"""Derived fields computed from stored ensemble variables.

Pure NumPy, no matplotlib and no torch, so a stats-only login-node run or a
dispatcher can import this without the heavy plotting chain.  Kept here rather
than inside ``plot_ensemble_member.py`` so the animation and any analysis
script that classifies members share one definition of each field — the
`scripts/_shared/README.md` convention.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6.371e6

#: Angular velocity of the Earth (rad/s), for planetary vorticity.
OMEGA = 7.292e-5


def coriolis_parameter(lat: np.ndarray) -> np.ndarray:
    """Planetary vorticity ``f = 2 * Omega * sin(phi)`` in s^-1, ``lat`` in degrees."""
    return 2.0 * OMEGA * np.sin(np.radians(np.asarray(lat, dtype=np.float64)))


def relative_vorticity(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Relative vorticity of a lat/lon wind field, in s^-1.

    Spherical form::

        zeta = (1 / (a cos(phi))) * (dv/dlambda - d(u cos(phi))/dphi)

    evaluated with centred differences on the coordinate vectors, which may be
    ascending or descending (ERA5 latitude runs 90 -> -90).

    Parameters
    ----------
    u, v
        Wind components, shape ``[n_frames, n_lat, n_lon]``.
    lat, lon
        Coordinate vectors in degrees, lengths ``n_lat`` and ``n_lon``.

    Returns
    -------
    numpy.ndarray
        Relative vorticity with the same shape as ``u``.

    Notes
    -----
    The zonal derivative is not wrapped at the 0/360 seam, so that single
    column is wrong.  No committed case's view box straddles it.  Values within
    ~1e-6 of the poles are clipped to avoid dividing by ``cos(phi) = 0``; every
    case domain is tropical, so this clips nothing inside a view box.
    """
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    lam = np.radians(np.asarray(lon, dtype=np.float64))
    cos_phi = np.cos(phi)[None, :, None]
    cos_phi = np.where(np.abs(cos_phi) < 1e-6, 1e-6, cos_phi)

    dv_dlam = np.gradient(v, lam, axis=2)
    du_cos_dphi = np.gradient(u * cos_phi, phi, axis=1)
    return (dv_dlam - du_cos_dphi) / (EARTH_RADIUS_M * cos_phi)
