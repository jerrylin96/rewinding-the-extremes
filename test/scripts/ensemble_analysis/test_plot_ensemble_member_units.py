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

"""Member plots draw values in the units their label names.

``plot_ensemble_member.py`` labels a variable through ``var_metadata.var_label``,
which names geopotential as geopotential height in m.  Storage (and the shared
palette's ``vmin``/``vmax``) keep m^2/s^2, so the field and its colour limits
are converted at plot time; without that a z500 colour bar would read "(m)"
over values ~9.8x too large.

The script's own code runs; only its heavy import-time dependencies (the
cBottle/earth2studio helpers in ``custom_utils``, and torch where it is not
installed) are stubbed, since no data is read: ``load_field`` is replaced by a
synthetic field.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_shared"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ensemble_analysis"))

for _dep in ("matplotlib", "cartopy", "cmasher", "zarr"):
    pytest.importorskip(_dep)

import var_metadata as vm  # noqa: E402

Z500_RAW = (51000.0, 58500.0)  # cases/_palettes.yaml z500 vmin/vmax, m^2/s^2


@pytest.fixture()
def pem(monkeypatch):
    try:
        importlib.import_module("torch")
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    cu = types.ModuleType("custom_utils")
    cu.__path__ = []
    loading = types.ModuleType("custom_utils.ensemble_loading")
    loading.variable_index = lambda root, var: 0
    interp = types.ModuleType("custom_utils.interpolation_utils")
    interp.CBottleUtils = object
    monkeypatch.setitem(sys.modules, "custom_utils", cu)
    monkeypatch.setitem(sys.modules, "custom_utils.ensemble_loading", loading)
    monkeypatch.setitem(sys.modules, "custom_utils.interpolation_utils", interp)
    sys.modules.pop("plot_ensemble_member", None)
    yield importlib.import_module("plot_ensemble_member")
    sys.modules.pop("plot_ensemble_member", None)


def _raw_z500(n_frames=12):
    lat = np.linspace(30.0, 60.0, 7)
    lon = np.linspace(200.0, 260.0, 9)
    field = np.full((n_frames, lat.size, lon.size), 55000.0)  # ~5608 m
    field[:, 0, 0] = 52000.0
    return field, lat, lon


def test_load_display_field_converts_geopotential_only(pem, monkeypatch):
    raw, lat, lon = _raw_z500()
    monkeypatch.setattr(pem, "load_field", lambda *a, **k: (raw, lat, lon))
    z, _, _ = pem.load_display_field(None, "z500", 0, 12, False, 6)
    np.testing.assert_allclose(z, raw / vm.DISPLAY_GRAVITY)
    msl, _, _ = pem.load_display_field(None, "msl", 0, 12, False, 6)
    np.testing.assert_array_equal(msl, raw)


def test_limits_convert_with_the_field(pem):
    assert pem.display_limit("z500", Z500_RAW[0]) == pytest.approx(
        Z500_RAW[0] / vm.DISPLAY_GRAVITY
    )
    assert pem.display_limit("msl", 96000) == 96000.0
    assert pem.display_limit("z500", None) is None


def test_grid_colour_bar_in_metres_over_metre_values(pem, monkeypatch, tmp_path):
    import matplotlib.figure

    raw, lat, lon = _raw_z500()
    monkeypatch.setattr(pem, "load_field", lambda *a, **k: (raw, lat, lon))
    monkeypatch.setattr(pem, "_hpx_level_from_root", lambda root: 6)
    monkeypatch.setattr(
        pem, "get_valid_time_labels", lambda root, n, is_era5=False: ["t"] * n
    )
    monkeypatch.setattr(pem, "_frame_provenance", lambda root, n: [(False, "")] * n)
    figs = []
    monkeypatch.setattr(pem.plt, "close", lambda fig=None: figs.append(fig))
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **k: None)

    class Root:
        def array_keys(self):
            return ["hpx", "data"]

        def __getitem__(self, key):
            return SimpleNamespace(shape=(2, 1, 12, 1, 48))

    args = SimpleNamespace(
        vmin=Z500_RAW[0],
        vmax=Z500_RAW[1],
        cmap="viridis",
        coastline_color="white",
        gridline_color="gray",
        extent=None,
        wind_overlay="none",
        grid_title=None,
        output="z500_grid.png",
        output_dir=str(tmp_path),
        dpi=50,
    )
    pem.plot_member_grid(Root(), None, "z500", 0, args)

    (fig,) = figs
    map_axes = [
        ax for ax in fig.axes if hasattr(ax, "projection")
    ]  # not the colour bar
    meshes = [c for ax in map_axes for c in ax.collections if c.get_array() is not None]
    assert len(meshes) == 12
    values = np.ma.concatenate([m.get_array().ravel() for m in meshes])
    assert 5000.0 < values.min() and values.max() < 6000.0  # metres, not m^2/s^2
    lo, hi = meshes[0].get_clim()
    assert (lo, hi) == pytest.approx(tuple(v / vm.DISPLAY_GRAVITY for v in Z500_RAW))
    labels = [ax.get_xlabel() for ax in fig.axes]
    assert vm.var_label("z500") in labels and vm.var_label("z500").endswith("(m)")
