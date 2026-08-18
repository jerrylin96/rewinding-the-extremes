"""Shared zarr loading helpers for ensemble + ERA5 stores.

Centralizes the ``zarr.open_group`` + variable-index lookup pattern that
was previously duplicated across the dispatch / plot / diagnostics scripts
and the analysis notebooks.

Layout conventions (matches the rest of the ensemble_run / ensemble_analysis pipeline):

* **ensemble** stores (HEALPix, post-pipeline-flip): ``data`` shape
  ``[N, 1, n_leads, n_vars, n_pix]``; coords include ``hpx`` (and
  ``hpx_level`` as a zarr attr).
* **ensemble** stores (legacy lat/lon): ``data`` shape
  ``[N, 1, n_leads, n_vars, n_lat, n_lon]``; coords include ``lat`` / ``lon``.
* **ERA5** stores: ``data`` shape ``[n_leads, 1, n_vars, n_lat, n_lon]``
  (always lat/lon — ERA5 is regridded to HEALPix at compute time when needed).

Ensemble loaders detect the grid from the zarr layout (presence of
``hpx`` vs ``lat``/``lon`` and ``data.ndim``) so callers don't have to.

All loaders return ``torch.Tensor`` (float32) on CPU so the output drops
into ``custom_utils.diagnostics`` / ``custom_utils.derived_variables``
without extra conversion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import zarr


@dataclass(frozen=True)
class ZarrInfo:
    """Lightweight metadata about a zarr store.

    ``grid`` is ``"latlon"`` or ``"hpx"``.  Lat/lon stores populate
    ``lat`` / ``lon``; HEALPix stores populate ``hpx`` / ``hpx_level``
    (the unused fields are ``None``).
    """

    path: str
    variables: list[str]
    lead_times: np.ndarray
    shape: tuple[int, ...]
    is_ensemble: bool
    grid: str
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    hpx: np.ndarray | None = None
    hpx_level: int | None = None

    @property
    def n_members(self) -> int | None:
        """Number of ensemble members, or ``None`` for ERA5 stores."""
        return self.shape[0] if self.is_ensemble else None

    @property
    def n_leads(self) -> int:
        return len(self.lead_times)

    @property
    def n_pix(self) -> int:
        """Number of spatial pixels (lat*lon for lat/lon, n_pix for hpx)."""
        if self.grid == "hpx":
            return int(self.hpx.shape[0])
        if self.grid == "latlon":
            return int(self.lat.shape[0]) * int(self.lon.shape[0])
        raise ValueError(f"unknown grid: {self.grid}")


def variable_index(root: Any, var: str) -> int:
    """Index of ``var`` in a zarr store's ``variable`` coordinate.

    Raises ``ValueError`` (with the available variables) if missing.
    """
    var_list = list(np.array(root["variable"][:]))
    if var not in var_list:
        raise ValueError(
            f"Variable '{var}' not found in zarr store. Available: {var_list}"
        )
    return var_list.index(var)


def _detect_grid(root: Any) -> str:
    """Return ``"hpx"`` or ``"latlon"`` based on which coord arrays are present."""
    keys = set(root.array_keys())
    if "hpx" in keys:
        return "hpx"
    if "lat" in keys and "lon" in keys:
        return "latlon"
    raise ValueError(
        f"zarr store does not expose either 'hpx' or 'lat'/'lon' coords; "
        f"available arrays: {sorted(keys)}"
    )


def inspect_zarr(zarr_path: str, label: str | None = None) -> ZarrInfo:
    """Open a zarr store and return its metadata.

    Auto-detects ensemble vs ERA5 layout (ensemble has a leading ``ensemble``
    axis, so ``ndim`` is one larger), and grid (lat/lon vs HEALPix) from the
    coord arrays.  Prints a short summary when ``label`` is given.
    """
    root = zarr.open_group(zarr_path, mode="r")
    grid = _detect_grid(root)
    variables = list(np.array(root["variable"][:]))
    lead_times = np.array(root["lead_time"][:])
    shape = tuple(root["data"].shape)
    # Ensemble stores have a leading members axis; ndim is 6 (lat/lon) or 5
    # (HEALPix).  ERA5 stores omit that axis: ndim 5 (lat/lon).
    if grid == "latlon":
        is_ensemble = root["data"].ndim == 6
    else:
        is_ensemble = root["data"].ndim == 5

    if grid == "latlon":
        info = ZarrInfo(
            path=zarr_path,
            variables=variables,
            lead_times=lead_times,
            shape=shape,
            is_ensemble=is_ensemble,
            grid="latlon",
            lat=np.array(root["lat"][:]),
            lon=np.array(root["lon"][:]),
        )
    else:
        info = ZarrInfo(
            path=zarr_path,
            variables=variables,
            lead_times=lead_times,
            shape=shape,
            is_ensemble=is_ensemble,
            grid="hpx",
            hpx=np.array(root["hpx"][:]),
            hpx_level=root.attrs.get("hpx_level"),
        )

    if label is not None:
        kind = "ensemble" if is_ensemble else "ERA5"
        print(f"[{label}] {kind} ({grid}) {zarr_path}")
        print(f"  data shape: {shape}")
        print(f"  variables ({len(variables)}): {variables}")
        print(f"  lead times ({info.n_leads}): {lead_times[0]} … {lead_times[-1]}")
        if grid == "hpx":
            print(f"  hpx pixels: {info.n_pix} (level={info.hpx_level})")
        if is_ensemble:
            print(f"  members: {info.n_members}")

    return info


def _ensemble_data_slice(root: Any, n: int, idx: int) -> np.ndarray:
    """Slice ``data[:n, 0, :, idx, ...]`` with explicit ndim handling.

    Layout is either ``[N, 1, n_leads, n_vars, n_lat, n_lon]`` (lat/lon, 6D)
    or ``[N, 1, n_leads, n_vars, n_pix]`` (HEALPix, 5D).  Returned tensor's
    trailing dims match the grid: ``(n_leads, n_lat, n_lon)`` or
    ``(n_leads, n_pix)``.
    """
    ndim = root["data"].ndim
    if ndim == 6:
        return np.array(root["data"][:n, 0, :, idx, :, :])
    if ndim == 5:
        return np.array(root["data"][:n, 0, :, idx, :])
    raise ValueError(f"unexpected ensemble data ndim: {ndim}")


def load_var_ensemble(
    zarr_path: str, var: str, n_members: int | None = 50
) -> torch.Tensor:
    """Load a single variable from an ensemble zarr store.

    Returns a float32 tensor of shape ``[N, n_leads, n_lat, n_lon]`` for
    lat/lon stores or ``[N, n_leads, n_pix]`` for HEALPix stores.
    Head-slices the first ``n_members`` (or all if ``n_members is None``).
    """
    root = zarr.open_group(zarr_path, mode="r")
    grid = _detect_grid(root)
    expected_ndim = 6 if grid == "latlon" else 5
    if root["data"].ndim != expected_ndim:
        raise ValueError(
            f"{zarr_path}: expected ensemble {grid} layout (ndim={expected_ndim}), "
            f"got ndim={root['data'].ndim}"
        )
    idx = variable_index(root, var)
    total = root["data"].shape[0]
    n = total if n_members is None else min(n_members, total)
    return torch.from_numpy(_ensemble_data_slice(root, n, idx)).float()


def load_var_era5(zarr_path: str, var: str) -> torch.Tensor:
    """Load a single variable from an ERA5 reference zarr store.

    Returns a float32 tensor of shape ``[n_leads, n_lat, n_lon]``.
    ERA5 stores are always lat/lon.
    """
    root = zarr.open_group(zarr_path, mode="r")
    if root["data"].ndim != 5:
        raise ValueError(
            f"{zarr_path}: expected ERA5 layout (ndim=5), "
            f"got ndim={root['data'].ndim}"
        )
    idx = variable_index(root, var)
    return torch.from_numpy(np.array(root["data"][:, 0, idx, :, :])).float()


def load_vars_ensemble(
    zarr_path: str,
    var_names: Sequence[str],
    n_members: int | None = 50,
) -> dict[str, Any]:
    """Load multiple variables from an ensemble zarr store, plus coords.

    Returns a dict with one entry per requested variable, each a float32
    tensor whose trailing dims match the store's grid:
    ``[N, n_leads, n_lat, n_lon]`` for lat/lon, ``[N, n_leads, n_pix]`` for
    HEALPix.

    Coord entries:
      * lat/lon stores: ``"lat"``, ``"lon"``, ``"lead_times"``
      * HEALPix stores: ``"hpx"``, ``"hpx_level"``, ``"lead_times"``

    Variable names must not collide with the reserved coordinate keys.
    """
    reserved = {"lat", "lon", "hpx", "hpx_level", "lead_times"}
    clash = reserved.intersection(var_names)
    if clash:
        raise ValueError(f"variable names collide with reserved keys: {sorted(clash)}")

    root = zarr.open_group(zarr_path, mode="r")
    grid = _detect_grid(root)
    expected_ndim = 6 if grid == "latlon" else 5
    if root["data"].ndim != expected_ndim:
        raise ValueError(
            f"{zarr_path}: expected ensemble {grid} layout (ndim={expected_ndim}), "
            f"got ndim={root['data'].ndim}"
        )

    total = root["data"].shape[0]
    n = total if n_members is None else min(n_members, total)

    out: dict[str, Any] = {}
    for name in var_names:
        idx = variable_index(root, name)
        out[name] = torch.from_numpy(_ensemble_data_slice(root, n, idx)).float()

    if grid == "latlon":
        out["lat"] = np.array(root["lat"][:])
        out["lon"] = np.array(root["lon"][:])
    else:
        out["hpx"] = np.array(root["hpx"][:])
        out["hpx_level"] = root.attrs.get("hpx_level")
    out["lead_times"] = np.array(root["lead_time"][:])
    return out


def load_vars_era5(
    zarr_path: str,
    var_names: Sequence[str],
) -> dict[str, Any]:
    """Load multiple variables from an ERA5 reference zarr store, plus coords.

    Returns a dict with one entry per requested variable, each a float32
    tensor of shape ``[n_leads, n_lat, n_lon]``, together with ``"lat"``,
    ``"lon"`` and ``"lead_times"`` as numpy arrays.
    """
    reserved = {"lat", "lon", "lead_times"}
    clash = reserved.intersection(var_names)
    if clash:
        raise ValueError(f"variable names collide with reserved keys: {sorted(clash)}")

    root = zarr.open_group(zarr_path, mode="r")
    if root["data"].ndim != 5:
        raise ValueError(
            f"{zarr_path}: expected ERA5 layout (ndim=5), "
            f"got ndim={root['data'].ndim}"
        )

    out: dict[str, Any] = {}
    for name in var_names:
        idx = variable_index(root, name)
        out[name] = torch.from_numpy(np.array(root["data"][:, 0, idx, :, :])).float()

    out["lat"] = np.array(root["lat"][:])
    out["lon"] = np.array(root["lon"][:])
    out["lead_times"] = np.array(root["lead_time"][:])
    return out
