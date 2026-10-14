"""Adapter: the cBottle end-conditioned ensemble as a lat/lon field source.

Presents the video-diffusion (cBottle) ensemble through the SAME interface as
:class:`makani_source.MakaniBackwardSource` -- ``lat``, ``lon``, ``lead_hours``,
``n_members``, ``n_frames``, ``channels``, and ``load_region(name, flat)`` -- so
the shared reducer (:mod:`latlon_metrics`) treats both models identically and any
difference in the curves is a difference in the models.

cBottle is native HEALPix (HPX64 / level 6); this source regrids it to the same
0.25 deg lat/lon grid makani is native on, at compute time and batched over members
(the whole ensemble is ~50 GB in lat/lon, so nothing is held whole).  Two design
points matching the manuscript:

* **Truth is the round-tripped ERA5** (``lat/lon -> HPX -> lat/lon``), not native
  ERA5 -- the per-model resolution-matched baseline (Text S5 ``era5_rt``): cBottle
  is scored against ERA5 at its own effective L6 resolution, while makani is scored
  against native-0.25 deg ERA5, so neither is penalised for scales it cannot reach.
* **No frame reversal.** cBottle's ``lead_time`` already runs forward (frame 0 =
  window start, frame 11 = end); makani was reversed onto this same axis, and the
  pinned frame is forward frame -1 for both.

Requires the heavy env (torch + earth2grid via ``CBottleUtils``); import is lazy so
the light-env tests never pull it in.  Reuses ``CBottleUtils.regrid_*`` verbatim --
the exact calls ``compute_ageostrophic.py`` / ``compute_power_spectra.py`` use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))

import _dispatch_lib as dl  # noqa: E402


def _decode(raw) -> List[str]:
    return [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]


class CBottleLatLonSource:
    """cBottle end-mode ensemble + round-tripped ERA5 truth, on lat/lon."""

    def __init__(
        self,
        case: Dict,
        *,
        mode: str = "end",
        size: Optional[int] = None,
        n_members: Optional[int] = None,
        batch_size: int = 32,
    ) -> None:
        self.case = case
        self.mode = mode
        self.batch_size = int(batch_size)
        size = dl.case_ensemble_size(case, size)
        self.ens_zarr = dl.ensemble_zarr_path(case["base"], mode, size)
        self.era5_zarr = dl.era5_zarr_path(case["base"], mode)
        self._read_metadata(n_members)

    def _read_metadata(self, n_members: Optional[int]) -> None:
        import zarr  # heavy-env only

        ens = zarr.open_group(self.ens_zarr, mode="r")
        era5 = zarr.open_group(self.era5_zarr, mode="r")
        if "hpx" not in ens.array_keys():
            raise ValueError(
                f"{self.ens_zarr}: expected HEALPix ensemble layout (with 'hpx'); "
                f"got arrays {sorted(ens.array_keys())}"
            )
        self._hpx_level = ens.attrs.get("hpx_level")
        if self._hpx_level is None:
            self._hpx_level = int(np.log2(int(ens["hpx"].shape[0]) // 12) / 2)
        total = int(ens["data"].shape[0])
        self.n_members = min(n_members, total) if n_members else total
        self.n_frames = int(ens["data"].shape[2])
        self.channels: List[str] = _decode(np.array(ens["variable"][:]))
        self._era5_channels: List[str] = _decode(np.array(era5["variable"][:]))
        self._lead_time = np.array(ens["lead_time"][:])
        self.lead_hours = np.asarray(dl.lead_hours(self._lead_time), dtype=float)
        # Standard regridder target grid (matches makani: 721x1440, 90->-90,
        # 0->359.75).  Captured explicitly so callers have lat/lon before any
        # regrid call.
        self.lat = np.linspace(90.0, -90.0, 721)
        self.lon = np.linspace(0.0, 360.0, 1440, endpoint=False)

    # -- regrid helpers (mirror compute_ageostrophic / compute_power_spectra) ----
    def _regrid_hpx_batch(self, hpx_batch, lead_times):
        """HPX ``[bs, n_leads, n_pix]`` -> lat/lon ``[bs, n_leads, H, W]``."""
        from custom_utils.interpolation_utils import CBottleUtils

        ens_4d = hpx_batch.unsqueeze(2)  # [bs, n_leads, 1, n_pix]
        hpx_coords = {
            "time": np.arange(hpx_batch.shape[0]),
            "lead_time": lead_times,
            "variable": np.array(["_"]),
            "hpx": np.arange(hpx_batch.shape[-1]),
        }
        latlon, _ = CBottleUtils.regrid_healpix_nest_to_latlon(
            ens_4d, hpx_coords, hpx_level=self._hpx_level)
        return latlon[:, :, 0, :, :]  # [bs, n_leads, H, W]

    def _era5_roundtrip(self, era5_latlon, lead_times):
        """ERA5 ``[n_leads, H, W]`` -> HPX -> lat/lon (the era5_rt baseline)."""
        from collections import OrderedDict

        import torch

        from custom_utils.interpolation_utils import CBottleUtils

        arr = era5_latlon.unsqueeze(0).unsqueeze(2)  # [1, n_leads, 1, H, W]
        coords = OrderedDict({
            "time": np.array([0]), "lead_time": lead_times,
            "variable": np.array(["_"]),
            "lat": self.lat, "lon": self.lon,
        })
        hpx, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
            arr, coords, hpx_level=self._hpx_level)
        rt, _ = CBottleUtils.regrid_healpix_nest_to_latlon(
            hpx, hpx_coords, hpx_level=self._hpx_level)
        return rt[0, :, 0, :, :]  # [n_leads, H, W]

    def load_region(self, name: str, flat_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(ens (N,L,P), truth (L,P))`` for one channel over the given pixels.

        Regrids the HPX ensemble ->lat/lon batched over members and keeps only the
        ``P`` region pixels; the truth is the round-tripped ERA5 for ``name``.
        """
        import torch
        import zarr

        flat = np.asarray(flat_indices, dtype=np.int64)
        ens = zarr.open_group(self.ens_zarr, mode="r")
        era5 = zarr.open_group(self.era5_zarr, mode="r")
        if name not in self.channels:
            raise KeyError(f"'{name}' not in cBottle ensemble; have {self.channels}")
        if name not in self._era5_channels:
            raise KeyError(f"'{name}' not in ERA5 reference; have {self._era5_channels}")
        vi = self.channels.index(name)
        ei = self._era5_channels.index(name)
        N, L = self.n_members, self.n_frames

        ens_parts: List[np.ndarray] = []
        for b0 in range(0, N, self.batch_size):
            b1 = min(b0 + self.batch_size, N)
            hpx = torch.from_numpy(
                np.array(ens["data"][b0:b1, 0, :, vi, :])).float()  # (bs,L,n_pix)
            ll = self._regrid_hpx_batch(hpx, self._lead_time)         # (bs,L,H,W)
            bs = ll.shape[0]
            ens_parts.append(
                ll.reshape(bs, L, -1)[:, :, flat].cpu().numpy())     # (bs,L,P)
        ens_sel = np.concatenate(ens_parts, axis=0)                  # (N,L,P)

        era5_latlon = torch.from_numpy(
            np.array(era5["data"][:, 0, ei, :, :])).float()          # (L,H,W)
        truth_rt = self._era5_roundtrip(era5_latlon, self._lead_time)  # (L,H,W)
        truth_sel = truth_rt.reshape(L, -1)[:, flat].cpu().numpy()    # (L,P)
        return ens_sel, truth_sel

    def iter_member_field_batches(self, names: List[str], *,
                                  batch_size: Optional[int] = None):
        """Yield ``(member_start, {name: torch (bs, n_frames, H, W)})`` regridded.

        Full-grid (not region-subset) batches over members, for diagnostics that
        need contiguous fields (e.g. geostrophic-wind derivatives).  The whole
        ensemble in lat/lon for three channels is far too big to hold, so this
        streams member batches and regrids each HPX->lat/lon on the fly.
        """
        import torch  # noqa: F401
        import zarr

        for n in names:
            if n not in self.channels:
                raise KeyError(f"'{n}' not in cBottle ensemble; have {self.channels}")
        ens = zarr.open_group(self.ens_zarr, mode="r")
        vis = {n: self.channels.index(n) for n in names}
        N = self.n_members
        bs_ = int(batch_size or self.batch_size)
        for b0 in range(0, N, bs_):
            b1 = min(b0 + bs_, N)
            out = {}
            for n, vi in vis.items():
                hpx = torch.from_numpy(
                    np.array(ens["data"][b0:b1, 0, :, vi, :])).float()  # (bs,L,n_pix)
                out[n] = self._regrid_hpx_batch(hpx, self._lead_time)   # (bs,L,H,W)
            yield b0, out

    def roundtrip_truth_fields(self, names: List[str]) -> Dict:
        """``{name: torch (n_frames, H, W)}`` round-tripped ERA5 (the era5_rt baseline)."""
        import torch  # noqa: F401
        import zarr

        era5 = zarr.open_group(self.era5_zarr, mode="r")
        out = {}
        for n in names:
            if n not in self._era5_channels:
                raise KeyError(f"'{n}' not in ERA5 reference; have {self._era5_channels}")
            ei = self._era5_channels.index(n)
            latlon = torch.from_numpy(
                np.array(era5["data"][:, 0, ei, :, :])).float()        # (L,H,W)
            out[n] = self._era5_roundtrip(latlon, self._lead_time)     # (L,H,W)
        return out

    def summary(self) -> Dict:
        return {
            "n_members": self.n_members, "n_frames": self.n_frames,
            "channels": self.channels, "hpx_level": self._hpx_level,
            "grid": (int(self.lat.shape[0]), int(self.lon.shape[0])),
            "lead_hours_forward": self.lead_hours.tolist(),
            "ens_zarr": self.ens_zarr, "era5_zarr": self.era5_zarr,
        }
