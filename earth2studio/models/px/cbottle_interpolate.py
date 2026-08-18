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

from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.cbottle_video import (
    HPX_LEVEL,
    CBottleVideo,
    DatasetModality,
    TimeStepperFunction,
)
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, TimeArray

try:
    import earth2grid
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.dataset_2d import encode_sst
    from cbottle.inference import MixtureOfExpertsDenoiser
except ImportError:
    OptionalDependencyFailure("cbottle")
    earth2grid = None
    Checkpoint = None
    encode_sst = None
    MixtureOfExpertsDenoiser = None


@check_optional_dependencies()
class CBottleInterpolate(CBottleVideo):
    """Version of CBottleVideo that allows for arbitrary frame conditioning.

    This subclass extends CBottleVideo to enable video interpolation by conditioning
    on specific frames. Instead of conditioning only on the first frame, you can
    specify which frames (0-11) to use as conditioning inputs, allowing the model
    to interpolate between known frames.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core CBottle model
    sst_ds : xr.Dataset
        Sea surface temperature dataset
    lat_lon : bool, optional
        If True, use lat/lon grid; if False, use native HEALPix grid, by default True
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : float, optional
        Noise amplitude for latent variables, by default 200
    seed : int, optional
        Random seed for latent variables, by default None
    conditioning_frames : list[int] | None, optional
        List of frame indices (0-11) to use as conditioning. If None or empty,
        behaves like standard CBottleVideo (first frame only), by default None

    Example
    -------
    >>> # Interpolate between frames 0, 6, and 11
    >>> model = CBottleInterpolate.load_model(
    ...     package,
    ...     conditioning_frames=[0, 6, 11]
    ... )
    """

    def __init__(self, *args, conditioning_frames: list[int] | None = None, lat_lon_out = True, deprecated_behavior = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lat_lon_out = lat_lon_out
        self.conditioning_frames = conditioning_frames if conditioning_frames is not None else []
        if not deprecated_behavior:
            self.full_sequence_generation = True
        else:
            print('Reverting to cbottle_video functionality')
            self.full_sequence_generation = False
        if self.full_sequence_generation and len(self.conditioning_frames) > 0:
            self.conditioning_frames = sorted(self.conditioning_frames)
            if not all(isinstance(frame, int) for frame in self.conditioning_frames):
                raise ValueError("frames must be passed as integers")
            elif min(self.conditioning_frames) < 0 or max(self.conditioning_frames) > 11:
                raise ValueError("Frames are out of bounds")

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        if self.lat_lon:
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.empty(0, dtype='timedelta64[ns]'),
                    "variable": np.array(self.VARIABLES),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440, endpoint=False),
                }
            )
        else:
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.empty(0, dtype='timedelta64[ns]'),
                    "variable": np.array(self.VARIABLES),
                    "hpx": np.arange(4**HPX_LEVEL * 12),
                }
            )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "time", 1)
        handshake_coords(input_coords, target_input_coords, "variable")

        if self.lat_lon:
            handshake_dim(input_coords, "lon", -1)
            handshake_dim(input_coords, "lat", -2)
            handshake_coords(input_coords, target_input_coords, "lon")
            handshake_coords(input_coords, target_input_coords, "lat")
        else:
            handshake_dim(input_coords, "hpx", -1)
            handshake_coords(input_coords, target_input_coords, "hpx")
        
        if self.lat_lon_out:
            output_coords = OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(self.VARIABLES),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440, endpoint=False),
                }
            )
        else:
            output_coords = OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(self.VARIABLES),
                    "hpx": np.arange(4**HPX_LEVEL * 12),
                }
            )
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        if self.full_sequence_generation:
            all_lead_times = [np.timedelta64(i * 6, 'h') for i in range(self._time_length)]
            output_coords["lead_time"] = np.array(all_lead_times)
        else:
            output_coords["lead_time"] = input_coords["lead_time"] + np.array(
                [self._time_step]
            )
        return output_coords

    def _forward(self, x: torch.Tensor, times: TimeArray) -> torch.Tensor:
        """Executes forward sample of the model given conditional tensor and time array

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to condition video model. Of size [time,1,45,721,1440] if
            lat_lon or [time,1,45,49152] if healpix
        times : TimeArray
            Time stamp array of size [time]

        Returns
        -------
        torch.Tensor
            12 forecast steps (initial step including) [time, 12, 45, 721, 1440] if
            lat_lon or [time, 1, 45, 49152] if healpix
        """
        # Small check to make sure

        device = self.device_buffer.device
        self.core_model.sigma_min = self.sigma_min
        self.core_model.sigma_max = self.sigma_max
        self.core_model.num_steps = self.sampler_steps
        self.core_model.time_stepper = TimeStepperFunction(self.time_stepper).value

        if self.lat_lon:
            x = self.condition_regridder(x.double())

        # CBottle video expects [time, vars, time-step, hpx]
        x = x.transpose(1, 2)
        input_batch = self.get_cbottle_input(
            x, times, dataset_modality=self.dataset_modality, device=device
        )
        out, _ = self.core_model.sample(input_batch, seed=self.seed)
        # Regrid if needed
        if self.lat_lon_out:
            out = self.output_regridder(out.contiguous().double())
        # [time, vars, lead, ...] -> [time, lead, vars, ...]
        out = out.transpose(1, 2)

        return out

    def get_cbottle_input(
        self,
        conditions: torch.Tensor,
        times: TimeArray,
        dataset_modality: DatasetModality = DatasetModality.ERA5,
        device: torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Creates batch input for cbottle

        Parameters
        ----------
        conditions : torch.Tensor
            HPX conditional tensor for first time-step of size
            [time, 45, 1, 4**HPX_LEVEL*12]. If all NaNs no condition will be used.
        times : TimeArray
            Array of np.datetime64 time stamps to be samples of size [time], must have
            SST that can be sampled from self.sst
        dataset_modality : DatasetModality, optional
            Dataset modality label, by default DatasetModality.ERA5
        device : torch.device, optional
            Torch device, by default "cpu"

        Returns
        -------
        dict[str, torch.Tensor]
            Input batch dictionary used in the CBottle repo
        """
        # Known support range for SST
        for time in times:
            if time < np.datetime64("1940-01-01") or time >= np.datetime64(
                "2022-12-12T12:00"
            ):
                logger.warning(
                    f"Requested time {time} is outside of the default SST support range"
                )

        time_steps = [i * self._time_step for i in range(self._time_length)]
        times = times[:, None] + np.array(time_steps, dtype=np.timedelta64)[None, :]

        time_arr = np.array(times, dtype="datetime64[ns]").reshape(-1)
        sst_data = torch.from_numpy(
            self.sst["tosbcs"].interp(time=time_arr, method="linear").values + 273.15
        ).to(self.device_buffer.device)
        sst_data = self.sst_regridder(sst_data)

        # TODO: Fix on off device in efficiency here
        cond = torch.zeros(
            times.shape[0],
            47,
            times.shape[1],
            4**HPX_LEVEL * 12,
            dtype=torch.double,
            device=device,
        )
        cond[:, -2, :, :] = torch.tensor(
            encode_sst(sst_data.cpu()).reshape(times.shape[0], times.shape[1], -1),
            device=device,
        )
        cond[:, self._nan_channels, :, :] = torch.nan
        # If initial state to condition the model
        if (
            not torch.isnan(conditions).all()
            and self.core_model.batch_info.center
            and self.core_model.batch_info.scales
        ):
            means = (
                torch.tensor(self.core_model.batch_info.center)
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            stds = (
                torch.tensor(self.core_model.batch_info.scales)
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            conditions = (conditions.to(device) - means) / stds
            if self.full_sequence_generation and len(self.conditioning_frames) > 0:
                for i, frame_idx in enumerate(self.conditioning_frames):
                    cond[:, :-2, frame_idx:frame_idx+1, :] = conditions[:, :, i:i+1, :]
                cond[:, -1, self.conditioning_frames, :] = 1
            elif self.full_sequence_generation:
                pass
            else:
                cond[:, :-2, :1, :] = conditions.to(device)
                cond[:, -1, :1, :] = 1  # Frame mask to 1

        def reorder(x: torch.Tensor) -> torch.Tensor:
            x = torch.as_tensor(x)
            x = earth2grid.healpix.reorder(
                x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
            )
            return x

        # Set up time tensors
        times0 = [
            datetime.fromtimestamp(
                t.astype("datetime64[s]").astype(int), tz=timezone.utc
            )
            for t in times.reshape(-1)
        ]
        second_of_day = np.array(
            [(t.hour * 3600) + (t.minute * 60) + t.second for t in times0]
        ).reshape(times.shape)
        day_of_year = np.array(
            [
                (t - datetime(t.year, 1, 1, tzinfo=timezone.utc)).total_seconds()
                / (86400.0)
                for t in times0
            ]
        ).reshape(times.shape)
        second_of_day = torch.tensor(second_of_day.astype(np.float32), device=device)
        day_of_year = torch.tensor(day_of_year.astype(np.float32), device=device)

        # Target tensor, not needed for video model since no infill
        target = torch.empty(1, device=device)

        # Label tensor
        dataset_modality = DatasetModality(dataset_modality)
        labels = torch.nn.functional.one_hot(
            torch.tensor(dataset_modality.value, device=device), num_classes=1024
        )
        labels = labels.unsqueeze(0).repeat(len(times), 1)

        out = {
            "target": target,
            "labels": labels,
            "condition": reorder(cond),
            "second_of_day": second_of_day,
            "day_of_year": day_of_year,
        }
        return out

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat_lon: bool = True,
        lat_lon_out: bool = True,
        sampler_steps: int = 18,
        sigma_max: float = 1000,
        seed: int | None = None,
        conditioning_frames: list[int] | None = None,
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        lat_lon : bool, optional
            Lat/lon toggle, if true prognostic input/output on a 0.25 deg lat/lon
            grid. If false, the native nested HealPix grid will be returned, by default
            True
        sampler_steps : int, optional
            Number of diffusion steps, by default 18
        sigma_max : float, optional
            Noise amplitude used to generate latent variables, by default 200
        seed : int, optional
            Random generator seed for latent variables. If None, no seed will be used,
            by default None
        conditioning_frames : list[int] | None, optional
            List of frame indices (0-11) to condition on for interpolation. If None,
            defaults to first frame only (standard CBottleVideo behavior), by default None

        Returns
        -------
        PrognosticModel
            Prognostic Model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        # https://github.com/NVlabs/cBottle/blob/4f44c125398896fad1f4c9df3d80dc845758befa/src/cbottle/inference.py#L810
        experts = []
        batch_info = None
        for path in [package.resolve("cBottle-video.zip")]:
            with Checkpoint(path) as c:
                model = c.read_model().eval()
                experts.append(model)
                batch_info = c.read_batch_info()
        core_model = MixtureOfExpertsDenoiser(
            experts, sigma_thresholds=(), batch_info=batch_info
        )

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="netcdf4",
            cache=False,
        ).load()

        return cls(
            core_model,
            sst_ds,
            lat_lon=lat_lon,
            lat_lon_out=lat_lon_out,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
            conditioning_frames=conditioning_frames,
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass through the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system. When conditioning is enabled,
            returns all 12 frames. Otherwise returns only the next frame.
        """
        output_coords = self.output_coords(coords)

        times = coords["time"].repeat(coords["batch"].shape[0])

        domain_shape = list(x.shape)[3:]  # Auto handle lat/lon vs healpix
        x = x.reshape(-1, coords["lead_time"].shape[0], *domain_shape)
        x = self._forward(x, times)
        x = x.reshape(
            coords["batch"].shape[0], coords["time"].shape[0], *x.shape[1:]
        )

        if self.full_sequence_generation:
            return x, output_coords
        else:
            return x[:, :, 1:2], output_coords
