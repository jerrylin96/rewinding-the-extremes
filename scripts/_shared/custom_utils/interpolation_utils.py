import calendar
import gc
import os
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import torch

from earth2studio.data import CBottle3D
from earth2studio.lexicon import WB2Lexicon, CBottleLexicon
from earth2studio.models.dx import CBottleTCGuidance
from earth2studio.models.px import CBottleInterpolate

import earth2grid
import xarray as xr

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmasher as cmr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import interp1d

HPX_LEVEL = 6

WB2_VARS = set(WB2Lexicon.VOCAB.keys())
CBOTTLE_VARS = set(CBottleLexicon.VOCAB.keys())
INPUT_VARIABLES = sorted(list(CBOTTLE_VARS.intersection(WB2_VARS)))
OUTPUT_VARIABLES = list(CBottleLexicon.VOCAB.keys())

# Curated subset of OUTPUT_VARIABLES retained in ensemble zarr stores.
# Persisting only this subset (instead of all 45 model output variables)
# reduces the on-disk footprint of an ensemble by ~2/3.  Must be a subset
# of INPUT_VARIABLES ∩ OUTPUT_VARIABLES so the ERA5-reference and ensemble
# zarrs can be subset identically and every downstream diagnostic finds
# its variables in both stores.  Enforced by the assertion below.
STORED_VARIABLES = [
    # Near-surface / column-integrated
    "u10m", "v10m", "t2m", "msl", "sst", "tcwv",
    # 200 hPa
    "u200", "v200",
    # 300 hPa — z300 enables the Zarzycki & Ullrich 2017 _DIFF(z300,z500)
    # warm-core test used by the tempest TC tracker.
    "z300",
    # 500 hPa
    "u500", "v500", "t500", "z500",
    # 850 hPa
    "u850", "v850",
]

_missing_input = [v for v in STORED_VARIABLES if v not in CBOTTLE_VARS or v not in WB2_VARS]
_missing_output = [v for v in STORED_VARIABLES if v not in CBOTTLE_VARS]
assert not _missing_input and not _missing_output, (
    f"STORED_VARIABLES must be ⊆ INPUT_VARIABLES ∩ OUTPUT_VARIABLES; "
    f"missing from input: {_missing_input}, missing from output: {_missing_output}"
)

ERA5_VAR_TO_IDX = {var: idx for idx, var in enumerate(INPUT_VARIABLES)}
MODEL_VAR_TO_IDX = {var: idx for idx, var in enumerate(OUTPUT_VARIABLES)}


def subset_variables(data_tensor, coords, target_variables=STORED_VARIABLES):
    """Select ``target_variables`` along the variable axis of ``data_tensor``.

    The variable axis is derived from ``coords`` (whichever key is
    ``"variable"``), so callers don't need to track it manually as the
    tensor layout evolves.

    Raises
    ------
    KeyError
        If any element of ``target_variables`` is missing from
        ``coords["variable"]``.  This is intentional: ``STORED_VARIABLES``
        is asserted at import time to be a subset of both ERA5 and model
        outputs, so a missing variable indicates a real configuration bug
        that should fail loudly rather than silently dropping data.

    Returns
    -------
    (subset_data, subset_coords)
        ``subset_coords`` is a shallow copy of ``coords`` with
        ``"variable"`` replaced by the selected subset in
        ``target_variables`` order.
    """
    var_axis = list(coords.keys()).index("variable")
    var_list = list(coords["variable"])
    missing = [v for v in target_variables if v not in var_list]
    if missing:
        raise KeyError(
            f"subset_variables: {missing} not found in coords['variable'] "
            f"(available: {var_list})"
        )
    indices = [var_list.index(v) for v in target_variables]
    idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=data_tensor.device)
    subset_data = torch.index_select(data_tensor, var_axis, idx_tensor)
    subset_coords = OrderedDict(coords)
    subset_coords["variable"] = np.array(target_variables)
    return subset_data, subset_coords

class CBottleUtils:
    """Core utility layer over CBottle model boilerplate.

    Handles device placement, package resolution, zarr I/O, coordinate
    construction, and tensor reshaping.  Models are returned to the caller
    (not stored), so you control what is loaded and when it is freed.

    Parameters
    ----------
    device : str or torch.device, optional
        Target device.  Defaults to ``"cuda"`` if available, else ``"cpu"``.
    random_seed : int
        Seed passed to model constructors.
    package_timeout : int
        Timeout in seconds for model package downloads.
    """

    def __init__(self, device=None, random_seed=None, package_timeout=600):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.random_seed = random_seed
        self.package_timeout = package_timeout

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_era5_zarr(data_root, start_year, start_month, start_day, start_hour):
        """Load a 12-frame (6-hourly) window of ERA5 data from a zarr store.

        Handles month and year boundary crossings automatically.

        Parameters
        ----------
        data_root : str
            Root directory containing ``{year}/era5_raw_{year}_{month:02d}.zarr``.
        start_year, start_month, start_day, start_hour : int
            Start of the 12-frame window.  ``start_hour`` must be in {0, 6, 12, 18}.

        Returns
        -------
        data : torch.Tensor
            Shape ``[12, 1, 34, 721, 1440]`` — (time, lead_time, variable, lat, lon).
        coords : OrderedDict
            Keys: ``time``, ``lead_time``, ``variable``, ``lat``, ``lon``.
        """
        import zarr

        assert start_hour in [0, 6, 12, 18], "start_hour must be 0, 6, 12, or 18"
        _, month_days = calendar.monthrange(start_year, start_month)
        assert start_day <= month_days, f"start_day must be <= {month_days}"

        start_idx = (start_day - 1) * 4 + start_hour // 6

        if start_idx + 12 > month_days * 4:
            zarr_path_1 = os.path.join(
                data_root, str(start_year),
                f"era5_raw_{start_year}_{start_month:02d}.zarr",
            )
            root_1 = zarr.open_group(zarr_path_1, mode="r")
            part_1 = torch.from_numpy(root_1["data"][start_idx:]).unsqueeze(1)

            if start_month != 12:
                next_year, next_month = start_year, start_month + 1
            else:
                next_year, next_month = start_year + 1, 1

            zarr_path_2 = os.path.join(
                data_root, str(next_year),
                f"era5_raw_{next_year}_{next_month:02d}.zarr",
            )
            root_2 = zarr.open_group(zarr_path_2, mode="r")
            remaining = 12 - part_1.shape[0]
            part_2 = torch.from_numpy(root_2["data"][:remaining]).unsqueeze(1)
            input_data = torch.cat([part_1, part_2], dim=0)
        else:
            zarr_path = os.path.join(
                data_root, str(start_year),
                f"era5_raw_{start_year}_{start_month:02d}.zarr",
            )
            root = zarr.open_group(zarr_path, mode="r")
            input_data = torch.from_numpy(
                root["data"][start_idx : start_idx + 12]
            ).unsqueeze(1)

        start_time = datetime(start_year, start_month, start_day, start_hour)
        time_array = np.array(
            [start_time + timedelta(hours=6 * i) for i in range(12)],
            dtype="datetime64[ns]",
        )

        coords = OrderedDict(
            {
                "time": time_array,
                "lead_time": np.array([0], dtype="timedelta64[ns]"),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        assert input_data.shape == (12, 1, 34, 721, 1440), (
            f"Expected shape (12, 1, 34, 721, 1440), got {input_data.shape}"
        )
        return input_data, coords

    # ------------------------------------------------------------------
    # SST override
    # ------------------------------------------------------------------

    @staticmethod
    def set_custom_sst(model, sst_ds):
        """Replace a model's default AMIP SST with a custom SST dataset.

        The SST dataset is used as the conditioning signal for the diffusion
        model (separate from the SST that appears as one of the 45 output
        variables).  This method swaps the dataset *and* rebuilds the
        lat-lon-to-HEALPix regridder so grids with different resolutions
        are handled correctly.

        Parameters
        ----------
        model : CBottleTCGuidance | CBottle3D | CBottleInterpolate
            Any loaded cBottle model that has a ``sst`` attribute.
        sst_ds : xr.Dataset
            Custom SST dataset.  Must contain a ``"tosbcs"`` variable with
            dimensions ``(time, lat, lon)`` in the same units as the default
            AMIP SST (degrees Celsius; the code adds 273.15 to convert to
            Kelvin).  The time coordinate must cover the range you intend to
            run inference over.

        Returns
        -------
        The same model, modified in-place (also returned for convenience).
        """
        model.sst = sst_ds

        # Rebuild the SST regridder to match the (possibly different) grid
        target_grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        lon_center = sst_ds.lon.values
        src_lon = lon_center - lon_center[0]
        target_lon = (target_grid.lon - lon_center[0]) % 360
        grid = earth2grid.latlon.LatLonGrid(sst_ds.lat.values, src_lon)
        model.sst_regridder = grid.get_bilinear_regridder_to(
            target_grid.lat, lon=target_lon
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_tc_model(self, lat_lon=False, sampler_steps=18, sigma_max=200.0,
                      sst_ds=None):
        """Load a CBottleTCGuidance model and return it.

        Parameters
        ----------
        lat_lon : bool
            If False (default), output is in native healpix for downstream
            model use.  Set True to get lat/lon output for visualization.
        sampler_steps : int
            Number of diffusion sampler steps.
        sigma_max : float
            Maximum noise level for the sampler.
        sst_ds : xr.Dataset, optional
            Custom SST dataset to use instead of the default AMIP SST.
            See ``set_custom_sst`` for format requirements.

        Returns
        -------
        CBottleTCGuidance
        """
        os.environ["EARTH2STUDIO_PACKAGE_TIMEOUT"] = str(self.package_timeout)
        tc_package = CBottleTCGuidance.load_default_package()
        tc_model = CBottleTCGuidance.load_model(
            package=tc_package,
            lat_lon=lat_lon,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=self.random_seed,
        )
        tc_model = tc_model.to(self.device)
        if sst_ds is not None:
            self.set_custom_sst(tc_model, sst_ds)
        return tc_model

    def load_datasource_model(
        self, lat_lon=False, sampler_steps=18, sigma_max=200.0, batch_size=4,
        sst_ds=None,
    ):
        """Load a CBottle3D data source model and return it.

        CBottle3D generates complete 45-variable climate fields from scratch,
        conditioned only on SST and time.  The returned frames can be used
        as conditioning for the interpolation model, similar to NaN-padded
        ERA5 or TC guidance frames.

        Parameters
        ----------
        lat_lon : bool
            If False (default), output is in native healpix for downstream
            model use.  Set True to get lat/lon output for visualization.
        sampler_steps : int
            Number of diffusion sampler steps.
        sigma_max : float
            Maximum noise level for the sampler.
        batch_size : int
            Batch size for generation.
        sst_ds : xr.Dataset, optional
            Custom SST dataset to use instead of the default AMIP SST.
            See ``set_custom_sst`` for format requirements.

        Returns
        -------
        CBottle3D
        """
        os.environ["EARTH2STUDIO_PACKAGE_TIMEOUT"] = str(self.package_timeout)
        package = CBottle3D.load_default_package()
        model = CBottle3D.load_model(
            package,
            lat_lon=lat_lon,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=self.random_seed,
            batch_size=batch_size,
        )
        model = model.to(self.device)
        if sst_ds is not None:
            self.set_custom_sst(model, sst_ds)
        return model

    def load_interpolation_model(
        self,
        conditioning_frames=None,
        lat_lon=False,
        lat_lon_out=True,
        sampler_steps=18,
        sigma_max=1000,
        sst_ds=None,
    ):
        """Load a CBottleInterpolate model and return it.

        Parameters
        ----------
        conditioning_frames : list[int] or None
            Frame indices (0–11) the model will condition on.  Pass ``[]``
            or ``None`` for unconditional generation.
        lat_lon : bool
            If False (default), input is in native healpix (from
            ``pad_era5_to_output``).  Set True if supplying lat/lon input
            directly.
        lat_lon_out : bool
            If True (default), output is on a 0.25-degree lat/lon grid
            for visualization.  Set False to stay in healpix.
        sampler_steps : int
            Number of diffusion sampler steps.
        sigma_max : float
            Maximum noise level for the sampler.
        sst_ds : xr.Dataset, optional
            Custom SST dataset to use instead of the default AMIP SST.
            See ``set_custom_sst`` for format requirements.

        Returns
        -------
        CBottleInterpolate
        """
        os.environ["EARTH2STUDIO_PACKAGE_TIMEOUT"] = str(self.package_timeout)
        interp_package = CBottleInterpolate.load_default_package()
        interp_model = CBottleInterpolate.load_model(
            package=interp_package,
            lat_lon=lat_lon,
            lat_lon_out=lat_lon_out,
            seed=self.random_seed,
            conditioning_frames=conditioning_frames,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
        )
        interp_model = interp_model.to(self.device)
        if sst_ds is not None:
            self.set_custom_sst(interp_model, sst_ds)
        return interp_model

    def pad_era5_to_output(self, era5_data, era5_coords):
        """Regrid ERA5 lat/lon frames to healpix and pad non-ERA5 vars with NaN.

        Drop-in replacement for the old ``run_infill``: produces a tensor in
        the ``OUTPUT_VARIABLES`` layout where ERA5-known variables are taken
        directly from ``era5_data`` (regridded to healpix) and the remaining
        variables are NaN.  No diffusion model is invoked, so ERA5-known
        variables are preserved exactly (modulo regridding) rather than being
        perturbed by the infill model.

        Parameters
        ----------
        era5_data : torch.Tensor
            Shape ``[T, 1, 34, 721, 1440]`` from ``load_era5_zarr``.
        era5_coords : OrderedDict
            Coordinates from ``load_era5_zarr``.

        Returns
        -------
        padded_data : torch.Tensor
            Shape ``[T, 1, 45, 49152]`` on the healpix NEST grid.
        padded_coords : OrderedDict
            Coordinates matching ``OUTPUT_VARIABLES`` and the healpix grid.
        """
        hpx_era5_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
            era5_data.to(self.device), era5_coords
        )

        n_time, n_lead, _, n_pixels = hpx_era5_data.shape
        padded_data = torch.full(
            (n_time, n_lead, len(OUTPUT_VARIABLES), n_pixels),
            float("nan"),
            dtype=hpx_era5_data.dtype,
            device=hpx_era5_data.device,
        )
        for era5_idx, var in enumerate(INPUT_VARIABLES):
            out_idx = MODEL_VAR_TO_IDX[var]
            padded_data[:, :, out_idx] = hpx_era5_data[:, :, era5_idx]

        padded_coords = OrderedDict(
            {
                "time": hpx_coords["time"],
                "lead_time": hpx_coords["lead_time"],
                "variable": np.array(OUTPUT_VARIABLES),
                "hpx": hpx_coords["hpx"],
            }
        )
        return padded_data, padded_coords

    def load_and_pad_frame(self, data_root, year, month, day, hour):
        """Load a single ERA5 frame and pad missing variables with NaN.

        Loads only the single requested frame from the zarr store, regrids it
        to the healpix grid, and embeds it into a ``len(OUTPUT_VARIABLES)``-
        channel tensor whose layout matches the interpolation model's input.
        Variables present in ERA5 are copied directly; variables absent from
        ERA5 are filled with NaN.

        Parameters
        ----------
        data_root : str
            Root directory for ERA5 zarr stores.
        year, month, day, hour : int
            Target datetime.  ``hour`` must be in {0, 6, 12, 18}.

        Returns
        -------
        frame : torch.Tensor
            Shape ``[1, 1, C, H]`` — a single frame on the healpix grid with
            ERA5 variables populated and non-ERA5 variables set to NaN.
        """
        import zarr

        assert hour in [0, 6, 12, 18], "hour must be 0, 6, 12, or 18"

        zarr_path = os.path.join(
            data_root, str(year), f"era5_raw_{year}_{month:02d}.zarr",
        )
        root = zarr.open_group(zarr_path, mode="r")
        idx = (day - 1) * 4 + hour // 6
        era5_data = torch.from_numpy(root["data"][idx : idx + 1]).unsqueeze(1)

        target_time = datetime(year, month, day, hour)
        era5_coords = OrderedDict(
            {
                "time": np.array([target_time], dtype="datetime64[ns]"),
                "lead_time": np.array([0], dtype="timedelta64[ns]"),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        padded_data, _ = self.pad_era5_to_output(era5_data, era5_coords)
        frame = padded_data.cpu()
        del era5_data, padded_data
        torch.cuda.empty_cache()
        return frame

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def build_conditioning(conditioning_data, conditioning_frames, start_coords, batch_size=1):
        """Reshape pre-selected conditioning frames for the interpolation model.

        This is purely a reshaping + coordinate construction utility.  You
        decide which frames to pass in and where they came from (NaN-padded
        ERA5, TC guidance, a different date window, etc.).

        For unconditional generation, pass ``conditioning_data=None`` and
        ``conditioning_frames=None`` (or ``[]``).  A dummy all-NaN tensor
        will be constructed so the model skips conditioning entirely.

        Parameters
        ----------
        conditioning_data : torch.Tensor or None
            The conditioning data, shape ``[N, 1, C, H]`` where *N* is the
            number of conditioning data (from any source), and the remaining
            dims match the padded ERA5 output (lead_time=1, variables, hpx).
            Pass ``None`` for unconditional generation.
        conditioning_frames : list[int] or None
            The frame positions (0–11) these correspond to.  Must have
            ``len(conditioning_frames) == N``.
            Pass ``None`` or ``[]`` for unconditional generation.
        start_coords : OrderedDict
            Coordinates used for start time. Expects ``OUTPUT_VARIABLES``-
            length variable list (the 45-variable model layout).
        batch_size : int
            The data is expanded along the batch dimension.

        Returns
        -------
        x_cond : torch.Tensor
            Shape ``[batch_size, 1, N, C, H]``.
        x_coords : OrderedDict
            Coordinates suitable for ``CBottleInterpolate.__call__``.
        """
        if (conditioning_data is None) != (not conditioning_frames):
            raise ValueError(
                "conditioning_data and conditioning_frames must both be provided or both be None/empty"
            )

        if not conditioning_frames:
            # Unconditional generation: produce an all-NaN dummy tensor
            n_vars = len(start_coords["variable"])
            n_hpx = 4**HPX_LEVEL * 12
            x_cond = torch.full((batch_size, 1, 1, n_vars, n_hpx), float('nan'))
            x_coords = OrderedDict(
                {
                    "batch": np.arange(batch_size),
                    "time": np.array([start_coords["time"][0]]),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": start_coords["variable"],
                    "hpx": np.arange(n_hpx),
                }
            )
            return x_cond, x_coords

        assert conditioning_frames == sorted(conditioning_frames), "conditioning_frames must be in ascending order"
        assert conditioning_data.shape[0] == len(conditioning_frames), (
            f"conditioning_data has {conditioning_data.shape[0]} frames but "
            f"{len(conditioning_frames)} indices were given"
        )

        x_cond = conditioning_data.squeeze(1)                       # [N, C, H]
        x_cond = x_cond.unsqueeze(0).unsqueeze(0)             # [1, 1, N, C, H]
        x_cond = x_cond.expand(batch_size, -1, -1, -1, -1)   # [B, 1, N, C, H]

        lead_times = np.array(
            [np.timedelta64(i * 6, "h") for i in range(12)]
        )[conditioning_frames]

        x_coords = OrderedDict(
            {
                "batch": np.arange(batch_size),
                "time": np.array([start_coords["time"][0]]),
                "lead_time": lead_times,
                "variable": start_coords["variable"],
                "hpx": np.arange(4**HPX_LEVEL * 12),
            }
        )
        return x_cond, x_coords

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_tc_guidance(self, tc_model, tc_latitudes, tc_longitudes, tc_time):
        """Run CBottleTCGuidance: lat/lon positions in, a single healpix frame out.

        This wraps the two-step TC workflow (create guidance tensor, then
        run inference) into one call.  The returned tensor is shaped so it
        can be passed directly into ``build_conditioning``.

        Parameters
        ----------
        tc_model : CBottleTCGuidance
            A loaded TC guidance model (from ``load_tc_model``).
        tc_latitudes : torch.Tensor
            TC latitude positions in degrees N.
        tc_longitudes : torch.Tensor
            TC longitude positions in degrees E (0–360 or -180–180).
        tc_time : datetime or np.datetime64
            The time the generated frame should represent.

        Returns
        -------
        tc_frame : torch.Tensor
            Shape ``[1, 1, 45, 49152]`` if healpix (default), or
            ``[1, 1, 45, 721, 1440]`` if ``lat_lon=True``.
            When in healpix, ready to concatenate with other frames and
            pass to ``build_conditioning``.
        """
        times = np.array([tc_time], dtype="datetime64[ns]")

        guidance_tensor, guidance_coords = tc_model.create_guidance_tensor(
            lat_coords=tc_latitudes,
            lon_coords=tc_longitudes,
            times=times,
        )

        with torch.no_grad():
            tc_data, tc_coords = tc_model(guidance_tensor, guidance_coords)

        return tc_data, tc_coords

    def run_datasource(self, datasource_model, times):
        """Run CBottle3D: generate complete climate fields from scratch.

        The returned tensor is shaped so it can be passed directly into
        ``build_conditioning``, just like NaN-padded ERA5 or TC guidance frames.

        Parameters
        ----------
        datasource_model : CBottle3D
            A loaded data source model (from ``load_datasource_model``).
        times : list[datetime] or np.ndarray
            Times to generate fields for.  Each time produces one frame.

        Returns
        -------
        data : torch.Tensor
            Shape ``[N, 1, 45, 49152]`` if healpix (default), or
            ``[N, 1, 45, 721, 1440]`` if ``lat_lon=True``.
            Ready to concatenate with other frames and pass to
            ``build_conditioning``.
        coords : OrderedDict
            Coordinates matching the output grid.
        """
        variables = list(CBottleLexicon.VOCAB.keys())

        with torch.no_grad():
            da = datasource_model(time=times, variable=variables)

        # Convert xr.DataArray -> torch.Tensor + OrderedDict
        tensor = torch.from_numpy(da.values)  # [time, variable, ...]

        # Add lead_time dim of size 1 to match padded-ERA5/TC output shape
        tensor = tensor.unsqueeze(1)  # [time, 1, variable, ...]

        if "hpx" in da.dims:
            coords = OrderedDict(
                {
                    "time": da.coords["time"].values,
                    "lead_time": np.array([0], dtype="timedelta64[ns]"),
                    "variable": da.coords["variable"].values,
                    "hpx": da.coords["hpx"].values,
                }
            )
        else:
            coords = OrderedDict(
                {
                    "time": da.coords["time"].values,
                    "lead_time": np.array([0], dtype="timedelta64[ns]"),
                    "variable": da.coords["variable"].values,
                    "lat": da.coords["lat"].values,
                    "lon": da.coords["lon"].values,
                }
            )

        return tensor, coords

    def run_interpolation(self, interp_model, x_cond, x_coords, conditioning_frames = None):
        """Run CBottleInterpolate on pre-built conditioning data.

        Parameters
        ----------
        interp_model : CBottleInterpolate
            A loaded interpolation model (from ``load_interpolation_model``).
        x_cond : torch.Tensor
            Conditioning tensor from ``build_conditioning``.
        x_coords : OrderedDict
            Coordinates from ``build_conditioning``.
        conditioning_frames : list[int] or None
            Frame indices to condition on.  Pass ``None`` or ``[]`` for
            unconditional generation.  When provided, overrides the model's
            current ``conditioning_frames``.

        Returns
        -------
        interp_data : torch.Tensor
            When the model was loaded with conditioning frames, shape is
            ``[batch, 1, 12, 45, ...]`` (all 12 frames).  The trailing
            dims are ``[721, 1440]`` if ``lat_lon_out=True``, or
            ``[49152]`` if healpix.
        interp_coords : OrderedDict
            Output coordinates with full ``lead_time`` array.
        """
        conditioning_frames = conditioning_frames if conditioning_frames is not None else []
        if conditioning_frames:
            assert x_cond.shape[2] == len(conditioning_frames), (
                f"Expected {len(conditioning_frames)} conditioning frames, but "
                f"x_cond has shape {x_cond.shape}"
            )
        if interp_model.conditioning_frames != conditioning_frames:
            print(f'Model conditioning frames are being changed from {interp_model.conditioning_frames} to {conditioning_frames}')
        interp_model.conditioning_frames = conditioning_frames
        print(f"Running interpolation with conditioning frames {interp_model.conditioning_frames}...")
        with torch.no_grad():
            interp_data, interp_coords = interp_model(x_cond.to(self.device), x_coords)
        return interp_data, interp_coords

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_valid_times(coords, time_idx=0):
        """Compute actual valid datetimes from time + lead_time.

        Parameters
        ----------
        coords : OrderedDict
            Must contain ``"time"`` (datetime64 array) and ``"lead_time"``
            (timedelta64 array).
        time_idx : int, optional
            Index into the ``"time"`` array.  Default 0.

        Returns
        -------
        np.ndarray
            Array of datetime64 values, one per lead_time entry.
        """
        init_time = coords["time"][time_idx]
        return init_time + coords["lead_time"]

    @staticmethod
    def format_valid_times(valid_times):
        """Format datetime64 array as human-readable strings.

        Parameters
        ----------
        valid_times : np.ndarray
            Array of datetime64 values (from ``get_valid_times``).

        Returns
        -------
        list[str]
            Formatted strings like ``"2024-01-15 06:00"``.
        """
        return [
            str(np.datetime_as_string(t, unit="m")).replace("T", " ")
            for t in valid_times
        ]

    # ------------------------------------------------------------------
    # Regridding and plotting
    # ------------------------------------------------------------------
    # NOTE: For interpolation output, the x-axis / frame titles must show
    # actual valid datetimes (time + lead_time), NOT "lead time from
    # initialization".  Use get_valid_times(coords) to compute them.

    @staticmethod
    def regrid_healpix_nest_to_latlon(hpx_data, hpx_coords, hpx_level=HPX_LEVEL):
        """
        Regrid healpix NEST data to lat/lon grid.
        
        Parameters
        ----------
        hpx_data : torch.Tensor
            Data on healpix NEST grid, shape [..., 4**hpx_level * 12]
        hpx_coords : OrderedDict
            Coordinates for hpx_data
        hpx_level : int
            HEALPix level (default 6 for 49152 pixels)
        
        Returns
        -------
        torch.Tensor
            Data on lat/lon grid, shape [..., 721, 1440]
        """
        print('Regridding healpix nest to lat/lon...')

        device = hpx_data.device
        
        # Create healpix grid (NEST ordering)
        hpx_grid = earth2grid.healpix.Grid(
            hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        
        # Create lat/lon grid
        nlat, nlon = 721, 1440
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            nlat, nlon, includes_south_pole=True
        )
        
        # Create regridder
        regridder = earth2grid.get_regridder(hpx_grid, latlon_grid).to(device)
        
        latlon_data = regridder(hpx_data.double()).float()
        latlon_coords = OrderedDict(
            {
                "time": hpx_coords["time"],
                "lead_time": hpx_coords["lead_time"],
                "variable": hpx_coords["variable"],
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        return latlon_data, latlon_coords

    @staticmethod
    def regrid_latlon_to_healpix_nest(latlon_data, latlon_coords, hpx_level=HPX_LEVEL):
        """
        Regrid lat/lon data to healpix NEST grid.

        The source lat/lon grid is inferred from the trailing two dimensions
        of ``latlon_data`` (assumed equiangular with the south pole included),
        so the function works for any lat/lon resolution.

        Parameters
        ----------
        latlon_data : torch.Tensor
            Data on lat/lon grid, shape [..., nlat, nlon]
        latlon_coords : OrderedDict
            Coordinates for latlon_data
        hpx_level : int
            HEALPix level (default 6 for 49152 pixels)

        Returns
        -------
        hpx_data : torch.Tensor
            Data on healpix NEST grid, shape [..., 4**hpx_level * 12]
        hpx_coords : OrderedDict
            Coordinates for hpx_data
        """
        print('Regridding lat/lon to healpix nest...')

        device = latlon_data.device

        nlat, nlon = latlon_data.shape[-2], latlon_data.shape[-1]
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            nlat, nlon, includes_south_pole=True
        )

        hpx_grid = earth2grid.healpix.Grid(
            hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )

        regridder = earth2grid.get_regridder(latlon_grid, hpx_grid).to(device)

        hpx_data = regridder(latlon_data.double()).float()
        hpx_coords = OrderedDict(
            {
                "time": latlon_coords["time"],
                "lead_time": latlon_coords["lead_time"],
                "variable": latlon_coords["variable"],
                "hpx": np.arange(4**hpx_level * 12),
            }
        )
        return hpx_data, hpx_coords

    @staticmethod
    def plot_healpix_nest(variable_dict, hpx_level=HPX_LEVEL):
        """
        Plot a single frame of HEALPix data (NEST ordering).

        Parameters
        ----------
        variable_dict : dict
            Dictionary with keys:
            - "data" : torch.Tensor
                Data on lat/lon grid
            - "coords" : OrderedDict
                Coordinates for healpix data. 
            - "var_name" : str (name of variable to plot)
            - "colormap" : str or matplotlib.colors.Colormap
            - "vmin" : float
            - "vmax" : float
        hpx_level : int
            HEALPix level (default 6).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from earth2grid import healpix
        coord_keys = list(variable_dict['coords'].keys())
        var_name = variable_dict["var_name"]
        var_dim = coord_keys.index("variable")
        hpx_dim = coord_keys.index("hpx")
        var_idx = np.where(variable_dict['coords']['variable'] == var_name)[0][0]
        cmap = variable_dict.get("colormap", "viridis")
        vmin = variable_dict.get("vmin", None)
        vmax = variable_dict.get("vmax", None)

        indexer = [0] * variable_dict.get("data").ndim
        indexer[var_dim] = var_idx
        indexer[hpx_dim] = slice(None)  # Keep all pixels
        # Extract data for time=0, lead=0
        # Shape: [pixel]
        data = variable_dict["data"][tuple(indexer)]
        
        # Move to CPU numpy
        data_np = data.detach().cpu().numpy()
        
        # Reorder NEST to RING
        data_ring = healpix.reorder(
            torch.from_numpy(data_np), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
        ).numpy()
        
        # Convert to 2D image for plotting
        img = healpix.to_double_pixelization(data_ring, fill_value=np.nan)
        
        # ROLL to shift from [-180, 180) to [0, 360)
        # 0 deg is at center of double pixelization, so rolling by W/2 brings it to left.
        shift = img.shape[1] // 2
        img = np.roll(img, shift, axis=1)
        
        img = np.ma.masked_invalid(img)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        im = ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest', extent=[0, 360, -90, 90])
        plt.colorbar(im, ax=ax, label=var_name)
        ax.set_title(f"{var_name} (HEALPix Level {hpx_level})")
        
        ax.axis('off')
        
        return fig, ax

    @staticmethod
    def plot_latlon(variable_dict, extent=None):
        """
        Plot a single frame of Lat/Lon data.

        Parameters
        ----------
        variable_dict : dict
            Dictionary with keys:
            - "data" : torch.Tensor
                Data on lat/lon grid
            - "coords" : OrderedDict
                Coordinates for latlon data.
            - "var_name": str
            - "colormap": str or colormap
            - "vmin": float
            - "vmax": float
        extent : list, optional
            Map extent [lon_min, lon_max, lat_min, lat_max].
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        coord_keys = list(variable_dict['coords'].keys())
        var_name = variable_dict["var_name"]
        var_dim = coord_keys.index("variable")
        lat_dim = coord_keys.index("lat")
        lon_dim = coord_keys.index("lon")
        var_idx = np.where(variable_dict['coords']['variable'] == var_name)[0][0]
        cmap = variable_dict.get("colormap", "viridis")
        vmin = variable_dict.get("vmin", None)
        vmax = variable_dict.get("vmax", None)

        indexer = [0] * variable_dict['data'].ndim
        indexer[var_dim] = var_idx
        indexer[lat_dim] = slice(None)
        indexer[lon_dim] = slice(None)

        data = variable_dict["data"][tuple(indexer)]
        data_np = data.detach().cpu().numpy()

        lons = variable_dict['coords']["lon"]
        lats = variable_dict['coords']["lat"]
        
        # Handle meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.coastlines(color = 'white')
        ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.pcolormesh(
            lon_grid, lat_grid, data_np,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm
        )

        plt.colorbar(im, ax=ax, label=var_name)
        ax.set_title(f"{var_name} (Lat/Lon)")

        return fig, ax

    @staticmethod
    def animate_healpix_nest(variable_dict, interpolated, hpx_level=HPX_LEVEL, output_filename=None, fps=2, interp_factor=1):
        """
        Animate HEALPix data over time with smooth interpolation.

        Parameters
        ----------
        variable_dict : dict
            Dictionary with keys:
            - "data" : torch.Tensor
                Data on lat/lon grid
            - "coords" : OrderedDict
                Coordinates for healpix data
            - "var_name" : str (name of variable to plot)
            - "colormap" : str or matplotlib.colors.Colormap
            - "vmin" : float
            - "vmax" : float
        interpolated : Boolean
            Whether the data is from interpolation model output
        hpx_level : int
            HEALPix level.
        output_filename : str, optional
            Path to save the animation (e.g., 'anim.mp4').
        fps : int
            Frames per second for output.
        interp_factor : int
            Temporal interpolation factor for smoother playback.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        from earth2grid import healpix
        coord_keys = list(variable_dict['coords'].keys())
        var_name = variable_dict["var_name"]
        var_dim = coord_keys.index("variable")
        hpx_dim = coord_keys.index("hpx")
        var_idx = np.where(variable_dict['coords']['variable'] == var_name)[0][0]
        cmap = variable_dict.get("colormap", "viridis")
        vmin = variable_dict.get("vmin", None)
        vmax = variable_dict.get("vmax", None)

        indexer = [0] * variable_dict.get("data").ndim
        indexer[var_dim] = var_idx
        indexer[hpx_dim] = slice(None)  # Keep all pixels
        if interpolated:
            temporal_dim = coord_keys.index("lead_time")
        else:
            temporal_dim = coord_keys.index("time")
        indexer[temporal_dim] = slice(None)  # Keep all time frames
        # Extract [time, pixel] (lead_time=0)
        data = variable_dict["data"][tuple(indexer)]
        data_np = data.detach().cpu().numpy()
        n_times, n_pixels = data_np.shape

        # Compute frame labels from actual valid datetimes
        coords = variable_dict["coords"]
        has_time_info = "time" in coords and len(coords["time"]) > 0
        if has_time_info and "lead_time" in coords:
            valid_times = CBottleUtils.get_valid_times(coords)
            frame_labels = CBottleUtils.format_valid_times(valid_times)
        elif has_time_info:
            frame_labels = CBottleUtils.format_valid_times(coords["time"])
        else:
            frame_labels = [str(i) for i in range(n_times)]

        # Temporal Interpolation (Cubic).  ``interp_factor <= 1`` plays the
        # original frames as-is; the cubic spline is skipped because it
        # propagates NaN globally (any NaN in ``data_np`` would poison the
        # entire output), which silently blanks out otherwise-valid frames.
        orig_indices = np.arange(n_times)
        if interp_factor > 1:
            new_indices = np.linspace(0, n_times - 1, (n_times - 1) * interp_factor + 1)
            # Interpolating on raw HEALPix data first is valid and efficient
            f = interp1d(orig_indices, data_np, axis=0, kind='cubic')
            data_smooth = f(new_indices)  # [n_frames, n_pixels]
        else:
            new_indices = orig_indices
            data_smooth = data_np

        n_frames = data_smooth.shape[0]

        # Map smoothed frame indices back to nearest original frame label
        smooth_frame_labels = []
        for i in range(n_frames):
            nearest_orig = min(int(round(new_indices[i])), len(frame_labels) - 1)
            smooth_frame_labels.append(frame_labels[nearest_orig])

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Initial frame
        # Reorder NEST to RING
        frame0_ring = healpix.reorder(
            torch.from_numpy(data_smooth[0]), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
        ).numpy()
        img0 = healpix.to_double_pixelization(frame0_ring, fill_value=np.nan)
        
        # ROLL to shift
        shift = img0.shape[1] // 2
        img0 = np.roll(img0, shift, axis=1)
        
        img0 = np.ma.masked_invalid(img0)
        
        im = ax.imshow(img0, cmap=cmap, norm=norm, interpolation='nearest', animated=True, extent=[0, 360, -90, 90])
        plt.colorbar(im, ax=ax, label=var_name, pad=0.08)
        title = ax.set_title(f"{var_name} — {smooth_frame_labels[0]}")
        ax.axis('off')

        def update(frame_idx):
            # Reorder
            frame_data = data_smooth[frame_idx]
            frame_ring = healpix.reorder(
                torch.from_numpy(frame_data), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
            ).numpy()
            img = healpix.to_double_pixelization(frame_ring, fill_value=np.nan)

            # ROLL
            img = np.roll(img, shift, axis=1)

            img = np.ma.masked_invalid(img)

            im.set_data(img)
            title.set_text(f"{var_name} — {smooth_frame_labels[frame_idx]}")
            return im, title

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=1000 // fps, blit=True
        )

        if output_filename:
            print(f"Saving animation to {output_filename}...")
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_filename, writer=writer)
            print("Done.")
        
        plt.close(fig)
        return anim

    @staticmethod
    def animate_latlon(variable_dict, interpolated, output_filename=None, fps=2, interp_factor=1, extent=None):
        """
        Animate Lat/Lon data over time with smooth interpolation.

        Parameters
        ----------
        variable_dict : dict
            Dictionary with keys:
            - "data" : torch.Tensor
                Data on lat/lon grid
            - "coords" : OrderedDict
                Coordinates for latlon data.
            - "var_name" : str (name of variable to plot)
            - "colormap" : str or matplotlib.colors.Colormap
            - "vmin" : float
            - "vmax" : float
        interpolated : Boolean
            Whether the data is from interpolation model output
        output_filename : str, optional
            Path to save animation.
        fps : int
            Frames per second.
        interp_factor : int
            Temporal interpolation factor for smoother playback.
        extent : list, optional
            Map extent [lon_min, lon_max, lat_min, lat_max].

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        coord_keys = list(variable_dict['coords'].keys())
        var_name = variable_dict["var_name"]
        var_dim = coord_keys.index("variable")
        lat_dim = coord_keys.index("lat")
        lon_dim = coord_keys.index("lon")
        var_idx = np.where(variable_dict['coords']['variable'] == var_name)[0][0]
        cmap = variable_dict.get("colormap", "viridis")
        vmin = variable_dict.get("vmin", None)
        vmax = variable_dict.get("vmax", None)

        indexer = [0] * variable_dict['data'].ndim
        indexer[var_dim] = var_idx
        indexer[lat_dim] = slice(None)
        indexer[lon_dim] = slice(None)
        if interpolated:
            temporal_dim = coord_keys.index("lead_time")
        else:
            temporal_dim = coord_keys.index("time")
        indexer[temporal_dim] = slice(None)

        # Extract [time, lat, lon]
        data = variable_dict["data"][tuple(indexer)]
        data_np = data.detach().cpu().numpy()
        n_times, n_lat, n_lon = data_np.shape

        # Compute frame labels from actual valid datetimes
        coords = variable_dict["coords"]
        has_time_info = "time" in coords and len(coords["time"]) > 0
        if has_time_info and "lead_time" in coords:
            valid_times = CBottleUtils.get_valid_times(coords)
            frame_labels = CBottleUtils.format_valid_times(valid_times)
        elif has_time_info:
            frame_labels = CBottleUtils.format_valid_times(coords["time"])
        else:
            frame_labels = [str(i) for i in range(n_times)]

        # Temporal Interpolation.  ``interp_factor <= 1`` plays the original
        # frames as-is; the cubic spline is skipped because it propagates NaN
        # globally (any NaN in ``data_np`` would poison the entire output),
        # which silently blanks out otherwise-valid frames.
        orig_indices = np.arange(n_times)
        if interp_factor > 1:
            new_indices = np.linspace(0, n_times - 1, (n_times - 1) * interp_factor + 1)
            f = interp1d(orig_indices, data_np, axis=0, kind='cubic')
            data_smooth = f(new_indices) # [n_frames, n_lat, n_lon]
        else:
            new_indices = orig_indices
            data_smooth = data_np
        n_frames = data_smooth.shape[0]

        # Map smoothed frame indices back to nearest original frame label
        smooth_frame_labels = []
        for i in range(n_frames):
            nearest_orig = min(int(round(new_indices[i])), len(frame_labels) - 1)
            smooth_frame_labels.append(frame_labels[nearest_orig])

        # Setup Plot
        lons = variable_dict['coords']["lon"]
        lats = variable_dict['coords']["lat"]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.coastlines(color = 'white')
        ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Initial frame
        im = ax.pcolormesh(
            lon_grid, lat_grid, data_smooth[0],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm
        )
        plt.colorbar(im, ax=ax, label=var_name, pad=0.08)
        title = ax.set_title(f"{var_name} — {smooth_frame_labels[0]}")
        def update(frame_idx):
            im.set_array(data_smooth[frame_idx].ravel())
            title.set_text(f"{var_name} — {smooth_frame_labels[frame_idx]}")
            return im, title

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=1000 // fps, blit=False
        )
        # blit=False often safer for cartopy/pcolormesh

        if output_filename:
            print(f"Saving animation to {output_filename}...")
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_filename, writer=writer)
            print("Done.")

        plt.close(fig)
        return anim
    
    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @staticmethod
    def free_model(model):
        """Delete a model and release GPU memory.

        Parameters
        ----------
        model : torch.nn.Module
            The model to free.  The caller's reference becomes stale after
            this call.
        """
        del model
        torch.cuda.empty_cache()
        gc.collect()
