"""Compute the ageostrophic-fraction diagnostics for one mode's ensemble.

Case-agnostic: takes whatever ensemble + ERA5 zarr pair you point it
at and produces per-member domain-mean ``|V_a| / |V|``, ``|V|``,
``|V_g|``, ``|V_a|`` traces plus the ERA5 reference.  The default
domain is the 30°-60° midlatitude band in both hemispheres at all
longitudes; configure the band via ``--lat-min``/``--lat-max`` and
(optionally) ``--lon-min``/``--lon-max``.

The domain mean uses ``cos(lat)`` area weighting because the band
spans a wide latitude range and grid cells shrink toward the poles —
without weighting, high latitudes would be over-represented in the
average.

This script writes the numerics for ONE conditioning mode. The aggregated
plot across all three modes is produced separately by
``aggregate_ageostrophic.py`` after all three per-mode jobs finish.

Outputs (in --output-dir):
    ageostrophic_fraction_<mode>.npz   per-member domain-mean traces

Usage:
    python compute_ageostrophic.py \\
        --ensemble-zarr /path/to/ensemble_f11_n1000.zarr \\
        --era5-zarr     /path/to/era5_reference.zarr \\
        --output-dir    /path/to/diagnostics/ageostrophic \\
        --mode          end \\
        --case-name     "<display name>"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent / "_shared"
sys.path.insert(0, str(SHARED_DIR))

from _dispatch_lib import lead_hours  # noqa: E402

from custom_utils.derived_variables import geostrophic_wind  # noqa: E402
from custom_utils.ensemble_loading import variable_index  # noqa: E402
from custom_utils.interpolation_utils import CBottleUtils  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_ensemble(path: str) -> zarr.Group:
    root = zarr.open_group(path, mode="r")
    if root["data"].ndim != 5 or "hpx" not in root.array_keys():
        raise ValueError(
            f"{path}: expected HEALPix ensemble layout (ndim=5 with 'hpx' "
            f"coord); got ndim={root['data'].ndim}, "
            f"arrays={sorted(root.array_keys())}"
        )
    return root


def _hpx_level_from_root(root: zarr.Group) -> int:
    level = root.attrs.get("hpx_level")
    if level is not None:
        return int(level)
    n_pix_arr = int(root["hpx"].shape[0])
    return int(np.log2(n_pix_arr // 12) / 2)


def _regrid_ens_batch_to_latlon(
    ens_batch: torch.Tensor,
    lead_times: np.ndarray,
    hpx_level: int,
) -> torch.Tensor:
    """Regrid an HPX ensemble batch ``[bs, n_leads, n_pix]`` to ``[bs, n_leads, n_lat, n_lon]``.

    The ageostrophic computation lives in lat/lon (Coriolis as a
    function of latitude, finite-difference derivatives in physical
    x/y), so each batch must be projected back to lat/lon at compute
    time.
    """
    bs = ens_batch.shape[0]
    ens_4d = ens_batch.unsqueeze(2)  # [bs, n_leads, 1, n_pix]
    hpx_coords = {
        "time": np.arange(bs),
        "lead_time": lead_times,
        "variable": np.array(["_"]),
        "hpx": np.arange(ens_batch.shape[-1]),
    }
    latlon_data, _ = CBottleUtils.regrid_healpix_nest_to_latlon(
        ens_4d,
        hpx_coords,
        hpx_level=hpx_level,
    )
    return latlon_data[:, :, 0, :, :]  # [bs, n_leads, n_lat, n_lon]


def _open_era5(path: str) -> zarr.Group:
    root = zarr.open_group(path, mode="r")
    if root["data"].ndim != 5:
        raise ValueError(
            f"{path}: expected ERA5 layout (ndim=5), got ndim={root['data'].ndim}"
        )
    return root


def _domain_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float | None,
    lon_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_idx, lon_idx) for the configured |lat| band.

    The latitude mask is applied to ``|lat|`` so ``[lat_min, lat_max]``
    selects both Northern and Southern Hemisphere bands in one shot.

    Longitude bounds, if given, are matched after wrapping any [0, 360)
    longitudes back into [-180, 180).
    """
    abs_lat = np.abs(lat)
    lat_mask = (abs_lat >= lat_min) & (abs_lat <= lat_max)

    if lon_min is None and lon_max is None:
        lon_mask = np.ones_like(lon, dtype=bool)
    else:
        lon_w = np.where(lon > 180, lon - 360, lon)
        lo = -180.0 if lon_min is None else lon_min
        hi = 180.0 if lon_max is None else lon_max
        lon_mask = (lon_w >= lo) & (lon_w <= hi)

    return np.where(lat_mask)[0], np.where(lon_mask)[0]


def _weighted_domain_means(
    spd_total: torch.Tensor,
    spd_geo: torch.Tensor,
    spd_ageo: torch.Tensor,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    lat_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``cos(lat)``-weighted means of ``|V|``, ``|V_g|``, ``|V_a|`` over the band.

    Shares one finite mask across all three fields (driven by ``V_g``,
    which is NaN at the equator/poles in ``geostrophic_wind``) so the
    returned means are mutually consistent: ``mean_va / mean_v`` matches
    the original ratio diagnostic, and the three lines stay coherent in
    the multi-panel plot.
    """
    tot = spd_total[..., lat_idx, :][..., :, lon_idx]
    geo = spd_geo[..., lat_idx, :][..., :, lon_idx]
    ageo = spd_ageo[..., lat_idx, :][..., :, lon_idx]
    finite = torch.isfinite(tot) & torch.isfinite(geo) & torch.isfinite(ageo)
    w = lat_weights.view(-1, 1) * finite.to(tot.dtype)
    w_sum = w.sum(dim=(-2, -1))
    tot_safe = torch.where(finite, tot, tot.new_zeros(()))
    geo_safe = torch.where(finite, geo, geo.new_zeros(()))
    ageo_safe = torch.where(finite, ageo, ageo.new_zeros(()))
    return (
        (tot_safe * w).sum(dim=(-2, -1)) / w_sum,
        (geo_safe * w).sum(dim=(-2, -1)) / w_sum,
        (ageo_safe * w).sum(dim=(-2, -1)) / w_sum,
    )


def _means_to_diag_dict(
    mean_v: torch.Tensor,
    mean_vg: torch.Tensor,
    mean_va: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Pack the three weighted means + the derived ratio into a numpy dict."""

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy().astype(np.float32)

    v_np, vg_np, va_np = _np(mean_v), _np(mean_vg), _np(mean_va)
    return {
        "ratio": va_np / v_np,
        "mean_v": v_np,
        "mean_vg": vg_np,
        "mean_va": va_np,
    }


def _format_band(
    lat_min: float, lat_max: float, lon_min: float | None, lon_max: float | None
) -> str:
    lat_part = f"{lat_min:g}°–{lat_max:g}° |lat|"
    if lon_min is None and lon_max is None:
        return f"{lat_part} (all longitudes)"
    lo = -180.0 if lon_min is None else lon_min
    hi = 180.0 if lon_max is None else lon_max
    return f"{lat_part}, lon ∈ [{lo:g}°, {hi:g}°]"


# ---------------------------------------------------------------------------
# Core computation — single zarr pass per dataset
# ---------------------------------------------------------------------------


def _load_era5_uvz(
    era_root: zarr.Group,
    var_u: str,
    var_v: str,
    var_z: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read ``(u, v, z)`` time series from the ERA5 zarr."""
    iu = variable_index(era_root, var_u)
    iv = variable_index(era_root, var_v)
    iz = variable_index(era_root, var_z)
    u = torch.from_numpy(np.array(era_root["data"][:, 0, iu, :, :])).float()
    v = torch.from_numpy(np.array(era_root["data"][:, 0, iv, :, :])).float()
    z = torch.from_numpy(np.array(era_root["data"][:, 0, iz, :, :])).float()
    return u, v, z


def _spds_from_uvz(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ``(|V|, |V_g|, |V_a|)`` field triple from ``(u, v, z)``."""
    ug, vg = geostrophic_wind(z, lat, lon)
    spd_total = torch.sqrt(u * u + v * v)
    spd_geo = torch.sqrt(ug * ug + vg * vg)
    spd_ageo = torch.sqrt((u - ug) ** 2 + (v - vg) ** 2)
    return spd_total, spd_geo, spd_ageo


def _round_trip_uvz_through_hpx(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``lat/lon -> HPX -> lat/lon`` to ``(u, v, z)``.

    The CBottle conditioning frame goes through this exact round trip
    (lat/lon ERA5 -> HPX inside ``pad_era5_to_output`` -> lat/lon at the
    interpolation model's output).  Round-tripping the ERA5 reference the
    same way gives an apples-to-apples baseline at the conditioning frame
    and an upper bound on regridding-induced smoothing for the others
    (which only see HPX -> lat/lon, i.e. one of the two passes).
    """
    from collections import OrderedDict

    from custom_utils.interpolation_utils import CBottleUtils

    stacked = torch.stack([u, v, z], dim=1)  # [T, 3, nlat, nlon]
    dummy_coords = OrderedDict(
        {
            "time": np.arange(stacked.shape[0]),
            "lead_time": np.array([0]),
            "variable": np.array(["u", "v", "z"]),
            "lat": lat,
            "lon": lon,
        }
    )
    hpx_data, hpx_coords = CBottleUtils.regrid_latlon_to_healpix_nest(
        stacked, dummy_coords
    )
    rt_data, _ = CBottleUtils.regrid_healpix_nest_to_latlon(hpx_data, hpx_coords)
    return rt_data[:, 0], rt_data[:, 1], rt_data[:, 2]


def compute_era5(
    era_root: zarr.Group,
    var_u: str,
    var_v: str,
    var_z: str,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    lat_weights: torch.Tensor,
    round_trip: bool = False,
) -> dict[str, np.ndarray]:
    """Ageostrophic diagnostics from ERA5, one value per lead.

    Returns ``{ratio, mean_v, mean_vg, mean_va}``.  When ``round_trip=True``,
    ``(u, v, z)`` are passed through the ``lat/lon -> HPX -> lat/lon`` round
    trip before the diagnostic, matching the smoothing applied to the
    CBottle conditioning frame.
    """
    u, v, z = _load_era5_uvz(era_root, var_u, var_v, var_z)
    if round_trip:
        u, v, z = _round_trip_uvz_through_hpx(u, v, z, lat, lon)
    spd_total, spd_geo, spd_ageo = _spds_from_uvz(u, v, z, lat, lon)
    return _means_to_diag_dict(
        *_weighted_domain_means(
            spd_total, spd_geo, spd_ageo, lat_idx, lon_idx, lat_weights
        )
    )


def compute_ensemble(
    ens_root: zarr.Group,
    var_u: str,
    var_v: str,
    var_z: str,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    lat_weights: torch.Tensor,
    n_members: int,
    batch_size: int,
    hpx_level: int,
) -> dict[str, np.ndarray]:
    """Per-member ageostrophic diagnostics.

    The ensemble is stored on HEALPix; each batch is regridded HPX -> lat/lon
    before the geostrophic-balance computation, which intrinsically lives in
    a lat/lon coordinate frame (Coriolis as a function of latitude, finite
    differences in physical x/y).

    Returns ``{ratio, mean_v, mean_vg, mean_va}`` with each array shaped
    ``(n_members, n_leads)``.
    """
    iu = variable_index(ens_root, var_u)
    iv = variable_index(ens_root, var_v)
    iz = variable_index(ens_root, var_z)

    n_leads = ens_root["data"].shape[2]
    lead_times = np.array(ens_root["lead_time"][:])
    out: dict[str, np.ndarray] = {
        k: np.empty((n_members, n_leads), dtype=np.float32)
        for k in ("ratio", "mean_v", "mean_vg", "mean_va")
    }

    data = ens_root["data"]
    t_start = time.time()
    for batch_start in range(0, n_members, batch_size):
        batch_end = min(batch_start + batch_size, n_members)
        bs = batch_end - batch_start

        u_hpx = torch.from_numpy(
            np.array(data[batch_start:batch_end, 0, :, iu, :])
        ).float()
        v_hpx = torch.from_numpy(
            np.array(data[batch_start:batch_end, 0, :, iv, :])
        ).float()
        z_hpx = torch.from_numpy(
            np.array(data[batch_start:batch_end, 0, :, iz, :])
        ).float()
        u = _regrid_ens_batch_to_latlon(u_hpx, lead_times, hpx_level)
        v = _regrid_ens_batch_to_latlon(v_hpx, lead_times, hpx_level)
        z = _regrid_ens_batch_to_latlon(z_hpx, lead_times, hpx_level)
        del u_hpx, v_hpx, z_hpx

        ug, vg = geostrophic_wind(z, lat, lon)
        del z
        spd_total = torch.sqrt(u * u + v * v)
        spd_geo = torch.sqrt(ug * ug + vg * vg)
        spd_ageo = torch.sqrt((u - ug) ** 2 + (v - vg) ** 2)
        del u, v, ug, vg

        means = _means_to_diag_dict(
            *_weighted_domain_means(
                spd_total, spd_geo, spd_ageo, lat_idx, lon_idx, lat_weights
            )
        )
        del spd_total, spd_geo, spd_ageo

        for k, arr in means.items():
            out[k][batch_start:batch_end] = arr

        elapsed = time.time() - t_start
        done = batch_end
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_members - done) / rate if rate > 0 else float("nan")
        print(
            f"  [{done:4d}/{n_members}]  batch={bs:3d}  "
            f"elapsed={elapsed:6.1f}s  rate={rate:5.2f} mem/s  eta={eta:6.1f}s",
            flush=True,
        )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    parser.add_argument("--ensemble-zarr", required=True)
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["start", "end", "both"],
        help="Conditioning mode; encoded in the output filename.",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=30.0,
        help="Lower edge of |lat| band (default 30°, i.e. tropical edge of midlatitudes).",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=60.0,
        help="Upper edge of |lat| band (default 60°, i.e. polar edge of midlatitudes).",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=None,
        help="Optional western longitude bound (default: all longitudes).",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=None,
        help="Optional eastern longitude bound (default: all longitudes).",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        required=True,
        help="Display name used in plot titles (e.g. 'Hurricane Sandy').",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Ensemble IC as an ISO 8601 UTC string (e.g. 2012-10-27T06:00:00Z). "
        "If omitted the lead-axis falls back to UTC labels starting at "
        "1970-01-01.  The dispatcher always supplies this.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="IANA timezone name for axis ticks (e.g. America/New_York).  "
        "Defaults to UTC when the case YAML omits a timezone.",
    )
    parser.add_argument("--level", type=int, default=500)
    parser.add_argument(
        "--n-members",
        type=int,
        default=None,
        help="Members to process (default: all in the zarr).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Members per batch (memory/throughput knob).",
    )
    parser.add_argument(
        "--conditioning-frame",
        type=int,
        action="append",
        default=None,
        dest="conditioning_frames",
        help="Frame index that is pinned to ERA5 in this run.  Repeatable "
        "(e.g. --conditioning-frame 0 --conditioning-frame 11 for "
        "both-mode runs).  Each value is drawn as a thin gray dashed "
        "vertical on every panel.",
    )

    args = parser.parse_args()

    if args.lat_min < 0 or args.lat_max <= args.lat_min:
        raise SystemExit(
            f"ERROR: --lat-min/--lat-max must satisfy 0 <= lat_min < lat_max "
            f"(got {args.lat_min}, {args.lat_max})."
        )

    conditioning_frames: list[int] = list(args.conditioning_frames or [])

    var_u = f"u{args.level}"
    var_v = f"v{args.level}"
    var_z = f"z{args.level}"

    print(f"[ageo] case        : {args.case_name}")
    print(f"[ageo] mode        : {args.mode}")
    print(f"[ageo] ensemble    : {args.ensemble_zarr}")
    print(f"[ageo] era5        : {args.era5_zarr}")
    print(f"[ageo] level       : {args.level}  ({var_u}, {var_v}, {var_z})")
    print(f"[ageo] cond frames : {conditioning_frames or '(none)'}")

    ens_root = _open_ensemble(args.ensemble_zarr)
    era_root = _open_era5(args.era5_zarr)

    # The ensemble lives on HEALPix and has no lat/lon coords; the lat/lon
    # grid for ageostrophic computation comes from ERA5.  ERA5's grid
    # matches the regridder's lat/lon target (721x1440), so the regridded
    # ensemble lands on the same axes.
    lat = np.array(era_root["lat"][:])
    lon = np.array(era_root["lon"][:])
    lead_times = np.array(ens_root["lead_time"][:])
    lead_h = lead_hours(lead_times)
    hpx_level = _hpx_level_from_root(ens_root)

    total_members = int(ens_root["data"].shape[0])
    n_members = (
        total_members if args.n_members is None else min(args.n_members, total_members)
    )
    print(f"[ageo] hpx level   : {hpx_level} ({12 * 4**hpx_level} pixels)")
    print(f"[ageo] members     : {n_members} / {total_members} in zarr")
    print(f"[ageo] batch size  : {args.batch_size}")
    print(f"[ageo] n_leads     : {len(lead_h)}")

    lat_idx, lon_idx = _domain_indices(
        lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise SystemExit(
            f"ERROR: configured band does not intersect the grid "
            f"(lat=[{args.lat_min},{args.lat_max}], "
            f"lon=[{args.lon_min},{args.lon_max}])."
        )
    lat_weights = torch.from_numpy(np.cos(np.deg2rad(lat[lat_idx])).astype(np.float32))
    band_label = _format_band(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[ageo] band        : {band_label}  "
        f"({lat_idx.size} lat × {lon_idx.size} lon = "
        f"{lat_idx.size * lon_idx.size} pts)"
    )

    print("[ageo] computing ERA5 diagnostics...", flush=True)
    t0 = time.time()
    era_diag = compute_era5(
        era_root, var_u, var_v, var_z, lat, lon, lat_idx, lon_idx, lat_weights
    )
    print(f"[ageo] ERA5 done in {time.time() - t0:.1f}s")

    print(
        "[ageo] computing ERA5 round-tripped (lat/lon -> HPX -> lat/lon) "
        "diagnostics...",
        flush=True,
    )
    t0 = time.time()
    era_rt_diag = compute_era5(
        era_root,
        var_u,
        var_v,
        var_z,
        lat,
        lon,
        lat_idx,
        lon_idx,
        lat_weights,
        round_trip=True,
    )
    print(f"[ageo] ERA5 round-trip done in {time.time() - t0:.1f}s")

    print("[ageo] computing ensemble per-member diagnostics...", flush=True)
    t0 = time.time()
    ens_diag = compute_ensemble(
        ens_root,
        var_u,
        var_v,
        var_z,
        lat,
        lon,
        lat_idx,
        lon_idx,
        lat_weights,
        n_members=n_members,
        batch_size=args.batch_size,
        hpx_level=hpx_level,
    )
    print(f"[ageo] ensemble done in {time.time() - t0:.1f}s")

    # Per-conditioning-frame sanity summary: at a conditioned frame the
    # ensemble should be pinned to (round-tripped) ERA5, so EnsRange ought
    # to be narrow and EnsMean ought to track ERA5_RT closely.
    for cf in conditioning_frames:
        if 0 <= cf < len(lead_h):
            print(f"[ageo] diagnostics at conditioned frame {cf} ({lead_h[cf]:.1f}h):")
            for label, key, fmt in [
                ("|V_a|/|V|", "ratio", ".3f"),
                ("mean |V|  ", "mean_v", ".2f"),
                ("mean |V_g|", "mean_vg", ".2f"),
                ("mean |V_a|", "mean_va", ".2f"),
            ]:
                print(
                    f"  {label}  ERA5={era_diag[key][cf]:{fmt}}  "
                    f"ERA5_RT={era_rt_diag[key][cf]:{fmt}}  "
                    f"EnsMean={ens_diag[key][:, cf].mean():{fmt}}  "
                    f"EnsRange=[{ens_diag[key][:, cf].min():{fmt}}, "
                    f"{ens_diag[key][:, cf].max():{fmt}}]"
                )

    npz_path = output_dir / f"ageostrophic_fraction_{args.mode}.npz"
    np.savez(
        npz_path,
        ratio_ens=ens_diag["ratio"],
        mean_v_ens=ens_diag["mean_v"],
        mean_vg_ens=ens_diag["mean_vg"],
        mean_va_ens=ens_diag["mean_va"],
        ratio_era=era_diag["ratio"],
        mean_v_era=era_diag["mean_v"],
        mean_vg_era=era_diag["mean_vg"],
        mean_va_era=era_diag["mean_va"],
        ratio_era_rt=era_rt_diag["ratio"],
        mean_v_era_rt=era_rt_diag["mean_v"],
        mean_vg_era_rt=era_rt_diag["mean_vg"],
        mean_va_era_rt=era_rt_diag["mean_va"],
        lead_hours=np.asarray(lead_h, dtype=np.float64),
        lat_min=np.float32(args.lat_min),
        lat_max=np.float32(args.lat_max),
        lon_min=np.float32(-180.0 if args.lon_min is None else args.lon_min),
        lon_max=np.float32(180.0 if args.lon_max is None else args.lon_max),
        level=np.int32(args.level),
        conditioning_frames=np.asarray(conditioning_frames, dtype=np.int32),
        case_name=np.array(args.case_name),
        band_label=np.array(band_label),
        mode=np.array(args.mode),
        start_time=np.array(args.start_time or "1970-01-01T00:00:00Z"),
        timezone=np.array(args.timezone),
    )
    print(f"[ageo] wrote {npz_path}")
    print("[ageo] done.")


if __name__ == "__main__":
    main()
