"""Derived atmospheric variables from raw fields.

All functions operate on PyTorch tensors with lat/lon as the **last two**
axes.  Arbitrary leading batch dimensions (ensemble, lead_time, etc.) are
supported transparently.

Latitude and longitude arrays are in **degrees** and assumed to form a
regular (equirectangular) grid.
"""

from __future__ import annotations

import numpy as np
import torch

EARTH_RADIUS_M = 6.371e6
EARTH_OMEGA = 7.2921e-5  # rad/s


# ---------------------------------------------------------------------------
# Helpers: spherical finite differences
# ---------------------------------------------------------------------------

def _spherical_d_dlon(
    field: torch.Tensor, lat_rad: torch.Tensor, dlon_rad: float
) -> torch.Tensor:
    """∂field/∂λ on a regular lat-lon grid using central differences.

    Longitude is periodic, so we use ``torch.roll``.
    Returns the derivative scaled by 1/(R cos φ).
    """
    df_dlon = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / (
        2.0 * dlon_rad
    )
    cos_lat = torch.cos(lat_rad)  # [n_lat]
    # Broadcast: (..., n_lat, n_lon) / (n_lat, 1)
    return df_dlon / (EARTH_RADIUS_M * cos_lat.unsqueeze(-1))


def _apply_pole_mask(
    field: torch.Tensor, lat_rad: torch.Tensor, pole_mask_deg: float
) -> torch.Tensor:
    """Set latitude rows within ``pole_mask_deg`` of ±90° to NaN.

    Vorticity, divergence, and the zonal-gradient term all divide by
    cos φ, which is zero (or tiny on offset grids) at the poles.
    """
    if pole_mask_deg <= 0.0:
        return field
    pole_rows = torch.abs(lat_rad) > np.deg2rad(90.0 - pole_mask_deg)
    if not bool(pole_rows.any()):
        return field
    field = field.clone()
    field[..., pole_rows, :] = float("nan")
    return field


def _spherical_d_dlat(field: torch.Tensor, dlat_rad: float) -> torch.Tensor:
    """∂field/∂φ on a regular lat-lon grid using central differences.

    Uses one-sided differences at the first and last latitude rows.
    Returns the derivative scaled by 1/R.
    """
    df = torch.zeros_like(field)
    # Central differences for interior
    df[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dlat_rad)
    # One-sided at boundaries
    df[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dlat_rad
    df[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dlat_rad
    return df / EARTH_RADIUS_M


# ---------------------------------------------------------------------------
# Wind speed
# ---------------------------------------------------------------------------

def wind_speed(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute wind speed from u and v components.

    Parameters
    ----------
    u, v : torch.Tensor
        Zonal and meridional wind components.  Same shape, with lat/lon
        as the last two axes.

    Returns
    -------
    torch.Tensor
        Wind speed, same shape as inputs.
    """
    return torch.sqrt(u ** 2 + v ** 2)


# ---------------------------------------------------------------------------
# Vorticity and divergence
# ---------------------------------------------------------------------------

def relative_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    pole_mask_deg: float = 0.5,
) -> torch.Tensor:
    r"""Relative vorticity on a lat-lon grid.

    .. math::
        \zeta = \frac{1}{R \cos\varphi}
        \left[\frac{\partial v}{\partial \lambda}
              - \frac{\partial (u \cos\varphi)}{\partial \varphi}\right]

    The 1/cos φ factor is singular at the poles, so latitudes within
    ``pole_mask_deg`` of ±90° are returned as NaN.

    Parameters
    ----------
    u, v : torch.Tensor
        Wind components, shape ``(..., n_lat, n_lon)``.
    lat, lon : np.ndarray
        1-D coordinate arrays in degrees.
    pole_mask_deg : float, optional
        Mask latitudes within this many degrees of the poles (default 0.5°).
        Set to 0 to disable masking.

    Returns
    -------
    torch.Tensor
        Relative vorticity, same shape as inputs.  NaN at masked polar rows.
    """
    lat_rad = torch.tensor(np.deg2rad(lat), dtype=u.dtype, device=u.device)
    dlon_rad = float(np.deg2rad(lon[1] - lon[0]))
    dlat_rad = float(np.deg2rad(lat[1] - lat[0]))

    cos_lat = torch.cos(lat_rad)  # [n_lat]

    # ∂v/∂λ  / (R cos φ)
    dv_dlon = _spherical_d_dlon(v, lat_rad, dlon_rad)

    # ∂(u cos φ) / ∂φ  / R
    u_cos = u * cos_lat.unsqueeze(-1)
    du_cos_dlat = _spherical_d_dlat(u_cos, dlat_rad)

    # ζ = (1/cos φ) * [above difference], but _spherical_d_dlat already
    # divides by R, so we just need the 1/cos φ factor on du_cos_dlat.
    zeta = dv_dlon - du_cos_dlat / cos_lat.unsqueeze(-1)
    return _apply_pole_mask(zeta, lat_rad, pole_mask_deg)


def horizontal_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    pole_mask_deg: float = 0.5,
) -> torch.Tensor:
    r"""Horizontal divergence on a lat-lon grid.

    .. math::
        D = \frac{1}{R \cos\varphi}
        \left[\frac{\partial u}{\partial \lambda}
              + \frac{\partial (v \cos\varphi)}{\partial \varphi}\right]

    The 1/cos φ factor is singular at the poles, so latitudes within
    ``pole_mask_deg`` of ±90° are returned as NaN.

    Parameters
    ----------
    u, v : torch.Tensor
        Wind components, shape ``(..., n_lat, n_lon)``.
    lat, lon : np.ndarray
        1-D coordinate arrays in degrees.
    pole_mask_deg : float, optional
        Mask latitudes within this many degrees of the poles (default 0.5°).
        Set to 0 to disable masking.

    Returns
    -------
    torch.Tensor
        Horizontal divergence, same shape as inputs.  NaN at masked polar rows.
    """
    lat_rad = torch.tensor(np.deg2rad(lat), dtype=u.dtype, device=u.device)
    dlon_rad = float(np.deg2rad(lon[1] - lon[0]))
    dlat_rad = float(np.deg2rad(lat[1] - lat[0]))

    cos_lat = torch.cos(lat_rad)  # [n_lat]

    # ∂u/∂λ  / (R cos φ)
    du_dlon = _spherical_d_dlon(u, lat_rad, dlon_rad)

    # ∂(v cos φ) / ∂φ  / R
    v_cos = v * cos_lat.unsqueeze(-1)
    dv_cos_dlat = _spherical_d_dlat(v_cos, dlat_rad)

    div = du_dlon + dv_cos_dlat / cos_lat.unsqueeze(-1)
    return _apply_pole_mask(div, lat_rad, pole_mask_deg)


# ---------------------------------------------------------------------------
# Geostrophic / ageostrophic wind
# ---------------------------------------------------------------------------

def geostrophic_wind(
    geopotential: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_mask_deg: float = 5.0,
    pole_mask_deg: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Geostrophic wind from the geopotential field.

    .. math::
        u_g = -\frac{1}{f}\frac{1}{R}\frac{\partial \Phi}{\partial \varphi},
        \quad
        v_g = \frac{1}{f}\frac{1}{R \cos\varphi}
              \frac{\partial \Phi}{\partial \lambda}

    where :math:`\Phi` is geopotential (m²/s²) and :math:`f = 2\Omega\sin\varphi`.

    Near the equator *f* → 0, so geostrophic balance is undefined;
    latitudes within ``lat_mask_deg`` of the equator are set to NaN.
    The zonal-gradient term also carries a 1/cos φ factor that is
    singular at the poles, so latitudes within ``pole_mask_deg`` of
    ±90° are likewise set to NaN.

    Parameters
    ----------
    geopotential : torch.Tensor
        Geopotential field (m²/s²), shape ``(..., n_lat, n_lon)``.
    lat, lon : np.ndarray
        1-D coordinate arrays in degrees.
    lat_mask_deg : float, optional
        Mask latitudes within this many degrees of the equator (default 5°).
    pole_mask_deg : float, optional
        Mask latitudes within this many degrees of the poles (default 0.5°).
        Set to 0 to disable polar masking.

    Returns
    -------
    u_g, v_g : torch.Tensor
        Geostrophic wind components, same shape as input.
        NaN at masked equatorial and polar latitudes.
    """
    lat_rad = torch.tensor(np.deg2rad(lat), dtype=geopotential.dtype,
                           device=geopotential.device)
    dlon_rad = float(np.deg2rad(lon[1] - lon[0]))
    dlat_rad = float(np.deg2rad(lat[1] - lat[0]))

    f = 2.0 * EARTH_OMEGA * torch.sin(lat_rad)  # [n_lat]

    # Mask near equator
    equator_mask = torch.abs(lat_rad) < np.deg2rad(lat_mask_deg)
    f_safe = f.clone()
    f_safe[equator_mask] = float("nan")
    inv_f = 1.0 / f_safe  # [n_lat]

    # ∂Φ/∂φ / R
    dphi_dlat = _spherical_d_dlat(geopotential, dlat_rad)

    # ∂Φ/∂λ / (R cos φ)
    dphi_dlon = _spherical_d_dlon(geopotential, lat_rad, dlon_rad)

    u_g = -inv_f.unsqueeze(-1) * dphi_dlat
    v_g = inv_f.unsqueeze(-1) * dphi_dlon

    u_g = _apply_pole_mask(u_g, lat_rad, pole_mask_deg)
    v_g = _apply_pole_mask(v_g, lat_rad, pole_mask_deg)
    return u_g, v_g


def ageostrophic_wind(
    u: torch.Tensor,
    v: torch.Tensor,
    geopotential: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_mask_deg: float = 5.0,
    pole_mask_deg: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ageostrophic wind: total wind minus geostrophic wind.

    Parameters
    ----------
    u, v : torch.Tensor
        Total wind components, shape ``(..., n_lat, n_lon)``.
    geopotential : torch.Tensor
        Geopotential field (m²/s²), same shape as u/v.
    lat, lon : np.ndarray
        1-D coordinate arrays in degrees.
    lat_mask_deg : float, optional
        Mask latitudes within this many degrees of the equator (default 5°).
    pole_mask_deg : float, optional
        Mask latitudes within this many degrees of the poles (default 0.5°).

    Returns
    -------
    u_ag, v_ag : torch.Tensor
        Ageostrophic wind components.  NaN at masked equatorial and polar
        latitudes (NaN propagates through the subtraction).
    """
    u_g, v_g = geostrophic_wind(geopotential, lat, lon, lat_mask_deg, pole_mask_deg)
    return u - u_g, v - v_g


# ---------------------------------------------------------------------------
# Convenience: load derived variable from zarr
# ---------------------------------------------------------------------------

def load_derived_from_zarr(
    zarr_path: str,
    derived_name: str,
    level: int,
    n_members: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Load raw variables from a zarr store and compute a derived field.

    This is a thin convenience wrapper that handles the multi-variable
    loading boilerplate.  The returned tensor has the same shape as a
    single-variable load from the zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.  Expected data shape for ensemble stores:
        ``[N, 1, n_leads, n_vars, n_lat, n_lon]``, or for ERA5:
        ``[n_leads, 1, n_vars, n_lat, n_lon]``.
    derived_name : str
        One of: ``"wind_speed"``, ``"vorticity"``, ``"divergence"``,
        ``"u_geostrophic"``, ``"v_geostrophic"``,
        ``"u_ageostrophic"``, ``"v_ageostrophic"``.
    level : int
        Pressure level (e.g. 500, 850).
    n_members : int | None, optional
        For ensemble stores, number of members to load.

    Returns
    -------
    data : torch.Tensor
        Derived field.  Shape matches the zarr store layout with the
        variable dimension removed.
    info : dict
        Dictionary with ``lat``, ``lon``, and ``lead_times`` arrays.
    """
    import zarr

    root = zarr.open_group(zarr_path, mode="r")
    var_list = list(np.array(root["variable"][:]))
    lat = np.array(root["lat"][:])
    lon = np.array(root["lon"][:])
    lead_times = np.array(root["lead_time"][:])
    is_ensemble = root["data"].ndim == 6

    def _load_var(name: str) -> torch.Tensor:
        idx = var_list.index(name)
        if is_ensemble:
            N = root["data"].shape[0]
            if n_members is not None:
                N = min(n_members, N)
            return torch.from_numpy(
                np.array(root["data"][:N, 0, :, idx, :, :])
            ).float()
        else:
            return torch.from_numpy(
                np.array(root["data"][:, 0, idx, :, :])
            ).float()

    # Variable naming convention from CBottle lexicon: u500, v500, z500, etc.
    u_name = f"u{level}"
    v_name = f"v{level}"
    z_name = f"z{level}"

    info = {"lat": lat, "lon": lon, "lead_times": lead_times}

    if derived_name == "wind_speed":
        return wind_speed(_load_var(u_name), _load_var(v_name)), info

    elif derived_name == "vorticity":
        return relative_vorticity(
            _load_var(u_name), _load_var(v_name), lat, lon
        ), info

    elif derived_name == "divergence":
        return horizontal_divergence(
            _load_var(u_name), _load_var(v_name), lat, lon
        ), info

    elif derived_name == "u_geostrophic":
        u_g, _ = geostrophic_wind(_load_var(z_name), lat, lon)
        return u_g, info

    elif derived_name == "v_geostrophic":
        _, v_g = geostrophic_wind(_load_var(z_name), lat, lon)
        return v_g, info

    elif derived_name == "u_ageostrophic":
        u_ag, _ = ageostrophic_wind(
            _load_var(u_name), _load_var(v_name),
            _load_var(z_name), lat, lon,
        )
        return u_ag, info

    elif derived_name == "v_ageostrophic":
        _, v_ag = ageostrophic_wind(
            _load_var(u_name), _load_var(v_name),
            _load_var(z_name), lat, lon,
        )
        return v_ag, info

    else:
        raise ValueError(
            f"Unknown derived variable {derived_name!r}. "
            f"Choose from: wind_speed, vorticity, divergence, "
            f"u_geostrophic, v_geostrophic, u_ageostrophic, v_ageostrophic."
        )
