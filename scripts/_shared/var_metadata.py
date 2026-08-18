"""Human-readable names and units for variable shorthands.

The Earth2Studio ``WB2Lexicon`` / ``CBottleLexicon`` map shorthand keys
(``msl``, ``z500``, ``t2m``, ...) to dataset-specific identifiers, but they
do not carry units or a presentation-ready long name. This module fills
that gap for plot labels, titles, and ``--help`` strings in the
``scripts/ensemble_analysis`` compute scripts.

Units are the values shown to a reader. They follow ERA5 / WB2 storage
conventions (pressures in Pa, temperatures in K, winds in m/s, vertical
velocity in Pa/s, specific humidity in kg/kg, total column water in kg/m^2,
total precipitation in m, relative humidity in percent) with one deliberate
exception: geopotential is stored as m^2/s^2 but reported as geopotential
height in m. Diagnostics stay in raw storage units everywhere; the division
by gravity happens only at plot time via :func:`to_display_units`, paired
with these labels so values and units cannot drift apart.

Kept lexicon-free on purpose so the module can be imported by plot-only
contexts without resolving the ``earth2studio`` install path. If you add
a variable to the WB2 / CBottle lexicons, add a matching entry here.
"""

from __future__ import annotations

import re
from typing import TypeVar

# Surface (single-level) shorthands -> (long name, units).
_SURFACE: dict[str, tuple[str, str]] = {
    "u10m": ("10 m u-wind", "m/s"),
    "v10m": ("10 m v-wind", "m/s"),
    "t2m": ("2-meter temperature", "K"),
    "msl": ("Mean sea level pressure", "Pa"),
    "sp": ("Surface pressure", "Pa"),
    "sst": ("Sea surface temperature", "K"),
    "tcwv": ("Total column water vapour", "kg/m^2"),
    "z": ("Surface geopotential height", "m"),
    "tclw": ("Total column liquid water", "kg/m^2"),
    "tciw": ("Total column ice water", "kg/m^2"),
    "tpf": ("Precipitation flux", "kg/m^2/s"),
    "rlut": ("Outgoing longwave radiation", "W/m^2"),
    "rsut": ("Outgoing shortwave radiation", "W/m^2"),
    "rsds": ("Downwelling shortwave at surface", "W/m^2"),
    "sic": ("Sea ice concentration", ""),
    "lsm": ("Land-sea mask", ""),
    "tp06": ("Total precipitation (6 h)", "m"),
    "tp12": ("Total precipitation (12 h)", "m"),
    "tp24": ("Total precipitation (24 h)", "m"),
}

# Pressure-level variable prefixes -> (long name, units).
_LEVEL_PREFIX: dict[str, tuple[str, str]] = {
    "u": ("u-wind", "m/s"),
    "v": ("v-wind", "m/s"),
    "w": ("Vertical velocity", "Pa/s"),
    "z": ("Geopotential height", "m"),
    "t": ("Temperature", "K"),
    "q": ("Specific humidity", "kg/kg"),
    "r": ("Relative humidity", "%"),
}

# TC-tracker-derived quantities (not raw lexicon variables).
_TC_DERIVED: dict[str, tuple[str, str]] = {
    "tc_msl": ("Storm central MSLP", "Pa"),
    "tc_w10m": ("Storm 10 m wind speed", "m/s"),
}

# MathText symbols for axis / colorbar labels, keyed the same way as the
# long-name tables above.  Only variables with established notation in the
# manuscript get an entry; everything else falls back to the spelled-out
# long name, so `symbol` is always safe to call.
_SURFACE_SYMBOLS: dict[str, str] = {
    "t2m": r"T_{\mathrm{2m}}",
    "u10m": r"u_{\mathrm{10m}}",
    "v10m": r"v_{\mathrm{10m}}",
    "msl": r"\mathrm{MSLP}",
    "sp": r"p_{\mathrm{s}}",
    "sst": r"\mathrm{SST}",
}

# Pressure-level prefixes -> symbol; the level is attached as a subscript.
_LEVEL_SYMBOLS: dict[str, str] = {
    "u": "u",
    "v": "v",
    "w": r"\omega",
    "z": "z",
    "t": "T",
    "q": "q",
    "r": r"\mathrm{RH}",
}


def _parse_level(var: str) -> tuple[str, int] | None:
    """Return ``(prefix, level_hPa)`` for names like ``"t500"``, else ``None``."""
    if len(var) < 2:
        return None
    prefix = var[0]
    if prefix not in _LEVEL_PREFIX:
        return None
    rest = var[1:]
    if not rest.isdigit():
        return None
    return prefix, int(rest)


def long_name(var: str) -> str:
    """Return a human-readable name (e.g. ``"Geopotential at 500 hPa"``).

    Falls back to the input string when the variable is unknown.
    """
    if var in _SURFACE:
        return _SURFACE[var][0]
    if var in _TC_DERIVED:
        return _TC_DERIVED[var][0]
    parsed = _parse_level(var)
    if parsed is not None:
        prefix, level = parsed
        return f"{_LEVEL_PREFIX[prefix][0]} at {level} hPa"
    return var


def units(var: str) -> str:
    """Return units in ERA5 / WB2 storage conventions, or ``""`` if unknown."""
    if var in _SURFACE:
        return _SURFACE[var][1]
    if var in _TC_DERIVED:
        return _TC_DERIVED[var][1]
    parsed = _parse_level(var)
    if parsed is not None:
        return _LEVEL_PREFIX[parsed[0]][1]
    return ""


_SUPERSCRIPT_RE = re.compile(r"\^(-?\d+)")


def units_tex(var: str) -> str:
    """Return units with matplotlib MathText superscripts (``m^2`` -> ``m$^2$``).

    Plain ASCII units (e.g. ``"m/s"``) pass through unchanged. The string is
    safe to embed directly in matplotlib axis labels.
    """
    return _SUPERSCRIPT_RE.sub(r"$^{\1}$", units(var))


def symbol(var: str) -> str:
    r"""Return a MathText symbol for ``var``, e.g. ``"$z_{500}$"``.

    Falls back to :func:`long_name` for variables with no established
    notation, so callers can use this unconditionally.

    Examples
    --------
    >>> symbol("t2m")
    '$T_{\\mathrm{2m}}$'
    >>> symbol("z500")
    '$z_{500}$'
    >>> symbol("tcwv")
    'Total column water vapour'
    """
    if var in _SURFACE_SYMBOLS:
        return f"${_SURFACE_SYMBOLS[var]}$"
    parsed = _parse_level(var)
    if parsed is not None:
        prefix, level = parsed
        if prefix in _LEVEL_SYMBOLS:
            return f"${_LEVEL_SYMBOLS[prefix]}_{{{level}}}$"
    return long_name(var)


# Standard gravity (WMO), m/s^2. The one place a magic conversion factor
# lives; see to_display_units.
DISPLAY_GRAVITY = 9.80665

_Field = TypeVar("_Field")


def is_geopotential(var: str) -> bool:
    """True for geopotential variables (``"z"``, ``"z500"``, ``"z850"``, ...).

    Geopotential is stored in m^2/s^2 (ERA5 / WB2 convention) but reported as
    geopotential height in m; see :func:`to_display_units`.
    """
    if var == "z":
        return True
    parsed = _parse_level(var)
    return parsed is not None and parsed[0] == "z"


def to_display_units(var: str, field: _Field, power: int = 1) -> _Field:
    """Scale a raw field into the units returned by :func:`units`.

    Diagnostics are computed and stored in raw ERA5 / WB2 storage units; the
    only unit conversion happens here, at plot time, paired with the labels in
    this module so values and their unit strings cannot drift apart.

    The sole conversion today is geopotential (m^2/s^2) -> geopotential height
    (m), i.e. division by standard gravity. ``power`` is the exponent of the
    field the quantity represents, so a variance-like quantity such as a power
    spectrum (``power=2``) scales by ``g**2``. Non-geopotential variables pass
    through unchanged.

    Physics that genuinely needs geopotential (e.g. geostrophic wind) must use
    the raw m^2/s^2 field and must NOT call this helper.
    """
    if is_geopotential(var):
        return field / DISPLAY_GRAVITY**power  # type: ignore[operator]
    return field


def var_label(var: str, suffix: str = "") -> str:
    """Return ``"Long name [suffix] (units)"`` for a matplotlib label.

    Parameters
    ----------
    var : str
        Variable shorthand, e.g. ``"msl"`` or ``"z500"``.
    suffix : str, optional
        Extra qualifier inserted before the units, e.g. ``"spread"`` or
        ``"RMSE"``. Empty by default.

    Examples
    --------
    >>> var_label("msl")
    'Mean sea level pressure (Pa)'
    >>> var_label("z500", "spread")
    'Geopotential height at 500 hPa spread (m)'
    """
    name = long_name(var)
    u = units_tex(var)
    head = f"{name} {suffix}".rstrip() if suffix else name
    return f"{head} ({u})" if u else head


def axis_label(var: str, suffix: str = "") -> str:
    r"""Return ``"<symbol> [suffix] (units)"`` for an axis or colorbar label.

    Same shape as :func:`var_label`, but leads with the MathText symbol
    instead of the spelled-out long name.  Use it where space is tight and
    the symbol is unambiguous -- axis and colorbar labels -- and keep
    :func:`var_label` for panel and figure titles, which have to read
    standalone.  Variables without an established symbol fall back to the
    long name, so the two agree wherever no notation exists.

    Examples
    --------
    >>> axis_label("z500", "spread")
    '$z_{500}$ spread (m)'
    >>> axis_label("t2m")
    '$T_{\\mathrm{2m}}$ (K)'
    """
    name = symbol(var)
    u = units_tex(var)
    head = f"{name} {suffix}".rstrip() if suffix else name
    return f"{head} ({u})" if u else head
