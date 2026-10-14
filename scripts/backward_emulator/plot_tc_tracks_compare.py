"""TC-track comparison figure: makani vs cBottle vs ERA5 (Hurricane Ian).

Reads the two tracks parquets (makani = `tc_tracks_makani_ian_end.parquet` from our
tracking job; cBottle = the paper's own `tc_tracks_end.parquet`), resolves each
member's storm of interest through the paper's track QC (`case_track_qc`, the rule
the paper's own track figure uses), keeps only the members it marks analyzable, and
draws a three-row figure:

  * top: two maps (makani | cBottle) of the member tracks, each model's ERA5 reference
    track, and its ERA5-closest member;
  * middle (only with PCA npzs): the leading-PC z500 precursor *pattern* per model;
  * bottom: the central-pressure (min-MSLP) trajectories, makani vs cBottle vs both
    ERA5 references.

**Per-model baseline.** makani is verified against *native* ERA5 (solid black); cBottle
against the *round-tripped* ERA5 it was scored on (dashed black) -- the same
resolution-matched-truth convention as the spread / ageostrophic / PCA diagnostics.
ERA5 is one colour (black) with the two resolutions distinguished by linestyle, rather
than a fourth colour competing with the two model colours.

**PC-percentile colouring (optional).** With `--pca-makani-npz` (and optionally
`--pca-cbottle-npz`) each member track is coloured by its leading-PC-score percentile
within its own ensemble, reproducing the cBottle-video-paper look. Each ensemble has its
OWN basis, so each gets its OWN colormap + colorbar (makani viridis, cBottle cool) on
the outer edges of the figure. The same percentile also tints each member's central-
pressure trace (low alpha) so the maps show *where* and the intensity panel shows *how
strong* under one key. `--color-pcs` sweeps a set of PCs, emitting one figure per PC
(e.g. `auto,1,2,3`); `auto` picks the leading-8 PC most correlated with the free-end
`tc_lon` (the axis Ian's fan spreads along).

By default each PC's sign is oriented so its score correlates *positively* with `tc_lon`
(so high percentile == eastward-tracking members == the pattern's positive lobe);
`--no-orient-by-tclon` disables this.  `tc_lon` is each member's free-end longitude
from the same track QC, never the path-0 targets baked into the paper's npz.

makani members are also censored to survivors: `--survival-npz` must be the 66 h
mask (`survival_<case>.npz`, one column per tracked lead) and is required unless
`--no-censor` is passed.  The PCA<->tracks join is by original member index: the
PCA's analyzable set is `alive[:, FREE_END_FRAME=0]` of the same mask, and a length
check falls back to flat colour rather than mis-colour.

Light where it can be (numpy reductions + the PC-percentile join + the pattern-map
reshape are unit-tested); the map + parquet read need `e2s-custom` (pandas, and cartopy
if available -- falls back to plain lat/lon axes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
EA_DIR = BW_DIR.parent / "ensemble_analysis"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(EA_DIR) not in sys.path:  # appended: never shadow our modules
    sys.path.append(str(EA_DIR))

import _dispatch_lib as dl  # noqa: E402
from case_track_qc import load_track_qc, qc_summary  # noqa: E402

_R_EARTH_KM = 6371.0

# Model identity colours (flat member tracks / spread band when NOT colouring by PC)
# and per-model percentile colormaps (distinct maps: each ensemble's PC is its OWN basis).
COLORS = {"makani": "#1b6ca8", "cbottle": "#c1440e"}
CMAPS = {"makani": "viridis", "cbottle": "cool"}
# Gold highlight for each model's ERA5-closest member: a deep gold that sits off
# viridis's thin yellow tip and entirely off cBottle's cool cyan->magenta range,
# so it stays distinct over the member spaghetti of EITHER map; the two models are
# told apart by linestyle (their own ERA5-baseline style).
CLOSEST_COLOR = "#E8A200"
# Closest-member linestyle per model: makani solid, cBottle a DENSE dash (tighter than
# the coarser round-trip ERA5 baseline "--") so the two closest members read apart yet
# the gold cBottle line stays easy to follow over its busy member field.
CLOSEST_LS: Dict[str, object] = {"makani": "-", "cbottle": (0, (3.5, 1.5))}


def _to_hpa(msl: np.ndarray) -> np.ndarray:
    """MSLP to hPa, auto-detecting Pa (median > 10000) vs already-hPa input."""
    msl = np.asarray(msl, dtype=np.float64)
    finite = msl[np.isfinite(msl)]
    if finite.size and np.nanmedian(finite) > 10000.0:
        return msl / 100.0
    return msl


def primary_msl_band(ens: np.ndarray, qlo: float = 10, qhi: float = 90):
    """Per-lead (p_lo, median, p_hi) of each member's path-0 min-MSLP (hPa).

    ``ens`` is ``[N, P, T, 4]`` already restricted-to-domain (path 0 = storm of
    interest).  Pure numpy -- the unit-tested reduction under the central-pressure
    panel.
    """
    msl = _to_hpa(ens[:, 0, :, 2])  # [N, T]
    with np.errstate(all="ignore"):
        lo = np.nanpercentile(np.where(np.isfinite(msl), msl, np.nan), qlo, axis=0)
        mid = np.nanpercentile(np.where(np.isfinite(msl), msl, np.nan), 50, axis=0)
        hi = np.nanpercentile(np.where(np.isfinite(msl), msl, np.nan), qhi, axis=0)
    return lo, mid, hi


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(lambda a: np.deg2rad(np.asarray(a, float)),
                                 (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def position_error_km(ens: np.ndarray, era5: np.ndarray) -> np.ndarray:
    """Median great-circle track error (km) vs ERA5 per lead, over members (path 0)."""
    lat = ens[:, 0, :, 0]
    lon = ens[:, 0, :, 1]
    elat = era5[0, 0, :, 0]
    elon = era5[0, 0, :, 1]
    d = _haversine_km(lat, lon, elat[None, :], elon[None, :])  # [N, T]
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(np.isfinite(d), d, np.nan), axis=0)


def closest_member_index(
    member_msl: np.ndarray,
    era5_msl: np.ndarray,
    eligible: Optional[np.ndarray] = None,
) -> Optional[int]:
    """Index of the member whose min-MSLP trace is closest (RMSE) to ``era5_msl``.

    ``member_msl`` is ``[N, T]`` (hPa, path-0 min-MSLP per member); ``era5_msl``
    is the ``[T]`` per-model ERA5 baseline (native for makani, round-trip for
    cBottle).  Distance is RMSE over the leads finite in BOTH the member and the
    reference, so a member tracked over only part of the window still competes on
    its overlap; a member with no overlap is ineligible.  ``eligible`` is an
    optional ``[N]`` bool mask (e.g. the drawn / finite-percentile members) that
    restricts the search.  Returns ``None`` when nothing is eligible or the
    shapes do not line up -- callers then simply skip the highlight.
    """
    member_msl = np.asarray(member_msl, float)
    era5 = np.asarray(era5_msl, float).ravel()
    if member_msl.ndim != 2 or era5.size != member_msl.shape[1]:
        return None
    diff = member_msl - era5[None, :]
    valid = np.isfinite(diff)
    counts = valid.sum(axis=1)
    with np.errstate(all="ignore"):
        sse = (np.where(valid, diff, 0.0) ** 2).sum(axis=1)
        rmse = np.sqrt(sse / np.maximum(counts, 1))
    rmse = np.where(counts > 0, rmse, np.inf)
    if eligible is not None:
        elig = np.asarray(eligible, bool)
        if elig.shape == rmse.shape:
            rmse = np.where(elig, rmse, np.inf)
    if not np.isfinite(rmse).any():
        return None
    return int(np.argmin(rmse))


# --------------------------------------------------------------------------- #
# PC-percentile colouring helpers (pure numpy; unit-tested)
# --------------------------------------------------------------------------- #
def score_percentile(col: np.ndarray) -> np.ndarray:
    """Rank-percentile [0, 100] of a 1-D score column (ties broken arbitrarily)."""
    col = np.asarray(col, float)
    n = col.size
    if n == 0:
        return col
    if n == 1:
        return np.array([50.0])
    order = np.argsort(col, kind="stable")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    return 100.0 * rank / (n - 1)


def choose_pc(pc_scores: np.ndarray, tc_lon: Optional[np.ndarray], n_pc: int) -> int:
    """0-based index of the leading-``n_pc`` PC most correlated with ``tc_lon``.

    Falls back to PC0 when no target is available or every correlation is
    undefined (constant column / all-NaN target).
    """
    d = min(n_pc, pc_scores.shape[1])
    if tc_lon is None:
        return 0
    y = np.asarray(tc_lon, float)
    good = np.isfinite(y)
    if good.sum() < 3:
        return 0
    best_j, best_r = 0, -1.0
    for j in range(d):
        x = pc_scores[:, j]
        m = good & np.isfinite(x)
        if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            continue
        r = abs(np.corrcoef(x[m], y[m])[0, 1])
        if np.isfinite(r) and r > best_r:
            best_j, best_r = j, r
    return best_j


def orient_by_target(score_col: np.ndarray, target: Optional[np.ndarray]) -> float:
    """+1/-1 sign so ``sign * score`` correlates non-negatively with ``target``.

    Makes 'high percentile' point the same physical way as ``+target`` (e.g. east).
    Returns +1 when no usable target / correlation.
    """
    if target is None:
        return 1.0
    x = np.asarray(score_col, float)
    y = np.asarray(target, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return 1.0
    r = np.corrcoef(x[m], y[m])[0, 1]
    return -1.0 if (np.isfinite(r) and r < 0) else 1.0


def pc_percentile_full(pc_scores: np.ndarray, analyzable: Optional[np.ndarray],
                       n_members: int, jstar: int, sign: float = 1.0) -> np.ndarray:
    """Percentile of PC ``jstar`` mapped back onto ``[n_members]`` (NaN off-analyzable).

    ``pc_scores`` is ``[Na, D]`` over the analyzable subset (in original-member
    order); ``analyzable`` is the ``[n_members]`` bool mask (None => identity, i.e.
    ``Na == n_members``).  ``sign`` flips the score before ranking (so the colour
    direction can be oriented to a physical axis).  Returns ``[n_members]`` so it can
    be censored by the same keep-mask as the tracks.
    """
    pct = score_percentile(sign * pc_scores[:, jstar])  # [Na]
    full = np.full(int(n_members), np.nan, dtype=float)
    if analyzable is None:
        if pct.size != n_members:
            raise ValueError(f"no analyzable mask and Na={pct.size} != N={n_members}")
        full[:] = pct
    else:
        idx = np.nonzero(np.asarray(analyzable, bool))[0]
        if idx.size != pct.size:
            raise ValueError(f"analyzable={idx.size} rows != pc_scores={pct.size} rows")
        full[idx] = pct
    return full


def resolve_var_index(variables, want: str = "z500") -> int:
    """Index of ``want`` in a PCA npz's ``variables`` list (fuzzy; falls back to 0)."""
    key = want.lower().replace("_", "")
    for i, v in enumerate(variables):
        if key in str(v).lower().replace("_", ""):
            return i
    return 0


def _scatter_to_box(vals: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    full_lat: np.ndarray, full_lon: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place scattered domain-pixel values onto their (lat, lon) product sub-grid.

    ``rows``/``cols`` index into ``full_lat``/``full_lon`` (the model's full grid).
    A synoptic box is a full rectangular sub-grid, so the unique rows x unique cols
    reshape is exact; any hole stays NaN.  Returns ``(field2d, lat1d, lon1d)``.
    """
    ur = np.unique(rows)
    uc = np.unique(cols)
    ri = np.searchsorted(ur, rows)
    ci = np.searchsorted(uc, cols)
    field = np.full((ur.size, uc.size), np.nan, dtype=float)
    field[ri, ci] = vals
    return field, full_lat[ur], full_lon[uc]


def load_pattern_map(npz_path: str, pc_idx: int, var_want: str = "z500"
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                Optional[np.ndarray], bool]:
    """Read a leading-PC spatial loading as a lat/lon box from either PCA schema.

    cBottle: ``pc_components_latlon [n_eof, n_var, nlat, nlon]`` (already on the box).
    makani: ``eof_loadings_domain [max_d, n_var, n_dom]`` + ``domain_flat`` reshaped
    onto its ``(domain_lat, domain_lon)`` full grid.  Returns
    ``(field2d, lat1d, lon1d, ok)``; ``ok=False`` when PC ``pc_idx`` isn't stored.
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    variables = list(z["variables"]) if "variables" in keys else []
    vi = resolve_var_index(variables, var_want)
    if "pc_components_latlon" in keys and np.asarray(z["pc_components_latlon"]).size:
        comp = np.asarray(z["pc_components_latlon"], float)  # [n_eof, n_var, nlat, nlon]
        if pc_idx >= comp.shape[0]:
            return None, None, None, False
        field = comp[pc_idx, min(vi, comp.shape[1] - 1)]
        return field, np.asarray(z["domain_lat"], float), np.asarray(z["domain_lon"],
                                                                     float), True
    if "eof_loadings_domain" in keys and np.asarray(z["eof_loadings_domain"]).size:
        eof = np.asarray(z["eof_loadings_domain"], float)  # [max_d, n_var, n_dom]
        if pc_idx >= eof.shape[0]:
            return None, None, None, False
        vals = eof[pc_idx, min(vi, eof.shape[1] - 1)]  # [n_dom]
        flat = np.asarray(z["domain_flat"], np.int64)
        full_lat = np.asarray(z["domain_lat"], float)
        full_lon = np.asarray(z["domain_lon"], float)
        nlon = full_lon.size
        field, lat1d, lon1d = _scatter_to_box(vals, flat // nlon, flat % nlon,
                                              full_lat, full_lon)
        return field, lat1d, lon1d, True
    return None, None, None, False


def _lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to [-180, 180] (monotone as long as the box doesn't wrap 180)."""
    lon = np.asarray(lon, float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def resolve_color_tokens(tokens: List[str], pca: Dict) -> List[Tuple[str, int]]:
    """Map each ``--color-pcs`` token to ``(label, jstar)`` for one figure apiece."""
    out: List[Tuple[str, int]] = []
    d = pca["pc_scores"].shape[1]
    for t in tokens:
        t = t.strip().lower()
        if not t:
            continue
        if t == "auto":
            j = choose_pc(pca["pc_scores"], pca["tc_lon"], pca["n_pc"])
            out.append((f"auto(pc{j + 1})", j))
        else:
            j = int(t) - 1  # 1-based on the CLI
            if 0 <= j < d:
                out.append((f"pc{j + 1}", j))
            else:
                print(f"[tc-plot] WARNING: PC {t} out of range 1..{d}; skipping.",
                      file=sys.stderr)
    return out


def survivor_mask(survival_npz: str, n_members: int, n_leads: int) -> np.ndarray:
    """``[N]`` makani members alive at the free-end frame (forward 0).

    The mask must cover exactly this figure's members and leads.  A full-horizon
    scan (``survival_full.npz``) has more leads and its frame 0 lies before the
    window, so it would drop members that stayed valid for the whole 66 h; any
    mismatch is an error, never a silent switch to uncensored.
    """
    alive = np.asarray(np.load(survival_npz)["alive"], bool)  # [N, L] forward order
    if alive.shape != (n_members, n_leads):
        raise SystemExit(
            f"[tc-plot] {survival_npz}: alive {alive.shape} != tracks "
            f"({n_members}, {n_leads}); pass the 66 h mask survival_<case>.npz.")
    return alive[:, 0]


def _load(parquet: str, case: Dict) -> Tuple[np.ndarray, np.ndarray, object]:
    """``(storm-of-interest tracks [N, 1, T, 4], ERA5 path 0, track QC)``.

    The paper's track figure draws exactly this: each member's anchor-resolved
    path, and ERA5's path 0 (the reference the QC anchors on).
    """
    ens, era5, qc = load_track_qc(parquet, case)
    return qc.storm_of_interest_array(ens), era5[:, 0:1], qc


def _panel_graticule(ax, ps, *, left, right, bottom, top, max_ticks=4):
    """Shared lat/lon graticule with per-edge label control.

    ``plot_style.draw_graticule`` hard-codes top/right labels off (its callers
    only ever want left/bottom), but the v3 track figure labels the *right*
    edge of the right-hand top map, so this local helper generalises the edge
    choice while reusing the shared graticule style constants so the look still
    matches the other paper maps.
    """
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=any((left, right, bottom, top)),
        linewidth=ps.GRATICULE_LW,
        alpha=ps.GRATICULE_ALPHA,
        linestyle=ps.GRATICULE_STYLE,
        color="gray",
    )
    gl.xlocator = mticker.MaxNLocator(max_ticks)
    gl.ylocator = mticker.MaxNLocator(max_ticks)
    gl.xlabel_style = {"size": ps.TICK_LABEL_SIZE_DENSE}
    gl.ylabel_style = {"size": ps.TICK_LABEL_SIZE_DENSE}
    gl.left_labels = bool(left)
    gl.right_labels = bool(right)
    gl.bottom_labels = bool(bottom)
    gl.top_labels = bool(top)
    return gl


def _cbar_on_edge(fig, mappable, ax, label, *, side, ps):
    """Colorbar filling one thin gridspec cell on the figure's outer edge."""
    cbar = fig.colorbar(mappable, cax=ax)
    cbar.set_label(label, fontsize=ps.COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=ps.COLORBAR_TICK_SIZE)
    if side == "left":
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
    return cbar


def _draw_map(
    ax,
    m,
    *,
    has_cartopy,
    proj,
    extent,
    cmap,
    norm,
    color_mode,
    ci,
    ncol,
    cfeature,
    ps,
    mark_closest=True,
):
    ens = m["ens"]
    c = COLORS.get(m["key"], "#444")
    pct = m.get("color_pct")
    for i in range(ens.shape[0]):
        lat, lon = ens[i, 0, :, 0], ens[i, 0, :, 1]
        if color_mode and pct is not None and np.isfinite(pct[i]):
            ax.plot(
                lon,
                lat,
                color=cmap(norm(pct[i])),
                lw=0.6,
                alpha=0.55,
                transform=proj if has_cartopy else None,
            )
        else:
            lc = "#999999" if color_mode else c
            ax.plot(
                lon,
                lat,
                color=lc,
                lw=0.5,
                alpha=0.15,
                transform=proj if has_cartopy else None,
            )
    era5 = m.get("era5")
    if era5 is not None:
        ax.plot(
            era5[0, 0, :, 1],
            era5[0, 0, :, 0],
            color="k",
            lw=2.0,
            ls=m.get("era5_ls", "--"),
            label=m.get("era5_label", "ERA5"),
            transform=proj if has_cartopy else None,
            zorder=8,
        )
    # The member whose TRACK is closest (mean great-circle error) to this
    # model's ERA5 track, gold, in the model's own ERA5-baseline linestyle -- the
    # same "closest to ERA5" highlight the central-pressure panel uses, so the
    # gold line ties the two panels together (there it is closest in min-MSLP).
    if mark_closest and era5 is not None:
        lat_all, lon_all = ens[:, 0, :, 0], ens[:, 0, :, 1]
        dist = _haversine_km(
            lat_all, lon_all, era5[0, 0, :, 0][None, :], era5[0, 0, :, 1][None, :]
        )
        elig = np.isfinite(pct) if (color_mode and pct is not None) else None
        bi = closest_member_index(dist, np.zeros(dist.shape[1]), eligible=elig)
        if bi is not None:
            ax.plot(
                lon_all[bi],
                lat_all[bi],
                color=CLOSEST_COLOR,
                lw=2.6,
                ls=CLOSEST_LS.get(m["key"], m.get("era5_ls", "-")),
                label=f"{m['label']} closest member",
                transform=proj if has_cartopy else None,
                zorder=9,
            )
    # Item 5 tick labels: in a two-map row the left map is labelled on its LEFT
    # edge only and the right map on its RIGHT edge only; a lone map keeps
    # left + bottom so it still reads standalone.
    if ncol == 1:
        left, right, bottom = True, False, True
    else:
        left, right, bottom = ci == 0, ci == ncol - 1, False
    if has_cartopy:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        _panel_graticule(ax, ps, left=left, right=right, bottom=bottom, top=False)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if right and not left:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        ax.set_ylabel("lat")
        if bottom:
            ax.set_xlabel("lon")
        ax.grid(alpha=0.3)
    # y=1.0 is required, not cosmetic: cartopy's Gridliner resolves an
    # auto-placed title to ``inf`` on a labelled axes with top labels off, which
    # drops the title and poisons the tight bbox (see plot_style.draw_graticule).
    ax.set_title(f"{m['label']}  (N={ens.shape[0]})", fontsize=ps.TITLE_SIZE, y=1.0)
    ax.legend(fontsize=ps.LEGEND_SIZE, loc="upper left")


def _draw_pattern(ax, pat, *, has_cartopy, proj, extent, cfeature, ps):
    """Diverging z500 loading map, normalized to its own max-|amplitude| (-1..1)."""
    field = pat["field"]
    amax = np.nanmax(np.abs(field))
    norm_field = field / amax if amax > 0 else field
    lon = _lon_to_180(pat["lon"])
    order = np.argsort(lon)
    mesh = ax.pcolormesh(
        lon[order],
        pat["lat"],
        norm_field[:, order],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
        transform=proj if has_cartopy else None,
    )
    if has_cartopy:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.4)
        # Item 5: pattern maps carry lon labels on the BOTTOM edge only.
        _panel_graticule(ax, ps, left=False, right=False, bottom=True, top=False)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("lon")
    # y=1.0: see the note in _draw_map (labelled cartopy axes + tight bbox).
    ax.set_title(pat["label"], fontsize=ps.AXIS_LABEL_SIZE, y=1.0)
    return mesh


def _plot(
    out_png,
    models: Dict[str, Dict],
    *,
    extent,
    lead_h,
    case_name,
    start_iso,
    tz_name,
    pc_label: Optional[str] = None,
    patterns: Optional[Dict[str, Dict]] = None,
    mark_closest: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    from local_time_axis import format_local_time_axis
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.PlateCarree()
        has_cartopy = True
    except Exception:  # noqa: BLE001
        has_cartopy = False
        proj = None
        cfeature = None

    mkeys = list(models)
    ncol = len(mkeys)
    color_mode = pc_label is not None
    pc_disp = pc_label.upper() if pc_label else None  # "pc1" -> "PC1" for display
    patterns = patterns or {}
    has_patterns = color_mode and any(patterns.get(k, {}).get("ok") for k in mkeys)
    norm = Normalize(vmin=0, vmax=100)

    # Columns: [Lcbar, gap, maps..., gap, Rcbar].  The gaps keep the maps' outer
    # lat/lon labels clear of the two tall percentile colorbars on the edges.
    cbar_w, gap_w = 0.045, 0.13
    width_ratios = [cbar_w, gap_w] + [1.0] * ncol + [gap_w, cbar_w]
    col_lcbar, col_rcbar = 0, ncol + 3

    # Rows: track maps, [pattern maps, loading colorbar], central pressure.  Size
    # the two map rows to the panels' OWN aspect so the cartopy maps fill their
    # cells instead of floating in tall boxes (that float was the main dead
    # space).  The loading colorbar rides in its own thin row, internal to the
    # middle -- never the right edge, where a tall percentile colorbar now lives.
    fig_w = 6.4 * ncol
    col_units = 2 * cbar_w + 2 * gap_w + ncol
    map_cell_w = fig_w / col_units  # inches across one map column
    lat_span = float(extent[3] - extent[2]) or 1.0
    map_aspect = float(extent[1] - extent[0]) / lat_span  # display width / height
    h_map = map_cell_w / map_aspect
    # A thin spacer row (h_gap) below the loading colorbar keeps its label off
    # the central-pressure panel underneath.
    h_load, h_gap, h_trace = 0.16, 0.34, 3.2
    heights = [h_map] + ([h_map, h_load, h_gap] if has_patterns else []) + [h_trace]
    nrow = len(heights)
    row_intensity = nrow - 1
    row_patt, row_loadcbar = 1, 2
    # A little headroom for the suptitle + the inter-row tick labels that
    # tight_layout carves out of the figure height (kept small so the maps stay
    # snug in their cells rather than floating).
    fig_h = sum(heights) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrow,
        ncol + 4,
        height_ratios=heights,
        width_ratios=width_ratios,
        hspace=0.14,
        wspace=0.12,
    )

    # --- track maps: the two percentile colorbars key BOTH these maps and the
    #     central-pressure traces, so they span the full figure height on the
    #     outer edges (viridis == Autoregressive left, cool == cBottle right).
    made_left = made_right = False
    top_axes = []
    for ci, mkey in enumerate(mkeys):
        cell = gs[0, 2 + ci]
        ax = (
            fig.add_subplot(cell, projection=proj)
            if has_cartopy
            else fig.add_subplot(cell)
        )
        top_axes.append(ax)
        m = dict(models[mkey], key=mkey)
        cmap = plt.get_cmap(CMAPS.get(mkey, "viridis"))
        _draw_map(
            ax,
            m,
            has_cartopy=has_cartopy,
            proj=proj,
            extent=extent,
            cmap=cmap,
            norm=norm,
            color_mode=color_mode,
            ci=ci,
            ncol=ncol,
            cfeature=cfeature,
            ps=ps,
            mark_closest=mark_closest,
        )
        if color_mode and models[mkey].get("color_pct") is not None:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            label = f"{models[mkey]['label']} — {pc_disp} percentile"
            if ci == 0 and not made_left:
                cax = fig.add_subplot(gs[:, col_lcbar])
                _cbar_on_edge(fig, sm, cax, label, side="left", ps=ps)
                made_left = True
            elif ci == ncol - 1 and not made_right:
                cax = fig.add_subplot(gs[:, col_rcbar])
                _cbar_on_edge(fig, sm, cax, label, side="right", ps=ps)
                made_right = True

    # --- pattern maps (optional) + one short horizontal loading colorbar
    #     centred beneath them ------------------------------------------------
    if has_patterns:
        mesh = None
        for ci, mkey in enumerate(mkeys):
            pat = patterns.get(mkey, {})
            if not pat.get("ok"):
                continue
            cell = gs[row_patt, 2 + ci]
            axp = (
                fig.add_subplot(cell, projection=proj)
                if has_cartopy
                else fig.add_subplot(cell)
            )
            try:
                mesh = _draw_pattern(
                    axp,
                    pat,
                    has_cartopy=has_cartopy,
                    proj=proj,
                    extent=extent,
                    cfeature=cfeature,
                    ps=ps,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[tc-plot] WARNING: pattern panel for {mkey} failed "
                    f"({exc}); skipping.",
                    file=sys.stderr,
                )
                axp.set_axis_off()
        if mesh is not None:
            # Thin bar spanning BOTH pattern maps (the full map-block width).
            cax = fig.add_subplot(gs[row_loadcbar, 2 : 2 + ncol])
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
            cb.set_label("normalized loading", fontsize=ps.COLORBAR_LABEL_SIZE)
            cb.ax.tick_params(labelsize=ps.COLORBAR_TICK_SIZE)

    # --- central pressure --------------------------------------------------
    # Widened past the map block into the right gap (its y-label lives in the
    # empty left gap) so the wall-clock tick labels get room to breathe.
    axc = fig.add_subplot(gs[row_intensity, 2 : ncol + 3])
    for mkey in mkeys:
        m = models[mkey]
        c = COLORS.get(mkey, "#444")
        pct = m.get("color_pct")
        msl = _to_hpa(m["ens"][:, 0, :, 2])  # [N, T]
        if color_mode and pct is not None:
            cmap = plt.get_cmap(CMAPS.get(mkey, "viridis"))
            for i in range(msl.shape[0]):
                if not np.isfinite(pct[i]):
                    continue
                axc.plot(lead_h, msl[i], color=cmap(norm(pct[i])), lw=0.4, alpha=0.1)
        else:
            lo, _, hi = primary_msl_band(m["ens"])
            axc.fill_between(lead_h, lo, hi, color=c, alpha=0.2, lw=0)
        # No ensemble median (it was clutter): the single member closest (in
        # min-MSLP RMSE) to THIS model's ERA5 baseline carries the story -- the
        # best-case intensity trajectory the ensemble produced.  Gold, in the
        # model's own ERA5-baseline linestyle, so the two models read apart and
        # each gold line pairs with the black ERA5 line it is scored against.
        era5 = m.get("era5")
        if mark_closest and era5 is not None:
            elig = np.isfinite(pct) if (color_mode and pct is not None) else None
            best = closest_member_index(msl, _to_hpa(era5[0, 0, :, 2]), eligible=elig)
            if best is not None:
                axc.plot(
                    lead_h,
                    msl[best],
                    color=CLOSEST_COLOR,
                    lw=2.2,
                    ls=CLOSEST_LS.get(mkey, m.get("era5_ls", "-")),
                    alpha=0.95,
                    zorder=7,
                    label=f"{m['label']} closest member",
                )
    seen = set()
    for mkey in mkeys:
        era5 = models[mkey].get("era5")
        lbl = models[mkey].get("era5_label", "ERA5")
        if era5 is not None and lbl not in seen:
            axc.plot(
                lead_h,
                _to_hpa(era5[0, 0, :, 2]),
                color="k",
                lw=2.0,
                ls=models[mkey].get("era5_ls", "--"),
                label=lbl,
                zorder=6,
            )
            seen.add(lbl)
    # Item 3: local wall-clock x-axis.  lead 0 == window start; time runs
    # left->right to the pinned present (66 h).  Natural (non-inverted) MSLP
    # axis: a deepening storm dips toward the bottom (familiar best-track look).
    format_local_time_axis(axc, lead_h, start_iso, tz_name)
    axc.set_ylabel("min MSLP (hPa)")
    # Two columns so the up-to-four entries (per-model closest member + the two
    # ERA5 baselines) sit in tidy rows without overrunning the panel width.
    axc.legend(
        fontsize=ps.LEGEND_SIZE_DENSE,
        ncol=2,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
        handletextpad=0.5,
    )
    axc.grid(alpha=0.3)
    ps.style_axes(
        axc,
        title_size=ps.TITLE_SIZE,
        label_size=ps.AXIS_LABEL_SIZE,
        tick_size=ps.TICK_LABEL_SIZE,
        legend_size=ps.LEGEND_SIZE,
    )

    # "storm tracks" rather than "TC tracks" (avoids jargon; "hurricane" would
    # just echo the case name).
    ttl = f"{case_name}: ensemble storm tracks & central pressure"
    if pc_disp:
        ttl += f"\ncolored by {pc_disp}"  # PC tag on its own second line
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    # Drop the title's bottom a fixed hair above wherever the top panels + their
    # titles ACTUALLY end (measured after layout) instead of guessing a y -- so
    # no reserved-band gap survives between the title and the panels regardless
    # of the panel aspect.  bbox_inches="tight" then crops the space above the
    # title (x=0.5 centres it; the y=1.0 panel titles guard the crop).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tops = []
    for ax in top_axes:
        tops.append(ax.get_window_extent(rend).ymax)
        if ax.title.get_text():
            tops.append(ax.title.get_window_extent(rend).ymax)
    y_panels = (max(tops) / fig.bbox.height) if tops else 0.95
    fig.suptitle(
        ttl,
        x=0.5,
        y=min(0.995, y_panels + 0.006),
        ha="center",
        va="bottom",
        fontsize=ps.TITLE_SIZE,
    )
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-parquet", required=True)
    p.add_argument("--cbottle-parquet", default=None)
    p.add_argument("--case", default="ian")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--survival-npz", default=None,
                   help="66 h makani survivor mask (survival_<case>.npz); members "
                   "dead at the free-end frame are dropped. Required unless "
                   "--no-censor.")
    p.add_argument("--no-censor", action="store_true",
                   help="Plot makani members without survivor censoring.")
    p.add_argument("--pca-makani-npz", default=None,
                   help="makani PCA npz; colour member tracks by PC percentile.")
    p.add_argument("--pca-cbottle-npz", default=None,
                   help="cBottle PCA npz; colour cBottle tracks symmetrically.")
    p.add_argument("--color-pcs", default="1",
                   help="Comma list of PCs to sweep (one figure each): 1-based "
                        "indices, e.g. '1,2,3' (or 'auto' to pick by tc_lon).")
    p.add_argument("--pattern-variable", default="z500",
                   help="Variable whose PC loading pattern is drawn (middle row).")
    p.add_argument("--no-orient-by-tclon", action="store_true",
                   help="Do not sign-orient each PC so high percentile == eastward.")
    p.add_argument("--no-mark-closest", action="store_true",
                   help="Do not highlight each model's ERA5-closest (min-MSLP) "
                        "member in the central-pressure panel.")
    p.add_argument("--margin", type=float, default=6.0,
                   help="Degrees to expand the track box for the map extent.")
    args = p.parse_args(argv)

    case = dl.load_case(args.case)
    box = dl.region_box(case, "track")
    if box is None:
        raise SystemExit(f"case '{args.case}' has no regions.track box.")
    lon_min, lon_max, lat_min, lat_max = box
    extent = [lon_min - args.margin, lon_max + args.margin,
              lat_min - args.margin, lat_max + args.margin]

    # makani: native ERA5 (solid black baseline).  Keep = analyzable under the
    # track QC AND alive at the free-end frame.
    m_ens, m_era5, m_qc = _load(args.makani_parquet, case)
    keep = m_qc.analyzable.copy()
    print(f"[tc-plot] makani track QC: {qc_summary(m_qc)}")
    if not args.no_censor:
        if not args.survival_npz:
            raise SystemExit("[tc-plot] --survival-npz is required (or pass "
                             "--no-censor to plot unstable makani members).")
        alive = survivor_mask(args.survival_npz, m_ens.shape[0], m_ens.shape[2])
        print(f"[tc-plot] makani survivors at the free end: {int(alive.sum())}/"
              f"{alive.size}")
        keep &= alive
    models: Dict[str, Dict] = {"makani": {
        "label": "Autoregressive", "ens": m_ens[keep], "era5": m_era5,
        "era5_ls": "-", "era5_label": "ERA5 (native)", "keep": keep, "qc": m_qc}}
    print(f"[tc-plot] makani members plotted: {int(keep.sum())}")

    # cBottle: round-tripped ERA5 (dashed black baseline).
    if args.cbottle_parquet and Path(args.cbottle_parquet).exists():
        c_ens, c_era5, c_qc = _load(args.cbottle_parquet, case)
        c_keep = c_qc.analyzable.copy()
        print(f"[tc-plot] cBottle track QC: {qc_summary(c_qc)}")
        models["cbottle"] = {
            "label": "Video diffusion (cBottle)", "ens": c_ens[c_keep], "era5": c_era5,
            "era5_ls": "--", "era5_label": "ERA5 (round-trip)",
            "keep": c_keep, "qc": c_qc}
        print(f"[tc-plot] cBottle members plotted: {int(c_keep.sum())}")
    else:
        print("[tc-plot] WARNING: no cBottle parquet; makani-only.", file=sys.stderr)

    T = m_ens.shape[2]
    lead_h = np.arange(T) * 6.0
    out = Path(args.out_dir)
    cname = case.get("display_name", args.case)
    # Local wall-clock x-axis for the central-pressure panel (lead 0 == the 66 h
    # window start; time flows left->right to the pinned present).
    start_iso = dl.start_time_iso_from_case(case)
    tz_name = dl.resolve_timezone(case)

    pca_by_model: Dict[str, Dict] = {}
    pca_path: Dict[str, str] = {}
    for mkey, npz in (("makani", args.pca_makani_npz),
                      ("cbottle", args.pca_cbottle_npz)):
        if npz and mkey in models:
            fe_lon = models[mkey]["qc"].free_end_targets()["fe_lon_deg"]  # [N]
            pca_by_model[mkey] = _load_pca_for_color(npz, fe_lon)
            pca_path[mkey] = npz

    def _attach(jstar_by_model: Dict[str, int], sign_by_model: Dict[str, float]) -> bool:
        any_ok = False
        for mkey, mdl in models.items():
            mdl["color_pct"] = None
            pca = pca_by_model.get(mkey)
            if pca is None or mkey not in jstar_by_model:
                continue
            full = pc_percentile_full(pca["pc_scores"], pca["analyzable"],
                                      pca["n_members"], jstar_by_model[mkey],
                                      sign=sign_by_model.get(mkey, 1.0))
            km = mdl["keep"]
            if full.shape[0] != km.shape[0]:
                print(f"[tc-plot] WARNING: {mkey} PCA N={full.shape[0]} != tracks "
                      f"N={km.shape[0]}; flat colour for {mkey}.", file=sys.stderr)
                continue
            mdl["color_pct"] = full[km]
            any_ok = True
        return any_ok

    figures: List[str] = []
    if pca_by_model:
        anchor = pca_by_model.get("makani") or next(iter(pca_by_model.values()))
        for label, j0 in resolve_color_tokens(args.color_pcs.split(","), anchor):
            jstar, sign = {}, {}
            for mkey, pca in pca_by_model.items():
                jstar[mkey] = (choose_pc(pca["pc_scores"], pca["tc_lon"], pca["n_pc"])
                               if label.startswith("auto") else j0)
                sign[mkey] = (1.0 if args.no_orient_by_tclon
                              else orient_by_target(pca["pc_scores"][:, jstar[mkey]],
                                                    pca["tc_lon"]))
            ok = _attach(jstar, sign)
            patterns = {}
            for mkey in models:
                if mkey in pca_path:
                    field, lat, lon, pok = load_pattern_map(
                        pca_path[mkey], jstar[mkey], args.pattern_variable)
                    if pok and sign.get(mkey, 1.0) < 0:
                        field = -field  # keep pattern polarity consistent with colour
                    patterns[mkey] = {
                        "ok": pok, "field": field, "lat": lat, "lon": lon,
                        "label": f"{models[mkey]['label']}: {args.pattern_variable} "
                                 f"PC{jstar[mkey] + 1} pattern"}
            suffix = label.replace("(", "").replace(")", "")
            png = out / f"tc_tracks_compare_{args.case}_{suffix}.png"
            _plot(
                png,
                models,
                extent=extent,
                lead_h=lead_h,
                case_name=cname,
                start_iso=start_iso,
                tz_name=tz_name,
                pc_label=(label if ok else None),
                patterns=patterns,
                mark_closest=not args.no_mark_closest,
            )
            figures.append(str(png))
            print(f"[tc-plot] wrote {png} (colour={label}, ok={ok})")
    else:
        png = out / f"tc_tracks_compare_{args.case}.png"
        _plot(
            png,
            models,
            extent=extent,
            lead_h=lead_h,
            case_name=cname,
            start_iso=start_iso,
            tz_name=tz_name,
            mark_closest=not args.no_mark_closest,
        )
        figures.append(str(png))

    stats: Dict[str, np.ndarray] = {"lead_h": lead_h}
    for mkey, m in models.items():
        if m.get("era5") is not None:
            stats[f"{mkey}::pos_err_km"] = position_error_km(m["ens"], m["era5"])
        lo, mid, hi = primary_msl_band(m["ens"])
        stats[f"{mkey}::msl_median_hpa"] = mid
    np.savez(out / f"tc_tracks_compare_{args.case}.npz", **stats)
    print(f"[tc-plot] wrote {len(figures)} figure(s) + "
          f"{out}/tc_tracks_compare_{args.case}.npz")


def _load_pca_for_color(npz_path: str, fe_lon: Optional[np.ndarray] = None) -> Dict:
    """Read a makani- or cBottle-schema PCA npz for track colouring.

    ``fe_lon`` is the ``[N]`` free-end longitude from the track QC (NaN for
    excluded members); it is mapped onto the PC-score rows as ``tc_lon``.  The
    npz's own baked track targets are deliberately not read (the paper's are
    path 0, which can be another storm).

    Returns ``{pc_scores [Na,D], analyzable [N]|None, n_members, tc_lon [Na]|None,
    era5_pc_percentile [D]|None, n_pc}``.
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    scores = np.asarray(z["pc_scores"], float)
    analyzable = np.asarray(z["analyzable"], bool) if "analyzable" in keys else None
    n_members = (int(z["n_members"]) if "n_members" in keys
                 else (int(analyzable.size) if analyzable is not None
                       else scores.shape[0]))
    tc_lon = None
    if fe_lon is not None and np.asarray(fe_lon).size == n_members:
        tc_lon = np.asarray(fe_lon, float)
        if analyzable is not None:
            tc_lon = tc_lon[analyzable]
    era5_pct = (np.asarray(z["era5_pc_percentile"], float)
                if "era5_pc_percentile" in keys else None)
    n_pc = int(z["n_pc"]) if "n_pc" in keys else 8
    return {"pc_scores": scores, "analyzable": analyzable, "n_members": n_members,
            "tc_lon": tc_lon, "era5_pc_percentile": era5_pct, "n_pc": n_pc}


if __name__ == "__main__":
    main()
