# `scripts/_shared/` — shared helpers for the ensemble pipeline

These modules are imported by both the dispatcher scripts in
`scripts/ensemble_run/` and `scripts/ensemble_analysis/` and by their
per-mode `compute_*` / `aggregate_*` submission scripts.  Pulling them
into a shared module keeps the figures and zarr layouts consistent
across cases: every aggregator's lead-time axis reads the same way,
every variable's units render the same way, every diagnostic looks up
its axis range from the same YAML.

Before adding a new figure helper to an aggregator, grep here first —
reusing the existing one keeps the convention enforced by code rather
than by reviewer attention.

## Figure helpers

Use these from any `aggregate_*.py` (or `compute_*.py` that draws a
figure directly).  Already wired into `aggregate_spread_rmse_crps`,
`aggregate_power_spectra`, `aggregate_ageostrophic`, `aggregate_tc_tracks`,
and `aggregate_free_end_states`.

| Helper | Path | Use it for |
|---|---|---|
| `format_local_time_axis(ax, lead_h, start_time_iso, tz_name)` | `local_time_axis.py` | Replace a raw `Lead time (h)` x-axis with local wall-clock ticks (`Mmm DD\nHH:MM`).  Each `compute_*.py` already accepts `--start-time` + `--timezone` from the dispatcher; pass the saved scalars straight in. |
| `format_local_time_title(start_time_iso, lead_h, tz_name)` | `local_time_axis.py` | Single-line local-time stamp (`Oct 30 00:00 EDT`) for figure or panel titles. |
| `var_label(var_name, suffix="")` | `var_metadata.py` | Spelled-out name + MathText units, e.g. `"Geopotential at 500 hPa spread (m$^{2}$/s$^{2}$)"`.  Always prefer this over hand-writing units.  Use it for **figure-level titles** (suptitles), which have to read standalone.  A dense per-panel map title may use `symbol` instead when the suptitle already spells the variable out and the long name would overflow the panel. |
| `axis_label(var_name, suffix="")` | `var_metadata.py` | Same shape, but leads with the MathText **symbol** — `"$z_{500}$ spread (m)"`, `"$T_{\mathrm{2m}}$ (K)"`.  Use it for **axis and colorbar labels**, where space is tight and the symbol is unambiguous.  Variables with no established notation fall back to the long name, so it is always safe to call. |
| `symbol(var_name)` | `var_metadata.py` | Just the symbol, no units — for composing a label inline (e.g. `"$z_{500}$ regression on $T_{\mathrm{2m}}$"`).  Add new entries to `_SURFACE_SYMBOLS` / `_LEVEL_SYMBOLS` only for notation the manuscript actually defines. |
| `units_tex(var_name)` | `var_metadata.py` | Just the units, MathText-aware.  Use when you want to compose the label inline. |
| `long_name(var_name)` | `var_metadata.py` | Human-readable variable name without units — for titles, colorbars, etc. |
| `TITLE_SIZE` / `..._DENSE` (and friends) | `plot_style.py` | Paper-ready font sizes.  Use `_DENSE` for ≥3-column × ≥3-row grids (free-end-states, ageostrophic). |
| `add_subplot_labels(axes, placement=...)` | `plot_style.py` | `a.)`, `b.)`, `c.)` panel labels with consistent positioning. |
| `style_axes(ax)`, `style_colorbar(cbar)` | `plot_style.py` | Bulk-apply the paper font sizes after artists are drawn (e.g. after `format_local_time_axis`, which doesn't take a fontsize kwarg). |
| `draw_graticule(ax, show_xlabels=, show_ylabels=, max_ticks=)` | `plot_style.py` | The shared dashed lat/lon graticule.  Label only the bottom row and left column of a multi-panel grid, and set `max_ticks` on small panels — cartopy's automatic step runs the labels together (`96°W93°W90°W…`).  **On any axes you label, set the title with an explicit `y` (`ax.set_title(..., y=1.0)`)**: on matplotlib >= 3.11 cartopy's `Gridliner` resolves an auto-placed title to `inf`, dropping it and breaking `bbox_inches="tight"` with a NaN bbox.  Harmless below that threshold — the automatic placement is exactly `y=1.0` — so pass it unconditionally.  Seven figures still carry an equivalent graticule inline; migrate each when something else touches that file. |
| `draw_quiver(ax, u, v, lat, lon, scale=, step=)`, `draw_quiver_key(ax, handle, mag)` | `plot_style.py` | Subsampled raw-flow arrows over a map panel — one convention, no styling knobs.  Take `step` from `quiver_step(lat, lon)`, the key magnitude from `nice_speed(ref)`, and the shared `scale` as `ref / QUIVER_REF_LEN_FRAC`; one key per figure, not per panel.  Used by `aggregate_synoptic_pca` and `plot_ensemble_member --wind-overlay`. |
| ↳ crop first | | Both the stride and `ref` come from the grid you pass in, so clip `u`/`v` to the panel's view box **before** deriving either.  A global field yields ~22 arrows around the planet, scaled by a jet the panel never shows. |
| `format_lon(x)`, `format_lat(x)`, `format_bbox(bbox)` | `plot_style.py` | Publication-ready coordinate strings — e.g. `"140°W"`, `"38°N"`, `"140°W–110°W, 38°N–55°N"`.  Cardinal-direction notation reads better in figures than negative numbers; integer-valued inputs format without a decimal. |
| `get_limits(analysis, variable, panel)` / `apply_limits(ax, ...)` | `axis_limits.py` | Cross-case-comparable axis ranges read from `axis_limits.yaml`.  Use when an aggregator's output sits next to its counterpart from another case. |

## Dispatch-side helpers (no matplotlib)

These live in `_dispatch_lib.py`; they keep the dispatcher scripts
matplotlib-free so they can run on the qsub head node without a heavy
import chain.

| Helper | Use it for |
|---|---|
| `MODE_FRAMES`, `MODE_COLORS_TAB`, `MODE_LABELS_SHORT` | Conditioning-mode constants — frame indices, plot colors, short display labels. |
| `mode_frames(mode)`, `frames_suffix(frames)` | Map a mode name to its frame indices or filename suffix. |
| `ensemble_zarr_path(base, mode, size)`, `era5_zarr_path(base, mode)` | Canonical zarr paths derived from the case YAML's `base:`. |
| `load_case(name)`, `list_cases()` | Read `cases/<name>.yaml`; validate required keys. |
| `load_palettes()` | Read `cases/_palettes.yaml` — shared `variable_sets` and `variable_defaults`. |
| `resolve_variable_set(case)` | Pick a case's diagnostic variable list: `diagnostics.variables` if set, else the palette's `standard` set. |
| `iter_modes(case)`, `parse_mode_list(csv, default)` | Iterate / filter conditioning modes. |
| `parse_member_spec(spec, n_members)` | Expand a `--members` CLI value (`all`, `0-99`, `5,6,7`, `0-3,10,20-25`) to a sorted index list, rejecting malformed / out-of-range entries up front.  Used by `plot_ensemble_member.py` and `member_field_qc.py`; reach for it before adding another member-selection flag. |
| `lead_hours(lead_times)` | Convert a zarr `lead_time` coord (timedelta64 or numeric) to float hours. |
| `case_ensemble_size(case, override)`, `start_time_iso_from_case(case)`, `resolve_timezone(case)` | Pull metadata out of the case YAML with sensible defaults. |
| `region_box(case, name)`, `require_region(case, name)`, `resolve_view(case, default="impact")` | Resolve a `regions.<name>` box to `(lon_min, lon_max, lat_min, lat_max)`; `require_region` raises if absent, `resolve_view` reads `plots.view` (region name or custom extent) for the member-plot viewport. |
| `validate_zarrs_exist(case, modes, size)` | Pre-flight check before submitting jobs. |
| `build_qsub_cmd(...)`, `submit_qsub(cmd, ...)` | Submit a SGE job with consistent log directory + hold-jid handling. |

## Case YAML schema (one-paragraph summary)

`cases/<event>.yaml` is the single source of truth for paths, ensemble
size, conditioning modes, spatial regions, and per-case diagnostic
overrides.  Top-level keys: `display_name`, `year`, `timezone` (IANA
name; consumed by `format_local_time_axis`), `base` (root output
directory), `ensemble` (size, modes, start_time, random_seed), `regions`
(purpose-named lat/lon boxes — see below), `plots` (members, optional
`view`), and optional per-diagnostic blocks (`tc_tracks`, `synoptic_pca`,
`diagnostics.variables`, ...).  See `pnw_heatwave.yaml` or `sandy.yaml`
for working examples.

### Spatial regions (`regions:`)

Each figure draws exactly the region it analyses, so the boxes are
purpose-named rather than overloaded — adjust the temperature-averaging
region without disturbing map viewports, and vice versa:

| Region | Role | Consumers |
|---|---|---|
| `synoptic` | large-scale cause / precursor | synoptic-PCA (PCA domain + precursor-map view); spread_rmse_crps (`--region synoptic` for height/mass variables) |
| `impact` | local effect for scalar reductions (may be land-only) | free_end_states (rank / bbox-mean / maps); synoptic-PCA (scalar impact mean + dashed annotation) — **one shared averaging, never two** |
| `track` | TC-track detection domain (TC cases only) | tc_tracks (detection + track-map view); synoptic-PCA (track-impact annotation) |

Boxes are `{ lon: [min, max], lat: [min, max] }`.  The member-plot
viewport is *not* an analysis region: `plots.view` references a region
name (or a custom `[lon_min, lon_max, lat_min, lat_max]`) and defaults to
`regions.impact`.  Resolve boxes with `region_box` / `require_region` /
`resolve_view` from `_dispatch_lib`.

A figure that reports both a cause and an effect uses **two** of these
boxes, so a diagnostic covering both kinds of variable needs a domain per
**variable, not per run**: for `pnw_heatwave` a z500 number belongs on
`synoptic` while a t2m number belongs on `impact` with the land mask, and
for a TC case z500 belongs on `synoptic` while `msl` — the storm itself —
belongs on `impact`.  Two diagnostics declare this per-variable:

| Key | Diagnostic | Shape |
|---|---|---|
| `diagnostics.free_end_states.mask` | free_end_states, synoptic-PCA | `{var: mask_kind}` |
| `diagnostics.spread_rmse_crps.domains` | spread_rmse_crps | `{var: {region, mask}}` |

Variables absent from either block get the permissive default (no mask /
global, respectively).  Because the bash submit scripts apply a single
`--region` / `--mask` to every variable they iterate over, both dispatchers
submit **one qsub per distinct domain value**; follow that pattern rather
than trying to thread a per-variable map through bash.

`custom_utils.diagnostics` (`hpx_land_mask_from_sst_hpx` /
`hpx_land_mask_from_era5`, `hpx_pixel_centers` / `domain_hpx_indices`) is
the intended single source of truth for which pixels a box-plus-mask
selects, so delegate there rather than re-deriving it.  (`compute_free_end_states.py`
still carries its own `_hpx_pixel_centers` / `_hpx_bbox_mask`, which
normalize longitude to `[-180, 180)` where the canonical helper yields
`(-180, 180]`; the two differ only for a box edge at exactly ±180, which no
case region has.  Consolidate when something else touches that file.)

Averaging a surface variable over unmasked ocean is not a cosmetic choice
in an *ensemble-spread* diagnostic: ocean t2m is driven by the prescribed
monthly-mean SST, which is identical across members, so those pixels
contribute almost no spread but full weight to the pixel mean.

## Adding a new aggregator?

A short checklist before opening the PR:

1. Reach for the helpers above instead of inlining font sizes, units
   strings, or lead-hour tick logic.
2. Time-series x-axes use `format_local_time_axis` (not raw hours).
3. Variable axis/colorbar labels use `var_label` / `long_name` (not
   hardcoded strings).
4. Multi-panel grids get `add_subplot_labels` for `a.)` `b.)` `c.)`.
5. Cross-case-comparable y-axes use `get_limits` / `apply_limits` so
   two cases' figures sit side-by-side honestly.
6. The compute script saves enough metadata in its npz/parquet
   (`start_time`, `timezone`, `lead_hours_all`, bbox, ...) for the
   aggregator to call the helpers above without re-deriving anything.
