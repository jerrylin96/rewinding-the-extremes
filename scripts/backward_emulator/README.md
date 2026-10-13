# `scripts/backward_emulator/` — side-by-side comparison with the autoregressive emulator

Tooling to compare this repo's **video-diffusion** end-conditioned ensembles
(cBottle) against the **autoregressive backward emulator** of
[run-it-back](https://github.com/jerrylin96/run-it-back) for the same
extreme-event case studies, running the analyses we already have on both methods
and drawing them side by side. The figures of the NeurIPS 2026 CCAI workshop
paper on that emulator (the Ian tracks and intensity, the two EOF regressions,
the PNW free-end states) are produced here. `run-it-back` produces the
emulator's ensembles (`analysis/backward_rollout_ensemble.sbatch` there); their
field dumps are transferred and read by `makani_source.py`.

The directory is self-contained apart from the shared helpers in
`scripts/_shared/`, `scripts/ensemble_analysis/` and `custom_utils/`. Both
methods are recomputed here the same way. `--makani-analysis-dir` arguments
point at the `analysis/` directory of a `run-it-back` checkout (named
`makani-aicr` on our cluster, which is why the recorded paths below say so).

## The two data formats

| | cBottle (this repo) | makani backward emulator |
|---|---|---|
| Container | zarr per mode (`ensemble_*.zarr`, `era5_reference.zarr`) | HDF5 `*fields*.h5` dumps |
| Ensemble | `data [N,1,n_leads,n_vars,n_pix]` | `pred [A,M,L,C,H,W]` |
| Truth | separate `era5_reference.zarr` (lat/lon) | bundled `truth [A,L,C,H,W]` |
| Grid | HEALPix NEST **level 6** (49 152 px) | equiangular **0.25° lat/lon** (721×1440) |
| Frames | 12, `lead_time` 0→66 h; frame 0 = start, 11 = end | frame 0 = present/anchor, `lead_hours_into_past` increasing; `step_hours=-6` |
| Members | 1000 in one store | 1000 as 250 files × 4 members (pool on member axis) |
| Units | raw (z=m²/s², msl=Pa, t=K; ÷g at plot time) | physical (same: z=m²/s², msl=Pa, t=K) |

**Window alignment (exact):** the makani anchor is the cBottle window *end*, so
the two index the same 12 valid times in opposite directions.
PNW: cBottle 2021-06-25 18Z + 66 h = **2021-06-28 12Z** = makani anchor.
Ian: 2022-09-28 00Z + 66 h = **2022-09-30 18Z** = makani anchor.
Frames are joined on **valid timestamp**, and makani's rollout (which runs
further back, −180 h) is truncated to the first 12 frames (66 h) because the
emulator destabilizes beyond that.

## Design decisions (locked)

1. **Compare in lat/lon, at native resolution.** HEALPix is only cBottle's
   transport grid; ERA5 (truth) and makani are both lat/lon-native, and five of
   the six diagnostics already regrid cBottle HPX→lat/lon at compute time
   (ageostrophic, power spectra, synoptic-PCA, free-end states, TC tracks — the
   last mandatorily, for TempestExtremes). So makani enters **native 0.25°**, no
   round-trip; cBottle keeps its existing HPX→lat/lon step; ERA5 stays untouched.
   makani's finer resolution is a result to advertise, not something to handicap.
2. **Per-model resolution-matched ERA5 baseline** for the *absolute*,
   resolution-dependent metrics (ageostrophic fraction, spectra, RMSE/CRPS
   magnitude): makani vs ERA5-0.25°, cBottle vs ERA5-L6 (the `era5_raw`/`era5_rt`
   pattern that `compute_power_spectra` already uses). This is what makes
   "makani's ageostrophic fraction is higher" read as *resolves real imbalance*
   rather than *just sharper*. Resolution-robust metrics (SSR, ACC, EOF patterns,
   TC track position) are compared head-to-head.
3. **`end` conditioning only.** makani is backward-from-present, which is
   cBottle's `end` mode; `start`/`both` are out of scope.
4. **Shared land mask** (PNW t2m) sourced once from ERA5 lat/lon SST — makani has
   no `sst` channel — so the masked pixels are identical for both methods.
   (makani's own land mask is the LSM invariant via `ensemble_horizon.domain_weights`;
   for the comparison we use one shared ERA5 mask so both models mask the same pixels.)

## Open scientific decision: survivor censoring

The autoregressive emulator's members go unstable and blow up (one member at
1e8 K destroys a 1000-member mean), so makani's settled methodology computes every
distributional statistic over **survivors** — a member is censored from its first
failing lead onward — and reports the **survival fraction** alongside. cBottle's
end-conditioned members are all valid, so they need no censoring.

Whether our comparison reducer needs censoring hinges on one empirical number:
**the survival fraction at 66 h** (likely ~all for the PNW ridge, fewer for the
faster Ian). Check it with makani's existing `analysis/ensemble_horizon.py` /
`plot_ensemble_horizon.py` (torch-free; the `range` detector works from the field
dumps alone) on the first 12 frames. If survival ≈ 100 % at 66 h, the naive
all-member stats stand; if not, `latlon_metrics.reduce_selected` gains a per-member
`alive` mask (all-True for cBottle) so both models reduce over survivors identically,
and a survival-fraction panel is added where makani appears.

## Memory

The full pooled ensemble for one channel is ~50 GB (1000 × 12 × 721 × 1440 f32),
so nothing loads it whole: `MakaniBackwardSource.load_region` streams per file and
keeps only a region box's pixels (~2 GB for the synoptic box), and the reducer runs
on that. `load_variable` (full field) is guarded and refuses above 8 GB — it's for
tests / coarse-grid maps only.

## Diagnostics in scope

spread/RMSE/CRPS · free-end states · synoptic-PCA · ageostrophic (500 hPa) ·
TC tracks (Ian) · power spectra. (Power spectra deferred in discussion but kept
in the channel manifest so no re-transfer is needed to add it.)

## Channel manifest

The union the six diagnostics consume — 8 channels, each present in cBottle's
`STORED_VARIABLES` too, so all are comparable:

`t2m, z500, msl, u500, v500, z300, u10m, v10m`

(`sst` is cBottle-only and not compared; the PNW land mask comes from ERA5.)

## Data locations (analysis HPC)

- PNW: `/projectnb/eb-general/jlin404/scratch/backward_emulator_data/2021_pnw_heatwave`
- Ian: `/projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian`

(cBottle bases are the sibling `.../scratch/{2021_pnw_heatwave,ian}`.)

## Components

| File | Role | Status |
|---|---|---|
| `trim_backward_fields.py` | **Source-side** channel-trim of `*fields*.h5` before Globus (numpy + h5py; ~4× cut). Now optional — see note. | ✅ done + tested |
| `makani_source.py` | `MakaniBackwardSource`: pool per-task dumps on the member axis, reverse/truncate frames to cBottle's forward 66 h window, serve native lat/lon fields. | ✅ done + tested |
| `latlon_metrics.py` | Shared cos-lat spread/skill/member-RMSE/mean-error/CRPS reducer — the fair seam both models feed. | ✅ done + tested |
| `inspect_bundle.py` | Morning-first PASS/WARN check on a transferred bundle. | ✅ done + tested |
| `survival.py` | Survivor censoring (makani `range` detector, exact ceilings) → per-member `alive` mask + survival fraction, in forward order. | ✅ done + tested |
| `compare_spread_rmse_crps.py` | adapter → censored reducer → npz + manuscript-style **5-panel + survival** figure (S6–S11 layout, cBottle-vs-makani series, display units, local-time axis). makani side wired; cBottle overlay is a documented HPC add-in. | ✅ makani side |
| `plot_survival_stability.py` | stability-only SI figure: both cases' full-horizon (−180 h) survival curves overlaid on ONE panel (hours-from-anchor axis, 66 h context line, survival-at-66 h + 50%-crossing stats in the legend), from `survival.py` npz files (scan via `submit_survival.sh --n-frames 0`, one qsub per case). `si_survival_stability.tex` is the matching SI figure block (numbers to fill from the script's stdout). | ✅ done + tested |
| `tests/` | 33 tests (schema, pooling/alignment, memory-safe paths, metric identities incl. per-lead bias-variance under censoring, forward/rollout ordering, driver E2E). | ✅ |
| `cbottle_source.py` | cBottle end-mode ensemble as a lat/lon source (batched HPX→lat/lon regrid; round-tripped ERA5 = matched-baseline truth); same interface as the makani source. Heavy env; untestable in the light env. | ✅ written, HPC-run |
| `smoke_cbottle_source.py` | Quick login-node gate for `cbottle_source`: regrids a few members, checks shapes/finiteness/physical range in seconds. Run before fanning out cBottle overlays. | ✅ |
| `make_land_mask.py` | Shared land mask (`land = isnan(sst)` on native lat/lon — the paper's convention — from the case `era5_reference`). zarr+numpy; seconds on a login node. | ✅ + tested |
| `submit_compare_spread_rmse_crps.sh` / `submit_survival.sh` | SGE wrappers (eb-general, 16 omp, e2s-custom): the per-(variable,region) compare job; the survivor-mask root job. | ✅ |
| `dispatch_compare.py` | Reads each case's `spread_rmse_crps.domains`; submits a survivor root + per-variable compare jobs (`-hold_jid` chained, cached mask). `--dry-run` previews. | ✅ + tested |
| `ageostrophic_compare.py` + `submit_ageostrophic_compare.sh` + `dispatch_ageostrophic.py` + `smoke_ageostrophic.py` | Ageostrophic imbalance (S13–S15): `\|V_a\|/\|V\|` over the 30–60° \|lat\| band, makani native vs cBottle regridded, each vs its resolution-matched ERA5, makani survivor-censored. Reuses `compute_ageostrophic`'s core verbatim. | ✅ + tested |
| `tc_tracks_compare.py` + `submit_tc_tracks_compare.sh` + `dispatch_tc_tracks_compare.py` + `smoke_tc_tracks.py` | TC tracks (Ian, Text S3): re-tracks makani natively through the paper's TempestExtremes runner (`track_member_with_tempest`, Z&U-2017 warm-core + WB2 orography), ERA5 ref = makani truth; writes the paper's parquet schema → feeds Ian's synoptic-PCA impact target. | ✅ + tested |
| `synoptic_pca_makani.py` + `synoptic_pca_compare.py` + `submit_synoptic_pca_makani.sh` + `dispatch_synoptic_pca_makani.py` | Per-ensemble EOF/PC regression (Text S2): makani on native lat/lon (√cos φ weighting), leading-8-PC → impact R² (in-sample + 10-fold CV), free-end-divergence censored. Compare vs the paper's cBottle `synoptic_pca_end.npz`. Ian consumes the TC parquet. | ✅ + tested |
| power spectra | Deferred (lowest priority; resolution asymmetry makes a raw overlay tautological). | ⏳ |

**Note on trimming:** now that the files transfer untrimmed, trimming buys ~no
analysis speed (the dumps are chunked one channel per chunk, so reading only the
8 we use is already cheap) — its only remaining value is reclaiming disk. Trim
post-transfer *only if `/projectnb` quota is tight* (trim then delete originals).
Keep the script for future makani re-runs, where trimming *before* transfer still
saves Globus time.

## Execution on the SCC: jobs vs. interactive

Follow the paper pipeline's convention — **submit batch jobs (`qsub`) for the heavy
compute, run the light aggregate/plot interactively.**

- **Batch (`qsub`, `e2s-custom`, `-P eb-general`, CPU only — no GPU, matching the
  existing `submission_scripts/`):** the survivor scan (~50 GB t2m I/O), the
  1000-member region reductions, and the cBottle HPX→lat/lon regrid. Use
  `submit_compare_spread_rmse_crps.sh`. Do **not** run these on the `scc1` login
  node.
- **Interactive (login node fine):** `inspect_bundle.py`, quick smoke tests, and
  re-plotting from saved `.npz` (seconds).

## Morning runbook

Once the transfer is confirmed complete:

```bash
# light env: numpy h5py matplotlib pyyaml  (no torch/earth2studio needed)
# 1. sanity-check what landed
python inspect_bundle.py .../backward_emulator_data/2021_pnw_heatwave --case pnw
python inspect_bundle.py .../backward_emulator_data/ian               --case ian

# 2. makani-side spread/RMSE/CRPS (real curves, per region)
python compare_spread_rmse_crps.py \
    --makani-dir .../backward_emulator_data/2021_pnw_heatwave \
    --case pnw_heatwave --variable z500 --region-name synoptic \
    --out-dir ./out/pnw/spread_rmse_crps

# 3. stability SI figure: FULL-HORIZON survival scans (batch, one per case).
#    --n-frames 0 scans every lead the dumps carry (the full rollout horizon,
#    whatever it is); write to a *_full.npz so the 12-frame survivor-mask cache
#    the compare jobs -hold_jid on stays untouched. The full scan reads ~2.5x
#    the 66 h window's t2m, so raise the wrapper's 3 h walltime on the CLI.
#    (Full literal paths on purpose — a pasted `...` placeholder has already
#    burned one round of jobs.)
qsub -l h_rt=08:00:00 submit_survival.sh \
    --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/ian \
    --n-frames 0 --out ./out/ian/survival_full.npz \
    --makani-analysis-dir /projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis
qsub -l h_rt=08:00:00 submit_survival.sh \
    --makani-dir /projectnb/eb-general/jlin404/scratch/backward_emulator_data/2021_pnw_heatwave \
    --n-frames 0 --out ./out/pnw/survival_full.npz \
    --makani-analysis-dir /projectnb/eb-general/jlin404/weather_interpolation/makani-aicr/analysis
# ... then the plot (interactive; seconds) once both npz files exist
python plot_survival_stability.py \
    --case-npz ian=./out/ian/survival_full.npz \
    --case-npz pnw_heatwave=./out/pnw/survival_full.npz \
    --out ./out/survival_stability.png
```

Then, in the full earth2studio env, add the cBottle overlay (the `# HPC` hook in
`compare_spread_rmse_crps.main`) and extend to the other diagnostics.

Trimmed/untrimmed dumps stay readable by both the makani analysis tools and this
adapter (verified by round-trip through makani's `FieldDumpWriter` /
`load_channel_members`).
