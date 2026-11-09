# Sandy reversed-endpoint conditioning experiment

This proposed replacement for Figure S12 swaps the two ERA5 endpoint states
used by the ordinary Sandy experiment. It tests how cBottle-video accommodates
reversed endpoint constraints. Whether the connecting evolution is dynamically
inadmissible must be assessed from the output; offshore motion alone does not
establish that claim.

| Frame | Forward model slot (UTC) | ERA5 conditioning source (UTC) |
| --- | --- | --- |
| F0 | 2012-10-30 00:00 | 2012-10-30 00:00, near New Jersey landfall |
| F1–F10 | Six-hourly after landfall | Unconditioned |
| F11 | 2012-11-01 18:00 | 2012-10-27 06:00, offshore precursor |

The model clock starts at landfall. Its calendar embeddings and linearly
interpolated monthly AMIP SST conditioning advance normally through the 66-hour
window. They are set by `ensemble.start_time`, independently of the ERA5 date
given to `custom_frames`. F0 matches its clock. F11's borrowed state is 132 hours
earlier than its model slot, including a 12-hour diurnal phase mismatch. SST
conditioning follows the forward model dates rather than the earlier final pin.
These mismatches must be disclosed when interpreting the stress test.

All ERA5 conditioning channels are supplied, including winds with their original
signs. Non-ERA5 channels remain NaN/unconditioned, as in the existing runner.
This is a new generative sample, not an existing video played backwards.

The named case is `scripts/_shared/cases/sandy_reversed.yaml`: both-end
conditioning only, 100 members, micro-batch size 1, seed 0, and a separate output
directory. The ordinary `sandy` case also uses seed 0; matching member indices
requires matching the micro-batch size and seed settings actually used in that
control run. No inference results are claimed here.

## Generate on the configured SCC environment

From the repository root, inspect the submission first:

```sh
uv run python scripts/ensemble_run/dispatch_ensemble.py sandy_reversed --quick --dry-run
```

Remove `--dry-run` to submit the eight-member pilot. Remove both `--quick` and
`--dry-run` for the configured 100-member run. The existing dispatcher splits
that run into two 50-member jobs and schedules their merge.

The runner needs October **and November 2012** raw ERA5 stores because it also
loads the natural forward reference window. The earlier final conditioning
state comes from October. The usual model weights and AMIP SST file are needed.

## Inspect the generated members

From `scripts/ensemble_analysis`, render the pilot on a common MSLP scale with
500 hPa wind arrows:

```sh
uv run python plot_ensemble_member.py --event sandy_reversed --mode both \
  --ensemble-size 8 --members 0-3 --layout grid --variable msl \
  --wind-overlay wind500
```

For the full run, omit `--ensemble-size 8`. Outputs land in
`output/sandy_reversed_plots/both/msl_member<N>_grid_wind500.png`.
The grid contains model output in every panel. Pinned panels name their ERA5
source; when source and model-slot dates differ, the title also names the slot.
Use the configured `z500` view to inspect the wider synoptic circulation.

Render a member from the existing ordinary Sandy both-end ensemble with the same
viewport, MSLP limits, colormap, and overlay for comparison. Explicit CLI mode
lets the colormap match the stress test (event mode uses the case's palette):

```sh
uv run python plot_ensemble_member.py \
  --ensemble-zarr /projectnb/eb-general/jlin404/scratch/sandy/both/ensemble_f0_11_n1000.zarr \
  --era5-zarr /projectnb/eb-general/jlin404/scratch/sandy/both/era5_reference.zarr \
  --member 0 --layout grid --variable msl --wind-overlay wind500 \
  --extent -85 -60 25 45 --vmin 93000 --vmax 103000 --cmap cmr.sapphire \
  --coastline-color 0.25 --output-dir output/sandy_control_plots/both
```

Inspect whether the initial storm remains continuous, decays and reforms, or
jumps toward the terminal constraint, and how its motion relates to the
environmental circulation. Any claim of dynamical inadmissibility needs
additional diagnostic support.

`era5_reference.zarr` is the actual October 30–November 1 forward trajectory,
not a reference for the deliberately reversed endpoints. Use the member grid
for this experiment; the default video juxtaposes that ordinary reference with
the generated member. Do not interpret RMSE, CRPS, coverage, or other verification
against that reference as skill for this stress test. Omitted analysis regions
block some case-study dispatchers, but do not block every diagnostic.

Inspect the generated results before replacing `conditioning_stress_test.png`
or updating the empirical description in `main.tex` and `si.tex`.
