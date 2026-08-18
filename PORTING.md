# Porting & Divergence Ledger (main ⇄ dev-clean)

Living document — the source of truth for porting between `main` (a private
mirror of [NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio); no
shared git history with `dev-clean`) and `dev-clean` (the research branch).
Read this before any porting session; update it whenever a port lands or a
divergence is added or resolved.

**Last reconciled:** 2026-07-16 — dev-clean `7dda89e` vs main `cbc8748`
(PRs #308/#309 initial port, PR #310 defect fixes + remaining safe ports).

**Porting rule:** copy from main verbatim, *except* the files in sections 1–3
below. After any port, run the verification checklist in section 6.

## 1. Custom research code — never overwrite from main

| Path | Divergence |
|---|---|
| `scripts/**`, `custom_utils/**`, `paper/**`, `test/scripts/**` | Research pipeline, analysis, and manuscript — dev-clean only |
| `installation_guide*.md`, `verify_env*.py`, `tox-full.ini`, `tox-smoke.ini` | Frozen-env tooling — dev-clean only |
| `earth2studio/models/dx/tc_tracking.py` | main's file **plus** additive custom code: `TCTrackerMinMSL`, `VARIABLES_TEMPEST_*` constants, and underscore output names (`tc_lat/tc_lon/tc_msl/tc_w10m` vs main's `tclat/tclon/tcmsl/tcw10m`) |
| `earth2studio/models/px/cbottle_interpolate.py` | Custom module, does not exist on main |
| `earth2studio/models/dx/cbottle_infill.py` | main's file **plus** additive `lat_lon_out` flag (default `True` preserves upstream behavior; candidate to upstream) |
| `earth2studio/models/dx/__init__.py`, `earth2studio/models/px/__init__.py` | main's file **plus** exports for the custom models above |
| `earth2studio/lexicon/base.py` | main's file **plus** four additive `E2STUDIO_VOCAB` entries describing the `tc_lat/...` tracker outputs |

## 2. Frozen environment — do not port while mid-paper

`pyproject.toml`, `uv.lock`, `requirements.txt`, and the `# /// script`
dependency headers in `examples/**` (torch 2.9.1 vs main's 2.11 / 0.17.0a0).

**Freeze riders** — deliberate adaptations that should revert to main-verbatim
when the environment is eventually unfrozen:

| File | Adaptation | Why |
|---|---|---|
| `tox.ini` | `basepython = python3.12`; `aifs2`/`aifs2ens` sync-and-test blocks omitted | extras absent from frozen `pyproject.toml` |
| `Makefile` | same two adaptations in `setup-ci` | same |
| `earth2studio/data/ufs.py` | `obstore` import guarded via `OptionalDependencyFailure` | `obstore` is a base dep on main, absent from the frozen lock |

## 3. Hand-merged files — merge, don't copy

- `CLAUDE.md` — custom project guide; fold in main's pointer changes manually.
- `.gitignore` — keep the union of main's and dev-clean's entries.

## 4. TC tracking cross-reference (checked 2026-07-16)

- Upstream's `recipes/tc_tracking/` (incl. `src/tempest_extremes.py`) and the
  local `scripts/ensemble_analysis/tempest_extremes_runner.py` are **parallel
  implementations**; the recipe does not replace the local pipeline.
- Configurations are the same Zarzycki & Ullrich (2017) setup (`searchbymin
  msl`, `mergedist 6°`, closed contour `msl,200,5.5`, stitch `range 8.0`,
  `mintime 54h`, `maxgap` 24 h, wind ≥ 10, |lat| ≤ 50) except:
  - warm-core contour max-search distance: local `1.0` (the canonical Z&U
    value) vs recipe `0` (stricter anchor at the MSL minimum);
  - the recipe always applies the orography stitch filter (`height ≤ 150 m`);
    local runs omit it when `zs` is not in the archived output — immaterial
    when tracks are matched to the target storm downstream.
- Sandy and Ian case YAMLs run the `tempest` tracker with `warm_core: true`.
- **Conclusion:** no main-side change alters local tracking; existing tracks
  remain valid; no rerun warranted.

## 5. Behavioral notes

- `Gaussian` perturbation (since PR #310) draws noise from a per-instance
  seeded generator instead of the global RNG: ensembles generated after that
  merge will not bit-match earlier runs. Existing outputs are unaffected.

## 6. Post-port verification checklist

1. `python -m compileall earth2studio test examples docs` passes.
2. Internal imports: every `from earth2studio.x import y` across
   `earth2studio/` and `test/` resolves to an existing module and top-level
   symbol (catches tests ported ahead of their source).
3. Third-party imports: every unguarded module-level import reachable from the
   package entry points (`earth2studio.data`, `models.*`, `io`, `perturbation`,
   `statistics`, `run`, `utils`, `lexicon`) resolves to a package in the frozen
   `uv.lock` (catches `obstore`-class breakage).
4. Tests-vs-source: any test file copied from main requires its source module
   to be copied too, or verified additively compatible.
5. `git diff main..dev-clean --name-only` afterwards must equal exactly the
   union of sections 1–3; anything else is either a port defect or a new
   divergence to record here.
