"""Synoptic-PCA on the makani backward ensemble (native lat/lon), paper-faithful.

Runs the paper's per-ensemble EOF/PC diagnostic (Text S2) on makani, so its output
npz mirrors the cBottle `synoptic_pca_<mode>.npz` and the two can be compared arm
for arm.  The method (unchanged from the paper):

* Decompose the ENSEMBLE's own free-end z500 field: build the member x pixel matrix
  over the synoptic domain, z-score per variable, SVD the member-centered matrix.
  `EOFn` = spatial loading, `PCn` = per-member score.  The basis is the ensemble's
  own -- there is NO shared/external basis.
* Project ERA5's free-end field INTO that member basis; report its percentile.
* Headline: predict the free-end impact target from the leading-8 PC scores by OLS;
  report in-sample and deterministic 10-fold-CV R^2 (Figs S2-S5).  Target =
  land-mean t2m (PNW) or the free-end TC storm properties from the tracks parquet
  (Ian).

Locked comparison choice (Option A): makani is decomposed on its NATIVE 0.25 deg
grid, so lat/lon pixels get **sqrt(cos phi) area weighting** (equivalent to the
paper's equal-area HEALPix); cBottle stays on HEALPix via the paper's own driver.
`build_feature_matrix_latlon` reduces exactly to the paper's `_build_feature_matrix`
when weights are uniform.

Member selection (censoring) -- a makani-specific, principled divergence from the
paper's all-1000-member decomposition, because the autoregressive rollout can go
numerically unstable while the video-diffusion ensemble cannot:

  * The paper decomposes all N=1000 cBottle members (Text S2); cBottle never blows
    up.  makani members can, and the free-end frame (forward frame 0, the window
    start 66 h from the pinned present) is the FURTHEST point of the backward
    rollout -- exactly where a diverged member's fields are most corrupt.  Left in,
    one blown-up member would dominate EOF1 with numerical garbage instead of
    physical precursor diversity.
  * A member is admitted iff it is ALIVE at the free-end frame under the pipeline's
    single divergence definition -- the survivor mask from `survival.py`
    (makani's own `ensemble_horizon` `range` detector: t2m outside [150, 350] K or
    non-finite marks divergence, applied MONOTONICALLY in rollout order).  Because
    the mask is monotone (a member that diverges stays dead), "alive at the
    free-end" == "a fully valid trajectory end to end", so only members with a
    physically valid precursor field enter the decomposition.
  * The count of admitted members is reported and saved as ``n_analyzable`` (the
    same bookkeeping the paper quotes, e.g. Ian "N=993 analyzable members").  This
    is a no-op for the PNW ridge (stable) and live for Ian.  ``--no-pca-censor``
    reverts to the paper's all-member convention.

This censoring is the same divergence criterion the spread/RMSE/CRPS and ageostrophic
diagnostics use, so "diverged" means one thing across the whole comparison.

Reuses `compute_synoptic_pca._pca` (SVD + sign-fix), `pca_stats` (OLS + CV), and
`tc_track_targets.free_end_target_arrays` verbatim.

Heavy env (the makani source is numpy/h5py, but `_pca`/`free_end_target_arrays`
live next to torch/pandas imports).  Free-end frame = forward frame 0 (the
unpinned window start; makani pins the present at forward -1, == cBottle end mode).
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
EA_SUB = EA_DIR / "submission_scripts"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _p in (str(EA_DIR), str(EA_SUB)):  # appended: never shadow our modules
    if _p not in sys.path:
        sys.path.append(_p)

import _dispatch_lib as dl  # noqa: E402
from latlon_metrics import cos_lat_weights, select_pixels  # noqa: E402
from makani_source import MakaniBackwardSource, resolve_expected_anchor  # noqa: E402

FREE_END_FRAME = 0  # forward frame 0 = the unpinned window start (end-mode precursor)


def build_feature_matrix_latlon(
    fields: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted PCA feature matrix for a lat/lon domain.

    ``fields`` is ``[N, V, P]`` (members, variables, domain pixels); ``weights`` is
    ``[P]`` (cos-lat, >= 0).  Per variable: cos-lat-weighted mean/std z-score, then
    scale each pixel column by ``sqrt(weight)`` so the SVD's implicit inner product
    is the area integral (the equal-area-HEALPix equivalent).  Returns
    ``(X [N, V*P], per_var_mean [V], per_var_std [V], sqrt_w [P])``.

    With uniform weights this is byte-for-byte the paper's `_build_feature_matrix`.
    """
    fields = np.asarray(fields, dtype=np.float64)
    N, V, P = fields.shape
    w = np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum())
    sqrt_w = np.sqrt(w)
    per_var_mean = np.empty(V, dtype=np.float32)
    per_var_std = np.empty(V, dtype=np.float32)
    X = np.empty((N, V, P), dtype=np.float32)
    for v in range(V):
        slab = fields[:, v, :]  # [N, P]
        mu = float((slab * w).sum() / (N * wsum))
        var = float((w * (slab - mu) ** 2).sum() / (N * wsum))
        sd = float(np.sqrt(var)) if var > 0 else 1.0
        per_var_mean[v] = mu
        per_var_std[v] = sd
        X[:, v, :] = (((slab - mu) / sd) * sqrt_w).astype(np.float32)
    return X.reshape(N, V * P), per_var_mean, per_var_std, sqrt_w


def project_era5(
    era5_fields: np.ndarray, per_var_mean, per_var_std, sqrt_w, feature_mean,
    components: np.ndarray,
) -> np.ndarray:
    """ERA5 free-end field ``[V, P]`` -> PC scores on the ensemble's basis (Text S2)."""
    V, P = era5_fields.shape
    z = np.empty((V, P), dtype=np.float64)
    for v in range(V):
        z[v] = ((era5_fields[v] - per_var_mean[v]) / per_var_std[v]) * sqrt_w
    centered = z.reshape(-1) - feature_mean
    return (centered @ components.T).astype(np.float32)


def regression_summary(scores: np.ndarray, target: np.ndarray, n_pc: int = 8) -> Dict:
    """In-sample + 10-fold-CV R^2 of ``target ~ leading-n_pc PC scores`` (Text S2)."""
    from pca_stats import _kfold_r2, _ols_fit, _r2_score, _regression_f_pvalue

    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(y)
    P = scores[:, :n_pc]
    Pf, yf = P[finite], y[finite]
    if yf.size <= n_pc + 1:
        return {"r2_insample": np.nan, "r2_cv": np.nan, "pvalue": np.nan,
                "n_used": int(yf.size)}
    beta, intercept = _ols_fit(Pf, yf)
    r2 = _r2_score(yf, intercept + Pf @ beta)
    try:  # scipy is lazy in pca_stats; the p-value is secondary to the R^2
        pvalue = float(_regression_f_pvalue(r2, int(yf.size), n_pc))
    except Exception:  # noqa: BLE001
        pvalue = float("nan")
    return {
        "r2_insample": float(r2),
        "r2_cv": float(_kfold_r2(Pf, yf, n_folds=10)),
        "pvalue": pvalue,
        "n_used": int(yf.size),
    }


def _percentile_of(scores_col: np.ndarray, value: float) -> float:
    n = scores_col.shape[0]
    return float(100.0 * (scores_col <= value).sum() / n) if n else np.nan


# ---- makani field loaders (free-end frame over a domain) --------------------
def _domain_freeend(makani, variables: List[str], region) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fields [N,V,P], weights [P], flat [P])`` at the free-end frame over a region."""
    flat, w = select_pixels(makani.lat, makani.lon, region)
    N = makani.n_members
    parts = []
    for name in variables:
        ens, _truth = makani.load_region(name, flat)  # (N, L, P)
        parts.append(ens[:, FREE_END_FRAME, :])       # (N, P) free-end
    return np.stack(parts, axis=1), w, flat            # (N, V, P)


def _era5_freeend(makani, variables: List[str], flat: np.ndarray) -> np.ndarray:
    """ERA5 (makani native truth) free-end field ``[V, P]`` over the domain pixels."""
    out = []
    for name in variables:
        _ens, truth = makani.load_region(name, flat)   # truth (L, P)
        out.append(truth[FREE_END_FRAME])              # (P,)
    return np.stack(out, axis=0)                        # (V, P)


def _impact_scalar_t2m(makani, region, land_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Free-end land-mean t2m per member (PNW impact target). Returns ``(y [N], era5)``."""
    flat, w = select_pixels(makani.lat, makani.lon, region, land_mask)
    ens, truth = makani.load_region("t2m", flat)       # (N,L,P), (L,P)
    wn = w / w.sum()
    y = (ens[:, FREE_END_FRAME, :] * wn).sum(axis=1)   # (N,)
    era5 = float((truth[FREE_END_FRAME] * wn).sum())
    return y.astype(np.float64), era5


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-dir", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variable", action="append", default=None, dest="variables",
                   help="Precursor variable(s) for the PCA (default z500).")
    p.add_argument("--region-name", default="synoptic")
    p.add_argument("--impact-kind", choices=["scalar", "track"], default=None,
                   help="Default: track for a case with tc_tracks, else scalar t2m.")
    p.add_argument("--impact-variable", default="t2m")
    p.add_argument("--impact-region", default="impact")
    p.add_argument("--land-mask", default=None, help="For scalar t2m (PNW).")
    p.add_argument("--tc-parquet", default=None,
                   help="makani tracks parquet for --impact-kind track (Ian).")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--expected-anchor", default=None,
                   help="ISO-hour prefix the dumps' anchor valid-time must match "
                   "(default: the case's window end, start_time + 66 h; "
                   "'none' disables the check).")
    p.add_argument("--max-d", type=int, default=10)
    p.add_argument("--n-pc", type=int, default=8, help="PCs in the regression (Text S2: 8).")
    p.add_argument("--pca-censor", dest="censor", action="store_true", default=True,
                   help="Exclude members diverged at the free-end frame from the EOF "
                   "(makani-specific; a no-op for a stable case). Default on.")
    p.add_argument("--no-pca-censor", dest="censor", action="store_false",
                   help="Decompose over ALL members (the paper's cBottle convention).")
    p.add_argument("--survival-npz", default=None,
                   help="Survivor-mask cache for censoring (shared with the spread run).")
    p.add_argument("--makani-analysis-dir", default=None)
    args = p.parse_args(argv)

    from compute_synoptic_pca import _pca  # heavy env

    case = dl.load_case(args.case)
    variables = args.variables or ["z500"]
    region = dl.region_box(case, args.region_name)
    if region is None:
        raise SystemExit(f"region '{args.region_name}' not in case '{args.case}'")
    impact_kind = args.impact_kind or ("track" if (case.get("tc_tracks")) else "scalar")

    makani = MakaniBackwardSource(args.makani_dir, n_frames=args.n_frames,
                                  expected_anchor_iso=resolve_expected_anchor(case, args.expected_anchor))
    print(f"[syn-pca-mk] {args.case} precursor={variables} region={args.region_name} "
          f"impact={impact_kind} N={makani.n_members}")

    # --- member selection: exclude free-end-diverged members (makani-specific) --
    analyzable = np.ones(makani.n_members, dtype=bool)
    if args.censor:
        from compare_spread_rmse_crps import _load_or_compute_survival
        cache = Path(args.survival_npz) if args.survival_npz else None
        surv = _load_or_compute_survival(
            makani, cache_path=cache, makani_analysis_dir=args.makani_analysis_dir)
        analyzable = np.asarray(surv["alive"], bool)[:, FREE_END_FRAME]  # alive at free-end
        print(f"[syn-pca-mk] analyzable members (alive at free-end frame): "
              f"{int(analyzable.sum())}/{makani.n_members}")

    # --- PCA on the free-end precursor field ---------------------------------
    fields_all, w, flat = _domain_freeend(makani, variables, region)  # (N,V,P)
    fields = fields_all[analyzable]                                    # analyzable subset
    X, per_var_mean, per_var_std, sqrt_w = build_feature_matrix_latlon(fields, w)
    scores, components, evr, feature_mean, eigenvalues = _pca(X, max_d=args.max_d)
    print(f"[syn-pca-mk] domain pixels={flat.size} EVR(1..3)="
          f"{[round(float(evr[i]), 3) for i in range(min(3, evr.size))]}")

    # ERA5 (native truth) projected into the ensemble basis.
    era5_fields = _era5_freeend(makani, variables, flat)  # (V,P)
    era5_pc = project_era5(era5_fields, per_var_mean, per_var_std, sqrt_w,
                           feature_mean, components[:args.n_pc])
    era5_pct = np.array([_percentile_of(scores[:, j], era5_pc[j])
                         for j in range(args.n_pc)], dtype=np.float32)

    # --- impact target(s) + regression ---------------------------------------
    reg: Dict[str, Dict] = {}
    payload: Dict = {}
    if impact_kind == "scalar":
        land_mask = np.load(args.land_mask).astype(bool) if args.land_mask else None
        y_all, era5_y = _impact_scalar_t2m(
            makani, dl.region_box(case, args.impact_region), land_mask)
        y = y_all[analyzable]  # align with the analyzable scores
        reg[args.impact_variable] = regression_summary(scores, y, n_pc=args.n_pc)
        payload["target_scalar"] = y.astype(np.float32)
        payload["target_era5"] = np.float32(era5_y)
        payload["target_names"] = np.array([args.impact_variable])
    else:
        if not args.tc_parquet:
            raise SystemExit("--impact-kind track needs --tc-parquet (the makani tracks).")
        from tc_track_targets import free_end_target_arrays
        landfall = dl.resolve_landfall_frame(case)
        anchor = landfall if 0 <= landfall < args.n_frames else args.n_frames - 1
        baked = free_end_target_arrays(args.tc_parquet, FREE_END_FRAME,
                                       makani.n_members, anchor_frame=anchor)
        tgt = np.asarray(baked["track_target_fe"], dtype=np.float64)[analyzable]  # [Na,4]
        names = ["tc_lat", "tc_lon", "tc_dist_km", "tc_msl_hpa"]
        for j, nm in enumerate(names):
            reg[nm] = regression_summary(scores, tgt[:, j], n_pc=args.n_pc)
        payload["target_track_fe"] = tgt.astype(np.float32)
        payload["target_names"] = np.array(names)
        payload["track_era5_fe"] = np.asarray(baked["track_era5_fe"], np.float32)

    for nm, r in reg.items():
        print(f"[syn-pca-mk]   {nm}: R2_insample={r['r2_insample']:.3f} "
              f"R2_cv={r['r2_cv']:.3f} (n={r['n_used']})")

    # --- EOF loading maps: undo z-score + area weight, keep on domain pixels --
    eof = components[:args.max_d].reshape(args.max_d, len(variables), flat.size).copy()
    for v in range(len(variables)):
        eof[:, v, :] = eof[:, v, :] / np.where(sqrt_w > 0, sqrt_w, 1.0) * per_var_std[v]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"synoptic_pca_makani_{args.case}"
    npz = out / f"{stem}.npz"
    np.savez(
        npz,
        case_name=np.array(case.get("display_name", args.case)),
        model=np.array("makani"), variables=np.array(variables),
        region_name=np.array(args.region_name), impact_kind=np.array(impact_kind),
        free_end_frame=np.int32(FREE_END_FRAME), n_members=np.int32(makani.n_members),
        censored=np.bool_(args.censor), n_analyzable=np.int32(int(analyzable.sum())),
        analyzable=analyzable,
        pc_scores=scores, explained_variance_ratio=evr, eigenvalues=eigenvalues,
        per_var_mean=per_var_mean, per_var_std=per_var_std,
        era5_pc_score=era5_pc, era5_pc_percentile=era5_pct,
        eof_loadings_domain=eof.astype(np.float32),
        domain_flat=flat.astype(np.int64),
        domain_lat=makani.lat.astype(np.float32), domain_lon=makani.lon.astype(np.float32),
        n_pc=np.int32(args.n_pc),
        r2_insample=np.array([reg[k]["r2_insample"] for k in reg], np.float32),
        r2_cv=np.array([reg[k]["r2_cv"] for k in reg], np.float32),
        r2_pvalue=np.array([reg[k]["pvalue"] for k in reg], np.float32),
        r2_target_names=np.array(list(reg.keys())),
        **payload,
    )
    print(f"[syn-pca-mk] wrote {npz}")


if __name__ == "__main__":
    main()
