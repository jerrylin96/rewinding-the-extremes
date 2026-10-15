"""Storm-of-interest resolution through the paper's track QC.

Locks the Ian bug: tracking is global and TempestExtremes numbers concurrent
systems arbitrarily, so ``path_id=0`` can be another storm.  19 of the 1000
cBottle Ian members had a western-Pacific (or eastern-Pacific / Bay of Bengal)
system on path 0, and the comparison read their "Ian" targets and colour
longitudes from it.  Every consumer now resolves the storm through
``case_track_qc`` (the paper's round-5 rule), checked here on a synthetic
parquet written with the paper's own ``save_tracks_parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

BW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BW))

import case_track_qc as ctq  # noqa: E402  (puts _shared + ensemble_analysis on the path)
import _dispatch_lib as dl  # noqa: E402
import plot_tc_tracks_compare as tp  # noqa: E402
import synoptic_pca_compare as spc  # noqa: E402
from tc_tracks_io import save_tracks_parquet  # noqa: E402

T = 12  # 66 h window, 6-hourly
IAN_FE = (24.5, -82.25)  # free end (frame 0)
IAN_ANCHOR = (33.0, -79.25)  # SC landfall (frame 11 = the case's landfall_frame)
WPAC = (26.0, 131.5)  # the system 15 of the 19 cBottle members carry on path 0


def _track(start, end, msl0=1000.0, msl1=990.0):
    """``[T, 4]`` straight-line track in tracker conventions (lon [0, 360), Pa)."""
    a = np.empty((T, 4))
    a[:, 0] = np.linspace(start[0], end[0], T)
    a[:, 1] = np.linspace(start[1], end[1], T) % 360.0
    a[:, 2] = np.linspace(msl0, msl1, T) * 100.0
    a[:, 3] = 30.0
    return a


def _ens():
    """5 members: 0, 3, 4 Ian on path 0; 1 WPac on path 0 + Ian on path 1;
    2 no Ian anywhere (only the WPac system)."""
    ian = _track(IAN_FE, IAN_ANCHOR)
    wpac = _track(WPAC, (WPAC[0] + 1.0, WPAC[1] - 2.0), 1009.0, 1008.0)
    ens = np.full((5, 2, T, 4), np.nan)
    for m in (0, 3, 4):
        ens[m, 0] = ian
    ens[1, 0], ens[1, 1] = wpac, ian
    ens[2, 0] = wpac
    return ens, ian[None, None].copy()


@pytest.fixture(scope="module")
def case():
    return dl.load_case("ian")


@pytest.fixture()
def parquet(tmp_path):
    ens, era5 = _ens()
    p = tmp_path / "tc_tracks_end.parquet"
    save_tracks_parquet(str(p), ensemble_arr=ens, era5_arr=era5)
    return str(p)


def test_anchor_frame_is_the_case_landfall(case):
    assert ctq.anchor_frame(case, T) == 11
    assert ctq.anchor_frame({"tc_tracks": {}}, T) == T - 1  # unset -> last lead


def test_alt_path_member_resolves_to_ian_and_no_track_is_excluded(case, parquet):
    _ens_arr, _era5, qc = ctq.load_track_qc(parquet, case)
    assert qc.analyzable.tolist() == [True, True, False, True, True]
    assert qc.resolved_path_id[1] == 1  # Ian, not path 0
    tgt = ctq.free_end_target_matrix(qc)
    np.testing.assert_allclose(tgt[1, :2], IAN_FE, atol=1e-3)  # not the WPac fix
    assert np.isnan(tgt[2]).all()


def test_figure_draws_the_resolved_storm(case, parquet):
    ens, era5, qc = tp._load(parquet, case)
    assert ens.shape == (5, 1, T, 4) and era5.shape == (1, 1, T, 4)
    np.testing.assert_allclose(ens[1, 0, 0, :2], (IAN_FE[0], IAN_FE[1] % 360), atol=1e-3)
    assert np.isnan(ens[2]).all()


def test_figure_colour_longitude_comes_from_track_qc(case, parquet, tmp_path):
    # The paper's npz bakes path 0: member 1's longitude is the WPac system's.
    baked = np.full((5, 4), np.nan)
    baked[[0, 3, 4], 1] = IAN_FE[1]
    baked[1, 1] = WPAC[1]
    npz = tmp_path / "synoptic_pca_end.npz"
    np.savez(npz, pc_scores=np.zeros((5, 3)), track_target_fe=baked)
    _e, _r, qc = tp._load(parquet, case)
    pca = tp._load_pca_for_color(str(npz), qc.free_end_targets()["fe_lon_deg"])
    assert pca["tc_lon"][1] == pytest.approx(IAN_FE[1], abs=1e-3)
    assert np.isnan(pca["tc_lon"][2])  # excluded member


def test_figure_colour_longitude_maps_onto_survivor_rows(tmp_path):
    npz = tmp_path / "pca_makani.npz"
    keep = np.array([True, False, True, True])
    np.savez(npz, pc_scores=np.zeros((3, 2)), analyzable=keep, n_members=4)
    pca = tp._load_pca_for_color(str(npz), np.array([10.0, 11.0, 12.0, 13.0]))
    np.testing.assert_array_equal(pca["tc_lon"], [10.0, 12.0, 13.0])


def test_survivor_mask_requires_the_66h_window(tmp_path):
    alive = np.ones((5, T), bool)
    alive[3, :2] = False
    ok = tmp_path / "survival_ian.npz"
    np.savez(ok, alive=alive)
    assert tp.survivor_mask(str(ok), 5, T).tolist() == [True, True, True, False, True]
    full = tmp_path / "survival_full.npz"  # full horizon: frame 0 is 180 h back
    np.savez(full, alive=np.ones((5, 31), bool))
    with pytest.raises(SystemExit, match="66 h"):
        tp.survivor_mask(str(full), 5, T)
    with pytest.raises(SystemExit):
        tp.survivor_mask(str(ok), 6, T)  # member count mismatch


def test_figure_requires_survival_mask(parquet, tmp_path):
    with pytest.raises(SystemExit, match="--survival-npz"):
        tp.main(["--case", "ian", "--makani-parquet", parquet,
                 "--out-dir", str(tmp_path)])


def _cbottle_track_npz(path, n=5):
    baked = np.zeros((n, 4))
    baked[1, :2] = WPAC
    np.savez(path, model="cbottle", case_name="Hurricane Ian", free_end_frame=0,
             pc_scores=np.random.default_rng(0).standard_normal((n, 10)),
             explained_variance_ratio=np.linspace(0.3, 0.0, 10),
             track_target_fe=baked)


def _makani_track_npz(path, source="track_qc"):
    extra = {"target_source": source} if source else {}
    np.savez(path, model="makani", case_name="Hurricane Ian", free_end_frame=0,
             pc_scores=np.random.default_rng(1).standard_normal((4, 10)),
             explained_variance_ratio=np.linspace(0.4, 0.0, 10),
             analyzable=np.array([True, True, False, True, True]),
             target_track_fe=np.zeros((4, 4)),
             target_names=np.array(["tc_lat", "tc_lon", "tc_dist_km", "tc_msl_hpa"]),
             **extra)


def test_compare_marks_track_target_sources(tmp_path):
    _cbottle_track_npz(tmp_path / "cb.npz")
    _makani_track_npz(tmp_path / "mk.npz")
    _makani_track_npz(tmp_path / "mk_old.npz", source=None)
    assert spc._read_pca_npz(str(tmp_path / "cb.npz"))["track_source"] == "path0"
    assert spc._read_pca_npz(str(tmp_path / "mk.npz"))["track_source"] == "track_qc"
    assert spc._read_pca_npz(str(tmp_path / "mk_old.npz"))["track_source"] == "path0"


def test_compare_accepts_a_track_qc_cbottle_npz(case, parquet, tmp_path):
    # An npz baked by the fixed compute_synoptic_pca (track_qc.free_end_npz_arrays)
    # needs no parquet: its targets already come from the track QC.
    _e, _r, qc = ctq.load_track_qc(parquet, case)
    np.savez(tmp_path / "cb.npz", model="cbottle", case_name="Hurricane Ian",
             free_end_frame=0, pc_scores=np.zeros((5, 10)),
             explained_variance_ratio=np.linspace(0.3, 0.0, 10),
             **qc.free_end_npz_arrays())
    cb = spc._read_pca_npz(str(tmp_path / "cb.npz"))
    assert cb["track_source"] == "track_qc"
    assert cb["targets"]["tc_lon"][1] == pytest.approx(IAN_FE[1], abs=1e-3)
    assert np.isnan(cb["targets"]["tc_lat"][2])


def test_compare_refuses_path0_targets_without_parquet(tmp_path):
    _cbottle_track_npz(tmp_path / "cb.npz")
    _makani_track_npz(tmp_path / "mk.npz")
    with pytest.raises(SystemExit, match="--cbottle-parquet"):
        spc.main(["--makani-npz", str(tmp_path / "mk.npz"),
                  "--cbottle-npz", str(tmp_path / "cb.npz"),
                  "--case", "ian", "--out-dir", str(tmp_path / "out")])


def test_compare_recomputes_targets_from_parquet(case, parquet, tmp_path):
    _cbottle_track_npz(tmp_path / "cb.npz")
    cb = spc._read_pca_npz(str(tmp_path / "cb.npz"))
    spc._apply_track_qc(cb, parquet, case)
    assert cb["track_source"] == "track_qc"
    assert cb["targets"]["tc_lon"][1] == pytest.approx(IAN_FE[1], abs=1e-3)
    assert np.isnan(cb["targets"]["tc_lat"][2])  # no Ian: dropped from the regression

    # makani rows are its survivors: targets follow the npz's analyzable mask.
    _makani_track_npz(tmp_path / "mk_old.npz", source=None)
    mk = spc._read_pca_npz(str(tmp_path / "mk_old.npz"))
    spc._apply_track_qc(mk, parquet, case)
    assert mk["targets"]["tc_lat"].shape == (4,)
    assert mk["targets"]["tc_lat"][1] == pytest.approx(IAN_FE[0], abs=1e-3)  # member 1
    assert np.isfinite(mk["targets"]["tc_lat"][2])  # member 3, not the dropped 2
