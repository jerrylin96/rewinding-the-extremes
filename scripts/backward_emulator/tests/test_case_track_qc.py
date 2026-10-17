"""Storm-of-interest resolution and population through the paper's track QC.

Locks the Ian bug: tracking is global and TempestExtremes numbers concurrent
systems arbitrarily, so ``path_id=0`` can be another storm.  19 of the 1000
cBottle Ian members had a western-Pacific (or eastern-Pacific / Bay of Bengal)
system on path 0, and the comparison read their "Ian" targets and colour
longitudes from it.  Every consumer now resolves the storm through
``case_track_qc`` (the paper's rule), never reads track targets stored in an
npz, and excludes a member whose storm has no fix at the free-end frame rather
than substituting a later fix.  Checked on a synthetic parquet written with the
paper's own ``save_tracks_parquet``.
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
IAN_ANCHOR = (33.0, -79.25)  # SC landfall (frame 11, the pinned frame)
WPAC = (26.0, 131.5)  # the system 15 of the 19 cBottle members carry on path 0


def _track(start, end, msl0=1000.0, msl1=990.0, first=0):
    """``[T, 4]`` straight-line track in tracker conventions (lon [0, 360), Pa)."""
    a = np.empty((T, 4))
    a[:, 0] = np.linspace(start[0], end[0], T)
    a[:, 1] = np.linspace(start[1], end[1], T) % 360.0
    a[:, 2] = np.linspace(msl0, msl1, T) * 100.0
    a[:, 3] = 30.0
    a[:first] = np.nan
    return a


def _ens():
    """6 members: 0, 3, 4 Ian on path 0; 1 WPac on path 0 + Ian on path 1;
    2 no Ian anywhere (only the WPac system); 5 Ian first detected at frame 1."""
    ian = _track(IAN_FE, IAN_ANCHOR)
    wpac = _track(WPAC, (WPAC[0] + 1.0, WPAC[1] - 2.0), 1009.0, 1008.0)
    ens = np.full((6, 2, T, 4), np.nan)
    for m in (0, 3, 4):
        ens[m, 0] = ian
    ens[1, 0], ens[1, 1] = wpac, ian
    ens[2, 0] = wpac
    ens[5, 0] = _track(IAN_FE, IAN_ANCHOR, first=1)
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


def test_frames_are_the_end_mode_rule(case, parquet):
    _e, _r, qc = ctq.load_track_qc(parquet, case)
    assert (qc.config["anchor_frame"], qc.config["free_end_frame"]) == (11, 0)


def test_alt_path_member_resolves_to_ian_and_excluded_members_are_nan(case, parquet):
    _ens_arr, _era5, qc = ctq.load_track_qc(parquet, case)
    assert qc.analyzable.tolist() == [True, True, False, True, True, False]
    assert qc.resolved_path_id[1] == 1  # Ian, not path 0
    assert qc.table["exclusion"].tolist()[5] == "no_free_end_fix"
    tgt = ctq.free_end_target_matrix(qc)
    np.testing.assert_allclose(tgt[1, :2], IAN_FE, atol=1e-3)  # not the WPac fix
    assert np.isnan(tgt[[2, 5]]).all()  # no frame-1 fix stands in for member 5


def test_figure_draws_only_analyzable_storms(case, parquet):
    ens, era5, qc = tp._load(parquet, case)
    assert ens.shape == (6, 1, T, 4) and era5.shape == (1, 1, T, 4)
    np.testing.assert_allclose(ens[1, 0, 0, :2], (IAN_FE[0], IAN_FE[1] % 360), atol=1e-3)
    assert np.isnan(ens[[2, 5]]).all()


def test_figure_colour_longitude_comes_from_track_qc(case, parquet, tmp_path):
    # The paper's npz bakes path 0: member 1's longitude is the WPac system's.
    baked = np.full((6, 4), np.nan)
    baked[[0, 3, 4, 5], 1] = IAN_FE[1]
    baked[1, 1] = WPAC[1]
    npz = tmp_path / "synoptic_pca_end.npz"
    np.savez(npz, pc_scores=np.zeros((6, 3)), track_target_fe=baked)
    _e, _r, qc = tp._load(parquet, case)
    pca = tp._load_pca_for_color(str(npz), qc.free_end_targets()["fe_lon_deg"])
    assert pca["tc_lon"][1] == pytest.approx(IAN_FE[1], abs=1e-3)
    assert np.isnan(pca["tc_lon"][[2, 5]]).all()  # excluded members


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


def _cbottle_track_npz(path, n=6):
    baked = np.zeros((n, 4))
    baked[1, :2] = WPAC
    np.savez(path, model="cbottle", case_name="Hurricane Ian", free_end_frame=0,
             impact_kind="track",
             pc_scores=np.random.default_rng(0).standard_normal((n, 10)),
             explained_variance_ratio=np.linspace(0.3, 0.0, 10),
             track_target_fe=baked)


def _makani_track_npz(path):
    np.savez(path, model="makani", case_name="Hurricane Ian", free_end_frame=0,
             impact_kind="track",
             pc_scores=np.random.default_rng(1).standard_normal((5, 10)),
             explained_variance_ratio=np.linspace(0.4, 0.0, 10),
             analyzable=np.array([True, True, False, True, True, True]),
             target_track_fe=np.zeros((5, 4)),
             target_names=np.array(["tc_lat", "tc_lon", "tc_dist_km", "tc_msl_hpa"]))


def test_compare_never_reads_track_targets_from_an_npz(tmp_path):
    _cbottle_track_npz(tmp_path / "cb.npz")
    _makani_track_npz(tmp_path / "mk.npz")
    for name in ("cb.npz", "mk.npz"):
        m = spc._read_pca_npz(str(tmp_path / name))
        assert m["impact_kind"] == "track" and m["targets"] == {}


@pytest.mark.parametrize("missing", ["makani", "cbottle"])
def test_compare_requires_both_parquets_for_a_track_case(parquet, tmp_path, missing):
    _cbottle_track_npz(tmp_path / "cb.npz")
    _makani_track_npz(tmp_path / "mk.npz")
    argv = ["--makani-npz", str(tmp_path / "mk.npz"),
            "--cbottle-npz", str(tmp_path / "cb.npz"),
            "--case", "ian", "--out-dir", str(tmp_path / "out")]
    other = "cbottle" if missing == "makani" else "makani"
    argv += [f"--{other}-parquet", parquet]
    with pytest.raises(SystemExit, match=f"--{missing}-parquet"):
        spc.main(argv)


def test_compare_resolves_targets_from_parquet(case, parquet, tmp_path, capsys):
    _cbottle_track_npz(tmp_path / "cb.npz")
    cb = spc._read_pca_npz(str(tmp_path / "cb.npz"))
    spc._track_targets(cb, parquet, case, name="cbottle")
    assert "[pca-cmp] cbottle track QC" in capsys.readouterr().out
    assert cb["targets"]["tc_lon"][1] == pytest.approx(IAN_FE[1], abs=1e-3)
    assert np.isnan(cb["targets"]["tc_lat"][[2, 5]]).all()  # dropped from the regression

    # makani rows are its survivors: targets follow the npz's analyzable mask.
    _makani_track_npz(tmp_path / "mk.npz")
    mk = spc._read_pca_npz(str(tmp_path / "mk.npz"))
    spc._track_targets(mk, parquet, case, name="makani")
    assert mk["targets"]["tc_lat"].shape == (5,)
    assert mk["targets"]["tc_lat"][1] == pytest.approx(IAN_FE[0], abs=1e-3)  # member 1
    assert np.isfinite(mk["targets"]["tc_lat"][2])  # member 3, not the dropped 2
    assert np.isnan(mk["targets"]["tc_lat"][4])  # member 5: no free-end fix
