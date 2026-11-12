"""Light-env tests for the ageostrophic comparison figure and its readout.

Fixtures are written by the compute script's own ``_save_npz``, so the reader is
tested against the schema the job actually writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BW))
sys.path.insert(0, str(BW.parent / "_shared"))

import _dispatch_lib as dl  # noqa: E402
import ageostrophic_compare as ac  # noqa: E402
import plot_ageostrophic_compare as pac  # noqa: E402

M, L = 5, 12
# Typical values, inside the pinned axis ranges of every row.
TYPICAL = {"ratio": 0.20, "mean_v": 17.0, "mean_vg": 19.0, "mean_va": 4.0}


def _write(out_root: Path, case: str, *, band="30-60 |lat|", dead=True,
           makani_offset=0.0, cbottle_offset=0.05) -> np.ndarray:
    """One case's npz.  Member 0 of makani is dead at frames 0-1 and holds 1e8
    there; the rest sit ``makani_offset`` above ERA5, cBottle ``cbottle_offset``.
    ERA5's round trip sits 0.02 below native ERA5.  Returns the alive mask."""
    era = {k: np.full(L, v) for k, v in TYPICAL.items()}
    rt = {k: v - 0.02 for k, v in era.items()}
    mk = {k: np.tile(v + makani_offset, (M, 1)) for k, v in era.items()}
    cb = {k: np.tile(v + cbottle_offset, (M, 1)) for k, v in era.items()}
    alive = np.ones((M, L), bool)
    if dead:
        for k in mk:
            mk[k][0, :2] = 1e8
        alive[0, :2] = False
    c = dl.load_case(case)
    meta = {"display_name": c["display_name"], "timezone": dl.resolve_timezone(c),
            "start_time": dl.start_time_iso_from_case(c)}
    ac._save_npz(out_root / case / "ageostrophic" / ac.npz_name(case, 500),
                 {"makani": {"ens": mk, "baseline": era},
                  "cbottle": {"ens": cb, "baseline": rt}},
                 lead_hours=np.arange(L) * 6.0, band_label=band, level=500,
                 pinned_frame=L - 1, case_meta=meta, alive=alive)
    return alive


def _load(out_root: Path, case: str):
    return pac.load(out_root / case / "ageostrophic" / ac.npz_name(case, 500))


def test_dead_member_frames_leave_the_mean(tmp_path):
    _write(tmp_path, "ian")
    d = _load(tmp_path, "ian")
    mean = pac.ensemble_mean(pac.member_traces(d, "makani", "ratio"))
    np.testing.assert_allclose(mean, TYPICAL["ratio"])  # the 1e8 never enters
    # cBottle is never censored.
    np.testing.assert_allclose(pac.ensemble_mean(pac.member_traces(d, "cbottle", "ratio")),
                               TYPICAL["ratio"] + 0.05)


def test_mean_is_nan_where_no_member_survives():
    tr = np.array([[1.0, np.nan], [3.0, np.nan]])
    mean = pac.ensemble_mean(tr)
    assert mean[0] == 2.0 and np.isnan(mean[1])


def test_departures_cover_only_the_unconditioned_frames(tmp_path):
    _write(tmp_path, "ian", makani_offset=0.003)
    dep = pac.departures(_load(tmp_path, "ian"))
    assert {len(x) for x in dep.values()} == {L - 1}
    np.testing.assert_allclose(dep["Rewind-AR - ERA5"], 0.003)
    np.testing.assert_allclose(dep["cBottle-video - ERA5"], 0.05)
    np.testing.assert_allclose(dep["cBottle-video - ERA5 round trip"], 0.07)


def test_readout_reports_survivors_and_departures(tmp_path, capsys):
    _write(tmp_path, "ian")
    pac.report("ian", _load(tmp_path, "ian"))
    out = capsys.readouterr().out
    assert "4 of 5 members survive to the free end" in out
    assert "cBottle-video N=5" in out
    assert "cBottle-video - ERA5 round trip" in out and "+0.0700" in out
    assert "WARNING" not in out


def test_figure_layout_axes_and_legend(tmp_path):
    for case in ("ian", "pnw_heatwave"):
        _write(tmp_path, case)
    fig = pac.draw({c: _load(tmp_path, c) for c in ("ian", "pnw_heatwave")})
    axes = np.array(fig.axes).reshape(len(pac.ROWS), 2)
    assert [a.get_title() for a in axes[0]] == [
        dl.load_case("ian")["display_name"], dl.load_case("pnw_heatwave")["display_name"]]
    # The original paper's pinned ranges (axis_limits.yaml, ageostrophic/uvz500).
    assert axes[0, 0].get_ylim() == pytest.approx((0.10, 0.32))
    assert axes[1, 1].get_ylim() == pytest.approx((14.0, 20.0))
    labels = [t.get_text() for t in axes[0, 0].get_legend().get_texts()]
    assert labels == ["Rewind-AR", "cBottle-video", "ERA5",
                      "ERA5 (HEALPix round trip)", "Conditioned frame"]
    # Member 0's dead frames are not drawn: its line starts at frame 2.
    member0 = axes[0, 0].get_lines()[0].get_ydata()
    assert np.isnan(member0[:2]).all() and np.isfinite(member0[2:]).all()


def test_main_writes_the_figure(tmp_path, capsys):
    for case in ("ian", "sandy"):
        _write(tmp_path, case)
    pac.main(["--case", "ian", "--case", "sandy", "--out-root", str(tmp_path)])
    assert (tmp_path / "ageostrophic_compare_L500.png").stat().st_size > 0
    assert capsys.readouterr().out.count("survive to the free end") == 2


def test_npz_without_survivor_mask_is_refused(tmp_path):
    p = tmp_path / "old.npz"
    np.savez(p, **{"makani::ratio_ens": np.zeros((2, 3))})
    with pytest.raises(SystemExit, match="no survivor mask"):
        pac.load(p)


def test_cases_on_different_bands_are_refused(tmp_path):
    _write(tmp_path, "ian")
    _write(tmp_path, "sandy", band="20-70 |lat|")
    with pytest.raises(SystemExit, match="band_label"):
        pac.draw({c: _load(tmp_path, c) for c in ("ian", "sandy")})
