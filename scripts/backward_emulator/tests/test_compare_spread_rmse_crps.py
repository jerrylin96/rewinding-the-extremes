"""End-to-end test of the makani-side spread/RMSE/CRPS driver.

Builds a synthetic case directory on a coarse-but-real lat/lon grid, resolves the
region from the actual ``pnw_heatwave`` cBottle case YAML, runs the driver, and
checks the npz/figure are written with the expected curves and that SSR is
spread/skill.  Exercises the adapter -> reducer -> save/plot path as one unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_spread_rmse_crps as drv  # noqa: E402

_CH = ["t2m", "z500", "msl", "u500", "v500", "z300", "u10m", "v10m"]


def _write_case(dirpath: Path, *, n_files=2, members_per_file=2, L=12,
                H=19, W=36, anchor_ts=b"2021-06-28T12"):
    dirpath.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(90, -90, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    rng = np.random.default_rng(0)
    base = {"z500": 52000.0, "t2m": 290.0, "msl": 101000.0, "z300": 90000.0}
    gid = 0
    for fi in range(n_files):
        M = members_per_file
        truth = np.empty((1, L, len(_CH), H, W), np.float32)
        for ci, c in enumerate(_CH):
            truth[0, :, ci] = base.get(c, 5.0)
        pred = np.empty((1, M, L, len(_CH), H, W), np.float32)
        for j in range(M):
            for ci, c in enumerate(_CH):
                b = base.get(c, 5.0)
                # growing spread away from the pinned (present) frame l=0. t2m gets
                # a tiny amplitude so it stays inside [150,350] K -- the default
                # survivor censoring is then a clean no-op on this fixture (real
                # censoring is exercised in test_survival.py).
                scale = 0.02 if c == "t2m" else 1.0
                for l in range(L):
                    amp = 0.0 if l == 0 else (l * 2.0 * scale)
                    pred[0, j, l, ci] = b + rng.standard_normal((H, W)) * amp
            pred[0, j, 0] = truth[0, 0]  # IC duplication
            gid += 1
        p = dirpath / f"backward_fields_{fi:03d}.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("truth", data=truth)
            f.create_dataset("pred", data=pred)
            f.create_dataset("channel",
                             data=np.array([c.encode() for c in _CH], dtype="S64"))
            f.create_dataset("lead_hours_into_past",
                             data=np.arange(L, dtype=np.int64) * 6)
            f.create_dataset("lat", data=lat)
            f.create_dataset("lon", data=lon)
            f.create_dataset("timestamps",
                             data=np.array([[anchor_ts] * L], dtype="S64"))
            f.create_dataset("anchor_index", data=np.array([0]))
            f.create_dataset("member_index", data=np.arange(M))
            f.attrs["step_hours"] = -6
            f.attrs["direction"] = "backward"


def test_driver_writes_npz_and_figure(tmp_path: Path):
    case_dir = tmp_path / "pnw"
    _write_case(case_dir)
    out = tmp_path / "out"

    drv.main([
        "--makani-dir", str(case_dir),
        "--case", "pnw_heatwave",
        "--variable", "z500",
        "--region-name", "synoptic",
        "--out-dir", str(out),
        "--no-cbottle",
    ])

    npz = out / "spread_rmse_crps_z500_synoptic.npz"
    png = out / "spread_rmse_crps_z500_synoptic.png"
    assert npz.exists() and png.exists()

    d = np.load(npz, allow_pickle=True)
    label = "Autoregressive (makani)"
    for key in ("spread", "skill", "member_rmse", "crps", "ssr", "lead_hours"):
        assert f"{label}::{key}" in d, key
    spread = d[f"{label}::spread"]
    skill = d[f"{label}::skill"]
    ssr = d[f"{label}::ssr"]
    assert spread.shape == (12,)
    # SSR is spread/skill where skill > 0.
    good = skill > 0
    np.testing.assert_allclose(ssr[good], spread[good] / skill[good], rtol=1e-6)
    # Pinned (last, forward) frame has ~zero spread by construction.
    assert spread[-1] < spread[:-1].mean()
    # Region actually selected a subset of the 19x36 grid.
    assert 0 < int(d[f"{label}::n_pix"]) < 19 * 36


def test_domain_label_degree_direction_form():
    # PNW synoptic box: negative lon = W, positive lat = N.
    assert drv._domain_label((-170.0, -100.0, 25.0, 70.0)) == (
        "170\N{DEGREE SIGN}W\N{EN DASH}100\N{DEGREE SIGN}W, "
        "25\N{DEGREE SIGN}N\N{EN DASH}70\N{DEGREE SIGN}N"
    )
    # Equator/antimeridian carry no hemisphere letter; None = global.
    assert drv._domain_label((0.0, 180.0, -10.0, 0.0)) == (
        "0\N{DEGREE SIGN}\N{EN DASH}180\N{DEGREE SIGN}, "
        "10\N{DEGREE SIGN}S\N{EN DASH}0\N{DEGREE SIGN}"
    )
    assert drv._domain_label(None) == "global"


def test_survivors_panel_draws_uncensored_flat_line(tmp_path: Path, monkeypatch):
    """A censored method plus an uncensored one with a recorded (flat)
    survival curve must both land in the survivors panel -- the flat line
    dashed -- and the suptitle must carry the exact domain box."""
    from matplotlib.figure import Figure

    captured = {}
    orig_savefig = Figure.savefig

    def grab(self, *a, **k):
        # Inspect before _plot closes the figure.
        captured["n_axes"] = len(self.axes)
        captured["suptitle"] = self._suptitle.get_text()
        surv_axes = [ax for ax in self.axes
                     if ax.get_ylabel() == "Survivors (%)"]
        captured["surv_styles"] = sorted(
            ln.get_linestyle() for ln in surv_axes[0].get_lines()
        ) if surv_axes else None
        return orig_savefig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", grab)

    lead = np.arange(12, dtype=float) * 6
    common = {
        "spread": np.ones(12), "skill": np.ones(12),
        "member_rmse": np.ones(12), "mean_error": np.zeros(12),
        "crps": np.ones(12), "ssr": np.ones(12), "lead_hours": lead,
    }
    results = {
        "Autoregressive (makani)": {
            **common, "censored": True,
            "survival": np.linspace(0.8, 1.0, 12),
        },
        "Video diffusion (cBottle)": {
            **common, "censored": False, "survival": np.ones(12),
        },
    }
    png = tmp_path / "fig.png"
    drv._plot(png, results, variable="z500",
              region=(-170.0, -100.0, 25.0, 70.0),
              case_meta={"display_name": "PNW", "timezone": "UTC",
                         "start_time": "2021-06-25T18:00:00"})
    assert png.exists()

    assert captured["n_axes"] == 6  # SSR, error, 3-panel triptych, survivors
    # Solid censored curve + dashed by-construction flat line.
    assert captured["surv_styles"] == ["-", "--"]
    assert "Domain: " + drv._domain_label((-170.0, -100.0, 25.0, 70.0)) \
        in captured["suptitle"]
    assert "synoptic" not in captured["suptitle"]


def test_survival_cache_roundtrip(tmp_path: Path):
    """The survivor mask is computed once, cached, then loaded (no rescan)."""
    from makani_source import MakaniBackwardSource  # noqa: E402

    case_dir = tmp_path / "pnw"
    _write_case(case_dir)
    src = MakaniBackwardSource(str(case_dir), n_frames=12)
    cache = tmp_path / "surv" / "survival_pnw.npz"

    # First call computes and populates the cache.
    s1 = drv._load_or_compute_survival(src, cache_path=cache,
                                       makani_analysis_dir=None)
    assert cache.is_file()
    assert s1["alive"].shape == (src.n_members, src.n_frames)
    assert not s1.get("cached", False)

    # Second call loads it (interchangeable schema), identical mask, no recompute.
    s2 = drv._load_or_compute_survival(src, cache_path=cache,
                                       makani_analysis_dir=None)
    assert s2.get("cached") is True
    np.testing.assert_array_equal(s1["alive"], s2["alive"])
    np.testing.assert_allclose(s1["survival"], s2["survival"])
    assert s2["n_diverged"] == s1["n_diverged"]


def test_survival_cache_shape_mismatch_recomputes(tmp_path: Path):
    """A stale cache (wrong N/L) is ignored and the mask recomputed."""
    from makani_source import MakaniBackwardSource  # noqa: E402

    case_dir = tmp_path / "pnw"
    _write_case(case_dir)
    src = MakaniBackwardSource(str(case_dir), n_frames=12)
    cache = tmp_path / "bad.npz"
    np.savez(cache, alive=np.ones((3, 5), dtype=bool))  # wrong shape
    s = drv._load_or_compute_survival(src, cache_path=cache,
                                      makani_analysis_dir=None)
    assert s["alive"].shape == (src.n_members, src.n_frames)
    assert not s.get("cached", False)


def test_land_mask_restricts_pixels_and_names_output(tmp_path: Path):
    """--land-mask intersects with the region, shrinks n_pix, and tags the stem."""
    case_dir = tmp_path / "pnw"
    _write_case(case_dir, H=19, W=36)
    out = tmp_path / "out"
    common = ["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
              "--variable", "t2m", "--region-name", "synoptic",
              "--no-cbottle", "--no-censor"]

    # Unmasked reference.
    drv.main(common + ["--out-dir", str(out / "full")])
    d_full = np.load(out / "full" / "spread_rmse_crps_t2m_synoptic.npz",
                     allow_pickle=True)
    n_full = int(d_full["Autoregressive (makani)::n_pix"])

    # Land mask that keeps only the northern half of the grid.
    lat = np.linspace(90, -90, 19)
    mask = np.broadcast_to((lat > 40)[:, None], (19, 36)).copy()
    mpath = tmp_path / "land_19x36.npy"
    np.save(mpath, mask)
    drv.main(common + ["--out-dir", str(out / "land"),
                       "--land-mask", str(mpath)])

    npz = out / "land" / "spread_rmse_crps_t2m_synoptic_land.npz"
    assert npz.exists(), "masked run should tag the stem with _land"
    d_land = np.load(npz, allow_pickle=True)
    n_land = int(d_land["Autoregressive (makani)::n_pix"])
    assert 0 < n_land < n_full


def test_land_mask_shape_mismatch_errors(tmp_path: Path):
    """A mask off the source grid is rejected with a clear error."""
    import pytest

    case_dir = tmp_path / "pnw"
    _write_case(case_dir, H=19, W=36)
    mpath = tmp_path / "bad_mask.npy"
    np.save(mpath, np.ones((10, 10), dtype=bool))
    with pytest.raises(SystemExit):
        drv.main(["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
                  "--variable", "t2m", "--region-name", "synoptic",
                  "--no-cbottle", "--no-censor", "--out-dir", str(tmp_path / "o"),
                  "--land-mask", str(mpath)])


def test_global_region_uses_all_pixels(tmp_path: Path):
    case_dir = tmp_path / "pnw"
    _write_case(case_dir, H=10, W=12)
    out = tmp_path / "out"
    drv.main([
        "--makani-dir", str(case_dir), "--case", "pnw_heatwave",
        "--variable", "msl", "--region-name", "global", "--out-dir", str(out),
        "--no-cbottle",
    ])
    d = np.load(out / "spread_rmse_crps_msl_global.npz", allow_pickle=True)
    assert int(d["Autoregressive (makani)::n_pix"]) == 10 * 12


def test_shifted_dump_is_refused_before_anything_is_written(tmp_path: Path):
    """A dump anchored on the wrong date must fail loudly, not produce plausible
    curves labelled with the case's dates: the join is positional, so this check is
    the only thing tying a dump to the case it is compared under."""
    import pytest

    case_dir = tmp_path / "pnw_shifted"
    _write_case(case_dir, anchor_ts=b"2021-06-29T12")  # one day late
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="window end"):
        drv.main(["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
                  "--variable", "z500", "--region-name", "synoptic",
                  "--no-cbottle", "--no-censor", "--out-dir", str(out)])
    assert not out.exists() or not list(out.glob("*.npz"))

    # The explicit opt-out still works, and an explicit expected anchor can override the case.
    drv.main(["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
              "--variable", "z500", "--region-name", "synoptic",
              "--no-cbottle", "--no-censor", "--out-dir", str(out),
              "--expected-anchor", "none"])
    assert (out / "spread_rmse_crps_z500_synoptic.npz").exists()
    drv.main(["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
              "--variable", "z500", "--region-name", "synoptic",
              "--no-cbottle", "--no-censor", "--out-dir", str(tmp_path / "out2"),
              "--expected-anchor", "2021-06-29T12"])


def test_cbottle_failure_is_loud_and_writes_nothing(tmp_path: Path, monkeypatch):
    """With the overlay requested (the default), a failing cBottle side must raise
    and leave no makani-only output behind; --no-cbottle is the only makani-only path."""
    import builtins
    import pytest

    case_dir = tmp_path / "pnw"
    _write_case(case_dir)
    out = tmp_path / "out"

    real_import = builtins.__import__

    def no_cbottle(name, *a, **k):
        if name == "cbottle_source":
            raise ImportError("heavy env missing (simulated)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cbottle)
    with pytest.raises(ImportError, match="simulated"):
        drv.main(["--makani-dir", str(case_dir), "--case", "pnw_heatwave",
                  "--variable", "z500", "--region-name", "synoptic",
                  "--no-censor", "--out-dir", str(out)])
    assert not out.exists() or not list(out.glob("*"))
