"""Unit tests for the makani->lat/lon pooling adapter.

Builds a few synthetic dumps (mirroring the ``backward_fields.h5`` schema) whose
values *encode* their global member id and makani lead index, so the pooled,
reversed, truncated output can be checked exactly:

* members pool across files on axis 0 in sorted-file order,
* the lead axis is truncated to the 66 h window and flipped to cBottle's forward
  order (forward frame f == makani frame n_frames-1-f),
* the pinned frame (makani frame 0, ``pred == truth``) lands at forward frame -1,
* grid / channel / anchor consistency across files is enforced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from makani_source import (  # noqa: E402
    DEFAULT_N_FRAMES,
    MakaniBackwardSource,
    find_field_dumps,
)
from latlon_metrics import (  # noqa: E402
    select_pixels,
    spread_rmse_crps_from_source,
    spread_rmse_crps_latlon,
)

_CHANNELS = ["t2m", "z500", "msl", "u500", "v500", "z300", "u10m", "v10m"]
_H, _W = 4, 5
_L = 14  # more leads than the 12-frame window, to test truncation


def _write_dump(path: Path, *, member_ids, n_leads=_L, n_anchors=1,
                anchor_ts=("2021-06-28T12",), channels=_CHANNELS):
    """Write a dump where truth[l] == l everywhere and pred[g, l>=1] == g.

    pred[:, 0] == truth[0] (== 0) encodes the IC-duplication property.
    """
    M, C = len(member_ids), len(channels)
    truth = np.empty((n_anchors, n_leads, C, _H, _W), np.float32)
    for l in range(n_leads):
        truth[:, l] = float(l)
    pred = np.empty((n_anchors, M, n_leads, C, _H, _W), np.float32)
    for j, g in enumerate(member_ids):
        for l in range(n_leads):
            pred[:, j, l] = 0.0 if l == 0 else float(g)  # frame 0 == truth[0]

    with h5py.File(path, "w") as f:
        f.create_dataset("truth", data=truth)
        f.create_dataset("pred", data=pred)
        f.create_dataset(
            "channel", data=np.array([c.encode() for c in channels], dtype="S64")
        )
        f.create_dataset(
            "lead_hours_into_past", data=np.arange(n_leads, dtype=np.int64) * 6
        )
        f.create_dataset("lat", data=np.linspace(90, -90, _H))
        f.create_dataset("lon", data=np.linspace(0, 360, _W, endpoint=False))
        ts = np.array(
            [[anchor_ts[a].encode()] * n_leads for a in range(n_anchors)], dtype="S64"
        )
        f.create_dataset("timestamps", data=ts)
        f.create_dataset("anchor_index", data=np.arange(n_anchors, dtype=np.int64))
        f.create_dataset("member_index", data=np.arange(M, dtype=np.int64))
        f.attrs["step_hours"] = -6
        f.attrs["direction"] = "backward"


def _three_file_case(tmp_path: Path) -> Path:
    d = tmp_path / "case"
    d.mkdir()
    # sorted filenames -> global member order 0..11
    _write_dump(d / "backward_fields_000.h5", member_ids=[0, 1, 2, 3])
    _write_dump(d / "backward_fields_001.h5", member_ids=[4, 5, 6, 7])
    _write_dump(d / "backward_fields_002.h5", member_ids=[8, 9, 10, 11])
    # a spectrum file that must be ignored by the globber
    (d / "backward_spectrum_000.h5").write_bytes(b"not an h5 we read")
    return d


def test_iter_member_fields_full_grid_forward(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    names = ["z500", "u500", "v500"]
    seen = {}
    for gid, fields in src.iter_member_fields(names):
        assert set(fields) == set(names)
        for n in names:
            assert fields[n].shape == (DEFAULT_N_FRAMES, _H, _W)
        # forward frame 0 == makani frame 11 == member id g; forward -1 == truth[0] == 0.
        assert np.all(fields["z500"][0] == gid)
        assert np.all(fields["z500"][-1] == 0.0)
        seen[gid] = True
    assert sorted(seen) == list(range(12))  # 3 files x 4 members, pooled


def test_truth_fields_native_baseline(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    tf = src.truth_fields(["z500", "u500"])
    assert tf["z500"].shape == (DEFAULT_N_FRAMES, _H, _W)
    # truth[l] == l; forward frame f == makani frame (nf-1-f), so forward 0 -> 11, -1 -> 0.
    assert np.all(tf["z500"][0] == DEFAULT_N_FRAMES - 1)
    assert np.all(tf["z500"][-1] == 0.0)


def test_find_field_dumps_skips_spectrum(tmp_path: Path):
    d = _three_file_case(tmp_path)
    files = find_field_dumps(d)
    assert [p.name for p in files] == [
        "backward_fields_000.h5", "backward_fields_001.h5", "backward_fields_002.h5"
    ]


def test_pooling_and_forward_alignment(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)

    assert src.n_members == 12
    assert src.n_frames == DEFAULT_N_FRAMES == 12
    assert src.channels == _CHANNELS
    assert src.summary()["grid"] == (_H, _W)
    np.testing.assert_array_equal(src.lead_hours, np.arange(12) * 6)  # [0..66]
    assert src.summary()["default_channels_missing"] == []

    ens, truth = src.load_variable("z500")
    assert ens.shape == (12, 12, _H, _W)
    assert truth.shape == (12, _H, _W)

    # truth[l] == l, reversed: forward frame f == makani frame 11-f == value 11-f.
    for f in range(12):
        assert np.all(truth[f] == float(11 - f)), f"truth forward frame {f}"

    # Members pooled in sorted-file order: global id g at every non-IC frame.
    # Forward frames 0..10 map to makani leads 11..1 (non-IC); frame 11 is the IC.
    for g in range(12):
        for f in range(11):
            assert np.all(ens[g, f] == float(g)), f"member {g} forward frame {f}"
        assert np.all(ens[g, 11] == 0.0)  # IC frame == truth[0]

    # Pinned frame: forward frame -1 (end/anchor) == truth across members.
    src.assert_ic_pinned("z500")


def test_truncation_uses_first_window(tmp_path: Path):
    # With n_frames=3 we keep makani leads 0,1,2 and flip -> truth 2,1,0.
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0, 1])
    src = MakaniBackwardSource(d, n_frames=3)
    _, truth = src.load_variable("t2m")
    assert truth.shape == (3, _H, _W)
    assert [float(truth[f].flat[0]) for f in range(3)] == [2.0, 1.0, 0.0]


def test_channel_and_grid_mismatch_raise(tmp_path: Path):
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0, 1])
    _write_dump(d / "backward_fields_001.h5", member_ids=[2, 3],
                channels=_CHANNELS[:-1] + ["tcwv"])  # different channel set
    with pytest.raises(ValueError, match="channels"):
        MakaniBackwardSource(d)


def test_anchor_timestamp_mismatch_raises(tmp_path: Path):
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0], anchor_ts=("2021-06-28T12",))
    _write_dump(d / "backward_fields_001.h5", member_ids=[1], anchor_ts=("2022-09-30T18",))
    with pytest.raises(ValueError, match="not the same case"):
        MakaniBackwardSource(d)


def test_too_few_leads_raises(tmp_path: Path):
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0], n_leads=8)
    with pytest.raises(ValueError, match="need the full 66 h window"):
        MakaniBackwardSource(d, n_frames=12)


_REGION = (0.0, 180.0, -50.0, 50.0)  # a box that selects a subset of the grid


def test_load_region_matches_full_subset(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    full_ens, full_truth = src.load_variable("z500")  # small grid -> under guard
    H, W = src.summary()["grid"]
    flat, _ = select_pixels(src.lat, src.lon, _REGION)
    ens_sel, truth_sel = src.load_region("z500", flat)
    np.testing.assert_array_equal(
        ens_sel, full_ens.reshape(src.n_members, src.n_frames, H * W)[:, :, flat]
    )
    np.testing.assert_array_equal(
        truth_sel, full_truth.reshape(src.n_frames, H * W)[:, flat]
    )


def test_from_source_reduction_matches_full(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    full_ens, full_truth = src.load_variable("z500")
    r_full = spread_rmse_crps_latlon(full_ens, full_truth, src.lat, src.lon,
                                     region=_REGION)
    r_src = spread_rmse_crps_from_source(src, "z500", region=_REGION)
    for k in ("spread", "skill", "member_rmse", "mean_error", "crps"):
        np.testing.assert_allclose(r_full[k], r_src[k], rtol=1e-10, atol=1e-12)
    assert r_full["n_pix"] == r_src["n_pix"]


def test_load_variable_memory_guard(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    src._FULL_LOAD_MAX_BYTES = 8  # force the guard on the tiny fixture
    with pytest.raises(MemoryError, match="load_region"):
        src.load_variable("z500")
    # load_region still works regardless of the full-load guard.
    flat, _ = select_pixels(src.lat, src.lon, _REGION)
    ens_sel, _t = src.load_region("z500", flat)
    assert ens_sel.shape[0] == src.n_members


def test_sampling_helpers_are_cheap_and_correct(tmp_path: Path):
    d = _three_file_case(tmp_path)
    src = MakaniBackwardSource(d)
    truth = src.sample_truth_field("z500")       # (nf, H, W)
    members = src.sample_member_window("z500")   # (M, nf, H, W) one file
    assert truth.shape[0] == src.n_frames
    assert members.shape[0] == 4  # first file has 4 members
    # forward truth frame f == value 11-f (see fixture); pinned frame is last.
    assert np.all(truth[-1] == 0.0)


def test_expected_anchor_selection(tmp_path: Path):
    # A dump with two anchors; select the second by its ISO prefix.
    d = tmp_path / "c"
    d.mkdir()
    p = d / "backward_fields_000.h5"
    _write_dump(p, member_ids=[0, 1], n_anchors=2,
                anchor_ts=("2021-06-28T12", "2022-09-30T18"))
    src = MakaniBackwardSource(d, expected_anchor_iso="2022-09-30T18")
    assert src.anchor_timestamp.startswith("2022-09-30T18")


def test_expected_anchor_mismatch_raises(tmp_path: Path):
    """The case-derived anchor must match the dump's valid-time, else the dump is refused."""
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0], anchor_ts=("2021-06-29T12",))
    with pytest.raises(ValueError, match="window end"):
        MakaniBackwardSource(d, expected_anchor_iso="2021-06-28T12")
    MakaniBackwardSource(d, expected_anchor_iso="2021-06-29T12")  # the right date passes


def test_expected_anchor_accepts_makani_space_separated_timestamps(tmp_path: Path):
    """makani writes str(datetime) with a space; the ISO prefix still matches."""
    d = tmp_path / "c"
    d.mkdir()
    _write_dump(d / "backward_fields_000.h5", member_ids=[0],
                anchor_ts=("2021-06-28 12:00:00+00:00",))
    src = MakaniBackwardSource(d, expected_anchor_iso="2021-06-28T12")
    assert src.anchor_timestamp.startswith("2021-06-28")


def test_expected_anchor_from_case_is_the_window_end():
    from makani_source import expected_anchor_iso_from_case, resolve_expected_anchor

    pnw = {"ensemble": {"start_time": {"year": 2021, "month": 6, "day": 25, "hour": 18}}}
    ian = {"ensemble": {"start_time": {"year": 2022, "month": 9, "day": 28, "hour": 0}}}
    assert expected_anchor_iso_from_case(pnw) == "2021-06-28T12"   # 06-25 18Z + 66 h
    assert expected_anchor_iso_from_case(ian) == "2022-09-30T18"   # 09-28 00Z + 66 h
    assert resolve_expected_anchor(pnw) == "2021-06-28T12"
    assert resolve_expected_anchor(pnw, "none") is None
    assert resolve_expected_anchor(pnw, "2022-09-30 18") == "2022-09-30T18"
    assert resolve_expected_anchor(None) is None
