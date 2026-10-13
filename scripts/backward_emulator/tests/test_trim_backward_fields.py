"""Unit tests for the source-side field-dump trimmer.

Builds tiny synthetic dumps in the exact ``backward_fields.h5`` schema (both the
current 6-D ``pred`` with a member axis and a pre-Aug-2026 5-D dump without one),
trims them, and checks the output is a faithful channel subset: right channels in
the dump's own order, right shapes, values preserved bit-for-bit for kept
channels, the IC-duplication property (``pred[..., 0] == truth[..., 0]``) intact,
and every non-channel dataset + attribute passed through.

Dependency-light on purpose (numpy + h5py), matching the script, so it runs in the
makani container where the real trimming happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trim_backward_fields import (  # noqa: E402
    COMPARISON_CHANNELS,
    trim_dump,
)

# A superset dump whose channel order deliberately interleaves manifest channels
# with extras, so the test proves the trimmer keeps dump order and drops extras.
_DUMP_CHANNELS = ["u10m", "v10m", "t2m", "msl", "tcwv", "u500", "v500", "z300", "z500"]
_KEEP = ["t2m", "z500", "msl", "u500", "v500", "z300", "u10m", "v10m"]


def _write_dump(path: Path, *, with_member_axis: bool, n_anchors=2, n_members=3,
                n_leads=4, H=5, W=6) -> np.ndarray:
    """Write a synthetic dump; return the full pred array for later comparison."""
    C = len(_DUMP_CHANNELS)
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((n_anchors, n_leads, C, H, W)).astype(np.float32)
    if with_member_axis:
        pred = rng.standard_normal(
            (n_anchors, n_members, n_leads, C, H, W)
        ).astype(np.float32)
        # IC duplication: frame 0 of every member equals truth frame 0.
        pred[:, :, 0] = truth[:, None, 0]
    else:
        pred = rng.standard_normal((n_anchors, n_leads, C, H, W)).astype(np.float32)
        pred[:, 0] = truth[:, 0]

    with h5py.File(path, "w") as f:
        f.create_dataset("truth", data=truth)
        f.create_dataset("pred", data=pred)
        f.create_dataset(
            "channel", data=np.array([c.encode() for c in _DUMP_CHANNELS], dtype="S64")
        )
        f.create_dataset(
            "lead_hours_into_past", data=np.arange(n_leads, dtype=np.int64) * 6
        )
        f.create_dataset("lat", data=np.linspace(90, -90, H))
        f.create_dataset("lon", data=np.linspace(0, 360, W, endpoint=False))
        ts = np.array(
            [[f"2021-06-{28 - l:02d}T12".encode() for l in range(n_leads)]
             for _ in range(n_anchors)],
            dtype="S64",
        )
        f.create_dataset("timestamps", data=ts)
        f.create_dataset("anchor_index", data=np.arange(n_anchors, dtype=np.int64))
        if with_member_axis:
            f.create_dataset(
                "member_index", data=np.arange(n_members, dtype=np.int64)
            )
        f.attrs["step_hours"] = -6
        f.attrs["direction"] = "backward"
        f.attrs["run_num"] = "fcn3_ens_10yr"
    return pred


@pytest.mark.parametrize("with_member_axis", [True, False])
def test_trim_preserves_schema_and_values(tmp_path: Path, with_member_axis: bool):
    src = tmp_path / "backward_fields.h5"
    dst = tmp_path / "out" / "backward_fields.h5"
    pred_full = _write_dump(src, with_member_axis=with_member_axis)

    trim_dump(src, dst, _KEEP, require_all=True)

    with h5py.File(src, "r") as fin, h5py.File(dst, "r") as fout:
        src_channels = [c.decode() for c in fin["channel"][...]]
        out_channels = [c.decode() for c in fout["channel"][...]]

        # Kept channels are exactly the manifest, in the DUMP's order.
        assert out_channels == [c for c in src_channels if c in set(_KEEP)]
        assert set(out_channels) == set(_KEEP)
        keep_idx = [src_channels.index(c) for c in out_channels]

        # Shapes: channel axis shrinks, everything else identical.
        assert fout["truth"].shape == (
            fin["truth"].shape[0], fin["truth"].shape[1],
            len(_KEEP), fin["truth"].shape[-2], fin["truth"].shape[-1],
        )
        assert fout["pred"].ndim == fin["pred"].ndim

        # Values preserved bit-for-bit for kept channels.
        np.testing.assert_array_equal(
            fout["truth"][...], fin["truth"][...][:, :, keep_idx]
        )
        if with_member_axis:
            np.testing.assert_array_equal(
                fout["pred"][...], pred_full[:, :, :, keep_idx]
            )
            # IC-duplication property survives the trim.  Broadcast the
            # (A, 1, C, H, W) truth frame explicitly to the (A, M, C, H, W) pred
            # frame -- numpy 2.x assert_array_equal no longer auto-broadcasts.
            p0 = fout["pred"][:, :, 0]
            truth0 = fout["truth"][...][:, None, 0]
            np.testing.assert_array_equal(p0, np.broadcast_to(truth0, p0.shape))
        else:
            np.testing.assert_array_equal(
                fout["pred"][...], pred_full[:, :, keep_idx]
            )
            np.testing.assert_array_equal(fout["pred"][:, 0], fout["truth"][:, 0])

        # Non-channel datasets passed through verbatim.
        for name in ("lead_hours_into_past", "lat", "lon", "timestamps",
                     "anchor_index"):
            np.testing.assert_array_equal(fout[name][...], fin[name][...])
        if with_member_axis:
            np.testing.assert_array_equal(
                fout["member_index"][...], fin["member_index"][...]
            )

        # Attributes preserved, plus the trim breadcrumb.
        assert fout.attrs["step_hours"] == -6
        assert fout.attrs["direction"] == "backward"
        assert fout.attrs["run_num"] == "fcn3_ens_10yr"
        assert [c.decode() for c in fout.attrs["trimmed_channels"]] == out_channels


def test_require_all_raises_on_missing(tmp_path: Path):
    src = tmp_path / "backward_fields.h5"
    _write_dump(src, with_member_axis=True)
    with pytest.raises(SystemExit):
        trim_dump(src, tmp_path / "o.h5", ["t2m", "nonexistent"], require_all=True)


def test_missing_channel_warns_but_keeps_present(tmp_path: Path):
    src = tmp_path / "backward_fields.h5"
    _write_dump(src, with_member_axis=True)
    dst = tmp_path / "o.h5"
    trim_dump(src, dst, ["t2m", "z500", "nonexistent"], require_all=False)
    with h5py.File(dst, "r") as f:
        assert [c.decode() for c in f["channel"][...]] == ["t2m", "z500"]


def test_default_manifest_is_the_eight_channels():
    assert COMPARISON_CHANNELS == [
        "t2m", "z500", "msl", "u500", "v500", "z300", "u10m", "v10m"
    ]


def test_overwrite_guard(tmp_path: Path):
    src = tmp_path / "backward_fields.h5"
    _write_dump(src, with_member_axis=True)
    dst = tmp_path / "o.h5"
    trim_dump(src, dst, _KEEP)
    with pytest.raises(SystemExit):
        trim_dump(src, dst, _KEEP)  # exists, no --overwrite
    trim_dump(src, dst, _KEEP, overwrite=True)  # ok
