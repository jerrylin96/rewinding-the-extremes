"""Channel-trim a makani backward-rollout field dump before transfer.

This is a **source-side** tool: it runs where the makani rollouts were produced
(the emulator HPC), *before* the Globus transfer to the analysis HPC, so only the
channels the side-by-side comparison actually needs cross the wire.  It is
deliberately dependency-light -- ``numpy`` + ``h5py`` only, no torch, no
earth2studio -- so it can be copied next to the raw dumps and run in the makani
container (mirroring the torch-free convention of ``analysis/backward_fields_io.py``
in the makani repo).

It reads each ``*fields*.h5`` dump written by
``backward_rollout.py --save_fields_anchors`` and rewrites a byte-for-byte
schema-compatible copy carrying only the requested channels.  Everything else --
``truth`` / ``pred`` layout, the member axis, ``lead_hours_into_past``,
``lat`` / ``lon``, ``timestamps``, ``anchor_index``, ``member_index`` and every
file attribute -- is preserved exactly, so both the makani analysis tools and the
earth2studio adapter read the trimmed file with no changes.

Why trim: a full 32-channel x ~1000-member x 12-lead x 721x1440 float32 dump is
~1.5 TB; the comparison needs 8 channels, a ~4x reduction that pays for itself in
transfer time and in analysis I/O.

The default keep-list is :data:`COMPARISON_CHANNELS` -- the union across the six
diagnostics in the earth2studio comparison (spread/RMSE/CRPS, free-end states,
synoptic-PCA, ageostrophic at 500 hPa, power spectra, and the Ian TC tracker).
Keep this list the single source of truth: the adapter imports it from here so the
trimmed dump and the analysis can never disagree about which channels exist.

Usage
-----
    # trim every *fields*.h5 under a run's scores dir into ./trimmed/
    python trim_backward_fields.py /path/to/scores --out-dir ./trimmed

    # a single file, custom channel list, fail if any is missing
    python trim_backward_fields.py dump.h5 --out-dir ./trimmed \\
        --channels t2m,z500,msl --require-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np

# Union of the channels the earth2studio side-by-side comparison consumes.
# Derived from the diagnostics themselves (do not pad casually -- every extra
# channel is member x lead x 721 x 1440 float32 across ~1000 members):
#   t2m           spread/RMSE/CRPS, free-end states, synoptic-PCA impact (PNW)
#   z500          spread/RMSE/CRPS, synoptic-PCA cause, ageostrophic, TC warm-core
#   msl           spread/RMSE/CRPS, free-end states, TC detection
#   u500, v500    ageostrophic wind at the default 500 hPa level
#   z300          TC warm-core thickness _DIFF(z300, z500) (Ian)
#   u10m, v10m    TC intensity _VECMAG(u10m, v10m) + stitch threshold (Ian)
# Every one of these exists in cBottle's STORED_VARIABLES too, so each is
# comparable on both sides.  (cBottle-only `sst` is not here: makani has no SST
# channel, and the PNW land mask is sourced from ERA5, not from the dump.)
COMPARISON_CHANNELS: List[str] = [
    "t2m",
    "z500",
    "msl",
    "u500",
    "v500",
    "z300",
    "u10m",
    "v10m",
]

# Copied datasets that do NOT carry the channel axis -- passed through verbatim.
_PASSTHROUGH_DATASETS = (
    "lead_hours_into_past",
    "lat",
    "lon",
    "timestamps",
    "anchor_index",
    "member_index",
)


def _decode_channels(raw: Sequence) -> List[str]:
    """Decode the ``channel`` dataset (``S64`` bytes) to a list of str."""
    return [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]


def _resolve_keep_indices(
    available: Sequence[str], requested: Sequence[str], *, require_all: bool
) -> List[int]:
    """Map requested channel names to indices in the dump's channel order.

    Preserves the dump's own channel order (not the requested order) so the
    output stays a straightforward slab subset of the input.  Missing channels
    warn by default; ``require_all`` turns that into a hard error so a silently
    incomplete trim can't reach the analysis HPC.
    """
    req = set(requested)
    missing = [c for c in requested if c not in available]
    if missing:
        msg = (
            f"requested channels not in dump: {missing}; "
            f"available: {list(available)}"
        )
        if require_all:
            raise SystemExit(f"ERROR: {msg}")
        print(f"  WARNING: {msg}", file=sys.stderr)
    keep = [i for i, c in enumerate(available) if c in req]
    if not keep:
        raise SystemExit(
            "ERROR: none of the requested channels are present in the dump; "
            "nothing to write."
        )
    return keep


def _iter_field_dumps(inputs: Sequence[str]) -> List[Path]:
    """Expand file / directory inputs to a sorted list of ``*fields*.h5`` paths.

    A directory is globbed for ``*fields*.h5`` (the field dumps), explicitly
    skipping ``*spectrum*.h5`` and any other ``.h5`` the bundle may carry.  An
    explicit file path is taken as-is (so an oddly-named dump can still be
    trimmed).
    """
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(f for f in p.glob("*fields*.h5")))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"ERROR: input path does not exist: {p}")
    if not out:
        raise SystemExit(
            f"ERROR: no *fields*.h5 dumps found under {list(inputs)}. "
            "(Directories are globbed for '*fields*.h5'; pass a file path "
            "explicitly if your dump is named differently.)"
        )
    return out


def trim_dump(src_path: Path, dst_path: Path, keep_channels: Sequence[str], *,
              require_all: bool = False, overwrite: bool = False) -> None:
    """Write a channel-subset copy of one field dump, preserving the schema.

    Copies slab-by-slab (per anchor, and per member for the 6-D ``pred``) so the
    trimmer never holds a full rollout in memory -- the same discipline the
    producer uses.  Handles both the current 6-D ``pred`` (A, M, L, C, H, W) and
    pre-Aug-2026 5-D dumps (A, L, C, H, W) with no member axis.
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists() and not overwrite:
        raise SystemExit(
            f"ERROR: output already exists: {dst_path} (use --overwrite to replace)"
        )

    with h5py.File(src_path, "r") as fin:
        available = _decode_channels(fin["channel"][...])
        keep_idx = _resolve_keep_indices(
            available, keep_channels, require_all=require_all
        )
        keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
        kept_names = [available[i] for i in keep_idx]

        pred = fin["pred"]
        truth = fin["truth"]
        has_member_axis = pred.ndim == 6
        H, W = truth.shape[-2], truth.shape[-1]
        C_new = len(keep_idx)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        with h5py.File(tmp_path, "w") as fout:
            # --- channel-bearing arrays, recreated with the trimmed C ---------
            truth_out = fout.create_dataset(
                "truth", (truth.shape[0], truth.shape[1], C_new, H, W),
                dtype=truth.dtype, chunks=(1, 1, 1, H, W),
            )
            if has_member_axis:
                A, M, L = pred.shape[0], pred.shape[1], pred.shape[2]
                pred_out = fout.create_dataset(
                    "pred", (A, M, L, C_new, H, W),
                    dtype=pred.dtype, chunks=(1, 1, 1, 1, H, W),
                )
            else:
                A, L = pred.shape[0], pred.shape[1]
                pred_out = fout.create_dataset(
                    "pred", (A, L, C_new, H, W),
                    dtype=pred.dtype, chunks=(1, 1, 1, H, W),
                )

            # truth has no member axis: copy one (L, C', H, W) slab per anchor.
            for a in range(truth.shape[0]):
                truth_out[a] = truth[a][:, keep_idx_arr]

            # pred: one (L, C', H, W) slab per (anchor[, member]).
            if has_member_axis:
                for a in range(A):
                    for m in range(M):
                        pred_out[a, m] = pred[a, m][:, keep_idx_arr]
            else:
                for a in range(A):
                    pred_out[a] = pred[a][:, keep_idx_arr]

            # --- channel names (subset), same dtype ---------------------------
            fout.create_dataset(
                "channel",
                data=np.array([n.encode() for n in kept_names],
                              dtype=fin["channel"].dtype),
            )

            # --- non-channel datasets: verbatim -------------------------------
            for name in _PASSTHROUGH_DATASETS:
                if name in fin:
                    fout.create_dataset(name, data=fin[name][...])

            # --- attributes: verbatim, plus a trim breadcrumb -----------------
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["trimmed_channels"] = np.array(
                [n.encode() for n in kept_names], dtype=fin["channel"].dtype
            )

    os.replace(tmp_path, dst_path)  # atomic: a killed job leaves no partial .h5
    src_mb = src_path.stat().st_size / 1e6
    dst_mb = dst_path.stat().st_size / 1e6
    dropped = [c for c in available if c not in set(kept_names)]
    print(
        f"  {src_path.name}: {len(available)}ch {src_mb:,.0f} MB -> "
        f"{len(kept_names)}ch {dst_mb:,.0f} MB "
        f"({dst_mb / src_mb:.0%})  kept={kept_names}  dropped={len(dropped)}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "inputs", nargs="+",
        help="Field dump file(s) and/or directories. Directories are globbed "
        "for '*fields*.h5' (spectrum/other .h5 skipped).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to write trimmed dumps into (basename preserved).",
    )
    parser.add_argument(
        "--channels", default=",".join(COMPARISON_CHANNELS),
        help="Comma-separated channels to keep (default: the comparison "
        f"manifest: {','.join(COMPARISON_CHANNELS)}).",
    )
    parser.add_argument(
        "--require-all", action="store_true",
        help="Fail if any requested channel is missing from a dump (default: "
        "warn and keep whatever is present).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing trimmed outputs instead of erroring.",
    )
    args = parser.parse_args(argv)

    keep = [c.strip() for c in args.channels.split(",") if c.strip()]
    out_dir = Path(args.out_dir)
    dumps = _iter_field_dumps(args.inputs)

    print(f"[trim] {len(dumps)} dump(s) -> {out_dir}  keep={keep}")
    for src in dumps:
        dst = out_dir / src.name
        trim_dump(src, dst, keep, require_all=args.require_all,
                  overwrite=args.overwrite)
    print("[trim] done.")


if __name__ == "__main__":
    main()
