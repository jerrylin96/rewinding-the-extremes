#!/usr/bin/env python3
"""Merge partitioned Zarr v3 stores into a single canonical Zarr v3 store.

Each chunk job writes ``ensemble_f<frames>_n<N>_offset<O>.zarr`` covering a
contiguous slice of the ensemble dimension.  Since the ensemble dim is chunked
with size 1 in ``create_ensemble_zarr``, every member maps to its own on-disk
chunk directory, so merging is a pure rename: copy the metadata from offset 0,
rewrite the ensemble-dim shape, then move each partial store's chunk
directories into the canonical ``data/c/<global_idx>/...`` layout.
"""

import argparse
import glob
import json
import os
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("ensemble_size", type=int)
    parser.add_argument("frames", type=str)
    parser.add_argument("chunks", type=int)
    return parser.parse_args()


def load_zarr_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    frames_tag = args.frames.replace(",", "_")
    prefix = f"ensemble_f{frames_tag}_n{args.ensemble_size}_offset"
    pattern = os.path.join(args.out_dir, f"{prefix}*.zarr")

    # 1. Validate that every chunk job wrote a complete store.
    zarr_dirs = sorted(glob.glob(pattern))
    if len(zarr_dirs) < args.chunks:
        print(
            f"ERROR: Expected {args.chunks} chunk directories, but found "
            f"{len(zarr_dirs)}.",
            file=sys.stderr,
        )
        print(f"Search pattern: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(zarr_dirs)} chunk arrays. Verifying metadata...")
    for zdir in zarr_dirs:
        group_meta_path = os.path.join(zdir, "zarr.json")
        if not os.path.exists(group_meta_path):
            print(
                f"ERROR: Missing zarr.json in {zdir}. Job likely crashed.",
                file=sys.stderr,
            )
            sys.exit(1)
        group_meta = load_zarr_json(group_meta_path)
        attrs = group_meta.get("attributes", {}) or {}
        if not attrs.get("complete", False):
            print(
                f"ERROR: Metadata for {zdir} shows incomplete. Upstream job "
                f"likely crashed or timed out.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 2. Seed the canonical store from offset 0, then rewrite the ensemble
    #    dimension length in the array's zarr.json.
    canonical_zarr = os.path.join(
        args.out_dir, f"ensemble_f{frames_tag}_n{args.ensemble_size}.zarr"
    )
    offset0 = os.path.join(args.out_dir, f"{prefix}0.zarr")
    if not os.path.exists(offset0):
        print(f"ERROR: offset 0 missing: {offset0}", file=sys.stderr)
        sys.exit(1)

    print(f"Creating canonical store at {canonical_zarr}")
    if os.path.exists(canonical_zarr):
        shutil.rmtree(canonical_zarr)
    shutil.copytree(offset0, canonical_zarr)

    array_meta_path = os.path.join(canonical_zarr, "data", "zarr.json")
    array_meta = load_zarr_json(array_meta_path)
    array_meta["shape"][0] = args.ensemble_size
    with open(array_meta_path, "w") as f:
        json.dump(array_meta, f)

    # Figure out the chunk_key_encoding separator ("/" by default in v3) so we
    # can build chunk paths the same way zarr did.
    cke = array_meta.get("chunk_key_encoding", {})
    separator = (cke.get("configuration") or {}).get("separator", "/")

    # 3. Re-parent chunk files from each partial store into the canonical one.
    print("Moving chunk files (zero-copy rename)...")
    canonical_data = os.path.join(canonical_zarr, "data")
    canonical_c = os.path.join(canonical_data, "c")
    os.makedirs(canonical_c, exist_ok=True)

    for zdir in zarr_dirs:
        offset = int(zdir.rsplit("_offset", 1)[-1].removesuffix(".zarr"))
        if offset == 0:
            # Already copied via copytree above; leave its chunks in place.
            continue

        src_c = os.path.join(zdir, "data", "c")
        if not os.path.isdir(src_c):
            print(
                f"ERROR: missing chunk directory {src_c}",
                file=sys.stderr,
            )
            sys.exit(1)

        if separator == "/":
            # Nested layout: c/<ensemble_idx>/<rest...>
            for local_idx_str in os.listdir(src_c):
                local_idx = int(local_idx_str)
                global_idx = offset + local_idx
                shutil.move(
                    os.path.join(src_c, local_idx_str),
                    os.path.join(canonical_c, str(global_idx)),
                )
        else:
            # Flat layout: c/<ensemble_idx>.<rest...>
            for name in os.listdir(src_c):
                parts = name.split(separator)
                parts[0] = str(offset + int(parts[0]))
                shutil.move(
                    os.path.join(src_c, name),
                    os.path.join(canonical_c, separator.join(parts)),
                )

    # 4. Tear down the partial stores.
    print("Cleaning up temporary directories...")
    for zdir in zarr_dirs:
        shutil.rmtree(zdir)

    print(f"Successfully constructed completed ensemble: {canonical_zarr}")


if __name__ == "__main__":
    main()
