"""One-off: rewrite the stored ``case_name`` of existing npz outputs to the new display names.

The compute jobs copy the case YAML's ``display_name`` into every npz they write as
``case_name``; figure titles and the PCA pairing check read it back.  Commit f338f00
renamed two display names to match the papers' text ("Hurricane Sandy" ->
"Superstorm Sandy", "2021 Pacific Northwest Heat Wave" -> "2021 Pacific Northwest
heatwave").  This rewrites that one field in files written before the rename, so they
hold exactly what a recompute with the current YAML would write.  Nothing else in a
file changes: with ``--apply`` each file is written to a temporary copy, every other
array is checked byte-identical to the original, and only then is the original
replaced.

Dry run by default (reads only ``case_name``); ``--apply`` writes.  Re-running is
harmless: migrated files report as already current.

Usage (from this directory):
    python migrate_case_names.py ROOT [ROOT ...]           # list what would change
    python migrate_case_names.py ROOT [ROOT ...] --apply   # change it
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dispatch_lib import load_case  # noqa: E402

# Old display name -> case key; the new name is that case's current display_name.
RENAMED = {"Hurricane Sandy": "sandy", "2021 Pacific Northwest Heat Wave": "pnw_heatwave"}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-level equality (NaN payloads included); object arrays via pickle."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == object:
        return pickle.dumps(a) == pickle.dumps(b)
    return a.tobytes() == b.tobytes()


def rewrite(path: Path, new_name: str) -> None:
    """Replace ``case_name`` in one npz, leaving every other array byte-identical."""
    with np.load(path, allow_pickle=True) as z:
        data: Dict[str, np.ndarray] = {k: z[k] for k in z.files}
    if "file" in data:  # np.savez's own first parameter
        raise SystemExit(f"ERROR: {path} has an array named 'file'; not handled.")
    data["case_name"] = np.array(new_name)
    tmp = path.with_name(path.name + ".migrating")
    with open(tmp, "wb") as f:  # a file object: np.savez would append .npz to a name
        np.savez(f, **data)
    with np.load(tmp, allow_pickle=True) as z:
        if sorted(z.files) != sorted(data):
            raise SystemExit(f"ERROR: {tmp} has keys {sorted(z.files)}; left in place.")
        for k in data:
            if not _same(z[k], data[k]):
                raise SystemExit(f"ERROR: {tmp}: '{k}' differs after writing; left in place.")
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("roots", nargs="+", help="Directories searched recursively for *.npz.")
    p.add_argument("--apply", action="store_true", help="Rewrite (default: list only).")
    args = p.parse_args(argv)

    new = {old: str(load_case(key)["display_name"]) for old, key in RENAMED.items()}
    for old, name in new.items():
        if name == old:
            raise SystemExit(f"ERROR: the case YAML still says {old!r}; pull first.")

    todo: List[Path] = []
    seen, tally = set(), Counter()
    for root in map(Path, args.roots):
        if not root.is_dir():
            raise SystemExit(f"ERROR: not a directory: {root}")
        for path in sorted(root.rglob("*.npz")):
            real = path.resolve()  # a symlinked file is handled once, at its target
            if real in seen:
                continue
            seen.add(real)
            with np.load(real, allow_pickle=True) as z:
                name = str(z["case_name"]) if "case_name" in z.files else None
            if name is None:
                tally["no case_name"] += 1
            elif name in new:
                todo.append(real)
                tally[f"{name!r} -> {new[name]!r}"] += 1
            elif name in new.values():
                tally[f"{name!r} (already current)"] += 1
            else:
                tally[f"{name!r} (other case, untouched)"] += 1

    for path in todo:
        print(f"[migrate] {'rewrite' if args.apply else 'would rewrite'} {path}")
    print(f"[migrate] {len(seen)} npz files under {len(args.roots)} root(s):")
    for label, n in sorted(tally.items()):
        print(f"[migrate]   {n:5d}  {label}")
    if not args.apply:
        print("[migrate] dry run; nothing written. Add --apply to rewrite.")
        return
    for path in todo:
        with np.load(path, allow_pickle=True) as z:
            old = str(z["case_name"])
        rewrite(path, new[old])
    print(f"[migrate] rewrote {len(todo)} file(s).")


if __name__ == "__main__":
    main()
