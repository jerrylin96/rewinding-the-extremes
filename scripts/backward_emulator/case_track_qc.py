"""The paper's track QC for the backward-emulator comparison, in one call.

Every comparison script that reads a tracks parquet (the TC-track figure, the
makani synoptic-PCA, and the PCA compare) resolves each member's storm of
interest here, through the paper's own rule (``ensemble_analysis/track_qc.py``)
for an end-conditioned case: the case's ``regions.track`` box, the anchor at the
pinned last frame, the free end at frame 0.  makani and cBottle therefore share
one population rule and one set of free-end targets.

Nothing reads a member's storm from ``path_id=0`` of the raw parquet, nor track
targets stored in an npz: tracking is global and TempestExtremes numbers
concurrent systems arbitrarily, so path 0 can be another storm (for Hurricane
Ian, 19 of the 1000 cBottle members carry a western-Pacific, eastern-Pacific or
Bay-of-Bengal system there).

`check_pca_npzs` is the matching guard for the PCA npz files those scripts pair
with the tracks: one case, this comparison's free-end frame, one impact kind.

Light: numpy + pandas (track_qc) + the parquet reader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
EA_DIR = BW_DIR.parent / "ensemble_analysis"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(EA_DIR) not in sys.path:  # appended: never shadow our modules
    sys.path.append(str(EA_DIR))

import _dispatch_lib as dl  # noqa: E402

MODE = "end"  # the comparison's ensembles are end-conditioned


def load_track_qc(
    parquet: str, case: Dict[str, Any], n_members: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """``(ens, era5, qc)`` for a raw all-paths tracks parquet of ``case``."""
    from track_qc import compute_track_qc

    box = dl.region_box(case, "track")
    if box is None:
        raise SystemExit(f"case '{case.get('name')}' has no regions.track box.")
    return compute_track_qc(parquet, track_box=box, mode=MODE, n_members=n_members)


def check_pca_npzs(paths: Sequence[str], case_name: Optional[str] = None) -> None:
    """Refuse PCA npz files that do not belong to one comparison.

    Every file must carry ``case_name``, ``free_end_frame`` and ``impact_kind``;
    both producers (``synoptic_pca_makani.py``, ``compute_synoptic_pca.py``) write
    all three, so a missing key is refused, never defaulted.  All files must name
    one case (``case_name``, the case YAML's display name, when given; otherwise
    the first file's), have their free end at this comparison's frame, and share
    one impact kind.  This catches a file from another case or another
    conditioning mode.  It cannot tell two ensembles of one case apart: the npz
    and the parquet carry no ensemble ID (GitHub issue #6).
    """
    free_end = dl.anchor_free_end_frames(MODE)[1]
    ref_case, case_src = case_name, "the case YAML"
    ref_kind, kind_src = None, None
    for path in paths:
        with np.load(path, allow_pickle=True) as z:  # as the readers load them
            missing = [k for k in ("case_name", "free_end_frame", "impact_kind")
                       if k not in z.files]
            if missing:
                raise SystemExit(f"[pca-pairing] {path}: no {', '.join(missing)}.")
            name, fe = str(z["case_name"]), int(z["free_end_frame"])
            kind = str(z["impact_kind"])
        if ref_case is None:
            ref_case, case_src = name, path
        if name != ref_case:
            raise SystemExit(f"[pca-pairing] {path}: case {name!r}, but "
                             f"{case_src} has {ref_case!r}.")
        if fe != free_end:
            raise SystemExit(f"[pca-pairing] {path}: free end at frame {fe}; this "
                             f"comparison's ({MODE} mode) is frame {free_end}.")
        if ref_kind is None:
            ref_kind, kind_src = kind, path
        if kind != ref_kind:
            raise SystemExit(f"[pca-pairing] {path}: impact kind {kind!r}, but "
                             f"{kind_src} has {ref_kind!r}.")


def free_end_target_matrix(qc) -> np.ndarray:
    """``[N, 4]`` free-end targets in ``track_qc.TARGET_COLUMNS`` order; excluded rows NaN."""
    from track_qc import TARGET_COLUMNS

    t = qc.free_end_targets()
    return np.column_stack([t[c] for c in TARGET_COLUMNS]).astype(np.float64)
