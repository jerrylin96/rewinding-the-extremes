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

Light: numpy + pandas (track_qc) + the parquet reader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def free_end_target_matrix(qc) -> np.ndarray:
    """``[N, 4]`` free-end targets in ``track_qc.TARGET_COLUMNS`` order; excluded rows NaN."""
    from track_qc import TARGET_COLUMNS

    t = qc.free_end_targets()
    return np.column_stack([t[c] for c in TARGET_COLUMNS]).astype(np.float64)
