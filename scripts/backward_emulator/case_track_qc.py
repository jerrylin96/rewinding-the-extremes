"""The paper's track QC for an end-conditioned case, in one call.

Every comparison script that reads a tracks parquet (the TC-track figure, the
makani synoptic-PCA, and the PCA compare) resolves each member's storm of
interest here.  makani and cBottle therefore go through the paper's own rule,
``ensemble_analysis/track_qc.py``, with the paper's own end-mode parameters
(the values ``aggregate_tc_tracks`` and ``aggregate_synoptic_pca`` use): the
case's ``regions.track`` box, the anchor at the case's landfall frame (last
lead when the case sets none), and the free end at frame 0.

Never take a member's storm from ``path_id=0`` of the raw parquet, nor from the
path-0 ``track_target_fe`` in a ``synoptic_pca_end.npz`` baked before
``compute_synoptic_pca.py`` switched to track QC (no ``track_target_source``
key; the paper's current figure code does not read it).  Tracking is global and
TempestExtremes numbers concurrent systems arbitrarily, so path 0 can be another
storm: for Hurricane Ian, 19 of the 1000 cBottle members have a western-Pacific,
eastern-Pacific or Bay-of-Bengal system on path 0 and Ian on another path.

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

FREE_END_FRAME = 0  # end mode: the unpinned window start
# Free-end target columns, in the order the PCA npz stores them.
TARGET_COLUMNS = ("fe_lat_deg", "fe_lon_deg", "fe_dist_era5_km", "fe_msl_hpa")


def anchor_frame(case: Dict[str, Any], n_leads: int) -> int:
    """End-mode anchor: the case's landfall frame, else the last lead."""
    landfall = dl.resolve_landfall_frame(case)
    return landfall if 0 <= landfall < n_leads else n_leads - 1


def track_qc_for_case(
    ens: np.ndarray,
    era5: np.ndarray,
    case: Dict[str, Any],
    n_members: Optional[int] = None,
):
    """Run the paper's track QC on raw all-paths tracker arrays for ``case``."""
    from track_qc import compute_track_qc_arrays

    box = dl.region_box(case, "track")
    if box is None:
        raise SystemExit(f"case '{case.get('name')}' has no regions.track box.")
    return compute_track_qc_arrays(
        ens,
        era5,
        track_box=box,
        anchor_frame=anchor_frame(case, int(ens.shape[2])),
        free_end_frame=FREE_END_FRAME,
        n_members=n_members,
    )


def load_track_qc(
    parquet: str, case: Dict[str, Any], n_members: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """``(ens, era5, qc)`` for a raw all-paths tracks parquet."""
    from tc_tracks_io import load_tracks_parquet

    d = load_tracks_parquet(parquet)
    ens, era5 = d.get("ensemble"), d.get("era5")
    if ens is None or era5 is None:
        raise SystemExit(f"{parquet}: missing ensemble or era5 tracks")
    return ens, era5, track_qc_for_case(ens, era5, case, n_members=n_members)


def free_end_target_matrix(qc) -> np.ndarray:
    """``[N, 4]`` free-end targets in ``TARGET_COLUMNS`` order; excluded rows NaN."""
    t = qc.free_end_targets()
    return np.column_stack([t[c] for c in TARGET_COLUMNS]).astype(np.float64)


def qc_summary(qc) -> str:
    """One-line exclusion count for logs."""
    c = qc.counts()
    return (
        f"{c['analyzable']}/{c['total']} analyzable (excluded: "
        f"no_track={c['no_track']} unanchored={c['unanchored']} "
        f"insufficient_fixes={c['insufficient_fixes']})"
    )
