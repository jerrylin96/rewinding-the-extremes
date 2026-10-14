"""Light-env tests for the free-end-states compare pre-render compatibility check.

The render itself is heavy-env (cartopy via the paper's ``make_figure``); here we
cover ``check_blocks_compatible`` — the guard that refuses stale npz pairs (an
impact-cropped makani npz from before the synoptic-view compute, or its 1970 epoch
``start_time`` placeholder) instead of rendering a silently lopsided figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_free_end_states_compare as pfc  # noqa: E402

SYNOPTIC = [-170.0, -100.0, 25.0, 70.0]
IMPACT = [-126.0, -110.0, 38.0, 55.0]


def _diag(view=SYNOPTIC, bbox=IMPACT, start="2021-06-25T18:00:00Z"):
    d = {
        "bbox": np.asarray(bbox, dtype=float),
        "start_time": np.array(start),
    }
    if view is not None:
        d["view_bbox"] = np.asarray(view, dtype=float)
    return d


def test_matching_blocks_pass():
    pfc.check_blocks_compatible(_diag(), _diag())


def test_view_extent_mismatch_is_refused():
    # the observed failure: makani npz cropped to the impact box, cBottle synoptic.
    with pytest.raises(SystemExit, match="different plot extents"):
        pfc.check_blocks_compatible(_diag(view=IMPACT, bbox=IMPACT), _diag())


def test_missing_view_bbox_falls_back_to_bbox():
    # pre-view_bbox npz: the extent make_figure draws is bbox itself.
    with pytest.raises(SystemExit, match="different plot extents"):
        pfc.check_blocks_compatible(_diag(view=None), _diag())


def test_impact_box_mismatch_is_refused():
    other = [-140.0, -110.0, 38.0, 55.0]
    with pytest.raises(SystemExit, match="different impact boxes"):
        pfc.check_blocks_compatible(_diag(bbox=other), _diag())


def test_epoch_start_time_placeholder_is_refused():
    with pytest.raises(SystemExit, match="start_time"):
        pfc.check_blocks_compatible(_diag(start="1970-01-01T00:00:00Z"), _diag())
