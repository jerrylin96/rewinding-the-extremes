"""Unit test for the land-mask convention (isnan(sst) = land)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_land_mask as mlm  # noqa: E402


def test_land_from_sst_marks_missing_as_land():
    # SST finite over ocean, NaN (and any non-finite) over land.
    sst = np.array([[280.0, np.nan],
                    [290.0, np.inf]], dtype=np.float32)
    land = mlm.land_from_sst(sst)
    assert land.dtype == bool
    np.testing.assert_array_equal(land, np.array([[False, True],
                                                  [False, True]]))
    # Ocean fraction is the complement.
    assert land.sum() == 2
