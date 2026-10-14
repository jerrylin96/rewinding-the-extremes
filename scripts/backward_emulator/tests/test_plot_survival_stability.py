"""Tests for the stability SI figure: npz round-trip, the hours-from-anchor
conversion (survival.py saves the FORWARD lead axis), the interpolated headline
stats (survival at the context edge, 50% crossing), the cross-case detector
consistency guard, and the end-to-end CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_survival_stability import (  # noqa: E402
    crossing_hour,
    load_survival_npz,
    main,
    plot,
    survival_at,
)


def _write_npz(path: Path, survival, *, step_h=6.0, n_members=10,
               channel="t2m", bounds=(150.0, 350.0), detector="range"):
    """Mimic survival.py's savez payload. ``survival`` is FORWARD-ordered
    (index 0 = farthest into the past, last = the anchor), matching the
    ``lead_hours`` axis it is saved with."""
    surv = np.asarray(survival, dtype=float)
    L = surv.size
    alive = np.zeros((n_members, L), dtype=bool)
    n_alive = np.round(surv * n_members).astype(np.int64)
    for lead, k in enumerate(n_alive):
        alive[:k, lead] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        alive=alive, survival=surv, n_alive=n_alive,
        fail_leads=np.full(n_members, -1, dtype=np.int64),
        lead_hours=np.arange(L, dtype=float) * step_h,
        channel=np.array(channel), bounds=np.asarray(bounds, float),
        detector=np.array(detector),
    )


def test_load_converts_forward_leads_to_hours_from_anchor(tmp_path):
    f = tmp_path / "s.npz"
    _write_npz(f, [0.2, 0.6, 1.0, 1.0], step_h=6.0)
    info = load_survival_npz(f)
    np.testing.assert_allclose(info["hours_from_anchor"], [-18, -12, -6, 0])
    assert info["survival"][-1] == 1.0  # the pinned anchor frame
    assert info["n_members"] == 10
    assert info["channel"] == "t2m" and info["detector"] == "range"
    assert info["bounds"] == (150.0, 350.0)


def test_survival_at_interpolates_between_frames():
    h = np.array([-18.0, -12.0, -6.0, 0.0])
    s = np.array([0.2, 0.6, 1.0, 1.0])
    assert survival_at(h, s, -12.0) == pytest.approx(0.6)   # exact frame
    assert survival_at(h, s, -9.0) == pytest.approx(0.8)    # midpoint
    assert survival_at(h, s, -15.0) == pytest.approx(0.4)


def test_crossing_hour_interpolates_first_drop_from_anchor():
    h = np.array([-18.0, -12.0, -6.0, 0.0])
    s = np.array([0.2, 0.6, 1.0, 1.0])
    # Scanning into the past, survival first drops below 0.5 between -12 (0.6)
    # and -18 (0.2): 0.5 sits a quarter of the way -12 -> -18.
    assert crossing_hour(h, s, 0.5) == pytest.approx(-13.5)
    # A frame exactly at the level is the crossing itself.
    assert crossing_hour(h, np.array([0.1, 0.5, 1.0, 1.0]), 0.5) == pytest.approx(-12.0)


def test_crossing_hour_none_when_never_below_level():
    h = np.array([-12.0, -6.0, 0.0])
    assert crossing_hour(h, np.array([0.9, 1.0, 1.0]), 0.5) is None
    assert crossing_hour(h, np.ones(3), 0.5) is None


def test_plot_rejects_mismatched_detectors(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_npz(a, [0.5, 1.0], bounds=(150.0, 350.0))
    _write_npz(b, [0.5, 1.0], bounds=(100.0, 400.0))
    cases = [("case_a", load_survival_npz(a)), ("case_b", load_survival_npz(b))]
    with pytest.raises(ValueError, match="bounds"):
        plot(tmp_path / "fig.png", cases)


def test_main_end_to_end_writes_figure(tmp_path, capsys):
    # Two cases over a 31-frame (-180 h) horizon: one crosses 50%, one never does.
    L = 31
    ian = np.linspace(0.0, 1.0, L) ** 0.5      # attrition deep into the past
    pnw = np.clip(np.linspace(0.7, 1.3, L), 0, 1)
    _write_npz(tmp_path / "ian.npz", ian)
    _write_npz(tmp_path / "pnw.npz", pnw)
    out = tmp_path / "fig" / "survival_stability.png"
    main([
        "--case-npz", f"case_ian={tmp_path / 'ian.npz'}",
        "--case-npz", f"case_pnw={tmp_path / 'pnw.npz'}",
        "--out", str(out),
    ])
    assert out.is_file() and out.stat().st_size > 0
    logged = capsys.readouterr().out
    assert "survival at -66 h" in logged
    assert "50% crossing = none" not in logged.splitlines()[1]  # ian crosses
    assert "detector=range" in logged
