"""Free-end-states comparison figure (paper Figure-1 analog): Autoregressive vs cBottle.

The paper's Figure 1 (``aggregate_free_end_states.make_figure``) stacks two
conditioning-mode blocks of ONE model (cBottle end / start).  Here we reuse that
function *verbatim* to stack two MODELS instead -- the autoregressive backward emulator
(top; ``free_end_states_makani.py`` output) over cBottle video diffusion (bottom; the
paper's own ``free_end_states_end.npz``) -- so the layout is identical to the reference
(same two-edge colorbars, bias-rank tail colouring, local-time axis, section headers)
with no re-implementation.

``make_figure`` is fully diag-driven: the ``"end"/"start"`` keys only choose the
top-to-bottom order and the ``MODE_LABEL`` section header (every field, extent, ERA5
tile, marker and reduction comes from each block's own npz).  So we drop our two
end-conditioned diags into those two slots and override ``MODE_LABEL`` with the model
names.  Per-model baseline is preserved because each block carries its own
``era5_field`` / ``era5_scalar`` (makani native ERA5; cBottle round-tripped).

Both npz files must share the schema (they do: ``free_end_states_makani.py`` mirrors
``compute_free_end_states.py``) and the same view/impact regions (both use
``regions.synoptic`` / ``regions.impact``).  Because ``make_figure`` draws whatever
each npz baked, a stale input renders a silently lopsided figure — e.g. a makani npz
from before the synoptic-view compute was cropped to the impact box, so its block
drew at a different extent than cBottle's, and carried the 1970 epoch ``start_time``
placeholder, so its local-time axis showed Dec 1969.  :func:`check_blocks_compatible`
refuses those inputs up front (``--skip-checks`` overrides).  The render needs
``e2s-custom`` (matplotlib + cartopy); the compatibility check is the only light-env
logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BW_DIR = Path(__file__).resolve().parent
SHARED = BW_DIR.parent / "_shared"
EA_SUB = BW_DIR.parent / "ensemble_analysis" / "submission_scripts"
for _p in (str(BW_DIR), str(SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(EA_SUB) not in sys.path:  # appended: never shadow our own modules
    sys.path.append(str(EA_SUB))

import _dispatch_lib as dl  # noqa: E402

# Top-to-bottom order + section labels (override the paper's mode labels).
TOP_KEY, BOTTOM_KEY = "end", "start"  # make_figure renders "end" above "start"
LABELS = {TOP_KEY: "Autoregressive", BOTTOM_KEY: "Video diffusion (cBottle)"}

# ``start_time`` free_end_states_makani.py bakes when no case start time resolves;
# rendering it puts the top block's local-time axis in Dec 1969.
EPOCH_PLACEHOLDER = "1970-01-01"


def _view_box(diag: Dict[str, np.ndarray]) -> np.ndarray:
    """The extent ``make_figure`` will draw for this block (``view_bbox``, else ``bbox``)."""
    return np.asarray(diag.get("view_bbox", diag["bbox"]), dtype=float)


def check_blocks_compatible(top: Dict[str, np.ndarray],
                            bottom: Dict[str, np.ndarray]) -> None:
    """Refuse npz pairs that would render a silently lopsided figure.

    ``make_figure`` draws each block exactly as its npz baked it, so nothing
    downstream catches a mismatch.  Both failure modes here mean the SAME thing —
    the makani npz predates the synoptic-view compute — and the same fix:
    re-run ``free_end_states_makani.py`` (it crops to ``regions.synoptic`` and
    resolves the case start time) before re-rendering.
    """
    vt, vb = _view_box(top), _view_box(bottom)
    if not np.allclose(vt, vb):
        raise SystemExit(
            f"ERROR: the two blocks bake different plot extents "
            f"(top view={vt.tolist()}, bottom view={vb.tolist()}), so the panels "
            f"would render at different fields of view. Re-run "
            f"free_end_states_makani.py (current code crops to regions.synoptic, "
            f"matching the paper's cBottle npz) and re-render, or pass "
            f"--skip-checks to force."
        )
    bt = np.asarray(top["bbox"], dtype=float)
    bb = np.asarray(bottom["bbox"], dtype=float)
    if not np.allclose(bt, bb):
        raise SystemExit(
            f"ERROR: the two blocks reduce over different impact boxes "
            f"(top={bt.tolist()}, bottom={bb.tolist()}); their bias rankings and "
            f"time series are not comparable. Recompute both npz files against "
            f"the same case regions, or pass --skip-checks to force."
        )
    for name, diag in (("top", top), ("bottom", bottom)):
        start = str(np.asarray(diag.get("start_time", "")).item() or "")
        if start.startswith(EPOCH_PLACEHOLDER):
            raise SystemExit(
                f"ERROR: the {name} npz carries the {EPOCH_PLACEHOLDER} start_time "
                f"placeholder, so its local-time axis would show Dec 1969. Re-run "
                f"its compute (current free_end_states_makani.py resolves the case "
                f"start time), or pass --skip-checks to force."
            )


def _load_diag(npz_path: str, afes) -> Dict[str, np.ndarray]:
    """Load a free-end-states npz as a diag dict (display-unit converted like the paper)."""
    diag = dict(np.load(npz_path, allow_pickle=True))
    afes._to_display_units(diag)  # no-op for non-geopotential (t2m); matches the paper
    return diag


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--makani-npz", required=True,
                   help="Autoregressive free-end-states npz (free_end_states_makani.py).")
    p.add_argument("--cbottle-npz", required=True,
                   help="cBottle free_end_states_end.npz (the paper's own output).")
    p.add_argument("--case", default="pnw_heatwave")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-checks", action="store_true",
                   help="Render even if the two npz blocks bake different extents / "
                   "impact boxes or a placeholder start time (see "
                   "check_blocks_compatible).")
    args = p.parse_args(argv)

    import aggregate_free_end_states as afes  # heavy env (matplotlib, cartopy, ...)

    # Reuse the reference figure: Autoregressive on top, cBottle below, relabelled.
    afes.MODE_LABEL = dict(LABELS)
    afes.MODE_ORDER = (TOP_KEY, BOTTOM_KEY)
    per_mode = {
        TOP_KEY: _load_diag(args.makani_npz, afes),
        BOTTOM_KEY: _load_diag(args.cbottle_npz, afes),
    }
    if not args.skip_checks:
        check_blocks_compatible(per_mode[TOP_KEY], per_mode[BOTTOM_KEY])

    case = dl.load_case(args.case)
    cname = case.get("display_name", args.case)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_png = out / f"free_end_states_compare_{args.case}.png"
    afes.make_figure(per_mode, out_png)
    print(f"[fes-cmp] wrote {out_png} ({cname}: Autoregressive vs cBottle)")


if __name__ == "__main__":
    main()
