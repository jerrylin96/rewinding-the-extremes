"""Adapter: a directory of makani backward-rollout dumps as one lat/lon source.

The makani case-study ensemble is produced as a SLURM job array -- many
``*fields*.h5`` dumps, each holding a few ensemble members for the *same* anchor
(e.g. 250 files x 4 members = 1000).  Every dump shares the anchor, grid,
channels and lead axis and differs only in its members' input-noise seed, so the
files pool into one ensemble by concatenating on the member axis (the exact
recipe makani's own ``load_channel_members`` docstring prescribes).

This class does that pooling once and presents the result on **cBottle's forward
lead axis**, so the comparison diagnostics can consume makani exactly like the
cBottle ensemble:

* **Native lat/lon.** makani is 0.25 deg lat/lon already; nothing is regridded
  here.  (cBottle stays HEALPix in storage and is regridded HPX->lat/lon at
  compute time, as it already is; ERA5 truth stays lat/lon.)
* **Frames reversed + truncated.** makani indexes frame 0 = present (the anchor)
  increasing INTO THE PAST; cBottle indexes frame 0 = window start increasing
  forward to frame 11 = end.  The makani anchor IS cBottle's window end, so the
  two label the same valid times in opposite order.  We take the first
  ``n_frames`` makani frames (the 66 h window; the rollout runs further back but
  destabilizes) and flip them, so forward frame f = makani frame (n_frames-1-f).
  After the flip the pinned frame (makani frame 0, where ``pred == truth``) lands
  at forward frame ``n_frames-1`` -- exactly where cBottle's end-conditioning
  pins it.

Torch-free (numpy + h5py) on purpose: ingestion has no reason to pull in the heavy
stack, and it stays unit-testable with a synthetic fixture.  Downstream metric
code adds torch / earth2studio where the actual reductions happen.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# Reuse the single source of truth for the comparison channel set.
try:  # importable both as a module and as a loose script next to the dumps
    from .trim_backward_fields import COMPARISON_CHANNELS
except ImportError:  # pragma: no cover - script-style import
    from trim_backward_fields import COMPARISON_CHANNELS

DEFAULT_N_FRAMES = 12  # 12 frames x 6 h = the 66 h comparison window


def expected_anchor_iso_from_case(case: Dict, n_frames: int = DEFAULT_N_FRAMES,
                                  step_hours: int = 6) -> str:
    """ISO-hour prefix (``YYYY-MM-DDTHH``) of the case's window END, which is the
    makani anchor by construction: ``ensemble.start_time`` + (n_frames - 1) x step.

    Always derived from the comparison window (12 frames = 66 h), never from a
    scan length -- ``survival.py --n-frames 0`` scans the whole rollout but the
    anchor it must match is still the window end.
    """
    start = case["ensemble"]["start_time"]
    t0 = datetime(int(start["year"]), int(start["month"]), int(start["day"]),
                  int(start["hour"]), tzinfo=timezone.utc)
    return (t0 + timedelta(hours=(n_frames - 1) * step_hours)).strftime("%Y-%m-%dT%H")


def resolve_expected_anchor(case: Optional[Dict], override: Optional[str] = None) -> Optional[str]:
    """Resolve the ``--expected-anchor`` CLI contract.

    ``override`` wins when given (``"none"`` disables the check -- an explicit,
    visible opt-out); otherwise the anchor is derived from the case; with neither
    there is nothing to check against.  The join of makani onto cBottle's window is
    positional, so this check is the only thing that ties a dump to the case it is
    compared under: without it a dump anchored on the wrong date produces
    plausible, correctly-labelled, silently time-shifted figures.
    """
    if override is not None:
        if override.strip().lower() == "none":
            return None
        return override.strip().replace(" ", "T")
    if case is None:
        return None
    return expected_anchor_iso_from_case(case)


def _decode(raw: Sequence) -> List[str]:
    return [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]


def find_field_dumps(path_or_paths) -> List[Path]:
    """Expand a directory (or explicit file list) to sorted ``*fields*.h5`` paths.

    A directory is globbed for ``*fields*.h5`` (skipping ``*spectrum*.h5`` and any
    other ``.h5``); an explicit file path is taken verbatim.
    """
    items = [path_or_paths] if isinstance(path_or_paths, (str, Path)) else list(
        path_or_paths
    )
    out: List[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*fields*.h5")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"input path does not exist: {p}")
    if not out:
        raise FileNotFoundError(
            f"no *fields*.h5 dumps found under {items}. (Directories are globbed "
            "for '*fields*.h5'; pass explicit file paths otherwise.)"
        )
    return out


class MakaniBackwardSource:
    """Pool a case's makani dumps into one lat/lon, forward-ordered ensemble."""

    def __init__(
        self,
        path_or_paths,
        *,
        n_frames: int = DEFAULT_N_FRAMES,
        expected_anchor_iso: Optional[str] = None,
        strict_truth: bool = True,
    ) -> None:
        self.files = find_field_dumps(path_or_paths)
        self.n_frames = int(n_frames)
        self.strict_truth = strict_truth
        self._per_file: List[Dict] = []  # {path, anchor_slot, n_members, ndim}
        self._read_and_validate(expected_anchor_iso)

    # -- metadata ---------------------------------------------------------------
    def _read_and_validate(self, expected_anchor_iso: Optional[str]) -> None:
        ref: Optional[Dict] = None
        total_members = 0
        for path in self.files:
            with h5py.File(path, "r") as f:
                channels = _decode(f["channel"][...])
                lat = f["lat"][...]
                lon = f["lon"][...]
                lhp = np.asarray(f["lead_hours_into_past"][...])
                pred = f["pred"]
                has_member_axis = pred.ndim == 6
                n_members = pred.shape[1] if has_member_axis else 1
                n_leads = pred.shape[2] if has_member_axis else pred.shape[1]
                ts = _decode(f["timestamps"][...].reshape(f["timestamps"].shape[0], -1)[:, 0]) \
                    if "timestamps" in f else [""] * pred.shape[0]
                step_hours = int(f.attrs.get("step_hours", -6))

                if n_leads < self.n_frames:
                    raise ValueError(
                        f"{path.name}: has {n_leads} leads but n_frames="
                        f"{self.n_frames} requested (need the full 66 h window)."
                    )
                # Resolve which anchor slot to use.
                anchor_slot = self._resolve_anchor_slot(f, expected_anchor_iso, path)
                anchor_ts = ts[anchor_slot] if anchor_slot < len(ts) else ""

                fp = {
                    "path": path,
                    "anchor_slot": anchor_slot,
                    "n_members": int(n_members),
                    "has_member_axis": has_member_axis,
                    "anchor_ts": anchor_ts,
                }
                if ref is None:
                    ref = {
                        "channels": channels,
                        "lat": np.asarray(lat, dtype=np.float64),
                        "lon": np.asarray(lon, dtype=np.float64),
                        "lead_hours_into_past": lhp,
                        "step_hours": step_hours,
                        "anchor_ts": anchor_ts,
                    }
                else:
                    self._assert_consistent(ref, channels, lat, lon, lhp,
                                            anchor_ts, path)
                self._per_file.append(fp)
                total_members += int(n_members)

        assert ref is not None
        self.channels: List[str] = ref["channels"]
        self.lat: np.ndarray = ref["lat"]
        self.lon: np.ndarray = ref["lon"]
        self.step_hours: int = ref["step_hours"]
        self.anchor_timestamp: str = ref["anchor_ts"]
        self.n_members: int = total_members

        # Forward lead axis (cBottle convention): frame 0 = window start.
        step_h = abs(self.step_hours)
        self.lead_hours: np.ndarray = np.arange(self.n_frames) * step_h
        # makani lead_hours_into_past over the kept window, reversed to forward.
        self._lhp_window = ref["lead_hours_into_past"][: self.n_frames]

    def _resolve_anchor_slot(
        self, f: h5py.File, expected_anchor_iso: Optional[str], path: Path
    ) -> int:
        n_anchors = f["pred"].shape[0]
        if expected_anchor_iso is None:
            return 0  # case-study dumps carry a single anchor
        ts = _decode(f["timestamps"][...][:, 0]) if "timestamps" in f else []
        # makani writes str(datetime): "2021-06-28 12:00:00+00:00" (a space); the
        # case-derived prefix is ISO "2021-06-28T12". Compare in one form.
        norm = [t.replace(" ", "T") for t in ts]
        want = expected_anchor_iso.replace(" ", "T")
        for a in range(n_anchors):
            if a < len(norm) and norm[a].startswith(want):
                return a
        raise ValueError(
            f"{path.name}: no anchor matching '{want}' (the case's window end); the "
            f"dump carries valid-time(s) {ts}. These dumps are not this case's -- "
            f"wrong --makani-dir? Pass --expected-anchor none to compare anyway."
        )

    @staticmethod
    def _assert_consistent(ref, channels, lat, lon, lhp, anchor_ts, path) -> None:
        if channels != ref["channels"]:
            raise ValueError(
                f"{path.name}: channels {channels} differ from first dump "
                f"{ref['channels']}"
            )
        if not np.array_equal(lat, ref["lat"]) or not np.array_equal(lon, ref["lon"]):
            raise ValueError(f"{path.name}: lat/lon grid differs from first dump")
        if not np.array_equal(lhp, ref["lead_hours_into_past"]):
            raise ValueError(f"{path.name}: lead_hours_into_past differs")
        if ref["anchor_ts"] and anchor_ts and anchor_ts != ref["anchor_ts"]:
            raise ValueError(
                f"{path.name}: anchor valid-time {anchor_ts!r} differs from "
                f"{ref['anchor_ts']!r} -- these dumps are not the same case."
            )

    # -- data -------------------------------------------------------------------
    def _channel_index(self, name: str) -> int:
        if name not in self.channels:
            raise KeyError(
                f"channel '{name}' not in dumps; available: {self.channels}"
            )
        return self.channels.index(name)

    def _forward(self, arr: np.ndarray) -> np.ndarray:
        """Truncate to the window and flip the lead axis (axis -3) to forward."""
        # arr: (..., n_leads, H, W) -> (..., n_frames, H, W) forward-ordered.
        windowed = arr[..., : self.n_frames, :, :]
        flipped = np.flip(windowed, axis=-3)
        return np.ascontiguousarray(flipped)

    # Guard: the full pooled ensemble for one channel is N*L*H*W floats.
    # At case-study scale (1000 x 12 x 721 x 1440) that is ~50 GB -- refuse to
    # build it in RAM and steer the caller to the region-restricted path.
    _FULL_LOAD_MAX_BYTES = 8_000_000_000  # 8 GB

    def _read_pred_forward(self, f: "h5py.File", fp: Dict, ci: int) -> np.ndarray:
        """Read one file's pred for channel ``ci``, forward-ordered: (M, nf, H, W).

        Slices to the window BEFORE reading so only ``n_frames`` leads leave disk.
        """
        a = fp["anchor_slot"]
        if fp["has_member_axis"]:
            block = np.asarray(f["pred"][a, :, : self.n_frames, ci])  # (M, nf, H, W)
        else:
            block = np.asarray(f["pred"][a, : self.n_frames, ci])[None]  # (1, nf,H,W)
        return self._forward(block)

    def _read_truth_forward(self, f: "h5py.File", fp: Dict, ci: int) -> np.ndarray:
        a = fp["anchor_slot"]
        return self._forward(np.asarray(f["truth"][a, : self.n_frames, ci]))  # (nf,H,W)

    def load_region(
        self, name: str, flat_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Memory-safe load of ONE channel restricted to a set of pixels.

        ``flat_indices`` indexes the flattened ``H*W`` grid (as returned by
        ``latlon_metrics.select_pixels``).  Returns ``ens (N, n_frames, P)`` and
        ``truth (n_frames, P)``.  Streams per file: only one file's planes are in
        RAM at a time, and only the ``P`` selected pixels are retained -- so a
        synoptic box (~2 GB pooled) is safe where the full field (~50 GB) is not.
        """
        ci = self._channel_index(name)
        flat = np.asarray(flat_indices, dtype=np.int64)
        ens_parts: List[np.ndarray] = []
        truth_ref: Optional[np.ndarray] = None
        for fp in self._per_file:
            with h5py.File(fp["path"], "r") as f:
                block = self._read_pred_forward(f, fp, ci)   # (M, nf, H, W)
                M, nf = block.shape[0], block.shape[1]
                ens_parts.append(block.reshape(M, nf, -1)[:, :, flat])  # (M,nf,P)
                truth = self._read_truth_forward(f, fp, ci).reshape(nf, -1)[:, flat]
                if truth_ref is None:
                    truth_ref = truth
                elif self.strict_truth and not np.array_equal(truth, truth_ref):
                    raise ValueError(
                        f"{fp['path'].name}: truth for '{name}' differs over the "
                        "selected pixels -- the pooled files are not the same "
                        "anchor/reanalysis."
                    )
        assert truth_ref is not None
        return np.concatenate(ens_parts, axis=0), truth_ref

    def load_variable(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Full-field ``(ens (N,nf,H,W), truth (nf,H,W))`` -- small grids only.

        Convenient for tests and coarse-grid maps.  Refuses to run when the full
        pooled tensor would exceed :data:`_FULL_LOAD_MAX_BYTES` -- use
        :meth:`load_region` for case-study-scale data.
        """
        H, W = int(self.lat.shape[0]), int(self.lon.shape[0])
        nbytes = self.n_members * self.n_frames * H * W * 4
        if nbytes > self._FULL_LOAD_MAX_BYTES:
            raise MemoryError(
                f"load_variable('{name}') would allocate ~{nbytes/1e9:.0f} GB "
                f"(N={self.n_members}, {self.n_frames}x{H}x{W}). Use load_region() "
                "with a box, or spread_rmse_crps over a region."
            )
        flat = np.arange(H * W, dtype=np.int64)
        ens, truth = self.load_region(name, flat)
        return ens.reshape(self.n_members, self.n_frames, H, W), truth.reshape(
            self.n_frames, H, W
        )

    def sample_pinned_frame(
        self, name: str, *, file_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cheap read of the pinned frame from ONE file: ``(pred (M,H,W), truth (H,W))``.

        The pinned frame is makani frame 0 (the present/anchor, ``pred == truth``),
        which our forward flip maps to forward frame -1.  Used to verify alignment
        without loading the ensemble.
        """
        ci = self._channel_index(name)
        fp = self._per_file[file_index]
        a = fp["anchor_slot"]
        with h5py.File(fp["path"], "r") as f:
            if fp["has_member_axis"]:
                pred0 = np.asarray(f["pred"][a, :, 0, ci])   # (M, H, W)
            else:
                pred0 = np.asarray(f["pred"][a, 0, ci])[None]
            truth0 = np.asarray(f["truth"][a, 0, ci])        # (H, W)
        return pred0, truth0

    def sample_truth_field(self, name: str, *, file_index: int = 0) -> np.ndarray:
        """Forward-ordered truth window ``(n_frames, H, W)`` from one file (~50 MB)."""
        ci = self._channel_index(name)
        fp = self._per_file[file_index]
        with h5py.File(fp["path"], "r") as f:
            return self._read_truth_forward(f, fp, ci)

    def sample_member_window(self, name: str, *, file_index: int = 0) -> np.ndarray:
        """Forward-ordered pred window ``(M, n_frames, H, W)`` for ONE file's
        members (~200 MB) -- a cheap blow-up / finiteness spot check."""
        ci = self._channel_index(name)
        fp = self._per_file[file_index]
        with h5py.File(fp["path"], "r") as f:
            return self._read_pred_forward(f, fp, ci)

    def iter_member_domain_series(
        self, name: str, flat_indices: Optional[np.ndarray] = None
    ):
        """Yield ``(global_member_id, series (n_frames, P))`` one member at a time.

        Streams per file/member with only one member's window in RAM -- the raw
        material for a survival scan (per-member, per-lead reductions) without
        pooling the whole 50 GB ensemble.  ``flat_indices`` restricts to a box.
        """
        ci = self._channel_index(name)
        H, W = int(self.lat.shape[0]), int(self.lon.shape[0])
        flat = (np.arange(H * W) if flat_indices is None
                else np.asarray(flat_indices, dtype=np.int64))
        gid = 0
        for fp in self._per_file:
            with h5py.File(fp["path"], "r") as f:
                block = self._read_pred_forward(f, fp, ci)  # (M, nf, H, W)
                M, nf = block.shape[0], block.shape[1]
                flat2d = block.reshape(M, nf, -1)
                for m in range(M):
                    yield gid, flat2d[m][:, flat]
                    gid += 1

    def iter_member_fields(self, names: Sequence[str]):
        """Yield ``(global_member_id, {name: (n_frames, H, W)})`` per member.

        Full-grid, forward-ordered, for one or more channels at once -- the raw
        material for diagnostics that need *contiguous* fields (spatial
        derivatives), where :meth:`load_region`'s scattered pixels won't do, e.g.
        the geostrophic-wind computation.  Streams per file, so only one file's
        planes for the requested channels are in RAM (~0.6 GB for 4 members x 12
        frames x 3 channels), never the pooled ~150 GB.
        """
        cis = {n: self._channel_index(n) for n in names}
        gid = 0
        for fp in self._per_file:
            with h5py.File(fp["path"], "r") as f:
                blocks = {n: self._read_pred_forward(f, fp, ci)
                          for n, ci in cis.items()}  # each (M, nf, H, W)
                M = next(iter(blocks.values())).shape[0]
                for m in range(M):
                    yield gid, {n: blocks[n][m] for n in names}  # each (nf, H, W)
                    gid += 1

    def truth_fields(self, names: Sequence[str]) -> Dict[str, np.ndarray]:
        """``{name: (n_frames, H, W)}`` native-ERA5 truth for the given channels.

        Read once from the first dump (truth is the shared reanalysis across the
        pooled files); this is makani's own resolution-matched baseline -- native
        0.25 deg, no HPX round trip.
        """
        fp = self._per_file[0]
        out: Dict[str, np.ndarray] = {}
        with h5py.File(fp["path"], "r") as f:
            for n in names:
                out[n] = self._read_truth_forward(f, fp, self._channel_index(n))
        return out

    def assert_ic_pinned(self, name: str, atol: float = 0.0) -> None:
        """Check the pinned frame equals truth on ONE file (cheap; alignment is
        deterministic, so one file suffices).  makani builds the rollout FROM truth
        (frame 0 pred == truth); our forward flip lands that at forward frame -1,
        mirroring cBottle's end-conditioning, so a mismatch means the alignment is
        off."""
        pred0, truth0 = self.sample_pinned_frame(name)
        target = np.broadcast_to(truth0, pred0.shape)
        ok = (np.allclose(pred0, target, atol=atol) if atol
              else np.array_equal(pred0, target))
        if not ok:
            raise AssertionError(
                f"pinned-frame check failed for '{name}': the present frame is not "
                "equal to truth across members -- frame alignment is off."
            )

    def summary(self) -> Dict:
        return {
            "n_files": len(self.files),
            "n_members": self.n_members,
            "n_frames": self.n_frames,
            "channels": self.channels,
            "grid": (int(self.lat.shape[0]), int(self.lon.shape[0])),
            "lat_range": (float(self.lat[0]), float(self.lat[-1])),
            "lon_range": (float(self.lon[0]), float(self.lon[-1])),
            "anchor_timestamp": self.anchor_timestamp,
            "lead_hours_forward": self.lead_hours.tolist(),
            "step_hours": self.step_hours,
            "default_channels_present": [
                c for c in COMPARISON_CHANNELS if c in self.channels
            ],
            "default_channels_missing": [
                c for c in COMPARISON_CHANNELS if c not in self.channels
            ],
        }
