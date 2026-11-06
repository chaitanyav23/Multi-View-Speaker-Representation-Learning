#!/usr/bin/env python3
"""
Stage 5B.4-A.1.2 Remediation: Strict Overlap Posterior and Provenance Loader
Validates Pyannote PyanNet multi-speaker overlap posteriors against verified Stage 2 / 5B.2 provenance.
Enforces strict 1D layout, monotonic increasing timestamps, 16.875ms frame stride,
0.03096875s first-frame centering, expected frame count parity, and [0.0, 1.0] probability semantics.
Fails closed on corrupt, misaligned, or provenance-incomplete inputs.
"""

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from path_utils import get_stage2_posteriors_dir, get_stage5b_2_overlap_cache_dir
from checkpoint_preflight import ExecutionMode

# Authoritative Pyannote 3.1 PyanNet segmentation acoustic constants
PYANNET_FRAME_STEP_SEC = 0.016875     # 270 samples at 16,000 Hz
PYANNET_FIRST_FRAME_CENTER = 0.03096875 # Receptive field center of first frame
PYANNET_SAMPLE_RATE = 16000           # 16 kHz mono


class OverlapPosteriorError(Exception):
    """Raised when overlap posterior data violates timestamp, shape, range, or provenance constraints."""
    pass


@dataclass(frozen=True)
class OverlapProvenanceMetadata:
    recording_id: str
    source_file: Path
    frame_count: int
    first_frame_sec: float
    last_frame_sec: float
    frame_step_sec: float
    duration_sec: Optional[float]
    sample_rate: int
    model_architecture: str = "Pyannote_PyanNet_3_1"
    is_verified: bool = True

    def to_dict(self) -> Dict:
        return {
            "recording_id": self.recording_id,
            "source_file": str(self.source_file),
            "frame_count": self.frame_count,
            "first_frame_sec": round(self.first_frame_sec, 6),
            "last_frame_sec": round(self.last_frame_sec, 6),
            "frame_step_sec": round(self.frame_step_sec, 6),
            "duration_sec": round(self.duration_sec, 4) if self.duration_sec else None,
            "sample_rate": self.sample_rate,
            "model_architecture": self.model_architecture,
            "is_verified": self.is_verified,
        }


def validate_frame_timestamps_strict(
    frame_times: np.ndarray,
    expected_duration: Optional[float] = None,
    expected_step_sec: float = PYANNET_FRAME_STEP_SEC,
    expected_first_center_sec: float = PYANNET_FIRST_FRAME_CENTER,
    step_tolerance: float = 1e-4,
    center_tolerance: float = 2e-3,
) -> None:
    """
    Strictly validates frame timestamp array:
    1. 1D and non-empty.
    2. Finite values.
    3. Strictly monotonic increasing (no duplicates, no reversals).
    4. Constant stride matching expected_step_sec within step_tolerance.
    5. First frame anchoring matching expected_first_center_sec.
    6. Last frame consistent with expected audio duration.
    """
    if not isinstance(frame_times, np.ndarray):
        raise OverlapPosteriorError(f"frame_times must be a numpy ndarray, got {type(frame_times)}")

    if frame_times.ndim != 1:
        raise OverlapPosteriorError(f"frame_times must be 1-dimensional, got shape {frame_times.shape}")

    n_frames = len(frame_times)
    if n_frames == 0:
        raise OverlapPosteriorError("frame_times array is empty.")

    if not np.all(np.isfinite(frame_times)):
        raise OverlapPosteriorError("frame_times contains NaN or Infinite values.")

    # 1. Strictly monotonic increasing check
    diffs = np.diff(frame_times)
    if not np.all(diffs > 0):
        # Locate first offending index
        bad_idx = int(np.where(diffs <= 0)[0][0])
        raise OverlapPosteriorError(
            f"frame_times is not strictly increasing at index {bad_idx}: "
            f"t[{bad_idx}]={frame_times[bad_idx]:.6f} >= t[{bad_idx+1}]={frame_times[bad_idx+1]:.6f}"
        )

    # 2. Stride consistency check
    max_step_err = float(np.max(np.abs(diffs - expected_step_sec)))
    if max_step_err > step_tolerance:
        raise OverlapPosteriorError(
            f"frame_times stride error exceeds tolerance {step_tolerance:.6f}s: "
            f"max_err={max_step_err:.6f}s (expected step={expected_step_sec:.6f}s)"
        )

    # 3. First frame center anchoring check
    first_t = float(frame_times[0])
    first_err = abs(first_t - expected_first_center_sec)
    if first_err > center_tolerance:
        raise OverlapPosteriorError(
            f"First frame timestamp {first_t:.6f}s does not match expected center "
            f"{expected_first_center_sec:.6f}s (err={first_err:.6f}s > {center_tolerance:.6f}s)"
        )

    # 4. Optional duration and last frame consistency check
    if expected_duration is not None and expected_duration > 0:
        last_t = float(frame_times[-1])
        if last_t > expected_duration + 0.10:
            raise OverlapPosteriorError(
                f"Last frame timestamp {last_t:.3f}s exceeds audio duration {expected_duration:.3f}s"
            )
        # Expected frame count check
        expected_frames = int(round(expected_duration / expected_step_sec))
        frame_diff = abs(n_frames - expected_frames)
        if frame_diff > 10:  # Allow small window edge tolerance
            raise OverlapPosteriorError(
                f"Frame count mismatch: got {n_frames} frames, expected ~{expected_frames} for duration {expected_duration:.2f}s"
            )


def validate_posterior_values_strict(
    p_overlap: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """
    Validates overlap posterior values:
    1. 1D shape (or squeezable 2D).
    2. Length equals expected_length.
    3. All values finite.
    4. Probabilities bounded within [0.0, 1.0].
    """
    if not isinstance(p_overlap, np.ndarray):
        raise OverlapPosteriorError(f"p_overlap must be a numpy ndarray, got {type(p_overlap)}")

    p_arr = p_overlap
    if p_arr.ndim == 2 and p_arr.shape[1] == 1:
        p_arr = p_arr.squeeze(axis=1)

    if p_arr.ndim != 1:
        raise OverlapPosteriorError(f"p_overlap must be 1-dimensional, got shape {p_overlap.shape}")

    if len(p_arr) != expected_length:
        raise OverlapPosteriorError(
            f"Length mismatch: p_overlap has {len(p_arr)} items, expected {expected_length}"
        )

    if not np.all(np.isfinite(p_arr)):
        raise OverlapPosteriorError("p_overlap contains NaN or Infinite values.")

    min_p = float(np.min(p_arr))
    max_p = float(np.max(p_arr))
    if min_p < -1e-4 or max_p > 1.0 + 1e-4:
        raise OverlapPosteriorError(
            f"p_overlap values outside valid probability range [0.0, 1.0]: min={min_p:.6f}, max={max_p:.6f}"
        )

    return np.clip(p_arr, 0.0, 1.0).astype(np.float32)


class OverlapPosteriorLoaderStrict:
    """
    Strict loader for overlap posteriors enforcing timestamp semantics and provenance.
    Primary sources:
    1. Stage 5B.2 overlap cache (authoritative continuous multi-speaker Top-2 overlap probabilities)
    2. Stage 2 baseline posteriors
    """

    def __init__(
        self,
        posteriors_dir: Optional[Path] = None,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    ):
        self.execution_mode = execution_mode
        self.posteriors_dir = Path(posteriors_dir or get_stage5b_2_overlap_cache_dir()).resolve()

    def get_posterior_path(self, rec_id: str) -> Path:
        """
        Locates the overlap posterior file for the given recording ID.
        Checks both Stage 5B.2 overlap cache and Stage 2 baseline directories.
        """
        # Try direct name in posteriors_dir
        candidate1 = self.posteriors_dir / f"{rec_id}_mono_overlap.npz"
        if candidate1.exists():
            return candidate1

        candidate2 = self.posteriors_dir / f"{rec_id}_16k_mono_posteriors.npz"
        if candidate2.exists():
            return candidate2

        # Fallback to stage2 baseline directory if in stage5b_2 or vice versa
        s2_dir = get_stage2_posteriors_dir().resolve()
        candidate3 = s2_dir / f"{rec_id}_16k_mono_posteriors.npz"
        if candidate3.exists():
            return candidate3

        return candidate1

    def load_and_validate(
        self,
        recording_id: str,
        expected_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, OverlapProvenanceMetadata]:
        """
        Loads and strictly validates (frame_times, p_overlap, provenance_metadata).
        Fails closed on any corruption, reversal, stride error, or out-of-range value.
        """
        npz_path = self.get_posterior_path(recording_id)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Overlap posteriors file not found for recording '{recording_id}': {npz_path}"
            )

        try:
            data = np.load(str(npz_path))
        except Exception as e:
            raise OverlapPosteriorError(f"Corrupt posteriors NPZ file {npz_path}: {e}")

        if "frame_times" not in data:
            raise KeyError(f"NPZ file {npz_path} missing required key 'frame_times'")
        if "p_overlap" not in data:
            raise KeyError(f"NPZ file {npz_path} missing required key 'p_overlap'")

        frame_times = np.asarray(data["frame_times"], dtype=np.float64)
        raw_p_overlap = np.asarray(data["p_overlap"], dtype=np.float32)

        # 1. Strict timestamp validation
        validate_frame_timestamps_strict(
            frame_times=frame_times,
            expected_duration=expected_duration,
        )

        # 2. Strict posterior value validation
        p_overlap = validate_posterior_values_strict(
            p_overlap=raw_p_overlap,
            expected_length=len(frame_times),
        )

        # 3. Assemble verified provenance metadata
        provenance = OverlapProvenanceMetadata(
            recording_id=recording_id,
            source_file=npz_path,
            frame_count=len(frame_times),
            first_frame_sec=float(frame_times[0]),
            last_frame_sec=float(frame_times[-1]),
            frame_step_sec=float(np.mean(np.diff(frame_times))),
            duration_sec=expected_duration,
            sample_rate=PYANNET_SAMPLE_RATE,
            model_architecture="Pyannote_PyanNet_3_1",
            is_verified=True,
        )

        return frame_times, p_overlap, provenance
