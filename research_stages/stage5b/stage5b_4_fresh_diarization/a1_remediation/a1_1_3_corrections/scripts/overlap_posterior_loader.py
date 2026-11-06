#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3 Remediation: Strict Overlap Posterior Loader & Multi-State Validator
Enforces Pyannote 3.1 PyanNet acoustic properties (16.875ms stride, 0.03096875s centering).
Coordinates structural validation and independent provenance verification.
Returns an immutable ValidatedPosteriorContainer carrying explicit multi-state validation:
- structurally_valid
- provenance_verified
- production_eligible
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from path_utils import get_stage5b_2_overlap_cache_dir, get_stage2_posteriors_dir
from checkpoint_preflight import ExecutionMode
from overlap_provenance import (
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_SAMPLE_RATE,
    OverlapProvenanceError,
    StructuralValidationState,
    ProvenanceValidationState,
    OverallValidationState,
    ValidatedPosteriorContainer,
    OverlapProvenanceVerifier,
)


def validate_frame_timestamps_strict(
    frame_times: np.ndarray,
    expected_duration: Optional[float] = None,
    expected_step_sec: float = PYANNET_FRAME_STEP_SEC,
    expected_first_center_sec: float = PYANNET_FIRST_FRAME_CENTER,
    step_tolerance: float = 1e-4,
    center_tolerance: float = 2e-3,
) -> StructuralValidationState:
    """
    Strictly validates acoustic frame timestamps:
    1. 1-dimensional and non-empty.
    2. Finite values (no NaN or Inf).
    3. Strictly monotonic increasing (no duplicates, no reversals).
    4. Constant stride matching expected_step_sec within step_tolerance.
    5. First frame center matching expected_first_center_sec within center_tolerance.
    6. Duration / frame count consistency.
    """
    errors: List[str] = []

    if not isinstance(frame_times, np.ndarray):
        raise OverlapProvenanceError(f"frame_times must be a numpy ndarray, got {type(frame_times)}")

    if frame_times.ndim != 1:
        raise OverlapProvenanceError(f"frame_times must be 1-dimensional, got shape {frame_times.shape}")

    n_frames = len(frame_times)
    if n_frames == 0:
        raise OverlapProvenanceError("frame_times array is empty.")

    if not np.all(np.isfinite(frame_times)):
        raise OverlapProvenanceError("frame_times contains NaN or Infinite values.")

    # 1. Monotonic increasing check
    diffs = np.diff(frame_times)
    if not np.all(diffs > 0):
        bad_idx = int(np.where(diffs <= 0)[0][0])
        raise OverlapProvenanceError(
            f"frame_times is not strictly increasing at index {bad_idx}: "
            f"t[{bad_idx}]={frame_times[bad_idx]:.6f} >= t[{bad_idx+1}]={frame_times[bad_idx+1]:.6f}"
        )

    # 2. Stride consistency check
    max_step_err = float(np.max(np.abs(diffs - expected_step_sec)))
    if max_step_err > step_tolerance:
        raise OverlapProvenanceError(
            f"frame_times stride error exceeds tolerance {step_tolerance:.6f}s: "
            f"max_err={max_step_err:.6f}s (expected step={expected_step_sec:.6f}s)"
        )

    # 3. First frame center anchoring check
    first_t = float(frame_times[0])
    first_err = abs(first_t - expected_first_center_sec)
    if first_err > center_tolerance:
        raise OverlapProvenanceError(
            f"First frame timestamp {first_t:.6f}s does not match expected center "
            f"{expected_first_center_sec:.6f}s (err={first_err:.6f}s > {center_tolerance:.6f}s)"
        )

    # 4. Audio duration and last frame consistency check
    if expected_duration is not None and expected_duration > 0:
        last_t = float(frame_times[-1])
        if last_t > expected_duration + 0.10:
            raise OverlapProvenanceError(
                f"Last frame timestamp {last_t:.3f}s exceeds audio duration {expected_duration:.3f}s"
            )
        expected_frames = int(round(expected_duration / expected_step_sec))
        frame_diff = abs(n_frames - expected_frames)
        if frame_diff > 10:
            raise OverlapProvenanceError(
                f"Frame count mismatch: got {n_frames} frames, expected ~{expected_frames} for duration {expected_duration:.2f}s"
            )

    return StructuralValidationState(
        structurally_valid=True,
        first_frame_sec=first_t,
        last_frame_sec=float(frame_times[-1]),
        frame_step_sec=float(np.mean(diffs)),
        frame_count=n_frames,
        min_probability=0.0,
        max_probability=1.0,
        validation_errors=[],
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
        raise OverlapProvenanceError(f"p_overlap must be a numpy ndarray, got {type(p_overlap)}")

    p_arr = p_overlap
    if p_arr.ndim == 2 and p_arr.shape[1] == 1:
        p_arr = p_arr.squeeze(axis=1)

    if p_arr.ndim != 1:
        raise OverlapProvenanceError(f"p_overlap must be 1-dimensional, got shape {p_overlap.shape}")

    if len(p_arr) != expected_length:
        raise OverlapProvenanceError(
            f"Length mismatch: p_overlap has {len(p_arr)} items, expected {expected_length}"
        )

    if not np.all(np.isfinite(p_arr)):
        raise OverlapProvenanceError("p_overlap contains NaN or Infinite values.")

    min_p = float(np.min(p_arr))
    max_p = float(np.max(p_arr))
    if min_p < -1e-4 or max_p > 1.0 + 1e-4:
        raise OverlapProvenanceError(
            f"p_overlap values outside valid probability range [0.0, 1.0]: min={min_p:.6f}, max={max_p:.6f}"
        )

    return np.clip(p_arr, 0.0, 1.0).astype(np.float32)


class OverlapPosteriorLoaderStrict:
    """
    Strict loader for overlap posteriors coordinating structural validation
    and cryptographic provenance verification.
    """

    def __init__(
        self,
        posteriors_dir: Optional[Path] = None,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
        provenance_manifest_path: Optional[Path] = None,
    ):
        self.execution_mode = execution_mode
        self.posteriors_dir = Path(posteriors_dir or get_stage5b_2_overlap_cache_dir()).resolve()
        self.verifier = OverlapProvenanceVerifier(manifest_path=provenance_manifest_path)

    def locate_posterior_file(self, rec_id: str, stream: str = "mono") -> Path:
        """Locates the overlap posterior file on disk."""
        candidate = self.posteriors_dir / f"{rec_id}_{stream}_overlap.npz"
        if candidate.exists():
            return candidate

        candidate2 = self.posteriors_dir / f"{rec_id}_16k_{stream}_posteriors.npz"
        if candidate2.exists():
            return candidate2

        # Fallback to stage2 baseline directory if in stage5b_2
        s2_dir = get_stage2_posteriors_dir().resolve()
        candidate3 = s2_dir / f"{rec_id}_{stream}_posteriors.npz"
        if candidate3.exists():
            return candidate3

        candidate4 = s2_dir / f"{rec_id}_16k_{stream}_posteriors.npz"
        if candidate4.exists():
            return candidate4

        return candidate

    def load_and_validate(
        self,
        recording_id: str,
        stream: str = "mono",
        expected_duration: Optional[float] = None,
        override_arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_synthetic: bool = False,
    ) -> ValidatedPosteriorContainer:
        """
        Loads and strictly validates posterior arrays and cryptographic provenance.
        Returns immutable ValidatedPosteriorContainer with multi-state validation.
        """
        reasons: List[str] = []
        is_synth = is_synthetic or (self.execution_mode == ExecutionMode.SYNTHETIC) or (override_arrays is not None)

        if override_arrays is not None:
            frame_times, raw_p_overlap = override_arrays
            npz_path = Path(f"/synthetic/{recording_id}_{stream}_overlap.npz")
        else:
            npz_path = self.locate_posterior_file(recording_id, stream=stream)
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Overlap posteriors file not found for recording '{recording_id}': {npz_path}"
                )
            try:
                data = np.load(str(npz_path))
            except Exception as e:
                raise OverlapProvenanceError(f"Corrupt posteriors NPZ file {npz_path}: {e}")

            if "frame_times" not in data:
                raise KeyError(f"NPZ file {npz_path} missing required key 'frame_times'")
            if "p_overlap" not in data:
                raise KeyError(f"NPZ file {npz_path} missing required key 'p_overlap'")

            frame_times = np.asarray(data["frame_times"], dtype=np.float64)
            raw_p_overlap = np.asarray(data["p_overlap"], dtype=np.float32)

        # 1. Structural Validation
        s_state = validate_frame_timestamps_strict(
            frame_times=frame_times,
            expected_duration=expected_duration,
        )
        p_overlap = validate_posterior_values_strict(
            p_overlap=raw_p_overlap,
            expected_length=len(frame_times),
        )

        # Update min and max probability in structural state
        s_state = StructuralValidationState(
            structurally_valid=s_state.structurally_valid,
            first_frame_sec=s_state.first_frame_sec,
            last_frame_sec=s_state.last_frame_sec,
            frame_step_sec=s_state.frame_step_sec,
            frame_count=s_state.frame_count,
            min_probability=float(np.min(p_overlap)),
            max_probability=float(np.max(p_overlap)),
            validation_errors=s_state.validation_errors,
        )

        # 2. Independent Provenance Verification
        p_state = self.verifier.verify_provenance(
            recording_id=recording_id,
            stream_name=stream,
            posterior_path=npz_path,
            execution_mode=self.execution_mode,
            is_synthetic=is_synth,
        )

        # 3. Multi-State Assessment
        structurally_valid = s_state.structurally_valid
        provenance_verified = p_state.provenance_verified

        if is_synth:
            production_eligible = False
            reasons.append("Synthetic posterior is never production-eligible.")
        elif not structurally_valid:
            production_eligible = False
            reasons.append("Acoustic structural validation failed.")
        elif not provenance_verified:
            production_eligible = False
            if p_state.unverified_fields:
                reasons.append(f"Unverified provenance fields: {p_state.unverified_fields}")
            if p_state.provenance_errors:
                reasons.extend(p_state.provenance_errors)
        else:
            production_eligible = True
            reasons.append("Structural and cryptographic provenance verified.")

        overall_state = OverallValidationState(
            structurally_valid=structurally_valid,
            provenance_verified=provenance_verified,
            production_eligible=production_eligible,
            is_synthetic=is_synth,
            reasons=reasons,
        )

        return ValidatedPosteriorContainer(
            frame_times=frame_times,
            p_overlap=p_overlap,
            structural_state=s_state,
            provenance_state=p_state,
            overall_state=overall_state,
            is_synthetic=is_synth,
        )
