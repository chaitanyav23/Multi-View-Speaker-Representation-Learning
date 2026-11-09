#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Strict Overlap Posterior Loader & Multi-State Validator
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
        validation_errors=errors,
    )


def validate_posterior_values_strict(p_overlap: np.ndarray, n_expected_frames: int) -> None:
    """
    Validates posterior array values:
    1. 1-dimensional and matches n_expected_frames.
    2. Finite (no NaN or Inf).
    3. Values bounded strictly within [0.0, 1.0].
    """
    if not isinstance(p_overlap, np.ndarray):
        raise OverlapProvenanceError(f"p_overlap must be a numpy ndarray, got {type(p_overlap)}")

    if p_overlap.ndim != 1:
        raise OverlapProvenanceError(f"p_overlap must be 1-dimensional, got shape {p_overlap.shape}")

    if len(p_overlap) != n_expected_frames:
        raise OverlapProvenanceError(
            f"Length mismatch: p_overlap has {len(p_overlap)} frames, expected {n_expected_frames} timestamps."
        )

    if not np.all(np.isfinite(p_overlap)):
        raise OverlapProvenanceError("p_overlap array contains NaN or Infinite values.")

    min_val, max_val = float(np.min(p_overlap)), float(np.max(p_overlap))
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        raise OverlapProvenanceError(
            f"p_overlap values outside valid probability range [0, 1]: min={min_val:.6f}, max={max_val:.6f}"
        )


class OverlapPosteriorLoaderStrict:
    """
    Loads, parses, and validates Stage 2 overlap posteriors.
    Enforces multi-state validation separating acoustic validity from cryptographic provenance.
    """

    def __init__(
        self,
        posteriors_dir: Optional[Path] = None,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
        provenance_manifest_path: Optional[Path] = None,
        source_audio_root: Optional[Path] = None,
        pyannet_checkpoint_path: Optional[Path] = None,
    ):
        self.execution_mode = execution_mode
        self.posteriors_dir = Path(posteriors_dir).resolve() if posteriors_dir else get_stage5b_2_overlap_cache_dir()
        self.verifier = OverlapProvenanceVerifier(
            manifest_path=provenance_manifest_path,
            source_audio_root=source_audio_root,
            pyannet_checkpoint_path=pyannet_checkpoint_path,
        )

    def locate_posterior_file(self, recording_id: str, stream: str = "mono") -> Path:
        """
        Locates posterior file:
        Tries posteriors_dir first: {recording_id}_{stream}_overlap.npz,
        then fallback to stage2_baseline: {recording_id}_{stream}_pyannote_posteriors.npz.
        """
        # Primary: Stage 5B.2 overlap cache naming convention
        p1 = self.posteriors_dir / f"{recording_id}_{stream}_overlap.npz"
        if p1.exists():
            return p1

        # Fallback: Stage 2 baseline naming convention
        stage2_dir = get_stage2_posteriors_dir()
        p2 = stage2_dir / f"{recording_id}_{stream}_pyannote_posteriors.npz"
        if p2.exists():
            return p2

        return p1

    def load_and_validate(
        self,
        recording_id: str,
        stream: str = "mono",
        expected_duration: Optional[float] = None,
        override_arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        is_synthetic: bool = False,
        source_audio_override: Optional[Path] = None,
        pyannet_checkpoint_override: Optional[Path] = None,
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

            data = np.load(str(npz_path))
            if "frame_times" in data:
                frame_times = data["frame_times"]
            else:
                raise OverlapProvenanceError(f"Missing 'frame_times' key in {npz_path}")

            if "p_overlap" in data:
                raw_p_overlap = data["p_overlap"]
            elif "probabilities" in data:
                # Stage 2 posteriors: multi-label speech probabilities [T, 3] -> continuous overlap top2
                probs = data["probabilities"]
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    sorted_p = np.sort(probs, axis=1)
                    raw_p_overlap = sorted_p[:, -2]  # Second highest speaker probability
                else:
                    raise OverlapProvenanceError(f"Unexpected shape for probabilities array: {probs.shape}")
            else:
                raise OverlapProvenanceError(f"No recognized overlap posterior array key in {npz_path}")

        # 1. Structural Acoustic Validation
        s_state = validate_frame_timestamps_strict(frame_times, expected_duration=expected_duration)
        validate_posterior_values_strict(raw_p_overlap, len(frame_times))
        p_overlap = raw_p_overlap.astype(np.float32)

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

        # 2. Independent Provenance Verification (Findings 2 and 3)
        p_state = self.verifier.verify_provenance(
            recording_id=recording_id,
            stream_name=stream,
            posterior_path=npz_path,
            execution_mode=self.execution_mode,
            is_synthetic=is_synth,
            source_audio_override=source_audio_override,
            pyannet_checkpoint_override=pyannet_checkpoint_override,
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
