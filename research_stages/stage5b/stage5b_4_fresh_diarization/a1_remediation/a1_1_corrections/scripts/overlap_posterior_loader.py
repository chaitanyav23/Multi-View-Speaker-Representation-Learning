#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Overlap Posterior Loader
Loads and strictly validates pyannote overlap posteriors from Stage 2 baseline.
Enforces shape consistency, timestamp alignment, and probability bounds [0.0, 1.0].
Guarantees explicit failure for M4 overlap conditions when posteriors are absent or corrupted,
while keeping M0 single-speaker baselines completely independent.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from path_utils import get_stage2_posteriors_dir


class OverlapPosteriorError(Exception):
    """Raised when overlap posterior data is corrupted, invalid, or misaligned."""
    pass


class OverlapPosteriorLoader:
    """
    Dedicated input adapter for loading frame-level overlap posteriors (p_overlap).
    Reads existing posteriors from stage2_baseline read-only.
    """

    def __init__(self, posteriors_dir: Optional[Path] = None):
        self.posteriors_dir = Path(posteriors_dir or get_stage2_posteriors_dir())

    def get_posterior_path(self, rec_id: str) -> Path:
        """Returns the expected path to the 16k_mono posteriors NPZ file."""
        return self.posteriors_dir / f"{rec_id}_16k_mono_posteriors.npz"

    def load_overlap_posteriors(
        self,
        rec_id: str,
        expected_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and validates (frame_times, p_overlap) for the specified recording ID.

        Raises:
            FileNotFoundError: If the posteriors file does not exist.
            KeyError: If 'p_overlap' or 'frame_times' keys are missing.
            OverlapPosteriorError: If dimensions mismatch or probabilities are invalid.
        """
        npz_path = self.get_posterior_path(rec_id)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Overlap posteriors file not found for recording '{rec_id}': {npz_path}"
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
        p_overlap = np.asarray(data["p_overlap"], dtype=np.float32)

        # 1. Dimensionality checks
        if frame_times.ndim != 1:
            raise OverlapPosteriorError(
                f"frame_times must be 1-dimensional, got shape {frame_times.shape}"
            )
        if p_overlap.ndim != 1:
            # Handle potential (N, 1) shape
            if p_overlap.ndim == 2 and p_overlap.shape[1] == 1:
                p_overlap = p_overlap.squeeze(axis=1)
            else:
                raise OverlapPosteriorError(
                    f"p_overlap must be 1-dimensional, got shape {p_overlap.shape}"
                )

        if len(frame_times) == 0:
            raise OverlapPosteriorError(f"Empty posteriors array in {npz_path}")

        if len(frame_times) != len(p_overlap):
            raise OverlapPosteriorError(
                f"Array length mismatch for '{rec_id}': "
                f"frame_times={len(frame_times)} != p_overlap={len(p_overlap)}"
            )

        # 2. Finite value checks
        if not np.all(np.isfinite(frame_times)):
            raise OverlapPosteriorError(f"frame_times contains NaN or Inf values in {npz_path}")
        if not np.all(np.isfinite(p_overlap)):
            raise OverlapPosteriorError(f"p_overlap contains NaN or Inf values in {npz_path}")

        # 3. Probability range validation [0.0, 1.0] with small epsilon tolerance
        min_p = float(np.min(p_overlap))
        max_p = float(np.max(p_overlap))
        if min_p < -1e-4 or max_p > 1.0 + 1e-4:
            raise OverlapPosteriorError(
                f"p_overlap values outside valid probability range [0.0, 1.0]: "
                f"min={min_p}, max={max_p} in {npz_path}"
            )

        # Clamp minor numerical float imprecision
        p_overlap = np.clip(p_overlap, 0.0, 1.0)

        # 4. Optional duration check
        if expected_duration is not None and expected_duration > 0:
            max_t = float(frame_times[-1])
            if max_t > expected_duration + 2.0:
                raise OverlapPosteriorError(
                    f"Frame timestamps extend beyond expected duration: "
                    f"last_frame={max_t:.2f}s > audio_duration={expected_duration:.2f}s"
                )

        return frame_times, p_overlap
