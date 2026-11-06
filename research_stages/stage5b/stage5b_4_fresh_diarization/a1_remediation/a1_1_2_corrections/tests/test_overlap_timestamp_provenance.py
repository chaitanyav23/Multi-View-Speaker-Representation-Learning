#!/usr/bin/env python3
"""
Test Suite: Strict Overlap Timestamp Validation and Provenance
Tests the rigorous validation of frame timestamps, step stride (16.875ms),
first-frame centering (0.03096875s), probability bounds [0, 1], and provenance tracking.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from overlap_posterior_loader import (
    validate_frame_timestamps_strict,
    validate_posterior_values_strict,
    OverlapPosteriorLoaderStrict,
    OverlapPosteriorError,
    OverlapProvenanceMetadata,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
)
from checkpoint_preflight import ExecutionMode


def make_valid_timestamps(n_frames: int = 100) -> np.ndarray:
    return PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC


def test_correct_ordered_timestamps():
    """Verify strictly ordered timestamps matching PyanNet parameters pass validation."""
    times = make_valid_timestamps(500)
    # Expected duration roughly 500 * 0.016875 = 8.4375s
    validate_frame_timestamps_strict(times, expected_duration=8.4375)


def test_duplicate_timestamps_rejected():
    """Verify that duplicate timestamps (diff == 0) raise OverlapPosteriorError."""
    times = make_valid_timestamps(100)
    times[50] = times[49]  # diff is 0
    with pytest.raises(OverlapPosteriorError, match="not strictly increasing at index 49"):
        validate_frame_timestamps_strict(times)


def test_decreasing_timestamps_rejected():
    """Verify that non-monotonic / decreasing timestamps raise OverlapPosteriorError."""
    times = make_valid_timestamps(100)
    times[60] = times[59] - 0.01
    with pytest.raises(OverlapPosteriorError, match="not strictly increasing at index 59"):
        validate_frame_timestamps_strict(times)


def test_nan_and_inf_timestamps_rejected():
    """Verify that NaN or Inf in timestamps raise OverlapPosteriorError."""
    times = make_valid_timestamps(100)
    times[10] = np.nan
    with pytest.raises(OverlapPosteriorError, match="NaN or Infinite"):
        validate_frame_timestamps_strict(times)

    times = make_valid_timestamps(100)
    times[20] = np.inf
    with pytest.raises(OverlapPosteriorError, match="NaN or Infinite"):
        validate_frame_timestamps_strict(times)


def test_incorrect_stride_rejected():
    """Verify that invalid frame step stride (e.g. 10ms instead of 16.875ms) raises OverlapPosteriorError."""
    times = PYANNET_FIRST_FRAME_CENTER + np.arange(100, dtype=np.float64) * 0.010  # 10ms stride
    with pytest.raises(OverlapPosteriorError, match="stride error exceeds tolerance"):
        validate_frame_timestamps_strict(times)


def test_incorrect_first_frame_center_rejected():
    """Verify that wrong first-frame offset (e.g. 0.0s instead of ~0.03096875s) raises OverlapPosteriorError."""
    times = 0.0 + np.arange(100, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    with pytest.raises(OverlapPosteriorError, match="does not match expected center"):
        validate_frame_timestamps_strict(times)


def test_frame_count_duration_mismatch_rejected():
    """Verify that frame count incompatible with audio duration raises OverlapPosteriorError."""
    times = make_valid_timestamps(100)  # ~1.68s of audio
    with pytest.raises(OverlapPosteriorError, match="exceeds audio duration|Frame count mismatch"):
        # Claim duration is 100.0s (expected ~5925 frames, got only 100)
        validate_frame_timestamps_strict(times, expected_duration=100.0)


def test_posterior_length_mismatch_rejected():
    """Verify that length mismatch between posterior array and timestamps raises OverlapPosteriorError."""
    p_arr = np.zeros(90, dtype=np.float32)
    with pytest.raises(OverlapPosteriorError, match="Length mismatch"):
        validate_posterior_values_strict(p_arr, expected_length=100)


def test_out_of_range_probabilities_rejected():
    """Verify that probabilities < 0 or > 1 raise OverlapPosteriorError."""
    p_neg = np.full(100, -0.05, dtype=np.float32)
    with pytest.raises(OverlapPosteriorError, match="outside valid probability range"):
        validate_posterior_values_strict(p_neg, expected_length=100)

    p_pos = np.full(100, 1.05, dtype=np.float32)
    with pytest.raises(OverlapPosteriorError, match="outside valid probability range"):
        validate_posterior_values_strict(p_pos, expected_length=100)


def test_posterior_nan_inf_rejected():
    """Verify NaN or Inf in posterior probabilities raise OverlapPosteriorError."""
    p_nan = np.full(100, 0.5, dtype=np.float32)
    p_nan[5] = np.nan
    with pytest.raises(OverlapPosteriorError, match="NaN or Infinite"):
        validate_posterior_values_strict(p_nan, expected_length=100)


def test_loader_synthetic_fixture_with_provenance(tmp_path):
    """Verify OverlapPosteriorLoaderStrict loads and validates synthetic fixture with full provenance."""
    rec_id = "R01_synthetic"
    n_frames = 200
    times = make_valid_timestamps(n_frames)
    p_overlap = (np.sin(np.linspace(0, np.pi, n_frames)) ** 2).astype(np.float32)

    # Save fixture as .npz
    fixture_path = tmp_path / f"{rec_id}_mono_overlap.npz"
    np.savez_compressed(
        fixture_path,
        frame_times=times,
        p_overlap=p_overlap,
        duration_sec=float(times[-1] + PYANNET_FRAME_STEP_SEC / 2),
    )

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=tmp_path,
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    frame_times, p_ov, metadata = loader.load_and_validate(rec_id, expected_duration=float(times[-1] + PYANNET_FRAME_STEP_SEC / 2))

    assert isinstance(metadata, OverlapProvenanceMetadata)
    assert len(frame_times) == n_frames
    assert len(p_ov) == n_frames
    assert metadata.is_verified is True
    assert metadata.frame_count == n_frames
    assert abs(metadata.frame_step_sec - PYANNET_FRAME_STEP_SEC) < 1e-5
    assert metadata.model_architecture == "Pyannote_PyanNet_3_1"


def test_loader_missing_posterior_fails(tmp_path):
    """Verify loader raises FileNotFoundError if posterior file does not exist."""
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=tmp_path,
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    with pytest.raises(FileNotFoundError, match="Overlap posteriors file not found"):
        loader.load_and_validate("R99_nonexistent")
