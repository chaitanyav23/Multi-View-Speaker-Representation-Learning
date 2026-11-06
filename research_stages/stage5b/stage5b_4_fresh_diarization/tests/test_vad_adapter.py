#!/usr/bin/env python3
"""Tests for Stage 5B.4 VAD Adapter."""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vad_adapter import (
    binarize_posteriors_with_hysteresis,
    canonical_interval_hash,
    VADAdapter,
)


def test_canonical_interval_hash_determinism():
    intervals1 = [
        {"start": 1.0, "end": 2.5, "duration": 1.5},
        {"start": 3.0, "end": 4.0, "duration": 1.0},
    ]
    # Same intervals in reverse order
    intervals2 = [
        {"start": 3.0, "end": 4.0, "duration": 1.0},
        {"start": 1.0, "end": 2.5, "duration": 1.5},
    ]
    h1 = canonical_interval_hash("R01", "C00", intervals1)
    h2 = canonical_interval_hash("R01", "C00", intervals2)
    assert h1 == h2
    assert len(h1) == 64

    # Different intervals produce different hash
    intervals3 = [
        {"start": 1.0, "end": 2.5001, "duration": 1.5001},
        {"start": 3.0, "end": 4.0, "duration": 1.0},
    ]
    h3 = canonical_interval_hash("R01", "C00", intervals3)
    assert h1 != h3

    # Different VAD ID produces different hash
    h_c50 = canonical_interval_hash("R01", "C50", intervals1)
    assert h1 != h_c50


def test_binarize_hysteresis_logic():
    # Synthetic frame posteriors: 0.0s to 1.0s, 100 frames (10ms each)
    frame_times = np.linspace(0.01, 1.00, 100)
    posteriors = np.zeros(100, dtype=np.float32)

    # Burst 1: 0.10s to 0.15s (50ms - below min_on=0.10s)
    posteriors[10:15] = 0.90

    # Burst 2: 0.30s to 0.50s (200ms - above min_on)
    posteriors[30:50] = 0.90

    # Burst 3: 0.55s to 0.75s (200ms - gap from Burst 2 is 50ms, below min_off=0.15s)
    posteriors[55:75] = 0.90

    segs = binarize_posteriors_with_hysteresis(
        posteriors=posteriors,
        frame_times=frame_times,
        onset=0.50,
        offset=0.50,
        min_duration_on=0.10,
        min_duration_off=0.15,
    )

    # Burst 1 should be filtered out
    # Burst 2 and Burst 3 should be bridged because gap is 50ms < min_off=0.15s
    assert len(segs) == 1
    assert segs[0]["start"] < 0.32
    assert segs[0]["end"] > 0.73


def test_vad_adapter_mismatch_rejection(tmp_path):
    # Create mock posterior directory with misaligned timestamps
    post_dir = tmp_path / "posteriors"
    post_dir.mkdir()

    f_times_1 = np.array([0.01, 0.03, 0.05])
    f_times_2 = np.array([0.01, 0.04, 0.05])  # Misaligned
    p_dummy = np.array([0.1, 0.8, 0.2])

    np.savez(post_dir / "R01_test_dfn_posteriors.npz", frame_times=f_times_1, p_speech=p_dummy)
    np.savez(post_dir / "R01_test_16k_mono_posteriors.npz", frame_times=f_times_2, p_speech=p_dummy)

    adapter = VADAdapter(posteriors_dir=post_dir)
    with pytest.raises(ValueError, match="Timestamp alignment error"):
        adapter.generate_vad_intervals("R01_test", "C62")
