#!/usr/bin/env python3
"""Tests for Task 7: VAD Boundary Parity with Audited Stage 5B.1 Implementation."""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vad_adapter_remediated import (
    binarize_posteriors_audited_reference,
    canonical_interval_hash,
)

# Reference function directly imported from historical Stage 5B.1 script
STAGE5B_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(STAGE5B_DIR / "stage5b_1_vad_optimization" / "scripts"))
from run_stage5b_1_experiments import binarize_posteriors as binarize_stage5b1_historical


def test_vad_parity_on_standard_case():
    times = np.linspace(0.0, 5.0, 500)
    probs = np.zeros(500, dtype=np.float32)
    # Turn 1: 0.5s to 1.5s
    probs[50:150] = 0.85
    # Turn 2: 2.0s to 3.0s
    probs[200:300] = 0.90

    res_audited = binarize_posteriors_audited_reference(times, probs, onset=0.481, offset=0.481, min_duration_on=0.10, min_duration_off=0.15)
    res_historical = binarize_stage5b1_historical(times, probs, onset=0.481, offset=0.481, min_duration_on=0.10, min_duration_off=0.15)

    assert len(res_audited) == len(res_historical)
    for a, h in zip(res_audited, res_historical):
        assert a["start"] == pytest.approx(h["start"], abs=1e-3)
        assert a["end"] == pytest.approx(h["end"], abs=1e-3)


def test_vad_parity_first_frame_active():
    # Speech active at frame 0
    times = np.linspace(0.014875, 2.0, 100)
    probs = np.zeros(100, dtype=np.float32)
    probs[:30] = 0.95  # Active from start to ~0.6s

    res_audited = binarize_posteriors_audited_reference(times, probs, onset=0.481, offset=0.481)
    res_historical = binarize_stage5b1_historical(times, probs, onset=0.481, offset=0.481)

    assert len(res_audited) == 1
    assert len(res_historical) == 1
    assert res_audited[0]["start"] == pytest.approx(res_historical[0]["start"], abs=1e-3)
    assert res_audited[0]["start"] == pytest.approx(times[0], abs=1e-3)


def test_vad_parity_last_frame_active():
    # Speech active at final frame
    times = np.linspace(0.0, 2.0, 100)
    probs = np.zeros(100, dtype=np.float32)
    probs[70:] = 0.95

    res_audited = binarize_posteriors_audited_reference(times, probs, onset=0.481, offset=0.481)
    res_historical = binarize_stage5b1_historical(times, probs, onset=0.481, offset=0.481)

    assert len(res_audited) == 1
    assert res_audited[0]["end"] == pytest.approx(times[-1], abs=1e-3)
    assert res_audited[0]["end"] == pytest.approx(res_historical[0]["end"], abs=1e-3)


def test_vad_parity_gap_bridging_thresholds():
    times = np.linspace(0.0, 3.0, 300)
    probs = np.zeros(300, dtype=np.float32)
    # Burst 1: 0.5s to 1.0s (indices 50..100)
    probs[50:100] = 0.90
    # Burst 2: gap of 0.10s (indices 110..160) -> gap is 0.10s < min_duration_off (0.15s)
    probs[110:160] = 0.90

    res_bridged = binarize_posteriors_audited_reference(times, probs, min_duration_off=0.15)
    # Gaps < 0.15s must be bridged into a single interval
    assert len(res_bridged) == 1
    assert res_bridged[0]["start"] == pytest.approx(times[50], abs=1e-3)
    assert res_bridged[0]["end"] == pytest.approx(times[160], abs=1e-3)
