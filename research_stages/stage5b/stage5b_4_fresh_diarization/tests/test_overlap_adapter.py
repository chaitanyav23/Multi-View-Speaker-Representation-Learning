#!/usr/bin/env python3
"""Tests for Stage 5B.4 Overlap Attribution Adapter."""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from overlap_adapter import (
    extract_contiguous_turns,
    attribute_m4_conservative,
    detect_overlap_intervals,
    apply_overlap_attribution,
)


def test_extract_contiguous_turns():
    # Frame resolution: 0.01s (10ms)
    # Frames: [spk0]*30 (0.30s), [None]*10 (0.10s), [spk1]*40 (0.40s)
    frame_spk = ["spk0"] * 30 + [None] * 10 + ["spk1"] * 40
    # Search forward from index 0 to 80
    runs = extract_contiguous_turns(
        frame_spk, search_start_idx=0, search_end_idx=80, direction="forward", resolution=0.01
    )
    assert len(runs) == 2
    assert runs[0]["speaker"] == "spk0"
    assert runs[0]["duration_sec"] == pytest.approx(0.30)
    assert runs[1]["speaker"] == "spk1"
    assert runs[1]["duration_sec"] == pytest.approx(0.40)


def test_m4_turn_alternation_success():
    # Setup: spk0 for 0.5s, overlap from 0.5s to 0.7s (primary=spk0), spk1 for 0.5s
    # Preceding: spk0 (0.50s >= 0.20s), Subsequent: spk1 (0.50s >= 0.20s)
    # Primary: spk0 -> secondary should be spk1!
    frame_spk = ["spk0"] * 50 + ["spk0"] * 20 + ["spk1"] * 50

    res = attribute_m4_conservative(
        frame_spk=frame_spk,
        overlap_interval=(0.50, 0.70),
        primary_spk="spk0",
        resolution=0.01,
        max_gap_sec=5.0,
        min_turn_dur_sec=0.20,
    )
    assert res["speaker"] == "spk1"
    assert res["reason"] == "turn_transition_forward"


def test_m4_same_speaker_adjacent_abstention():
    # Setup: spk0 before overlap, overlap, and spk0 after overlap (speaker pause)
    frame_spk = ["spk0"] * 50 + ["spk0"] * 20 + ["spk0"] * 50

    res = attribute_m4_conservative(
        frame_spk=frame_spk,
        overlap_interval=(0.50, 0.70),
        primary_spk="spk0",
        resolution=0.01,
        max_gap_sec=5.0,
        min_turn_dur_sec=0.20,
    )
    # Must abstain because same speaker before and after
    assert res["speaker"] is None
    assert res["reason"] == "same_speaker_adjacent"


def test_m4_short_turn_abstention():
    # Setup: subsequent turn is only 0.10s (below min_turn_dur_sec=0.20s)
    frame_spk = ["spk0"] * 50 + ["spk0"] * 20 + ["spk1"] * 10 + [None] * 40

    res = attribute_m4_conservative(
        frame_spk=frame_spk,
        overlap_interval=(0.50, 0.70),
        primary_spk="spk0",
        resolution=0.01,
        max_gap_sec=5.0,
        min_turn_dur_sec=0.20,
    )
    # Must abstain due to insufficient context
    assert res["speaker"] is None
    assert res["reason"] == "insufficient_context"


def test_apply_overlap_m0_baseline():
    hyp_segs = [
        {"start": 0.0, "end": 2.0, "speaker": "spk0"},
        {"start": 2.0, "end": 4.0, "speaker": "spk1"},
    ]
    final_segs, stats = apply_overlap_attribution(
        single_speaker_segments=hyp_segs,
        p_overlap=None,
        frame_times=None,
        method_id="M0_no_overlap",
    )
    assert len(final_segs) == 2
    assert stats["secondary_attributed"] == 0
