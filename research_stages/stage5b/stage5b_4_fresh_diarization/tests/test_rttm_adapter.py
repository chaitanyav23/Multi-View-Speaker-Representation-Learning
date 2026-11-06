#!/usr/bin/env python3
"""Tests for Stage 5B.4 RTTM Adapter."""

import pytest
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rttm_adapter import (
    subsegments_to_single_speaker_segments,
    write_rttm_file,
    read_rttm_file,
    rasterize_segments_to_frame_spk,
)


def test_subsegments_to_single_speaker_segments_merging():
    subsegs = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.25, "end": 0.75},  # Overlaps with prev, same speaker
        {"start": 1.0, "end": 1.5},    # New speaker
        {"start": 1.25, "end": 1.75},  # Same as spk1
    ]
    labels = ["spk0", "spk0", "spk1", "spk1"]

    merged = subsegments_to_single_speaker_segments(subsegs, labels)
    assert len(merged) == 2
    assert merged[0]["speaker"] == "spk0"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 0.75

    assert merged[1]["speaker"] == "spk1"
    assert merged[1]["start"] == 1.0
    assert merged[1]["end"] == 1.75


def test_rttm_file_roundtrip(tmp_path):
    rttm_file = tmp_path / "test.rttm"
    segs = [
        {"start": 0.50, "end": 2.50, "duration": 2.0, "speaker": "spk0"},
        {"start": 3.00, "end": 4.50, "duration": 1.5, "speaker": "spk1"},
    ]
    write_rttm_file(segs, "R01_test", rttm_file)
    assert rttm_file.exists()

    loaded = read_rttm_file(rttm_file)
    assert len(loaded) == 2
    assert loaded[0]["speaker"] == "spk0"
    assert loaded[0]["start"] == 0.50
    assert loaded[0]["end"] == 2.50
    assert loaded[1]["speaker"] == "spk1"


def test_rasterize_segments_to_frame_spk():
    segs = [
        {"start": 0.10, "end": 0.30, "speaker": "spk0"},
        {"start": 0.50, "end": 0.70, "speaker": "spk1"},
    ]
    frame_spk = rasterize_segments_to_frame_spk(segs, total_duration_sec=1.0, resolution=0.01)
    assert len(frame_spk) == 100
    assert frame_spk[0] is None
    assert frame_spk[15] == "spk0"
    assert frame_spk[40] is None
    assert frame_spk[60] == "spk1"
    assert frame_spk[80] is None
