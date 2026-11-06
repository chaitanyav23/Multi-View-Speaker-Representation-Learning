#!/usr/bin/env python3
"""Tests for Stage 5B.4 Dynamic Subsegment Adapter."""

import pytest
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from subsegment_adapter import (
    SCALES,
    generate_subsegments_for_interval,
    generate_multiscale_subsegments,
    generate_all_scales,
)


def test_scale_definitions():
    assert len(SCALES) == 5
    assert SCALES[0] == (0, 1.75, 0.50, 0.225)
    assert SCALES[1] == (1, 1.50, 0.50, 0.225)
    assert SCALES[2] == (2, 1.25, 0.50, 0.225)
    assert SCALES[3] == (3, 1.00, 0.50, 0.225)
    assert SCALES[4] == (4, 0.50, 0.25, 0.100)  # Base scale


def test_short_interval_subsegmentation():
    # An interval of 0.30s is shorter than window of 0.50s
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=10.0,
        end=10.3,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 1
    assert subsegs[0]["start"] == 10.0
    assert subsegs[0]["end"] == 10.3
    assert subsegs[0]["duration"] == 0.3


def test_sliding_window_with_trailing_remainder():
    # Interval of 1.35s:
    # win=0.50, shift=0.25:
    # 0.00-0.50, 0.25-0.75, 0.50-1.00, 0.75-1.25 -> slice_st=1.00, end_time=1.35.
    # end_time - slice_st = 0.35 >= 0.05 -> trailing subsegment: [1.35-0.50, 1.35] = [0.85, 1.35]
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=1.35,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 5
    assert subsegs[-1]["start"] == 0.85
    assert subsegs[-1]["end"] == 1.35


def test_generate_all_scales_coverage():
    mock_vad = [
        {"start": 1.0, "end": 4.0, "duration": 3.0},
        {"start": 5.0, "end": 7.0, "duration": 2.0},
    ]
    all_scales = generate_all_scales(mock_vad)
    assert len(all_scales) == 5
    for k in range(5):
        assert len(all_scales[k]) > 0
        # All subsegments must have scale attribute and parent_interval_idx
        for sub in all_scales[k]:
            assert sub["scale"] == k
            assert sub["parent_interval_idx"] in [0, 1]
            assert sub["start"] >= 1.0
            assert sub["end"] <= 7.0
