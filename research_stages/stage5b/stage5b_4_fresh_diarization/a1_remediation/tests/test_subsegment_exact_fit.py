#!/usr/bin/env python3
"""Tests for Task 4: Fix Exact-Fit Subsegment Duplication."""

import pytest
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from subsegment_generator_remediated import (
    generate_subsegments_for_interval,
    generate_all_scales,
)


def test_exact_fit_no_duplication():
    """
    CRITICAL AUDIT CASE:
    Interval [0.0, 2.5] with win=0.50s, shift=0.25s.
    Expected: Exactly 9 distinct subsegments:
    [0.00, 0.50], [0.25, 0.75], [0.50, 1.00], [0.75, 1.25],
    [1.00, 1.50], [1.25, 1.75], [1.50, 2.00], [1.75, 2.25], [2.00, 2.50]
    MUST NOT DUPLICATE [2.00, 2.50]!
    """
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=2.5,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 9

    # Verify uniqueness of (start, end)
    boundaries = [(s["start"], s["end"]) for s in subsegs]
    assert len(set(boundaries)) == 9
    assert boundaries[-1] == (2.00, 2.50)
    assert boundaries[-2] == (1.75, 2.25)


def test_exact_multiple_of_window():
    # Interval [0.0, 1.0] with win=0.50s, shift=0.25s
    # 0.0-0.5, 0.25-0.75, 0.5-1.0 -> exactly 3 subsegments
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=1.0,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 3
    assert [(s["start"], s["end"]) for s in subsegs] == [
        (0.0, 0.5), (0.25, 0.75), (0.5, 1.0)
    ]


def test_exact_multiple_of_shift():
    # Interval [0.0, 0.75] with win=0.50s, shift=0.25s
    # 0.0-0.5, 0.25-0.75 -> exactly 2 subsegments
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=0.75,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 2
    assert [(s["start"], s["end"]) for s in subsegs] == [(0.0, 0.5), (0.25, 0.75)]


def test_trailing_remainder_thresholds():
    # Remainder < 0.05s: Interval [0.0, 0.53], win=0.50, shift=0.50
    # slice_st=0.0 -> [0.0, 0.50], slice_st becomes 0.50. Remainder = 0.03s (< 0.05s).
    # Remainder should be ignored!
    subsegs_short = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=0.53,
        window=0.50,
        shift=0.50,
    )
    assert len(subsegs_short) == 1
    assert (subsegs_short[0]["start"], subsegs_short[0]["end"]) == (0.0, 0.50)

    # Remainder >= 0.05s: Interval [0.0, 0.55], win=0.50, shift=0.50
    # slice_st=0.0 -> [0.0, 0.50], slice_st becomes 0.50. Remainder = 0.05s (>= 0.05s).
    # Remainder clamped: [0.05, 0.55]
    subsegs_long = generate_subsegments_for_interval(
        parent_idx=0,
        start=0.0,
        end=0.55,
        window=0.50,
        shift=0.50,
    )
    assert len(subsegs_long) == 2
    assert (subsegs_long[1]["start"], subsegs_long[1]["end"]) == (0.05, 0.55)


def test_interval_shorter_than_window():
    subsegs = generate_subsegments_for_interval(
        parent_idx=0,
        start=1.2,
        end=1.5,
        window=0.50,
        shift=0.25,
    )
    assert len(subsegs) == 1
    assert subsegs[0]["start"] == 1.2
    assert subsegs[0]["end"] == 1.5
    assert subsegs[0]["duration"] == 0.3


def test_zero_or_negative_duration():
    assert generate_subsegments_for_interval(0, 1.0, 1.0, 0.5, 0.25) == []
    assert generate_subsegments_for_interval(0, 2.0, 1.0, 0.5, 0.25) == []


def test_multiscale_uniqueness():
    mock_vad = [
        {"start": 0.0, "end": 2.5, "duration": 2.5},
        {"start": 3.0, "end": 5.0, "duration": 2.0},
    ]
    all_scales = generate_all_scales(mock_vad)
    for scale_k, subsegs in all_scales.items():
        bounds = [(s["start"], s["end"]) for s in subsegs]
        assert len(bounds) == len(set(bounds)), f"Duplicate subsegment found at scale {scale_k}!"
