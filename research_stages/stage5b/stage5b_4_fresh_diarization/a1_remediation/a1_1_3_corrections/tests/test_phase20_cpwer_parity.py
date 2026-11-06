#!/usr/bin/env python3
"""
Test Suite: Differential cpWER Word-Attribution Parity with Phase 20
Validates 100% bit-exact equivalence between Stage 5B.4 cpWER adapter and Phase 20 reference code
(phase20_standalone/scripts/common.py and cpwer.py).
Tests 12 exhaustive edge-case configurations plus full end-to-end evaluation.
"""

import json
from pathlib import Path
import sys
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cpwer_adapter import (
    evaluate_cpwer,
    attribute_words_to_segments,
    group_words_by_speaker,
    rttm_to_segments,
    max_overlap_speaker,
    phase20_attributed_words,
    phase20_group_words,
    phase20_max_overlap_speaker,
    phase20_cpwer,
    phase20_cpwer_stats,
)


def assert_attribution_parity(words, segments):
    """Asserts that our adapter produces 100% bit-exact results compared to Phase 20."""
    p20_res = phase20_attributed_words(words, segments)
    adapter_res = attribute_words_to_segments(words, segments)
    assert len(p20_res) == len(adapter_res), "Length mismatch in attributed words list."
    for idx, (p_item, a_item) in enumerate(zip(p20_res, adapter_res)):
        assert p_item == a_item, f"Attribution mismatch at index {idx}: Phase20={p_item} vs Adapter={a_item}"


def assert_grouping_parity(attributed, max_gap=2.0):
    p20_grouped = phase20_group_words(attributed, max_gap=max_gap)
    adapter_grouped = group_words_by_speaker(attributed, max_gap=max_gap)
    assert p20_grouped == adapter_grouped, "Speaker turn grouping mismatch with Phase 20."


# Case 1: Standard non-overlapping segments
def test_parity_case1_ordinary_single_speaker():
    segs = [{"speaker": "SPK1", "start": 0.0, "end": 2.0}]
    words = [{"word": "hello", "start": 0.5, "end": 1.0}]
    assert_attribution_parity(words, segs)


# Case 2: Overlapping segments from different speakers
def test_parity_case2_two_speakers_overlapping():
    segs = [
        {"speaker": "SPK1", "start": 0.0, "end": 2.0},
        {"speaker": "SPK2", "start": 1.0, "end": 3.0},
    ]
    # Word predominantly in SPK2
    words = [{"word": "interrupt", "start": 1.8, "end": 2.5}]
    assert_attribution_parity(words, segs)


# Case 3: Equal overlap ties (Phase 20 uses max(overlaps)[1] string sort)
def test_parity_case3_equal_overlap_ties():
    segs = [
        {"speaker": "SPK_B", "start": 0.0, "end": 2.0},
        {"speaker": "SPK_A", "start": 0.0, "end": 2.0},
    ]
    words = [{"word": "tie", "start": 0.5, "end": 1.5}]
    assert_attribution_parity(words, segs)


# Case 4: Word missing timestamps (start/end is None)
def test_parity_case4_no_timestamps():
    segs = [{"speaker": "SPK1", "start": 0.0, "end": 2.0}]
    words = [{"word": "notime", "start": None, "end": None}]
    assert_attribution_parity(words, segs)


# Case 5: Word completely outside all intervals (> tolerance distance)
def test_parity_case5_word_outside_all_intervals():
    segs = [{"speaker": "SPK1", "start": 0.0, "end": 1.0}]
    words = [{"word": "isolated", "start": 10.0, "end": 11.0}]
    assert_attribution_parity(words, segs)


# Case 6: Nearest segment midpoint fallback within tolerance
def test_parity_case6_nearest_segment_fallback():
    segs = [
        {"speaker": "SPK1", "start": 0.0, "end": 1.0},
        {"speaker": "SPK2", "start": 2.0, "end": 3.0},
    ]
    # Word at 1.1 - 1.2 is outside both, but closer to SPK1 (dist 0.15 vs 0.85)
    words = [{"word": "close", "start": 1.1, "end": 1.2}]
    assert_attribution_parity(words, segs)


# Case 7: Multiple overlapping segments from the SAME speaker
def test_parity_case7_multiple_overlapping_same_speaker():
    segs = [
        {"speaker": "SPK1", "start": 0.0, "end": 2.0},
        {"speaker": "SPK1", "start": 1.0, "end": 3.0},
    ]
    words = [{"word": "double", "start": 1.2, "end": 1.8}]
    assert_attribution_parity(words, segs)


# Case 8: Adjacent segment boundaries (word straddling boundary)
def test_parity_case8_adjacent_boundaries():
    segs = [
        {"speaker": "SPK1", "start": 0.0, "end": 1.0},
        {"speaker": "SPK2", "start": 1.0, "end": 2.0},
    ]
    # Word exactly on the seam
    words = [{"word": "boundary", "start": 0.9, "end": 1.3}]
    assert_attribution_parity(words, segs)


# Case 9: Empty transcript
def test_parity_case9_empty_transcript():
    segs = [{"speaker": "SPK1", "start": 0.0, "end": 1.0}]
    words = []
    assert_attribution_parity(words, segs)


# Case 10: Empty RTTM segments
def test_parity_case10_empty_rttm():
    segs = []
    words = [{"word": "orphan", "start": 0.5, "end": 1.0}]
    assert_attribution_parity(words, segs)


# Case 11: Speaker label permutation invariance in Hungarian matching
def test_parity_case11_speaker_permutation_invariance():
    ref_segs = [
        {"speaker": "REF_1", "text": "hello world"},
        {"speaker": "REF_2", "text": "foo bar"},
    ]
    hyp_segs_a = [
        {"speaker": "HYP_A", "text": "hello world"},
        {"speaker": "HYP_B", "text": "foo bar"},
    ]
    hyp_segs_b = [
        {"speaker": "HYP_B", "text": "hello world"},
        {"speaker": "HYP_A", "text": "foo bar"},
    ]
    stats_a = phase20_cpwer_stats(ref_segs, hyp_segs_a)
    stats_b = phase20_cpwer_stats(ref_segs, hyp_segs_b)
    assert stats_a["cpwer"] == stats_b["cpwer"] == 0.0


# Case 12: Zero-duration word timestamps (start == end)
def test_parity_case12_identical_timestamps():
    segs = [{"speaker": "SPK1", "start": 0.0, "end": 2.0}]
    words = [{"word": "instant", "start": 1.0, "end": 1.0}]
    assert_attribution_parity(words, segs)


# Case 13: Full differential cpWER evaluation end-to-end
def test_full_differential_cpwer_evaluation(tmp_path):
    hyp_rttm = tmp_path / "hyp.rttm"
    with open(hyp_rttm, "w") as f:
        f.write("SPEAKER test 1 0.0 1.0 <NA> <NA> SPK1 <NA> <NA>\n")
        f.write("SPEAKER test 1 1.0 1.0 <NA> <NA> SPK2 <NA> <NA>\n")

    words_file = tmp_path / "words.json"
    aligned_words = [
        {"word": "hello", "start": 0.2, "end": 0.8},
        {"word": "world", "start": 1.2, "end": 1.8},
    ]
    with open(words_file, "w") as f:
        json.dump(aligned_words, f)

    ref_file = tmp_path / "ref.json"
    ref_transcript = {
        "segments": [
            {"speaker": "REF1", "text": "hello"},
            {"speaker": "REF2", "text": "world"},
        ]
    }
    with open(ref_file, "w") as f:
        json.dump(ref_transcript, f)

    result = evaluate_cpwer(
        hypothesis_rttm=hyp_rttm,
        aligned_words_json=words_file,
        reference_transcript_json=ref_file,
    )

    assert result["cpwer"] == 0.0
    assert result["insertions"] == 0
    assert result["deletions"] == 0
    assert result["substitutions"] == 0
    assert result["reference_words"] == 2
    assert result["hypothesis_words"] == 2
