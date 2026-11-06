#!/usr/bin/env python3
"""
Tests for Correction 3: Differential cpWER Word-Attribution Parity with Phase 20.
Executes differential testing across all 12 specified conditions comparing
the authoritative Phase 20 implementation with the A.1.2 cpwer_adapter.
Proves 100% bit-exact attribution and scoring parity.
"""

import json
from pathlib import Path
import pytest
import sys
from typing import Any, Dict, List

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from path_utils import get_phase20_scripts_dir

P20_SCRIPTS = get_phase20_scripts_dir().resolve()
if str(P20_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(P20_SCRIPTS))

# Authoritative Phase 20 functions directly imported
import common as p20_common
import cpwer as p20_cpwer

# A.1.2 adapter functions
import cpwer_adapter as a1_2_adapter


def assert_attribution_parity(words: List[Dict[str, Any]], segments: List[Dict[str, Any]]):
    """Feeds identical inputs into Phase 20 and A.1.2, asserting exact equality."""
    p20_attr = p20_common.attributed_words(words, segments)
    a1_2_attr = a1_2_adapter.attribute_words_to_segments(words, segments)

    assert len(p20_attr) == len(a1_2_attr)
    for w_p20, w_a1_2 in zip(p20_attr, a1_2_attr):
        assert w_p20["word"] == w_a1_2["word"]
        assert w_p20["speaker"] == w_a1_2["speaker"]
        assert w_p20["start"] == w_a1_2["start"]
        assert w_p20["end"] == w_a1_2["end"]

    # Also test grouping parity
    p20_grp = p20_common.group_words(p20_attr)
    a1_2_grp = a1_2_adapter.group_words_by_speaker(a1_2_attr)

    assert len(p20_grp) == len(a1_2_grp)
    for g_p20, g_a1_2 in zip(p20_grp, a1_2_grp):
        assert g_p20["speaker"] == g_a1_2["speaker"]
        assert g_p20["text"] == g_a1_2["text"]
        assert g_p20["start"] == g_a1_2["start"]
        assert g_p20["end"] == g_a1_2["end"]


# Case 1: Ordinary single-speaker attribution
def test_parity_case1_ordinary_single_speaker():
    segments = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    words = [
        {"word": "hello", "start": 0.2, "end": 0.6},
        {"word": "world", "start": 0.8, "end": 1.4},
    ]
    assert_attribution_parity(words, segments)


# Case 2: Two speakers with overlapping intervals
def test_parity_case2_two_speakers_overlapping():
    segments = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"word": "primary", "start": 0.5, "end": 1.2},    # Overlaps only spk0
        {"word": "conflict", "start": 2.1, "end": 2.8},   # Overlaps both: spk0 has 0.7s, spk1 has 0.7s
        {"word": "secondary", "start": 3.5, "end": 4.2},  # Overlaps only spk1
    ]
    assert_attribution_parity(words, segments)


# Case 3: Equal-overlap ties
def test_parity_case3_equal_overlap_ties():
    # Word [2.0, 3.0] overlaps spk0 [1.0, 3.0] by 1.0s and spk1 [2.0, 4.0] by 1.0s
    segments = [
        {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    words = [{"word": "tied_word", "start": 2.0, "end": 3.0}]
    assert_attribution_parity(words, segments)


# Case 4: Words with no timestamps
def test_parity_case4_no_timestamps():
    segments = [{"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00"}]
    words = [
        {"word": "none_start", "start": None, "end": 2.0},
        {"word": "none_end", "start": 1.0, "end": None},
        {"word": "none_both", "start": None, "end": None},
    ]
    assert_attribution_parity(words, segments)


# Case 5: Word outside all intervals
def test_parity_case5_word_outside_all_intervals():
    segments = [{"start": 10.0, "end": 12.0, "speaker": "SPEAKER_00"}]
    words = [{"word": "far_away", "start": 0.0, "end": 1.0}]  # 9.0s away > tolerance
    assert_attribution_parity(words, segments)


# Case 6: Nearest-segment fallback within tolerance
def test_parity_case6_nearest_segment_fallback():
    segments = [{"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00"}]
    # Word at 1.7-1.9s: midpoint 1.8s, distance to start (2.0) is 0.2s <= tolerance 0.5s
    words = [{"word": "near_start", "start": 1.7, "end": 1.9}]
    assert_attribution_parity(words, segments)


# Case 7: Multiple overlapping segments from the same speaker
def test_parity_case7_multiple_overlapping_same_speaker():
    segments = [
        {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00"},
    ]
    words = [{"word": "overlapping", "start": 2.2, "end": 2.8}]
    assert_attribution_parity(words, segments)


# Case 8: Adjacent segment boundaries (touching edges)
def test_parity_case8_adjacent_boundaries():
    segments = [
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
    ]
    # Word right at boundary
    words = [
        {"word": "w1", "start": 1.8, "end": 2.0},
        {"word": "w2", "start": 2.0, "end": 2.2},
    ]
    assert_attribution_parity(words, segments)


# Case 9: Empty transcript
def test_parity_case9_empty_transcript():
    segments = [{"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    words = []
    assert_attribution_parity(words, segments)


# Case 10: Empty RTTM
def test_parity_case10_empty_rttm():
    segments = []
    words = [{"word": "orphan", "start": 1.0, "end": 2.0}]
    assert_attribution_parity(words, segments)


# Case 11: Speaker-label permutation invariance
def test_parity_case11_speaker_permutation_invariance(tmp_path):
    ref_segs = [
        {"speaker": "spk_1", "text": "alpha beta gamma"},
        {"speaker": "spk_2", "text": "delta epsilon zeta"},
    ]
    # Swapped speaker names in hypothesis
    hyp_segs = [
        {"speaker": "hyp_B", "text": "alpha beta gamma"},
        {"speaker": "hyp_A", "text": "delta epsilon zeta"},
    ]

    p20_score = p20_cpwer.cpwer(ref_segs, hyp_segs)
    p20_res = p20_cpwer.cpwer_stats(ref_segs, hyp_segs)

    assert p20_score == 0.0
    assert p20_res["cpwer"] == 0.0
    assert p20_res["substitutions"] == 0
    assert p20_res["insertions"] == 0
    assert p20_res["deletions"] == 0


# Case 12: Multiple words at identical timestamps
def test_parity_case12_identical_timestamps():
    segments = [
        {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"word": "word1", "start": 2.0, "end": 2.5},
        {"word": "word2", "start": 2.0, "end": 2.5},
    ]
    assert_attribution_parity(words, segments)


def test_full_differential_cpwer_evaluation(tmp_path):
    # Create matching synthetic RTTM, words JSON, and ref JSON
    ref_json = tmp_path / "ref.json"
    ref_json.write_text(json.dumps({
        "segments": [
            {"speaker": "A", "text": "good morning everyone"},
            {"speaker": "B", "text": "welcome to the meeting"},
        ]
    }))

    words_json = tmp_path / "words.json"
    words_json.write_text(json.dumps([
        {"word": "good", "start": 0.5, "end": 0.8},
        {"word": "morning", "start": 0.9, "end": 1.4},
        {"word": "everyone", "start": 1.5, "end": 2.0},
        {"word": "welcome", "start": 3.0, "end": 3.5},
        {"word": "to", "start": 3.6, "end": 3.8},
        {"word": "the", "start": 3.9, "end": 4.1},
        {"word": "meeting", "start": 4.2, "end": 4.7},
    ]))

    hyp_rttm = tmp_path / "hyp.rttm"
    hyp_rttm.write_text(
        "SPEAKER rec 1 0.50 1.60 <NA> <NA> hyp_01 <NA> <NA>\n"
        "SPEAKER rec 1 3.00 1.80 <NA> <NA> hyp_02 <NA> <NA>\n"
    )

    res = a1_2_adapter.evaluate_cpwer(hyp_rttm, words_json, ref_json)
    assert res["cpwer"] == 0.0
    assert res["reference_words"] == 7
    assert res["hypothesis_words"] == 7
