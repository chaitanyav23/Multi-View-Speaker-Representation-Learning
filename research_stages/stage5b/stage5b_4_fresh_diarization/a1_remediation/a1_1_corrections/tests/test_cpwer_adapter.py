#!/usr/bin/env python3
"""Tests for Task 1: Audited cpWER Evaluation and Word Attribution."""

import json
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from cpwer_adapter import (
    attribute_words_to_segments,
    group_words_by_speaker,
    rttm_to_segments,
    evaluate_cpwer,
)


def test_rttm_parsing(tmp_path):
    rttm = tmp_path / "test.rttm"
    rttm.write_text(
        "SPEAKER rec 1 0.50 1.20 <NA> <NA> spk_01 <NA> <NA>\n"
        "# comment\n"
        "SPEAKER rec 1 2.00 0.80 <NA> <NA> spk_02 <NA> <NA>\n"
    )
    segs = rttm_to_segments(rttm)
    assert len(segs) == 2
    assert segs[0] == {"start": 0.5, "end": 1.7, "speaker": "spk_01"}
    assert segs[1] == {"start": 2.0, "end": 2.8, "speaker": "spk_02"}


def test_word_attribution_and_grouping():
    diar_segs = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"word": "hello", "start": 0.2, "end": 0.6},
        {"word": "world", "start": 0.8, "end": 1.2},
        {"word": "goodbye", "start": 2.2, "end": 2.7},
    ]

    attributed = attribute_words_to_segments(words, diar_segs)
    assert len(attributed) == 3
    assert attributed[0]["speaker"] == "SPEAKER_00"
    assert attributed[1]["speaker"] == "SPEAKER_00"
    assert attributed[2]["speaker"] == "SPEAKER_01"

    grouped = group_words_by_speaker(attributed)
    assert len(grouped) == 2
    assert grouped[0]["speaker"] == "SPEAKER_00"
    assert grouped[0]["text"] == "hello world"
    assert grouped[1]["speaker"] == "SPEAKER_01"
    assert grouped[1]["text"] == "goodbye"


def test_cpwer_perfect_match(tmp_path):
    # Reference transcript
    ref_json = tmp_path / "ref.json"
    ref_json.write_text(json.dumps({
        "segments": [
            {"speaker": "ref_A", "text": "one two three"},
            {"speaker": "ref_B", "text": "four five six"},
        ]
    }))

    # Words
    words_json = tmp_path / "words.json"
    words_json.write_text(json.dumps([
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.6, "end": 1.0},
        {"word": "three", "start": 1.1, "end": 1.5},
        {"word": "four", "start": 2.0, "end": 2.5},
        {"word": "five", "start": 2.6, "end": 3.0},
        {"word": "six", "start": 3.1, "end": 3.5},
    ]))

    # Hypothesis RTTM matching reference intervals
    hyp_rttm = tmp_path / "hyp.rttm"
    hyp_rttm.write_text(
        "SPEAKER test 1 0.0 1.6 <NA> <NA> hyp_01 <NA> <NA>\n"
        "SPEAKER test 1 2.0 1.6 <NA> <NA> hyp_02 <NA> <NA>\n"
    )

    res = evaluate_cpwer(hyp_rttm, words_json, ref_json)
    assert res["cpwer"] == 0.0
    assert res["insertions"] == 0
    assert res["deletions"] == 0
    assert res["substitutions"] == 0
    assert res["reference_words"] == 6


def test_cpwer_speaker_permutation_invariance(tmp_path):
    # If hypothesis speaker labels are permuted (hyp_02 spoke first, hyp_01 spoke second),
    # Hungarian matching should still achieve cpWER = 0.0!
    ref_json = tmp_path / "ref.json"
    ref_json.write_text(json.dumps({
        "segments": [
            {"speaker": "spk_1", "text": "alpha beta"},
            {"speaker": "spk_2", "text": "gamma delta"},
        ]
    }))

    words_json = tmp_path / "words.json"
    words_json.write_text(json.dumps([
        {"word": "alpha", "start": 0.0, "end": 0.5},
        {"word": "beta", "start": 0.6, "end": 1.0},
        {"word": "gamma", "start": 2.0, "end": 2.5},
        {"word": "delta", "start": 2.6, "end": 3.0},
    ]))

    hyp_rttm = tmp_path / "hyp.rttm"
    # Note: spk_1 words attributed to "speaker_B", spk_2 words attributed to "speaker_A"
    hyp_rttm.write_text(
        "SPEAKER test 1 0.0 1.2 <NA> <NA> speaker_B <NA> <NA>\n"
        "SPEAKER test 1 2.0 1.2 <NA> <NA> speaker_A <NA> <NA>\n"
    )

    res = evaluate_cpwer(hyp_rttm, words_json, ref_json)
    assert res["cpwer"] == 0.0
    # Assignment maps spk_1 -> speaker_B, spk_2 -> speaker_A
    assert len(res["assignment"]) == 2


def test_cpwer_with_errors(tmp_path):
    ref_json = tmp_path / "ref.json"
    ref_json.write_text(json.dumps({
        "segments": [
            {"speaker": "spk_A", "text": "apple banana cherry"},
        ]
    }))

    words_json = tmp_path / "words.json"
    # "banana" is missing (deletion), "orange" is substituted for "cherry"
    words_json.write_text(json.dumps([
        {"word": "apple", "start": 0.0, "end": 0.5},
        {"word": "orange", "start": 0.6, "end": 1.0},
    ]))

    hyp_rttm = tmp_path / "hyp.rttm"
    hyp_rttm.write_text("SPEAKER test 1 0.0 1.2 <NA> <NA> hyp_spk <NA> <NA>\n")

    res = evaluate_cpwer(hyp_rttm, words_json, ref_json)
    assert res["cpwer"] > 0.0
    assert res["reference_words"] == 3
    assert res["hypothesis_words"] == 2


def test_missing_files_fail_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_cpwer(tmp_path / "missing.rttm", tmp_path / "words.json", tmp_path / "ref.json")
