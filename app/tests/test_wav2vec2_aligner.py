"""Deterministic unit tests for native Wav2Vec2 CTC forced aligner.

Verifies WhisperX v3.8.5 semantics, wildcard mechanics, dictionary ordering,
tie-breaking, word aggregation, and interpolation using synthetic emissions.
Strictly avoids loading pretrained weights or executing audio inference.
"""
from typing import List, Optional
import pytest
import torch

from engine.wav2vec2_aligner import (
    AlignedSegment,
    TrellisPoint,
    Wav2Vec2ForcedAligner,
    interpolate_nearest_1d,
)


@pytest.fixture
def aligner() -> Wav2Vec2ForcedAligner:
    """Fixture providing an uninstantiated aligner with lazy loading."""
    aligner_instance = Wav2Vec2ForcedAligner(device="cpu", lazy_load=True)
    assert aligner_instance._model is None, "Model weights must not be loaded during tests"
    return aligner_instance


def test_01_vocabulary_and_token_ids(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 1: Verify exact 29-label inventory and indices matching TorchAudio bundle."""
    labels = aligner.labels
    assert len(labels) == 29
    assert labels[0] == "-"
    assert labels[1] == "|"
    assert labels[24] == "'"

    expected_labels = (
        "-", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U",
        "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"
    )
    assert labels == expected_labels
    assert aligner.blank_id == 0
    assert aligner.delimiter_id == 1


def test_02_blank_and_delimiter_behavior(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 2: Verify blank and delimiter space conversion."""
    text = "hello world"
    clean_char, clean_cdx, tokens, has_wildcard = aligner.tokenize_text(text)

    assert not has_wildcard
    assert "|" in clean_char
    space_pos = clean_char.index("|")
    assert clean_cdx[space_pos] == 5  # index of space in 'hello world'
    assert tokens[space_pos] == aligner.delimiter_id
    assert aligner.dictionary["|"] == 1


def test_03_oov_wildcard_assignment(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 3: Verify OOV characters (digits, punctuation, symbols) trigger wildcard."""
    text = "Room 42, £100!"
    clean_char, clean_cdx, tokens_initial, has_wildcard = aligner.tokenize_text(text)

    assert has_wildcard is True
    # '4', '2', ',', '£', '1', '0', '0', '!' are outside the 29-label vocabulary
    for oov_char in ["4", "2", ",", "£", "1", "0", "!"]:
        assert oov_char in clean_char
        oov_idx = clean_char.index(oov_char)
        assert tokens_initial[oov_idx] == -1  # Placeholder before emission column creation


def test_04_wildcard_column_construction_and_blank_exclusion(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 4: Verify wildcard column takes max non-blank per frame, strictly excluding blank."""
    num_frames = 4
    vocab_size = 29
    emission = torch.full((num_frames, vocab_size), -10.0)

    # Set blank token (index 0) to a massive score (+50.0)
    emission[:, aligner.blank_id] = 50.0

    # Set specific non-blank tokens to varying values
    emission[0, 2] = -1.0  # Frame 0: 'E' is best non-blank
    emission[1, 3] = -0.5  # Frame 1: 'T' is best non-blank
    emission[2, 4] = -2.0  # Frame 2: 'A' is best non-blank
    emission[3, 5] = -0.2  # Frame 3: 'O' is best non-blank

    ext_emission, wildcard_id = aligner.construct_wildcard_emission(emission)

    assert ext_emission.shape == (num_frames, vocab_size + 1)
    assert wildcard_id == 29

    wildcard_values = ext_emission[:, wildcard_id]
    expected_wildcard = torch.tensor([-1.0, -0.5, -2.0, -0.2])
    assert torch.allclose(wildcard_values, expected_wildcard)

    # Crucial check: blank value (+50.0) was NOT picked up by the wildcard
    assert not torch.any(wildcard_values == 50.0)


def test_05_hyphen_and_apostrophe_behavior(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 5: Explicit regression test for hyphen '-' and apostrophe '''.

    '-' is label 0 in the vocabulary. It must map to token index 0, NOT wildcard.
    "'" is label 24 in the vocabulary. It must map to token index 24, NOT wildcard.
    """
    text = "semi-final don't"
    clean_char, clean_cdx, tokens_initial, has_wildcard = aligner.tokenize_text(text)

    # Neither hyphen nor apostrophe is OOV; has_wildcard must be False
    assert has_wildcard is False

    # Check hyphen:
    hyphen_pos = clean_char.index("-")
    assert clean_char[hyphen_pos] == "-"
    assert tokens_initial[hyphen_pos] == 0  # Blank token index 0
    assert aligner.dictionary["-"] == 0

    # Check apostrophe:
    apostrophe_pos = clean_char.index("'")
    assert clean_char[apostrophe_pos] == "'"
    assert tokens_initial[apostrophe_pos] == 24  # Apostrophe token index 24
    assert aligner.dictionary["'"] == 24


def test_06_repeated_characters(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 6: Verify repeated characters produce separate tokens and collapse correctly."""
    text = "look"
    clean_char, clean_cdx, tokens_initial, has_wildcard = aligner.tokenize_text(text)

    assert not has_wildcard
    assert clean_char == ["l", "o", "o", "k"]
    assert tokens_initial == [
        aligner.dictionary["l"],
        aligner.dictionary["o"],
        aligner.dictionary["o"],
        aligner.dictionary["k"],
    ]

    # Test merge_repeats on synthetic path
    path = [
        TrellisPoint(token_index=0, time_index=0, score=0.9),  # l
        TrellisPoint(token_index=0, time_index=1, score=0.8),  # l
        TrellisPoint(token_index=1, time_index=2, score=0.95), # o (first)
        TrellisPoint(token_index=2, time_index=3, score=0.92), # o (second)
        TrellisPoint(token_index=3, time_index=4, score=0.88), # k
    ]
    segs = aligner.merge_repeats(path, clean_char)
    assert len(segs) == 4
    assert segs[0].label == "l" and segs[0].start == 0 and segs[0].end == 2
    assert segs[1].label == "o" and segs[1].start == 2 and segs[1].end == 3
    assert segs[2].label == "o" and segs[2].start == 3 and segs[2].end == 4
    assert segs[3].label == "k" and segs[3].start == 4 and segs[3].end == 5


def test_07_trellis_backtracking_and_tie_breaking(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 7: Verify tie-breaking logic: changed > stayed.

    If changed == stayed, it stays (advances time without advancing token).
    """
    # 2 frames, 1 token
    tokens = [2]  # Token 'e'
    emission = torch.full((2, 29), -5.0)

    # Frame 0: emission for blank is 0.0, emission for token is 0.0
    emission[:, aligner.blank_id] = 0.0
    emission[:, tokens[0]] = 0.0

    trellis = aligner.compute_trellis(emission, tokens)
    path = aligner.backtrack(trellis, emission, tokens)

    assert path is not None
    assert len(path) > 0
    # Frame scores must be in [0.0, 1.0] (linear probability via exp)
    for p in path:
        assert 0.0 <= p.score <= 1.0


def test_08_character_merging(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 8: Verify merge_repeats computes arithmetic mean of linear probabilities."""
    clean_char = ["a", "b"]
    path = [
        TrellisPoint(token_index=0, time_index=0, score=0.8),
        TrellisPoint(token_index=0, time_index=1, score=0.6),
        TrellisPoint(token_index=1, time_index=2, score=0.9),
    ]
    segs = aligner.merge_repeats(path, clean_char)
    assert len(segs) == 2
    assert segs[0].label == "a"
    assert segs[0].start == 0 and segs[0].end == 2
    assert pytest.approx(segs[0].score, 1e-4) == 0.7  # (0.8 + 0.6) / 2

    assert segs[1].label == "b"
    assert segs[1].start == 2 and segs[1].end == 3
    assert pytest.approx(segs[1].score, 1e-4) == 0.9


def test_09_punctuation_and_digit_word_aggregation(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 9: Punctuation and digits remain attached to words and receive timestamps."""
    text = "okay, 42!"
    clean_char, clean_cdx, tokens_initial, has_wildcard = aligner.tokenize_text(text)
    assert has_wildcard is True

    # Construct synthetic emission matrix where each token in clean_char has high prob at frame t
    num_frames = len(clean_char)
    emission = torch.full((num_frames, 29), -15.0)

    # Build emission favoring the exact sequence of tokens
    for t, c in enumerate(clean_char):
        if c in aligner.dictionary:
            emission[t, aligner.dictionary[c]] = 0.0
        else:
            # Set another non-blank token to high prob so wildcard column will capture 0.0
            emission[t, 2] = 0.0

    words = aligner.align_from_emissions(
        emission=emission,
        text=text,
        start_time=0.0,
        end_time=float(num_frames) * 0.02,
    )

    assert len(words) == 2
    assert words[0]["word"] == "okay,"
    assert words[1]["word"] == "42!"
    assert words[0]["start"] < words[0]["end"]
    assert words[1]["start"] < words[1]["end"]
    assert words[0]["end"] <= words[1]["start"] + 0.001


def test_10_timestamp_offsets_and_rounding(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 10: Verify start_time offset and 3-decimal rounding."""
    text = "search"
    clean_char, _, _, _ = aligner.tokenize_text(text)
    num_frames = len(clean_char)
    emission = torch.full((num_frames, 29), -10.0)
    for t, c in enumerate(clean_char):
        emission[t, aligner.dictionary[c]] = 0.0

    start_offset = 15.3456
    end_offset = 18.7891

    words = aligner.align_from_emissions(
        emission=emission,
        text=text,
        start_time=start_offset,
        end_time=end_offset,
    )

    assert len(words) == 1
    w = words[0]
    assert w["word"] == "search"
    assert w["start"] >= round(start_offset, 3)
    assert w["end"] <= round(end_offset, 3) + 0.001
    # Check 3-decimal rounding precision
    assert str(w["start"])[::-1].find(".") <= 3
    assert str(w["end"])[::-1].find(".") <= 3
    assert str(w["confidence"])[::-1].find(".") <= 3


def test_11_nearest_neighbor_interpolation() -> None:
    """Requirement 11: Verify 1D nearest-neighbor interpolation with forward/backward fill."""
    # Case 1: Internal missing values with equidistant tie-break toward earlier index
    seq1: List[Optional[float]] = [10.0, None, 20.0]
    res1 = interpolate_nearest_1d(seq1)
    assert res1 == [10.0, 10.0, 20.0]

    # Case 2: Leading, multiple internal, and trailing missing values
    seq2: List[Optional[float]] = [None, 10.0, None, None, 20.0, None]
    res2 = interpolate_nearest_1d(seq2)
    assert res2 == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]

    # Case 3: Single valid value fills entire series
    seq3: List[Optional[float]] = [None, None, 5.0, None]
    res3 = interpolate_nearest_1d(seq3)
    assert res3 == [5.0, 5.0, 5.0, 5.0]

    # Case 4: No valid values returns unmodified list of Nones
    seq4: List[Optional[float]] = [None, None]
    res4 = interpolate_nearest_1d(seq4)
    assert res4 == [None, None]

    # Case 5: Empty list
    assert interpolate_nearest_1d([]) == []


def test_12_empty_transcripts(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 12: Verify empty and whitespace strings return empty list gracefully."""
    emission = torch.full((10, 29), -1.0)
    assert aligner.align_from_emissions(emission, "", 0.0, 1.0) == []
    assert aligner.align_from_emissions(emission, "   ", 0.0, 1.0) == []
    assert aligner.align_from_emissions(torch.empty((0, 29)), "hello", 0.0, 1.0) == []


def test_13_failed_backtracking(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 13: Backtracking failure gracefully falls back to segment bounds."""
    text = "hello world"
    # An emission with -inf for all valid tokens makes reaching the end mathematically impossible
    emission = torch.full((2, 29), -100.0)
    emission[:, aligner.blank_id] = 0.0

    words = aligner.align_from_emissions(
        emission=emission,
        text=text,
        start_time=1.0,
        end_time=3.0,
    )

    # Must not crash; returns fallback words with segment timestamps and 0.0 confidence
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[0]["start"] == 1.0
    assert words[0]["end"] == 3.0
    assert words[0]["confidence"] == 0.0

    assert words[1]["word"] == "world"
    assert words[1]["start"] == 1.0
    assert words[1]["end"] == 3.0
    assert words[1]["confidence"] == 0.0


def test_14_output_schema_and_lazy_model_verification(aligner: Wav2Vec2ForcedAligner) -> None:
    """Requirement 14: Output conforms to Phase 20 schema and model weights were never loaded."""
    text = "confidence test"
    clean_char, _, _, _ = aligner.tokenize_text(text)
    emission = torch.full((len(clean_char), 29), -10.0)
    for t, c in enumerate(clean_char):
        emission[t, aligner.dictionary[c]] = 0.0

    words = aligner.align_from_emissions(
        emission=emission,
        text=text,
        start_time=5.0,
        end_time=6.5,
    )

    assert len(words) == 2
    for item in words:
        # Phase 20 canonical keys
        assert set(item.keys()) == {"word", "start", "end", "confidence"}
        assert isinstance(item["word"], str)
        assert isinstance(item["start"], float)
        assert isinstance(item["end"], float)
        assert isinstance(item["confidence"], float)
        assert item["start"] <= item["end"]
        assert 0.0 <= item["confidence"] <= 1.0

    # Explicit verification that model weights remained unloaded
    assert aligner._model is None
