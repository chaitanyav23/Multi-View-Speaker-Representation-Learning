"""Unit tests for canonical data models and serialization."""
from __future__ import annotations

import json
from engine.schemas import (
    AlignedWord,
    CalibrationMetadata,
    ConfidenceFeatures,
    ConversationTurn,
    TranscriptResult,
)


def test_aligned_word_schema():
    feat = ConfidenceFeatures(
        acoustic_score=-0.15,
        duration=0.35,
        speech_ratio=0.9,
        overlap_detected=False,
        energy_level=0.85,
    )
    word = AlignedWord(
        word="officer",
        start=1.2,
        end=1.55,
        speaker="SPEAKER_00",
        confidence=0.92,
        raw_confidence=0.88,
        is_calibrated=True,
        features=feat,
    )
    assert word.word == "officer"
    assert word.confidence == 0.92
    assert word.features.energy_level == 0.85

    # Test serialization
    data = word.model_dump()
    assert data["word"] == "officer"
    assert "features" in data


def test_transcript_result_serialization():
    calib = CalibrationMetadata(method="isotonic_regression", brier_score=0.05, ece=0.03)
    word = AlignedWord(
        word="hello", start=0.0, end=0.5, speaker="SPEAKER_01", confidence=0.95, raw_confidence=0.9
    )
    turn = ConversationTurn(
        turn_id="turn_001",
        speaker="SPEAKER_01",
        start=0.0,
        end=0.5,
        text="hello",
        words=[word],
        avg_confidence=0.95,
    )
    result = TranscriptResult(
        recording_id="rec_123",
        filename="test.wav",
        duration=1.0,
        calibration=calib,
        speakers=["SPEAKER_01"],
        segments=[turn],
        words=[word],
    )

    dumped = result.model_dump()
    json_str = json.dumps(dumped)
    loaded = TranscriptResult.model_validate_json(json_str)

    assert loaded.recording_id == "rec_123"
    assert len(loaded.words) == 1
    assert loaded.segments[0].turn_id == "turn_001"
