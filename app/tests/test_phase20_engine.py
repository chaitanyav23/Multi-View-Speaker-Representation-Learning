"""Unit and integration tests for Phase20SpeechEngine."""
from __future__ import annotations

import pytest
from pathlib import Path
from engine.phase20_engine import Phase20SpeechEngine
from engine.schemas import AlignedWord, ConfidenceFeatures, TranscriptResult


def test_phase20_engine_initialization():
    engine = Phase20SpeechEngine(model_size="tiny")
    info = engine.get_model_info()
    assert info["engine"] == "Phase20SpeechEngine"
    assert info["model_size"] == "tiny"
    assert info["compute_type"] in ("int8", "float16")
    assert 0.0 < info["confidence_threshold"] < 1.0


def test_phase20_speaker_attribution():
    engine = Phase20SpeechEngine(model_size="tiny")
    raw_words = [
        {"word": "hello", "start": 1.0, "end": 1.5, "confidence": 0.95},
        {"word": "officer", "start": 1.6, "end": 2.1, "confidence": 0.88},
        {"word": "license", "start": 5.0, "end": 5.6, "confidence": 0.72},
    ]
    diar_segments = [
        {"start": 0.8, "end": 2.5, "speaker": "DRIVER"},
        {"start": 4.8, "end": 7.0, "speaker": "OFFICER"},
    ]

    attributed = engine._attribute_words(raw_words, diar_segments, tolerance=0.5)
    assert len(attributed) == 3
    assert attributed[0]["speaker"] == "DRIVER"
    assert attributed[1]["speaker"] == "DRIVER"
    assert attributed[2]["speaker"] == "OFFICER"


def test_phase20_turn_grouping():
    engine = Phase20SpeechEngine(model_size="tiny")
    feat = ConfidenceFeatures(acoustic_score=0.9, duration=0.5)
    words = [
        AlignedWord(word="hello", start=1.0, end=1.5, speaker="SPEAKER_00", confidence=0.9, raw_confidence=0.9, features=feat),
        AlignedWord(word="there", start=1.6, end=2.0, speaker="SPEAKER_00", confidence=0.85, raw_confidence=0.85, features=feat),
        AlignedWord(word="good", start=3.0, end=3.4, speaker="SPEAKER_01", confidence=0.92, raw_confidence=0.92, features=feat),
        AlignedWord(word="morning", start=3.5, end=4.0, speaker="SPEAKER_01", confidence=0.89, raw_confidence=0.89, features=feat),
    ]

    turns = engine._group_into_turns(words, max_gap=1.5)
    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "hello there"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].text == "good morning"
    assert turns[0].turn_id == "turn_000"
    assert turns[1].turn_id == "turn_001"


def test_phase20_real_audio_inference():
    """Verify real Faster-Whisper inference on actual benchmark audio segment."""
    audio_candidates = [
        Path("../speech_segments/bwc/bwc_seg01_46.612_52.812.wav").resolve(),
        Path("storage/uploads/481a3a07-b2bc-4708-8fd8-e6e4a6ed224d.wav").resolve(),
    ]
    test_audio = next((p for p in audio_candidates if p.exists()), None)
    if not test_audio:
        pytest.skip("Benchmark audio fixture is absent; skipping real audio inference test.")

    engine = Phase20SpeechEngine(model_size="tiny")
    progress = []

    def cb(val, label):
        progress.append((val, label))

    result = engine.process(
        audio_path=str(test_audio),
        recording_id="test_real_rec",
        progress_callback=cb,
    )

    assert isinstance(result, TranscriptResult)
    assert result.recording_id == "test_real_rec"
    assert len(result.words) > 0
    assert len(result.segments) > 0
    assert 0.0 <= result.summary["overall_confidence"] <= 1.0
    assert len(progress) >= 4
