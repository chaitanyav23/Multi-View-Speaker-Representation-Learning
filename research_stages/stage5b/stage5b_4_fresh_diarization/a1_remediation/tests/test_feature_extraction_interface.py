#!/usr/bin/env python3
"""Tests for Task 1: Waveform-to-Embedding Extraction Interfaces."""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from feature_extraction_interface import (
    validate_waveform_input,
    extract_titanet_embedding,
    extract_wespeaker_embedding,
    EmbeddingResult,
    InvalidAudioError,
    DimensionMismatchError,
    ModelExecutionError,
    MockTitaNetModel,
    MockWeSpeakerModel,
    TITANET_EMBEDDING_DIM,
    WESPEAKER_EMBEDDING_DIM,
)


def test_waveform_input_validation():
    # Valid 16kHz mono audio (1.0s = 16000 samples)
    valid_wav = np.zeros(16000, dtype=np.float32)
    res = validate_waveform_input(valid_wav, sample_rate=16000)
    assert res.ndim == 1
    assert len(res) == 16000

    # 2D tensor shape (1, 16000)
    tensor_wav = torch.zeros(1, 16000)
    res_t = validate_waveform_input(tensor_wav, sample_rate=16000)
    assert res_t.ndim == 1
    assert len(res_t) == 16000

    # Wrong sample rate
    with pytest.raises(InvalidAudioError, match="Unsupported sample rate"):
        validate_waveform_input(valid_wav, sample_rate=44100)

    # Audio too short (< 0.05s = 800 samples)
    short_wav = np.zeros(400, dtype=np.float32)
    with pytest.raises(InvalidAudioError, match="shorter than minimum required"):
        validate_waveform_input(short_wav, sample_rate=16000)

    # Audio contains NaN
    nan_wav = np.zeros(16000, dtype=np.float32)
    nan_wav[50] = np.nan
    with pytest.raises(InvalidAudioError, match="NaN or Infinite"):
        validate_waveform_input(nan_wav, sample_rate=16000)

    # Empty audio
    with pytest.raises(InvalidAudioError, match="empty"):
        validate_waveform_input(np.zeros(0, dtype=np.float32), sample_rate=16000)


def test_titanet_mock_extraction_contract():
    model = MockTitaNetModel(seed=42)
    wav = np.random.default_rng(42).standard_normal(16000).astype(np.float32)

    result = extract_titanet_embedding(wav, sample_rate=16000, model=model)
    assert isinstance(result, EmbeddingResult)
    assert result.dimension == TITANET_EMBEDDING_DIM
    assert result.embedding.shape == (192,)
    assert result.is_valid is True
    # Normalized L2 norm must be approximately 1.0
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)
    # Pre-normalization L2 norm must be recorded
    assert result.l2_norm > 0.0


def test_wespeaker_mock_extraction_contract():
    model = MockWeSpeakerModel(seed=100)
    wav = np.random.default_rng(100).standard_normal(16000).astype(np.float32)

    result = extract_wespeaker_embedding(wav, sample_rate=16000, model=model)
    assert isinstance(result, EmbeddingResult)
    assert result.dimension == WESPEAKER_EMBEDDING_DIM
    assert result.embedding.shape == (256,)
    assert result.is_valid is True
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)
    assert result.l2_norm > 0.0


def test_dimension_mismatch_error():
    # Model returning wrong dimension (e.g. 128 instead of 192)
    bad_model = MockTitaNetModel(return_dim=128)
    wav = np.zeros(16000, dtype=np.float32)
    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        extract_titanet_embedding(wav, sample_rate=16000, model=bad_model)


def test_model_execution_error():
    class BrokenModel:
        def forward(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory simulation")

    wav = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ModelExecutionError, match="model execution failed"):
        extract_titanet_embedding(wav, sample_rate=16000, model=BrokenModel())
