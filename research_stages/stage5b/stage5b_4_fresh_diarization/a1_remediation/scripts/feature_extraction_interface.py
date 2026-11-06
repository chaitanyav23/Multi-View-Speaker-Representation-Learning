#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Waveform-to-Embedding Extraction Interfaces
Production-facing extraction contracts for NVIDIA TitaNet-Large (192d) and
WeSpeaker ResNet293-LM (256d). Supports rigorous input validation, shape verification,
and mocked model fixtures for non-inference testing.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


TITANET_EMBEDDING_DIM = 192
WESPEAKER_EMBEDDING_DIM = 256
EXPECTED_SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION_SEC = 0.05  # 50ms absolute floor for feature extraction


class ExtractionError(Exception):
    """Base exception for feature extraction failures."""
    pass


class InvalidAudioError(ExtractionError):
    """Raised when waveform input fails validation (sample rate, shape, NaN, duration)."""
    pass


class DimensionMismatchError(ExtractionError):
    """Raised when model output does not match the expected embedding dimension."""
    pass


class ModelExecutionError(ExtractionError):
    """Raised when the encoder model fails during forward execution."""
    pass


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: np.ndarray  # 1D float32 array, L2 normalized
    l2_norm: float         # Euclidean norm prior to normalization
    dimension: int
    encoder_name: str
    sample_rate: int
    duration_sec: float
    is_valid: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embedding_shape"] = list(self.embedding.shape)
        del d["embedding"]  # Exclude heavy numpy array from light dict representation
        return d


def validate_waveform_input(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    min_duration_sec: float = MIN_SEGMENT_DURATION_SEC,
    expected_sample_rate: int = EXPECTED_SAMPLE_RATE,
) -> np.ndarray:
    """
    Strictly validates waveform input:
    - Sample rate must match expected rate (16000 Hz)
    - Waveform must be 1D (mono) or (1, N)
    - Values must be finite (zero NaN or Inf)
    - Duration must be >= min_duration_sec
    Returns 1D float32 numpy array.
    """
    if sample_rate != expected_sample_rate:
        raise InvalidAudioError(
            f"Unsupported sample rate {sample_rate} Hz. Expected {expected_sample_rate} Hz."
        )

    if isinstance(waveform, torch.Tensor):
        wav_np = waveform.detach().cpu().numpy()
    elif isinstance(waveform, np.ndarray):
        wav_np = waveform
    else:
        raise InvalidAudioError(f"Unsupported waveform type: {type(waveform)}. Expected np.ndarray or torch.Tensor.")

    # Convert to float32
    wav_np = wav_np.astype(np.float32)

    # Flatten if shape is (1, N) or (N, 1)
    if wav_np.ndim == 2:
        if wav_np.shape[0] == 1:
            wav_np = wav_np.squeeze(0)
        elif wav_np.shape[1] == 1:
            wav_np = wav_np.squeeze(1)
        else:
            raise InvalidAudioError(f"Multi-channel audio not supported: shape {wav_np.shape}")
    elif wav_np.ndim != 1:
        raise InvalidAudioError(f"Waveform array must be 1D, got ndim={wav_np.ndim}")

    if len(wav_np) == 0:
        raise InvalidAudioError("Waveform array is empty (0 samples).")

    if not np.all(np.isfinite(wav_np)):
        raise InvalidAudioError("Waveform contains NaN or Infinite values.")

    duration_sec = len(wav_np) / float(sample_rate)
    if duration_sec < min_duration_sec - 1e-6:
        raise InvalidAudioError(
            f"Segment duration {duration_sec:.4f}s is shorter than minimum required {min_duration_sec:.4f}s."
        )

    return wav_np


def extract_titanet_embedding(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    model: Any,
    device: str = "cpu",
    encoder_name: str = "titanet_large",
    min_duration_sec: float = MIN_SEGMENT_DURATION_SEC,
) -> EmbeddingResult:
    """
    Extracts a 192-dimensional speaker embedding using NVIDIA TitaNet-Large.
    """
    wav_1d = validate_waveform_input(waveform, sample_rate, min_duration_sec)
    duration_sec = round(len(wav_1d) / float(sample_rate), 4)

    try:
        wav_t = torch.from_numpy(wav_1d).unsqueeze(0).to(device)  # shape (1, T)
        len_t = torch.tensor([len(wav_1d)], dtype=torch.int64, device=device)

        if hasattr(model, "forward"):
            # Standard NeMo EncDecSpeakerLabelModel
            out = model.forward(input_signal=wav_t, input_signal_length=len_t)
            # NeMo returns (logits, embs)
            raw_emb = out[1] if isinstance(out, (tuple, list)) else out
        elif callable(model):
            out = model(wav_t, len_t)
            raw_emb = out[1] if isinstance(out, (tuple, list)) else out
        else:
            raise ModelExecutionError(f"Model {type(model)} is neither callable nor has forward().")

        if isinstance(raw_emb, torch.Tensor):
            raw_emb_np = raw_emb.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            raw_emb_np = np.asarray(raw_emb, dtype=np.float32).squeeze()

    except Exception as e:
        if isinstance(e, ExtractionError):
            raise
        raise ModelExecutionError(f"TitaNet model execution failed: {e}") from e

    # Validation
    if raw_emb_np.ndim != 1 or len(raw_emb_np) != TITANET_EMBEDDING_DIM:
        raise DimensionMismatchError(
            f"TitaNet output dimension mismatch: expected ({TITANET_EMBEDDING_DIM},), got {raw_emb_np.shape}"
        )

    if not np.all(np.isfinite(raw_emb_np)):
        raise ExtractionError("TitaNet embedding contains NaN or Inf.")

    l2_norm = float(np.linalg.norm(raw_emb_np))
    if l2_norm < 1e-12:
        raise ExtractionError("TitaNet embedding has near-zero L2 norm (degenerate output).")

    normalized_emb = (raw_emb_np / l2_norm).astype(np.float32)

    return EmbeddingResult(
        embedding=normalized_emb,
        l2_norm=round(l2_norm, 6),
        dimension=TITANET_EMBEDDING_DIM,
        encoder_name=encoder_name,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        is_valid=True,
        metadata={"device": device, "input_samples": len(wav_1d)},
    )


def extract_wespeaker_embedding(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    model: Any,
    device: str = "cpu",
    encoder_name: str = "wespeaker_resnet293_lm",
    min_duration_sec: float = MIN_SEGMENT_DURATION_SEC,
) -> EmbeddingResult:
    """
    Extracts a 256-dimensional speaker embedding using WeSpeaker ResNet293-LM.
    Expects 80-channel Kaldi Fbank features.
    """
    wav_1d = validate_waveform_input(waveform, sample_rate, min_duration_sec)
    duration_sec = round(len(wav_1d) / float(sample_rate), 4)

    try:
        import torchaudio.compliance.kaldi as kaldi

        wav_t = torch.from_numpy(wav_1d).unsqueeze(0).float()
        # Compute 80-dim Kaldi Fbank matching WeSpeaker standard
        feats = kaldi.fbank(
            wav_t,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            sample_frequency=float(sample_rate),
            window_type="hamming",
            use_energy=False,
        )
        # CMN subtractive mean
        feats = feats - torch.mean(feats, dim=0, keepdim=True)
        feats = feats.unsqueeze(0).to(device)  # shape (1, T, 80)

        if hasattr(model, "forward"):
            out = model.forward(feats)
            raw_emb = out[1] if isinstance(out, (tuple, list)) else out
        elif callable(model):
            out = model(feats)
            raw_emb = out[1] if isinstance(out, (tuple, list)) else out
        else:
            raise ModelExecutionError(f"Model {type(model)} is neither callable nor has forward().")

        if isinstance(raw_emb, torch.Tensor):
            raw_emb_np = raw_emb.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            raw_emb_np = np.asarray(raw_emb, dtype=np.float32).squeeze()

    except Exception as e:
        if isinstance(e, ExtractionError):
            raise
        raise ModelExecutionError(f"WeSpeaker model execution failed: {e}") from e

    # Validation
    if raw_emb_np.ndim != 1 or len(raw_emb_np) != WESPEAKER_EMBEDDING_DIM:
        raise DimensionMismatchError(
            f"WeSpeaker output dimension mismatch: expected ({WESPEAKER_EMBEDDING_DIM},), got {raw_emb_np.shape}"
        )

    if not np.all(np.isfinite(raw_emb_np)):
        raise ExtractionError("WeSpeaker embedding contains NaN or Inf.")

    l2_norm = float(np.linalg.norm(raw_emb_np))
    if l2_norm < 1e-12:
        raise ExtractionError("WeSpeaker embedding has near-zero L2 norm (degenerate output).")

    normalized_emb = (raw_emb_np / l2_norm).astype(np.float32)

    return EmbeddingResult(
        embedding=normalized_emb,
        l2_norm=round(l2_norm, 6),
        dimension=WESPEAKER_EMBEDDING_DIM,
        encoder_name=encoder_name,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        is_valid=True,
        metadata={"device": device, "input_samples": len(wav_1d), "fbank_dim": 80},
    )


# ==============================================================================
# MOCK MODELS FOR NON-INFERENCE UNIT TESTING
# ==============================================================================

class MockTitaNetModel:
    """Mock NeMo TitaNet model producing deterministic 192d embeddings for testing."""

    def __init__(self, seed: int = 42, return_dim: int = TITANET_EMBEDDING_DIM):
        self.seed = seed
        self.return_dim = return_dim

    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        batch_size = input_signal.shape[0]
        rng = np.random.default_rng(self.seed + int(input_signal.shape[-1]))
        emb_np = rng.standard_normal((batch_size, self.return_dim)).astype(np.float32)
        # Simulate unnormalized output with L2 norm ~ 10-15
        emb_np = emb_np * 12.0
        return None, torch.from_numpy(emb_np).to(input_signal.device)


class MockWeSpeakerModel:
    """Mock WeSpeaker model producing deterministic 256d embeddings for testing."""

    def __init__(self, seed: int = 100, return_dim: int = WESPEAKER_EMBEDDING_DIM):
        self.seed = seed
        self.return_dim = return_dim

    def forward(self, feats: torch.Tensor):
        batch_size = feats.shape[0]
        rng = np.random.default_rng(self.seed + int(feats.shape[1]))
        emb_np = rng.standard_normal((batch_size, self.return_dim)).astype(np.float32)
        emb_np = emb_np * 8.0
        return torch.from_numpy(emb_np).to(feats.device)
