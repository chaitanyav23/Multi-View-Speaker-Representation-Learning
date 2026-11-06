#!/usr/bin/env python3
"""
Stage 5B.4 Dual-Encoder Feature Extraction Interface
Defines validated extraction interfaces for:
1. NVIDIA TitaNet-Large (192-dim, DFN audio)
2. WeSpeaker VoxCeleb ResNet293-LM (256-dim, Raw Mono audio)
Guarded with explicit non-inference safeguards to prevent premature model loading.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


TITANET_EMBEDDING_DIM = 192
WESPEAKER_EMBEDDING_DIM = 256
SAMPLE_RATE = 16000

# Canonical local checkpoint locations
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
TITANET_CHECKPOINT_PATH = Path(
    os.path.expanduser("~/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo")
)
WESPEAKER_PYTORCH_PATH = Path(
    os.path.expanduser("~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet293-LM/snapshots/6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/avg_model.pt")
)
WESPEAKER_ONNX_PATH = Path(
    os.path.expanduser("~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet293-LM/snapshots/6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/voxceleb_resnet293_LM.onnx")
)


def validate_embedding_matrix(
    embeddings: np.ndarray,
    expected_dim: int,
    expected_count: Optional[int] = None,
) -> bool:
    """
    Validates that an embedding matrix satisfies strict mathematical properties:
    - 2D floating-point array
    - Exact expected feature dimension
    - Exact expected row count (if provided)
    - Zero NaN or Infinite values
    - Non-zero vectors
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Embeddings must be np.ndarray, got {type(embeddings)}")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D matrix (N, D), got shape {embeddings.shape}")

    n_rows, dim = embeddings.shape
    if dim != expected_dim:
        raise ValueError(f"Feature dimension mismatch: expected {expected_dim}, got {dim}")

    if expected_count is not None and n_rows != expected_count:
        raise ValueError(f"Row count mismatch: expected {expected_count}, got {n_rows}")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain NaN or Inf values")

    # Check for all-zero rows
    norms = np.linalg.norm(embeddings, axis=-1)
    if np.any(norms < 1e-12):
        zero_rows = int(np.sum(norms < 1e-12))
        raise ValueError(f"Found {zero_rows} degenerate (all-zero or near-zero) embedding vectors")

    return True


def slice_waveform(
    waveform: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Slices a continuous 1D float32 waveform between start_sec and end_sec.
    """
    n_samples = len(waveform)
    st_idx = max(0, int(round(start_sec * sample_rate)))
    en_idx = min(n_samples, int(round(end_sec * sample_rate)))

    if en_idx <= st_idx:
        return np.zeros(0, dtype=np.float32)

    return waveform[st_idx:en_idx]


class DualEncoderInterface:
    """
    Interface for speaker embedding extraction.
    In Stage 5B.4-A, model instantiation is explicitly guarded and deferred.
    """

    def __init__(self, allow_model_loading: bool = False):
        self.allow_model_loading = allow_model_loading
        self._titanet_model = None
        self._wespeaker_model = None

    def load_titanet(self, device: str = "cuda"):
        """
        Loads NVIDIA TitaNet-Large from local NeMo MSDD checkpoint.
        Guarded against accidental execution during non-inference stages.
        """
        if not self.allow_model_loading:
            raise RuntimeError(
                "BLOCKED: Model loading is not authorized in Stage 5B.4-A. "
                "Explicit execution authorization is required."
            )
        import torch
        from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

        if not TITANET_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"TitaNet checkpoint missing: {TITANET_CHECKPOINT_PATH}")

        msdd_full = EncDecDiarLabelModel.restore_from(str(TITANET_CHECKPOINT_PATH), map_location=device)
        msdd_sd = msdd_full.state_dict()
        prefix = "msdd._speaker_model."
        spk_sd = {k.replace(prefix, ""): v for k, v in msdd_sd.items() if k.startswith(prefix)}
        model = EncDecSpeakerLabelModel.from_config_dict(msdd_full.cfg.speaker_model_cfg)
        model.load_state_dict(spk_sd, strict=True)
        model.to(torch.device(device))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._titanet_model = model
        return self._titanet_model

    def load_wespeaker(self, device: str = "cuda"):
        """
        Loads WeSpeaker ResNet293-LM PyTorch checkpoint.
        Guarded against accidental execution during non-inference stages.
        """
        if not self.allow_model_loading:
            raise RuntimeError(
                "BLOCKED: Model loading is not authorized in Stage 5B.4-A. "
                "Explicit execution authorization is required."
            )
        import torch
        from wespeaker.models.resnet import ResNet293

        if not WESPEAKER_PYTORCH_PATH.exists():
            raise FileNotFoundError(f"WeSpeaker checkpoint missing: {WESPEAKER_PYTORCH_PATH}")

        model = ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP")
        sd = torch.load(str(WESPEAKER_PYTORCH_PATH), map_location=device)
        model.load_state_dict(sd)
        model.to(torch.device(device))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._wespeaker_model = model
        return self._wespeaker_model


def generate_synthetic_embeddings(
    n_subsegments: int,
    encoder_name: str,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic L2-normalized embeddings and L2-norms for testing.
    Zero neural model inference.
    """
    dim = TITANET_EMBEDDING_DIM if "titanet" in encoder_name.lower() else WESPEAKER_EMBEDDING_DIM
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_subsegments, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    normalized = raw / np.clip(norms, 1e-12, None)
    return normalized, norms.squeeze(-1)
