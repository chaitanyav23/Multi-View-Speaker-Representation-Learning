#!/usr/bin/env python3
"""Download external model artifacts used by Phase 20.

This script intentionally mirrors the model names used by the pipeline. It does
not change inference code or model choices.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    env_path = ROOT / ".env"
    if not token and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    if not token:
        raise RuntimeError("HF_TOKEN is required. Copy .env.example to .env and set HF_TOKEN.")
    return token


def main() -> int:
    token = require_token()
    cache = ROOT / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub

    print("Downloading pyannote/voice-activity-detection")
    download_from_hf_hub("pyannote/voice-activity-detection", AssetFileName.Pipeline, token=token)

    print("Downloading WhisperX large-v3 and English alignment model")
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisperx.load_model("large-v3", device, compute_type=compute_type, language="en")
    whisperx.load_align_model(language_code="en", device=device)

    print("Downloading NeMo speaker and MSDD models")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel, NeuralDiarizer

    EncDecSpeakerLabelModel.from_pretrained("titanet_large")
    try:
        NeuralDiarizer.from_pretrained("diar_msdd_telephonic")
    except AttributeError:
        print("NeuralDiarizer has no from_pretrained method in this NeMo version; it downloads during diarize().")

    print("Model download complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"download_models.py failed: {exc}", file=sys.stderr)
        raise
