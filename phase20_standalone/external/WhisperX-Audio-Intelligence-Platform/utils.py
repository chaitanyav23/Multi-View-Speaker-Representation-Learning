import whisperx
import torch
from dotenv import dotenv_values
import os

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = (ENV_VALUES.get("HF_TOKEN") or "").strip() or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"

import librosa
import whisperx
import torch

from whisperx.diarize import DiarizationPipeline

def load_diarization_model():

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is required for diarization. "
            "Set it in the .env file (loaded via python-dotenv) and accept "
            "access to 'pyannote/speaker-diarization-community-1'."
        )

    print("Loading diarization model...")

    diarize_model = DiarizationPipeline(
        token=HF_TOKEN,
        device=DEVICE
    )

    return diarize_model

def validate_audio_length(audio_path, max_duration=120):
    """
    Limit audio duration for HF Spaces.
    """

    duration = librosa.get_duration(path=audio_path)

    if duration > max_duration:
        raise ValueError(
            f"Audio too long. "
            f"Maximum allowed length is "
            f"{max_duration} seconds."
        )

def load_models(model_size="large-v3"):
    """
    Load WhisperX ASR model.
    """

    print(f"Using device: {DEVICE}")

    asr_model = whisperx.load_model(
        model_size,
        DEVICE,
        compute_type=COMPUTE_TYPE
    )

    return asr_model

def format_transcript(segments):
    """
    Convert segments into readable speaker transcript.
    """

    formatted = []

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        text = seg["text"]

        formatted.append(
            f"[{start}s - {end}s] {speaker}: {text}"
        )

    return "\n".join(formatted)
