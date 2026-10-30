import torch
from pyannote.audio import Pipeline
from dotenv import dotenv_values
from pathlib import Path

PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

models = [
    "pyannote/voice-activity-detection",
    "pyannote/segmentation-3.0",
    "pyannote/segmentation",
]

for m in models:
    try:
        print(f"Testing {m}...")
        p = Pipeline.from_pretrained(m, token=HF_TOKEN)
        print(f"SUCCESS: {m}")
        print(f"Parameters: {p.parameters()}")
        break
    except Exception as e:
        print(f"FAILED: {m}: {e}")
