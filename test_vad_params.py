import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from dotenv import dotenv_values
from pathlib import Path

PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
pipeline = VoiceActivityDetection(segmentation=model)
print(f"Pipeline parameters: {pipeline.parameters()}")
