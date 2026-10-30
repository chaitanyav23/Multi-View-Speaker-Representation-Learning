import torch
from pyannote.audio import Pipeline
from pathlib import Path
from dotenv import dotenv_values

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
pipeline.to(torch.device(DEVICE))

audio_path = "experiments/karen_11A_audit/audio/karen_normalized.wav"
print(f"Processing {audio_path}...")
vad = pipeline(audio_path)

segments = []
for s in vad.get_timeline().support():
    segments.append({"start": s.start, "end": s.end})

print(f"Found {len(segments)} segments.")
if segments:
    print(f"First segment: {segments[0]}")
