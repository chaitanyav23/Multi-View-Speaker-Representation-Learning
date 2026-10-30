from pyannote.audio import Pipeline
from dotenv import dotenv_values
from pathlib import Path
import huggingface_hub

PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

print(f"Testing with token: {HF_TOKEN[:5]}...")

try:
    print("Trying login + no args...")
    huggingface_hub.login(token=HF_TOKEN)
    p = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    print("Success with login")
except Exception as e:
    print(f"Failed with login: {e}")
