import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

def get_audio_audit(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # RMS and Peak
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # LUFS (requires pyloudnorm, but I'll use a simple approximation or skip if not available)
    # Actually, I can use ffmpeg for LUFS
    lufs = -20 # Placeholder
    
    return {
        "File": os.path.basename(file_path),
        "Duration": duration,
        "Sample Rate": sr,
        "LUFS": lufs,
        "RMS": rms,
        "Peak": peak,
        "Spectral Centroid": spectral_centroid,
        "Spectral Roll-off": spectral_rolloff
    }

# Run ffmpeg to get LUFS
def get_lufs(file_path):
    import subprocess
    cmd = f"ffmpeg -i {file_path} -af loudnorm=print_format=json -f null -"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    import json
    import re
    # Find the JSON part in stderr
    match = re.search(r'\{.*\}', result.stderr, re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        return float(data["input_i"])
    return None

file_path = "experiments/phase13_karen_generalization/audio/karen_normalized.wav"
audit_data = get_audio_audit(file_path)
audit_data["LUFS"] = get_lufs(file_path)

df = pd.DataFrame([audit_data])
df.to_csv("experiments/phase13_karen_generalization/reports/audio_audit.csv", index=False)
print("Audio audit report generated.")
print(df.to_string())
