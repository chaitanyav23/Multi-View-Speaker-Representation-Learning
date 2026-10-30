import os
import subprocess
import hashlib
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path

raw_audio = "raw_audio/karen.mp3"
normalized_audio = "experiments/phase12_karen_evaluation/audio_preprocessed/karen_normalized.wav"
audit_path = "experiments/phase12_karen_evaluation/audits/audio_provenance_report.md"

def compute_hash(filepath, algo='md5'):
    h = hashlib.new(algo)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize():
    print("Normalizing audio...")
    # Clean the MP3 first since it's corrupted, then decode
    tmp_clean = "experiments/phase12_karen_evaluation/audio_preprocessed/tmp_clean.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw_audio, "-c:a", "copy", tmp_clean])
    
    # Now normalize
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", tmp_clean,
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        normalized_audio
    ]
    subprocess.run(cmd)
    
    if os.path.exists(tmp_clean):
        os.remove(tmp_clean)

def analyze_audio():
    print("Analyzing audio...")
    data, rate = sf.read(normalized_audio)
    
    # LUFS
    meter = pyln.Meter(rate) # create BS.1770 meter
    lufs = meter.integrated_loudness(data)
    
    # RMS
    rms = np.sqrt(np.mean(data**2))
    
    # Peak
    peak = np.max(np.abs(data))
    
    # Hashes
    md5 = compute_hash(normalized_audio, 'md5')
    sha256 = compute_hash(normalized_audio, 'sha256')
    
    with open(audit_path, "w") as f:
        f.write("# AUDIO PROVENANCE REPORT\n\n")
        f.write(f"- **Source File:** {raw_audio}\n")
        f.write(f"- **Output File:** {normalized_audio}\n")
        f.write(f"- **Normalization Filter:** `loudnorm=I=-20:TP=-1.5:LRA=11`\n\n")
        f.write(f"### Metrics\n")
        f.write(f"- **LUFS:** {lufs:.2f}\n")
        f.write(f"- **RMS:** {rms:.4f}\n")
        f.write(f"- **Peak:** {peak:.4f}\n\n")
        f.write(f"### Hashes\n")
        f.write(f"- **MD5:** {md5}\n")
        f.write(f"- **SHA256:** {sha256}\n")
        
    print(f"Report saved to {audit_path}")

if __name__ == "__main__":
    normalize()
    analyze_audio()
