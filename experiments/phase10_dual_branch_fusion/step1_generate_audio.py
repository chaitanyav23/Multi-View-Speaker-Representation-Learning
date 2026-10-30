import os
import subprocess
from pathlib import Path
import hashlib
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
import json
import csv

BASE_DIR = Path("experiments/phase10_dual_branch_fusion")
AUDIO_DIR = BASE_DIR / "audio_variants"
REPORT_DIR = BASE_DIR / "reports"

RAW_BWC = Path("raw_audio/bwc.wav")
NORM_WAV = AUDIO_DIR / "bwc_normalized.wav"
DFN_WAV = AUDIO_DIR / "bwc_deepfilternet.wav"
SB_WAV = AUDIO_DIR / "bwc_speechbrain.wav"

def compute_hash(filepath, algo='md5'):
    h = hashlib.new(algo)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_audio(filepath):
    # Load audio
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # LUFS using pyloudnorm
    meter = pyln.Meter(sr) # create BS.1770 meter
    lufs = meter.integrated_loudness(y)
    
    # RMS and Peak
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    
    # Spectral Centroid and Roll-off
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    return {
        "file": filepath.name,
        "md5": compute_hash(filepath, 'md5'),
        "sha256": compute_hash(filepath, 'sha256'),
        "duration": float(duration),
        "sample_rate": int(sr),
        "LUFS": float(lufs),
        "RMS": float(rms),
        "Peak": float(peak),
        "Spectral_Centroid_Mean": float(np.mean(cent)),
        "Spectral_Rolloff_Mean": float(np.mean(rolloff))
    }

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Generating Normalized Audio (Phase 6 V2 settings)...")
    if not NORM_WAV.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(RAW_BWC),
            "-ar", "16000", "-ac", "1",
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            str(NORM_WAV)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Normalized audio exists, skipping...")

    print("2. Generating DeepFilterNet Enhanced Audio...")
    if not DFN_WAV.exists():
        # df.enhance saves as {name}_DeepFilterNet3.wav in output-dir
        cmd = [
            "python3", "-m", "df.enhance", 
            str(NORM_WAV), 
            "--output-dir", str(AUDIO_DIR)
        ]
        subprocess.run(cmd, check=True)
        # Rename output
        dfn_out = AUDIO_DIR / f"{NORM_WAV.stem}_DeepFilterNet3.wav"
        if dfn_out.exists():
            os.rename(dfn_out, DFN_WAV)
    else:
        print("DeepFilterNet audio exists, skipping...")

    print("3. Generating SpeechBrain Enhanced Audio...")
    if not SB_WAV.exists():
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhance_model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank", 
            run_opts={"device": device}
        )
        noisy = enhance_model.load_audio(str(NORM_WAV)).unsqueeze(0).to(device)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]).to(device))
        torchaudio.save(str(SB_WAV), enhanced.cpu(), 16000)
    else:
        print("SpeechBrain audio exists, skipping...")

    print("4. Running Audio Audit...")
    audit_results = []
    for f in [NORM_WAV, DFN_WAV, SB_WAV]:
        print(f"  Analyzing {f.name}...")
        audit_results.append(analyze_audio(f))
    
    # Save to CSV
    csv_path = REPORT_DIR / "audio_audit_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=audit_results[0].keys())
        writer.writeheader()
        writer.writerows(audit_results)
    
    # Save to Markdown
    md_path = REPORT_DIR / "audio_audit_report.md"
    with open(md_path, 'w') as f:
        f.write("# Phase 10 Audio Audit Report\n\n")
        f.write("| File | MD5 | Sample Rate | Duration | LUFS | RMS | Peak | Centroid | Roll-off |\n")
        f.write("|------|-----|-------------|----------|------|-----|------|----------|----------|\n")
        for r in audit_results:
            f.write(f"| {r['file']} | {r['md5'][:8]}... | {r['sample_rate']} | {r['duration']:.2f}s | {r['LUFS']:.2f} | {r['RMS']:.4f} | {r['Peak']:.4f} | {r['Spectral_Centroid_Mean']:.0f} | {r['Spectral_Rolloff_Mean']:.0f} |\n")

    print(f"Audit complete. Reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()
