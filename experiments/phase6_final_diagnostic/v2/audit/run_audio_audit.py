
import os
import hashlib
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path

def get_hashes(file_path):
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
            sha256_hash.update(byte_block)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def get_audio_metrics(file_path):
    data, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=data, sr=sr)
    
    # LUFS
    meter = pyln.Meter(sr)
    # pyloudnorm expects (samples, channels)
    if data.ndim == 1:
        data_pyln = data[:, np.newaxis]
    else:
        data_pyln = data.T
    lufs = meter.integrated_loudness(data_pyln)
    
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    crest_factor = peak / rms if rms > 0 else 0
    
    # Spectral
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr))
    
    # Dominant Frequency (simplified)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sr)
    dominant_freq = freqs[np.argmax(np.abs(fft))]
    
    return {
        "duration": duration,
        "lufs": lufs,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "dominant_freq": dominant_freq
    }

def audit_files(files_dict):
    results = []
    for label, path in files_dict.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        print(f"Auditing: {label}")
        md5, sha256 = get_hashes(path)
        metrics = get_audio_metrics(path)
        metrics.update({
            "label": label,
            "path": str(path),
            "md5": md5,
            "sha256": sha256
        })
        results.append(metrics)
    return results

if __name__ == "__main__":
    V2_DIR = Path("experiments/phase6_final_diagnostic/v2/audio_preprocessed")
    ORIG_DIR = Path("experiments/phase6_final_diagnostic/audio_preprocessed")
    
    files_to_audit = {
        "v2_norm": V2_DIR / "bwc_norm_v2.wav",
        "v2_dfn": V2_DIR / "bwc_deepfilternet_v2.wav",
        "v2_sb": V2_DIR / "bwc_speechbrain_v2.wav",
        "orig_norm": ORIG_DIR / "bwc_norm.wav",
        "orig_dfn": ORIG_DIR / "bwc_deepfilternet.wav",
        "orig_sb": ORIG_DIR / "bwc_speechbrain.wav"
    }
    
    results = audit_files(files_to_audit)
    df = pd.DataFrame(results)
    
    # Save to CSV for easy comparison
    report_path = Path("experiments/phase6_final_diagnostic/v2/audit/audio_provenance_data.csv")
    df.to_csv(report_path, index=False)
    
    # Generate Markdown report
    md_report_path = Path("experiments/phase6_final_diagnostic/v2/audit/audio_provenance_report.md")
    with open(md_report_path, "w") as f:
        f.write("# Audio Provenance Audit Report\n\n")
        f.write("## Audio Metrics Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Comparison Analysis\n\n")
        
        # Check if v2_norm is different from orig_norm
        v2_norm_md5 = df[df['label'] == 'v2_norm']['md5'].values[0]
        orig_norm_md5 = df[df['label'] == 'orig_norm']['md5'].values[0]
        
        if v2_norm_md5 != orig_norm_md5:
            f.write("### ALERT: bwc_norm.wav and bwc_norm_v2.wav are DIFFERENT.\n")
            f.write(f"Original MD5: {orig_norm_md5}\n")
            f.write(f"V2 MD5: {v2_norm_md5}\n\n")
        else:
            f.write("### SUCCESS: bwc_norm.wav and bwc_norm_v2.wav are identical.\n\n")

    print(f"Audit complete. Reports saved to {report_path.parent}")
