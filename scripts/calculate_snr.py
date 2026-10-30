import json
import numpy as np
import librosa
from pathlib import Path
import argparse

def calculate_snr(audio_path, grounding_path):
    y, sr = librosa.load(audio_path, sr=None)
    with open(grounding_path, 'r') as f:
        grounding = json.load(f)
    
    segments = grounding['segments']
    if not segments:
        return None
    
    # Create masks for signal and noise
    signal_mask = np.zeros_like(y, dtype=bool)
    for seg in segments:
        start_idx = int(seg['start'] * sr)
        end_idx = int(seg['end'] * sr)
        signal_mask[start_idx:end_idx] = True
        
    signal_part = y[signal_mask]
    noise_part = y[~signal_mask]
    
    if len(signal_part) == 0 or len(noise_part) == 0:
        return None
        
    rms_signal = np.sqrt(np.mean(signal_part**2))
    rms_noise = np.sqrt(np.mean(noise_part**2))
    
    if rms_noise == 0:
        return float('inf')
        
    snr_db = 20 * np.log10(rms_signal / rms_noise)
    return snr_db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    grounding_dir = Path(args.grounding_dir)
    
    results = []
    # Match based on grounding files
    for g_path in grounding_dir.glob("*.json"):
        stem = g_path.stem.replace("_grounding", "")
        audio_path = audio_dir / f"{stem}.wav"
        
        if audio_path.exists():
            snr = calculate_snr(audio_path, g_path)
            results.append({"file": stem, "SNR_dB": round(snr, 2) if snr is not None else "N/A"})
            
    print(f"\n--- SNR Analysis: {args.label} ---")
    for res in results:
        print(f"{res['file']}: {res['SNR_dB']} dB")

if __name__ == "__main__":
    main()
