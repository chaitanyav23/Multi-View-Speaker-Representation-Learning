"""
preprocess_audio.py
===================
Audio preprocessing script for BWC pipeline evaluation.
Supports Loudness Normalization and Bandpass Filtering.
"""

import argparse
import json
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfilt
from datetime import datetime

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Ensure cutoffs are within (0, 1) normalized range
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def normalize_audio(data, target_db=-20.0):
    """Peak normalization to target dB."""
    rms = np.sqrt(np.mean(data**2))
    if rms == 0:
        return data, 0.0
    
    curr_db = 20 * np.log10(rms)
    gain_db = target_db - curr_db
    gain = 10**(gain_db / 20)
    
    # Apply gain and clip to prevent distortion if necessary
    # (though target_db is usually low enough)
    normalized = data * gain
    return normalized, gain_db

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for BWC pipeline")
    parser.add_argument("--input-dir", default="raw_audio", help="Input audio directory")
    parser.add_argument("--output-dir", required=True, help="Output audio directory")
    parser.add_argument("--normalize", action="store_true", help="Apply loudness normalization")
    parser.add_argument("--target-db", type=float, default=-20.0, help="Target dB for normalization")
    parser.add_argument("--low-cutoff", type=float, help="Low cutoff for bandpass filter (Hz)")
    parser.add_argument("--high-cutoff", type=float, help="High cutoff for bandpass filter (Hz)")
    parser.add_argument("--filter-order", type=int, default=4, help="Order of the butterworth filter")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs = []
    
    for audio_path in input_dir.glob("*.wav"):
        print(f"Processing: {audio_path.name}")
        
        # Load audio (preserve original sr)
        data, sr = librosa.load(str(audio_path), sr=None, mono=False)
        
        # Convert to mono for processing if stereo, though BWC is often mono
        # We should preserve channels as per requirement
        was_mono = (len(data.shape) == 1)
        
        processed_data = data.copy()
        gain_applied = 0.0
        
        # 1. Normalization
        if args.normalize:
            if was_mono:
                processed_data, gain_applied = normalize_audio(processed_data, args.target_db)
            else:
                # Normalize each channel by the same gain (based on max RMS)
                rms_vals = [np.sqrt(np.mean(processed_data[i]**2)) for i in range(data.shape[0])]
                max_rms = max(rms_vals)
                if max_rms > 0:
                    curr_db = 20 * np.log10(max_rms)
                    gain_db = args.target_db - curr_db
                    gain = 10**(gain_db / 20)
                    processed_data *= gain
                    gain_applied = gain_db

        # 2. Bandpass Filter
        if args.low_cutoff and args.high_cutoff:
            if was_mono:
                processed_data = bandpass_filter(processed_data, args.low_cutoff, args.high_cutoff, sr, args.filter_order)
            else:
                for i in range(data.shape[0]):
                    processed_data[i] = bandpass_filter(processed_data[i], args.low_cutoff, args.high_cutoff, sr, args.filter_order)

        # Save output
        out_path = output_dir / audio_path.name
        # soundfile expect [frames, channels]
        if was_mono:
            sf.write(out_path, processed_data, sr)
        else:
            sf.write(out_path, processed_data.T, sr)
            
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "input_path": str(audio_path),
            "output_path": str(out_path),
            "gain_db_applied": round(gain_applied, 2),
            "low_cutoff": args.low_cutoff,
            "high_cutoff": args.high_cutoff,
            "filter_order": args.filter_order,
            "normalized": args.normalize
        })

    # Save metadata log
    log_df = pd.DataFrame(logs)
    log_path = output_dir / "preprocessing_metadata.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Metadata log saved to: {log_path}")

if __name__ == "__main__":
    main()
