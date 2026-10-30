"""
experiments/phase6_final_diagnostic/generate_spectrograms.py
=========================================================
Generates Mel-spectrograms for BWC audio variants to visualize 
the impact of normalization and enhancement.
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_spectrogram(audio_path, output_path, title):
    print(f"Generating spectrogram for: {title}")
    # Load audio at 16k SR (consistent with project)
    y, sr = librosa.load(str(audio_path), sr=16000)
    
    # Compute Mel-spectrogram
    # n_mels=128 provides high frequency resolution
    # fmax=8000 (Nyquist for 16k SR)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(15, 5))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = Path("experiments/phase6_final_diagnostic")
    output_dir = base_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("raw_audio/bwc.wav", "BWC Raw Audio"),
        ("experiments/phase6_final_diagnostic/audio_preprocessed/bwc_norm.wav", "BWC Normalized (Phase 6)"),
        ("experiments/phase6_final_diagnostic/audio_preprocessed/bwc_deepfilternet.wav", "BWC DeepFilterNet Enhanced"),
        ("experiments/phase6_final_diagnostic/audio_preprocessed/bwc_speechbrain.wav", "BWC SpeechBrain Enhanced"),
    ]
    
    for audio_path, title in files:
        if not Path(audio_path).exists():
            print(f"Warning: {audio_path} not found.")
            continue
            
        out_name = Path(audio_path).stem + "_spectrogram.png"
        generate_spectrogram(audio_path, output_dir / out_name, title)

    print(f"Spectrograms saved to: {output_dir}")

if __name__ == "__main__":
    main()
