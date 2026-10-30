
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_spectrogram(audio_path, output_path, title):
    print(f"Generating spectrogram for: {title}")
    y, sr = librosa.load(str(audio_path), sr=16000)
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
    V2_DIR = Path("experiments/phase6_final_diagnostic/v2")
    output_dir = V2_DIR / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("experiments/phase6_final_diagnostic/audio_preprocessed/bwc_norm.wav", "BWC Normalized (Original Phase 6)"),
        ("experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_norm_v2.wav", "BWC Normalized (Phase 6 V2 Clean)"),
        ("experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_deepfilternet_v2.wav", "BWC DeepFilterNet (V2)"),
        ("experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_speechbrain_v2.wav", "BWC SpeechBrain (V2)"),
    ]
    
    for audio_path, title in files:
        if not Path(audio_path).exists():
            continue
        out_name = Path(audio_path).stem + "_spectrogram.png"
        generate_spectrogram(audio_path, output_dir / out_name, title)

    print(f"Spectrograms saved to: {output_dir}")

if __name__ == "__main__":
    main()
