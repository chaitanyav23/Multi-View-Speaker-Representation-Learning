import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def normalize_audio(data, target_db=-20.0):
    rms = np.sqrt(np.mean(data**2))
    if rms == 0:
        return data, 0.0
    curr_db = 20 * np.log10(rms)
    gain_db = target_db - curr_db
    gain = 10**(gain_db / 20)
    normalized = data * gain
    return normalized, gain_db

def main():
    INPUT_PATH = Path("raw_audio/rich_karen.mp3")
    OUTPUT_PATH = Path("experiments/phase18_tich_karen/audio/rich_karen_normalized.wav")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {INPUT_PATH}...")
    # Load and resample to 16kHz for NeMo/Whisper
    data, sr = librosa.load(str(INPUT_PATH), sr=16000)
    
    print("Normalizing audio...")
    normalized_data, gain = normalize_audio(data)
    
    print(f"Saving to {OUTPUT_PATH} (Gain applied: {gain:.2f} dB)...")
    sf.write(str(OUTPUT_PATH), normalized_data, 16000)
    print("Done.")

if __name__ == "__main__":
    main()
