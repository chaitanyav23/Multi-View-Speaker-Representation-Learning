import librosa
import soundfile as sf
from pathlib import Path

def main():
    INPUT_PATH = Path("raw_audio/rich_karen.mp3")
    # Store in the new no_norm sub-directory
    OUTPUT_PATH = Path("experiments/phase18_tich_karen/no_norm/audio/rich_karen_no_norm.wav")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {INPUT_PATH} (NO NORMALIZATION)...")
    # Load and resample to 16kHz (mandatory for models) but do NOT normalize
    data, sr = librosa.load(str(INPUT_PATH), sr=16000)
    
    print(f"Saving to {OUTPUT_PATH}...")
    sf.write(str(OUTPUT_PATH), data, 16000)
    print("Done.")

if __name__ == "__main__":
    main()
