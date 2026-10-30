import librosa
import numpy as np
import sys

def analyze_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # LUFS (approximate using RMS)
    # Note: librosa doesn't have native LUFS, using pyloudnorm if available, otherwise RMS
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr) # create BS.1770 meter
        lufs = meter.integrated_loudness(y)
    except ImportError:
        lufs = "pyloudnorm not installed"
        
    # RMS
    rms = np.sqrt(np.mean(y**2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
    
    # Peak
    peak = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    
    # Dynamic Range (Peak to RMS ratio, Crest Factor)
    crest_factor = 20 * np.log10(peak / rms) if rms > 0 else 0
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_cent = np.mean(cent)
    
    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mean_rolloff = np.mean(rolloff)

    # Simple Frequency Response (Mean magnitude per frequency bin)
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_mag = np.mean(D, axis=1)
    
    # Find dominant frequency bin
    dom_freq_idx = np.argmax(mean_mag)
    dominant_freq = freqs[dom_freq_idx]

    return {
        "LUFS": lufs,
        "RMS (dB)": rms_db,
        "Peak (dB)": peak_db,
        "Dynamic Range (Crest Factor dB)": crest_factor,
        "Mean Spectral Centroid (Hz)": mean_cent,
        "Mean Spectral Roll-off (Hz)": mean_rolloff,
        "Dominant Frequency (Hz)": dominant_freq
    }

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    res1 = analyze_audio(file1)
    res2 = analyze_audio(file2)
    
    print(f"--- Analysis for {file1} ---")
    for k, v in res1.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    print(f"\n--- Analysis for {file2} ---")
    for k, v in res2.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
