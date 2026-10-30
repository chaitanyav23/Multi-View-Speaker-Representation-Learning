import librosa
import numpy as np

norm, sr = librosa.load(
    "experiments/phase6_final_diagnostic/audio_preprocessed/bwc_norm.wav",
    sr=None
)

dfn, sr2 = librosa.load(
    "experiments/phase6_final_diagnostic/audio_preprocessed/bwc_deepfilternet.wav",
    sr=None
)

min_len = min(len(norm), len(dfn))

corr = np.corrcoef(
    norm[:min_len],
    dfn[:min_len]
)[0, 1]

print(f"Correlation: {corr:.6f}")
