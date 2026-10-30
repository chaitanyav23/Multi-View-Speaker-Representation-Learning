# Phase 15: Future Direction

## Recommended Next Steps

### 1. Adaptive Normalization
Instead of a "one-size-fits-all" LoudNorm (Step 1), explore adaptive normalization that only triggers if the input signal's peak or RMS falls below a certain threshold. This would preserve the integrity of high-quality recordings while fixing problematic ones.

### 2. MSDD Parameter Optimization
Given the high Confusion rate in Phase 15, future research should focus on tuning the MSDD hyper-parameters specifically for non-normalized audio.
- **Multiscale Weights:** Experiment with different weights for the multiscale embedding extractor in `step5_msdd.py`.
- **Sigmoid Thresholds:** Adjust the `sigmoid_threshold` in the MSDD config to be more or less conservative based on the signal strength.

### 3. VAD Sensitivity Tuning
The 68.9% False Alarm rate in VAD is a major area for optimization.
- **Threshold Adjustment:** Experiment with higher `onset`/`offset` thresholds in the Pyannote pipeline to reduce noise-to-speech classification.
- **Preprocessing Filters:** Test the impact of simple high-pass or band-pass filters in Step 0 (Audio Repair) to remove low-frequency rumble that might be tripping up the VAD.

### 4. Hybrid Normalization/Ablation Study
Perform a direct "side-by-side" comparison on the *same* dataset (e.g., Phase 15 vs. a hypothetical Phase 15B with normalization) to definitively quantify the "Normalization Tax." This would involve running identical pipeline parameters, with the only variable being the presence/absence of Step 1.

### 5. Speaker Embedding Refinement
Investigate using alternative embedding models (e.g., Wespeaker, ECAPA-TDNN) in the MSDD stage to see if any are inherently more robust to non-normalized signals than TitaNet-Large.
