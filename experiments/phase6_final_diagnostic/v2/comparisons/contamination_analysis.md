# Phase 6 vs Phase 6 V2 Contamination Analysis

## Metric Comparison

| Enhancer      |   Orig_WER |   V2_WER |   WER_Delta_% |   Orig_DER |   V2_DER |   DER_Delta_% |   Orig_VAD_F1 |   V2_VAD_F1 |   VAD_F1_Delta_% |
|:--------------|-----------:|---------:|--------------:|-----------:|---------:|--------------:|--------------:|------------:|-----------------:|
| baseline      |     0.4074 |   0.3008 |        -26.17 |     0.6991 |   0.6064 |        -13.26 |        0.7729 |      0.8022 |             3.79 |
| deepfilternet |     0.3996 |   0.4121 |          3.13 |     0.7041 |   0.6647 |         -5.6  |        0.7752 |      0.7695 |            -0.74 |
| speechbrain   |     0.425  |   0.4922 |         15.81 |     0.6898 |   0.7057 |          2.31 |        0.7566 |      0.7114 |            -5.97 |

## Key Findings

### 1. Is bwc_norm.wav different from bwc_norm_v2.wav?
YES. MD5 hashes and acoustic metrics (Spectral Centroid, RMS) differ significantly.

### 2. Is bwc_norm.wav acoustically closer to bwc_deepfilternet.wav or bwc_norm_v2.wav?
bwc_norm.wav is acoustically much closer to bwc_deepfilternet.wav (1732 Hz vs 1672 Hz) than to the clean bwc_norm_v2.wav (1309 Hz).
This confirms that the original baseline was contaminated with enhanced audio.

### 3. Did the original contamination materially affect WER/DER/etc.?
YES. Baseline WER improved from 0.4074 to 0.3008 (a -26.17% reduction in error).
Baseline DER improved from 0.6991 to 0.6064 (a -13.26% reduction in error).

### 4. Do the original Phase 6 conclusions still hold?
NO. The original conclusions suggested that the baseline was poor and enhancement provided marginal gains. The clean rerun shows the baseline is actually quite strong, and current enhancement methods (DeepFilterNet, SpeechBrain) actually DEGRADE performance relative to clean normalization.
