# Error Analysis: Phase 18 (Tich Karen)

This document provides a deep dive into the specific failure modes identified during the `rich_karen` stress test.

## 1. VAD False Alarms (Environmental Noise)
- **Symptom**: 36.49% False Alarm rate in Diarization.
- **Root Cause**: The `rich_karen.mp3` file contains significant background noise, including police sirens, wind noise, and distant chatter. Pyannote 3.1's default sensitivity triggers on these non-speech acoustic patterns.
- **Impact**: Because the VAD passes these "noisy" segments to the diarizer, the MSDD model is forced to assign a speaker label to them. This results in "phantom speech" in the final transcript where sirens are transcribed as hallucinated text or assigned to a random speaker ID.

## 2. Speaker Over-Counting
- **Symptom**: The MSDD model predicted 8 speakers, while the Ground Truth contains ~4.
- **Root Cause**: In multi-scale diarization, noisy segments with distinct acoustic profiles (e.g., a siren vs. a wind gust) can be interpreted as unique speakers if their embeddings are sufficiently distant from the main speech clusters. 
- **Evidence**: PCA projections in `embedding_analysis/` show tight clusters for the main speakers (Speakers 0, 1, 2) but numerous "diffuse" clusters for the remaining 5 IDs.

## 3. Speaker Confusion (Acoustic Similarity)
- **Symptom**: 15.85% Confusion rate.
- **Root Cause**: This occurs primarily during overlapping speech and fast turn-taking. When two speakers have similar vocal ranges (e.g., two adult females in the "Karen" audio), the TitaNet-Large embeddings may overlap in the latent space.
- **Spatial Analysis**: UMAP visualizations show that the boundaries between Speaker 1 and Speaker 2 are blurred, indicating that the multi-scale decoder is struggling with speaker assignment at the segment boundaries.

## 4. Forced Alignment Drift
- **Symptom**: Words at the beginning or end of segments occasionally have incorrect timestamps.
- **Root Cause**: WhisperX forced alignment relies on a phoneme model. In very noisy sections, the phoneme alignment can "slip" if the acoustic signal of the word is masked by a siren, leading to words being assigned to the wrong speaker during the Fusion step.

## Mitigation Strategies (Technical Recommendations)
1.  **Denoising (DeepFilterNet)**: Apply a neural speech enhancement model *before* VAD. This would suppress the sirens and wind, likely reducing False Alarms by 20-30%.
2.  **Threshold Tuning**: Increase the `sigmoid_threshold` in MSDD from 0.7 to 0.85 to be more conservative about assigning speaker labels in low-confidence regions.
3.  **Speaker Limiting**: If the expected speaker count is known (e.g., for BWC, usually 2-3), hard-cap the `max_num_speakers` parameter in the NeMo config.
