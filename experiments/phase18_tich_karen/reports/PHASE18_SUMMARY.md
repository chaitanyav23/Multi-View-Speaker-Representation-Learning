# Phase 18: Tich Karen Analysis Summary

## Pipeline Overview
- **Audio**: `rich_karen.mp3` (Normalized to 16kHz WAV)
- **Ground Truth**: `rich_karen.txt`
- **Architecture**: Phase 13 MSDD + WhisperX Large-v3

## Evaluation Results

### 1. Voice Activity Detection (VAD)
- **F1 Score**: 0.7940
- **Precision**: 0.7105
- **Recall**: 0.8996
- **Note**: The pipeline detected more speech (497s) than the GT (393s), likely due to background noise or the model being more sensitive.

### 2. Automatic Speech Recognition (ASR)
- **Word Error Rate (WER)**: 0.2233
- **Word Information Loss (WIL)**: 0.2952
- **Note**: Decent performance on a noisy body-worn camera style audio.

### 3. Diarization (MSDD)
- **Diarization Error Rate (DER)**: 0.6286
- **Missed Detection**: 0.1051
- **False Alarm**: 0.3649
- **Speaker Confusion**: 0.1585
- **Note**: High DER is largely driven by False Alarms and Confusion. The MSDD model predicted 8 speakers while the GT had 10 speakers.

## Diagnostics
- **Embedding Analysis**: PCA and UMAP projections have been generated in `embedding_analysis/`.
- **Interactive Tool**: `analyze_embeddings_interactive.py` can be used to explore the clusters.

## Conclusion
The pipeline successfully processed the new `rich_karen` sample. While ASR and VAD are performing reasonably well, the diarization (MSDD) is struggling with speaker count overestimation, leading to a high DER. Further tuning of the `sigmoid_threshold` or `max_num_speakers` in Step 6 might improve results.
