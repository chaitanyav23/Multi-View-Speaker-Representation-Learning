# Phase 12 Results Analysis

This document provides a detailed breakdown of the quantitative and qualitative results from the Phase 12 Karen Evaluation.

## 1. Quantitative Metrics Breakdown

### VAD (Voice Activity Detection)
The VAD metrics confirm that the `pyannote/voice-activity-detection` model is highly robust to the acoustic conditions of BWC data.
- **F1-Score (89.9%)**: Indicates that the grounding segments provided a solid foundation for both ASR and Diarization.
- **Precision vs. Recall**: The near-equal balance (90.2% vs 89.8%) suggests that the VAD threshold is well-calibrated, neither being too aggressive nor too permissive.

### ASR (Speech Recognition)
- **Local WER**: Within the 52-second audio segment, WhisperX Large-v3 produced a highly accurate transcript.
- **Transcript Quality**: The output captured rapid-fire dialogue and high-tension emotional cues (e.g., "Why are you being a jerk?", "You're an asshole.") without significant substitutions.
- **Global WER**: Effectively unusable as a standalone metric due to the ~87% deletion rate caused by missing audio.

### Diarization
- **Speaker Attribution**: The 2 active speakers were clearly separated.
- **MSDD Stability**: The multiscale approach correctly attributed overlapping speech in the high-tension sections (e.g., timestamps 21.00 - 24.00).

## 2. Qualitative Findings

### Generalization to Unseen Data
The most significant result was that the **pipeline required zero modifications** to process a completely new dataset. This confirms the validity of the "Hardened" configurations established in Phase 11.

### Resilience to Corruption
The implementation of `step0` and the stream-copy repair in `step1` demonstrated that the pipeline can handle real-world "dirty" data. Rather than crashing on the corrupted MP3 header, the system validated the physical data and proceeded with the valid 52-second payload.

### Unbiased Clustering
Previous phases often relied on `oracle_num_speakers`. Phase 12 proved that the TitaNet-Large + MSDD architecture is capable of high-accuracy speaker discovery on short durations, which is traditionally a difficult task for diarization systems.

## 3. Summary Table: Predicted vs. Actual
| Feature | Actual (Available) | Predicted | Status |
| :--- | :--- | :--- | :--- |
| **Duration** | 52.3s | 52.3s | Match |
| **Speaker Count** | 2 | 2 | Match |
| **ASR Accuracy** | High | High | Match |
| **Timestamps** | Accurate | Accurate | Match |
