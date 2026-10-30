# Evaluation Analysis: Phase 18 (Tich Karen)

## Methodology
Phase 18 utilizes a multi-metric evaluation framework to assess the health of each pipeline stage. The evaluation is performed by comparing the pipeline's automated outputs against a manually verified Ground Truth (GT) derived from `rich_karen.txt`.

## 1. Voice Activity Detection (VAD) Evaluation
- **Primary Metric**: F1 Score (Harmonic mean of Precision and Recall).
- **Secondary Metrics**: Precision (How much of the detected speech was actual speech), Recall (How much of the actual speech was detected).
- **Temporal Tolerance**: 100ms (collars).
- **Analysis**:
    - **F1 Score**: 0.7940
    - **Recall**: 0.8996 (Excellent; the model captures almost all speech).
    - **Precision**: 0.7105 (Moderate; the model hallucinates speech in noisy environments).
- **Observation**: The gap between Recall and Precision confirms that the system is "over-sensitive," triggering on environmental noise.

## 2. Automatic Speech Recognition (ASR) Evaluation
- **Primary Metric**: Word Error Rate (WER).
- **Secondary Metric**: Word Information Loss (WIL).
- **Normalization**: Standardized text cleaning (removing punctuation, lowercase) is applied before WER calculation to ensure acoustic accuracy is measured rather than formatting differences.
- **Analysis**:
    - **WER**: 22.33%
    - **WIL**: 29.52%
- **Observation**: A 22% WER on BWC audio is considered acceptable, as BWC audio often features overlapping speech and muffled acoustics which Whisper `large-v3` handles with high robustness.

## 3. Diarization (DER) Evaluation
- **Primary Metric**: Diarization Error Rate (DER).
- **Components**:
    - **Missed Detection**: Speech present in GT but not in output.
    - **False Alarm (FA)**: Speech present in output but not in GT.
    - **Speaker Confusion**: Speech assigned to the wrong speaker.
- **Analysis**:
    - **Standard DER**: 62.86% (Driven by 36% False Alarms and 15% Confusion).
    - **Oracle DER**: 18.10% (The theoretical performance of the diarizer if VAD was perfect).
- **Observation**: The massive difference between Standard DER (62%) and Oracle DER (18%) mathematically proves that **VAD is the primary failure mode**, not the speaker embedding or clustering logic.

## 4. Diagnostic Metrics
- **Speaker Overlap**: The pipeline identifies the percentage of the file where multiple speakers are detected simultaneously.
- **Silhouette Score (Internal)**: Measure of how well the speaker embeddings (TitaNet) are clustered in the latent space.
