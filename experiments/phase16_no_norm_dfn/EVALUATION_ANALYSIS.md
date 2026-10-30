# Evaluation Analysis: Phase 16

Phase 16 utilizes a comprehensive suite of evaluation metrics to assess the impact of the No-Normalization and DFN approach.

## 1. Voice Activity Detection (VAD)
- **Methodology:** Frame-level comparison (10ms resolution) between system mask and ground truth mask.
- **Primary Metrics:**
  - **Recall:** Measures the ability to capture all speech. A high recall indicates low **Miss Rate**.
  - **Precision:** Measures the accuracy of detected speech. A high precision indicates low **False Alarm (FA) Rate**.
  - **F1-Score:** The harmonic mean of Precision and Recall.

## 2. Automated Speech Recognition (ASR)
- **Methodology:** JIWER-based alignment of system transcript vs. ground truth transcript.
- **Metrics:**
  - **Word Error Rate (WER):** (Substitutions + Deletions + Insertions) / Total Words.
  - **Sub/Del/Ins Breakdown:** Critical for understanding if the system is "hearing" the wrong things (Sub) or "missing" speech (Del).

## 3. Diarization Error Rate (DER)
- **Calculation:** `DER = (Miss + False Alarm + Speaker Confusion) / Total Speech Time`.
- **Nuance in Phase 16:** Since MSDD uses the system's VAD segments (via `oracle_vad: True`), the **Miss** and **False Alarm** components of DER are directly inherited from the VAD performance, while **Speaker Confusion** reflects the MSDD model's ability to distinguish speakers within those segments.

## 4. Speaker Attribution Accuracy
- **Methodology:** Evaluates if the word-level speaker labels in the final output match the ground truth.
- **Metric:** Percentage of correctly attributed words.

## 5. Conversational Metrics (Joint Evaluation)
- **cpWER (Concatenated Permutation WER):** Evaluates ASR quality while accounting for speaker assignments. It handles speaker-level permutations to find the best match between system and reference.
- **WDER (Word Diarization Error Rate):** Measures the percentage of words that were correctly transcribed but assigned to the wrong speaker.
- **SA-WER (Speaker-Attributed WER):** Similar to cpWER, provides a holistic view of the "who said what" accuracy.
