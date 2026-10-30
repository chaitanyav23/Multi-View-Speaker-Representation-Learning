# Phase 2: Audio Enhancement Benchmarking

## Phase Objective
Phase 2 aimed to improve pipeline performance by preprocessing raw audio with neural enhancement models. The core goal was to reduce the False Alarms observed in Phase 1 (Pyannote VAD) and improve ASR clarity in high-noise Body-Worn Camera (BWC) footage.

## Historical Motivation
Phase 1 results showed that environmental noise in BWC data leads to significant False Alarms and "hallucinated" words in the ASR stage. It was hypothesized that removing this noise before the VAD and ASR stages would lead to cleaner segments and more accurate transcriptions.

## Hypothesis being Tested
Neural audio enhancement (DeepFilterNet3 and MetricGAN+) will significantly reduce WER and DER in noisy BWC audio by suppressing non-speech background noise while preserving the spectral characteristics of the target speakers.

## What Changed from Previous Phase
*   **Preprocessing Stage:** Added an explicit enhancement step before Stage 1 (VAD).
*   **Models:** Integrated `DeepFilterNet3` (state-of-the-art real-time denoising) and `SpeechBrain MetricGAN+` (generative enhancement).

## Expected Outcomes
*   Significant reduction in VAD False Alarms.
*   Improved ASR accuracy (lower WER) on BWC data.
*   Clearer separation of speakers (lower DER).

## Actual Outcomes
*   **Unexpected Degradation:** Enhancement models generally **increased WER** (from 0.30 to 0.37+).
*   **Artifact Introduction:** While noise was reduced, the models introduced robotic artifacts and over-suppressed quiet speech, causing the ASR (WhisperX) to fail more frequently.
*   **Conclusion:** Neural denoising at this stage is "destructive" for conversational ASR on forensic data.
