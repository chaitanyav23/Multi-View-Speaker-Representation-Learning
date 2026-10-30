# Phase 4: Grounded Enhancement Study

## Phase Objective
Phase 4 was a critical refinement phase aimed at resolving the conflicting results of Phase 2 (Enhancement) and Phase 3 (Diarization). The goal was to test if **normalization** combined with **denoising** could improve the overall conversational metrics (cpWER) by aiding the diarization stage, even if it slightly degraded raw ASR accuracy.

## Historical Motivation
Phase 2 showed that denoising hurt ASR, but Phase 3 showed that Diarization (identity collapse) was a massive issue. Phase 4 hypothesized that denoising might be a "necessary evil" that sacrifices some word-level clarity to achieve better speaker separation.

## Hypothesis being Tested
Applying DeepFilterNet3 to peak-normalized audio will result in a lower **cpWER** (Conversational WER) compared to raw audio, because the improvement in speaker attribution (lower WDER) will outweigh the degradation in transcription (higher WER).

## What Changed from Previous Phase
*   **Normalization:** Introduced a mandatory librosa peak-normalization step to -20dB before any other processing.
*   **Grounded Evaluation:** Used a "Master Grounding" (pre-vetted VAD segments) to isolate the impact of enhancement on the ASR and Diarization stages specifically.
*   **Metric Shift:** Shifted primary focus from WER to **cpWER** as the true measure of pipeline utility.

## Outcomes
*   **The cpWER Breakthrough:** DeepFilterNet (DFN) achieved a **relative improvement in cpWER (0.5054 -> 0.4711)** on BWC data.
*   **WDER Reduction:** DFN halved the Word Diarization Error Rate (0.1033 -> 0.0558), proving that denoising is highly effective for the diarization embedding model (Pyannote).
*   **Confirmation of ASR Degradation:** Confirmed that DFN still increases base WER (from 0.29 to 0.36), but the diarization gains make the overall transcript more usable.
*   **SpeechBrain Rejection:** SpeechBrain was officially rejected as it degraded both ASR and Diarization significantly.
