# Phase 15: No-Normalization Experiment

## Overview
Phase 15 is a critical diagnostic phase designed to evaluate the impact of audio normalization on the overall pipeline performance. In previous phases, loudness normalization (specifically `ffmpeg-loudnorm`) was a standard preprocessing step (Step 1). Phase 15 intentionally bypasses this step, moving directly from basic audio repair (resampling and mono-conversion) to Voice Activity Detection (VAD).

The primary goal is to establish a performance baseline for the hardened pipeline architecture (VAD -> ASR -> MSDD -> Fusion) when operating on "raw" (but repaired) audio. This helps determine if the normalization process introduced artifacts or if it was strictly beneficial for the downstream ASR and diarization models.

## Objectives
- **Ablation of Normalization:** Isolate the effect of removing the `loudnorm` stage.
- **Baseline Establishment:** Provide high-fidelity metrics for the `karen.wav` dataset using the latest pipeline configuration.
- **Robustness Check:** Test the resilience of WhisperX (ASR) and NeMo MSDD (Diarization) to variations in input signal levels.

## Key Changes
- **Step 1 Removal:** The `step1_loudnorm.py` script is omitted from the execution flow.
- **Step 0 Modification:** `run_phase15.py` performs audio repair (16kHz, mono) but does not apply any gain adjustments or dynamic range compression.
- **Direct Input:** The pipeline consumes the repaired but un-normalized audio file `karen_repaired.wav`.

## Results Summary
The phase successfully processed the `karen.wav` dataset, yielding the following core metrics:
- **WER:** 14.35%
- **DER:** 40.93%
- **cpWER:** 26.47%

Detailed analysis of these results compared to previous normalized runs allows for a definitive assessment of the necessity of the LoudNorm stage.
