# Phase 17: BWC No-Normalization Experiment

## Overview
Phase 17 is a targeted experimental phase designed to isolate and evaluate the impact of audio normalization (specifically `LoudNorm`) on the end-to-end processing pipeline for Body Worn Camera (BWC) audio. By bypassing the standard normalization step used in previous iterations (e.g., Phase 11A), this phase provides a direct comparison of how raw (but repaired) signal characteristics influence Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Multi-Scale Diarization Decoder (MSDD).

## Objective
The primary objective is to determine if the non-linear transformations introduced by LoudNorm are beneficial or detrimental to high-noise, low-fidelity BWC audio. Key questions addressed:
- Does raw audio improve VAD sensitivity at the cost of False Alarms?
- How does the absence of normalization affect WhisperX's word error rate?
- Does the diarization component (NeMo MSDD) perform better on original dynamic ranges?

## Key Findings
- **VAD Sensitivity:** Skipping normalization significantly reduced the **Miss Rate** (from 28.78% to 20.49%), indicating that LoudNorm was likely suppressing low-amplitude speech segments.
- **False Alarm Trade-off:** The reduction in Miss Rate came at the cost of a doubled **False Alarm Rate** (from 7.66% to 16.74%), as background noise was more frequently misidentified as speech.
- **Diarization Improvement:** Despite the ASR regression (WER increased from 30.0% to 33.7%), the overall **Diarization Error Rate (DER)** improved (from 46.79% to 44.93%), primarily driven by better speech coverage and lower confusion.
- **Conversational Metrics:** WDER and cpWER showed marginal improvements, suggesting that the "No-Norm" approach might be superior for structural speaker attribution even if lexical accuracy slightly suffers.

## Implementation Summary
The phase utilizes a modular Python-based pipeline:
1. **Audio Repair:** Downsampling to 16kHz Mono.
2. **VAD:** Pyannote 3.1.
3. **ASR:** WhisperX (Large-v2).
4. **Diarization:** NeMo MSDD.
5. **Fusion:** Time-based alignment of words and speaker turns.
