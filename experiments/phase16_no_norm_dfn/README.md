# Phase 16: No-Normalization + DeepFilterNet (DFN) Experiment

## Overview
Phase 16 explores the impact of omitting the standard loudness normalization step and instead utilizing **DeepFilterNet3 (DFN)** for real-time speech enhancement and noise suppression. This experiment is conducted on the "Karen" dataset, which presents significant acoustic challenges including high ambient noise and varying speaker distances.

The primary hypothesis is that loudness normalization may inadvertently amplify background noise or introduce artifacts that degrade Voice Activity Detection (VAD) and Speaker Diarization performance in noisy environments. By using DFN directly on the raw repaired audio, we aim to achieve cleaner speech signals for downstream processing.

## Objective
- **Ablation of Normalization:** Determine if skipping the traditional normalization step improves signal integrity for VAD.
- **DFN Integration:** Evaluate the efficacy of DeepFilterNet3 as a primary enhancement stage.
- **End-to-End Pipeline Validation:** Measure the impact of these preprocessing changes on ASR accuracy (WER), Diarization (DER), and Speaker Attribution.

## Key Components
1. **Audio Repair:** Standard downsampling to 16kHz Mono.
2. **DeepFilterNet3:** Deep-learning based noise suppression.
3. **Pyannote 3.1 VAD:** State-of-the-art voice activity detection.
4. **WhisperX ASR:** Segmented transcription and word-level alignment.
5. **NeMo MSDD:** Multi-scale Speaker Diarization Decoder using TitaNet-L embeddings.
6. **Dual-Branch Fusion:** Mapping speaker identity to aligned word segments.

## Summary of Findings
- **VAD Improvement:** Significant reduction in Miss Rate compared to Phase 11A baseline, indicating better capture of low-volume speech.
- **ASR Trade-off:** Slight increase in Word Error Rate (WER), likely due to DFN-induced artifacts or the lack of gain-normalization affecting WhisperX's sensitivity.
- **Diarization Complexity:** Increased Speaker Confusion suggests that while DFN improves detection, it may subtly alter speaker embeddings, making multi-speaker separation more challenging.
