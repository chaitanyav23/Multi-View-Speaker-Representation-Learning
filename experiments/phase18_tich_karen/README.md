# Phase 18: Tich Karen - Conversational Intelligence Pipeline Stress Test

## Overview
Phase 18 represents a critical stress test of the Conversational Intelligence pipeline (derived from Phase 13) using high-complexity, noisy body-worn camera (BWC) style audio. The primary goal was to evaluate the robustness of the standard architecture against environmental noise (sirens, background chatter, and movement) and to identify the primary failure modes in the diarization and transcription chain.

## Objectives
- **Robustness Testing**: Evaluate the performance of the Phase 13 architecture on the `rich_karen` dataset.
- **Ablation & Comparison**: Systematically compare four variations of the pipeline (Normalized, No-Norm, Oracle-VAD, and ASR-Grounded) to isolate performance bottlenecks.
- **Root Cause Analysis**: Use embedding analysis and diagnostic metrics to understand high Diarization Error Rate (DER) drivers.
- **Strategic Direction**: Determine the next evolutionary step for the pipeline (e.g., Speech Enhancement).

## Implementation Summary
The phase is implemented as a 10-step sequential pipeline:
1.  **Preprocessing**: Audio normalization and resampling.
2.  **GT Processing**: Formatting ground truth for evaluation.
3.  **VAD Grounding**: Execution of Pyannote 3.1 VAD.
4.  **Oracle Generation**: Creation of RTTM files from ground truth for upper-bound testing.
5.  **WhisperX Transcription**: ASR and phoneme alignment.
6.  **MSDD Diarization**: Multi-scale Diarization Decoder using TitaNet-Large.
7.  **Fusion**: Merging ASR and Diarization results.
8.  **VAD Evaluation**: Precision, Recall, and F1 scoring.
9.  **ASR Evaluation**: WER and WIL scoring.
10. **Diarization Evaluation**: DER, Miss, False Alarm, and Confusion analysis.

## Key Findings
- **VAD Bottleneck**: The primary source of error is Voice Activity Detection (VAD) "False Alarms" caused by environmental noise.
- **Diarizer Integrity**: The MSDD model remains highly accurate (18% DER) when provided with clean (Oracle) VAD segments.
- **Normalization Stability**: RMS normalization at -20dB is beneficial for ASR and does not negatively impact diarization.

## Directory Structure
- `scripts/`: Implementation steps (1-10).
- `audio/`: Processed audio files.
- `reports/`: Detailed evaluation metrics and comparison reports.
- `outputs/`: Final JSON/TXT transcripts and RTTM files.
- `diagnostics/`: Embedding projections and interactive analysis tools.
