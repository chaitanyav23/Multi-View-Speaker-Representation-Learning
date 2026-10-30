# Phase 13: Karen End-to-End Generalization Benchmark

## Overview
Phase 13 focuses on evaluating the generalization capabilities of the conversational intelligence pipeline developed in previous phases (specifically the Phase 11A / Phase 10C architecture) on an unseen dataset, referred to as the "Karen" dataset. This phase is critical for verifying that the pipeline's performance is not over-fitted to the Body Worn Camera (BWC) data and can handle different acoustic environments, speaker counts, and conversational dynamics.

## Objectives
- **Generalization Testing:** Apply the best-validated pipeline to a new, higher-quality recording.
- **System Stability:** Verify that the multi-scale diarization (MSDD) and maximum-overlap fusion logic remain robust.
- **Performance Benchmarking:** Compare metrics (WER, DER, cpWER) against the BWC baseline to identify environment-specific performance deltas.
- **Bottleneck Identification:** Determine if the primary failure modes in the Karen dataset differ from those in the BWC dataset.

## Key Results
- **ASR Improvement:** WER on the Karen dataset was **14.67%**, a significant improvement over the ~30% WER typical for BWC data, indicating high intelligibility.
- **Diarization Performance:** DER was recorded at **40.70%**, driven primarily by speaker confusion and over-clustering.
- **Speaker Discovery:** The MSDD model discovered **6 speakers**, whereas the ground truth contains 4, confirming a tendency towards over-clustering in this specific configuration.
- **Conversational Accuracy:** cpWER was **26.85%**, which is a substantial improvement over the BWC Phase 11A baseline (50.32%).

## Implementation Summary
The pipeline utilizes a modular architecture:
1.  **Preprocessing:** FFmpeg LoudNorm for audio consistency.
2.  **VAD:** Pyannote 3.1 for robust speech activity detection.
3.  **ASR:** WhisperX Large-v3 with Wav2Vec2-based alignment for precise word-level timestamps.
4.  **Diarization:** NeMo Multi-scale Diarization Decoder (MSDD) with TitaNet_Large embeddings.
5.  **Fusion:** A custom maximum-overlap fusion algorithm to attribute speakers to ASR words.

## Directory Structure
- `audio/`: Normalized audio files.
- `grounding/`: VAD-based grounding and oracle RTTM files.
- `diarization/`: MSDD outputs and raw RTTMs.
- `outputs/`: Final speaker-attributed transcripts in JSON and TXT formats.
- `reports/`: Categorized metrics (ASR, VAD, Diarization, Attribution).
- `diagnostics/`: Detailed error regions and confusion matrices.
