# Phase 14: BWC Fix (Hardened Pipeline)

## Overview
Phase 14 represents the implementation of the "Hardened Pipeline" specifically optimized for Body-Worn Camera (BWC) audio. This phase addresses the unique challenges of forensic audio, including extreme volume disparities between the officer and subject, high background noise, and acoustic drift.

The core objective of this phase is to move beyond generic diarization and ASR by implementing a **Grounded Diarization** approach using Oracle Voice Activity Detection (VAD) injection into a Multi-Scale Diarization Decoder (MSDD).

## Key Objectives
- **Solve Identity Collapse:** Prevent the diarizer from merging the officer and subject into a single speaker label due to noise or volume differences.
- **Improve Attribution Accuracy:** Utilize word-level alignment fused with multiscale embeddings for precise speaker attribution.
- **Forensic Reliability:** Produce a transcript that is legally defensible and structurally sound for law enforcement oversight.

## Components
The pipeline is divided into 7 distinct steps:
1.  **Step 0: Audio Repair:** Downsampling and mono conversion for consistency.
2.  **Step 1: Normalization:** Loudness normalization to -20dB to stabilize embeddings.
3.  **Step 2: VAD Grounding:** High-recall voice activity detection using Pyannote 3.1.
4.  **Step 3: Oracle RTTM:** Conversion of VAD segments into a proposal format for the diarizer.
5.  **Step 4: WhisperX ASR:** Large-v3 transcription with phoneme-level alignment.
6.  **Step 5: MSDD Speaker Discovery:** Neural diarization using NVIDIA NeMo's TitaNet and MSDD models.
7.  **Step 6: Fusion:** Heuristic-based merging of word timestamps and speaker boundaries.

## Location
- **Implementation:** `experiments/phase14_bwc_fix/`
- **Output Data:** `experiments/phase14_bwc_fix/outputs/`
- **Diarization Data:** `experiments/phase14_bwc_fix/diarization/`
