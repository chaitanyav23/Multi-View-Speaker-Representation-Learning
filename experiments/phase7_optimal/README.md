# Phase 7: The "Optimal" Architecture Synthesis

## Phase Objective
Phase 7 aimed to construct the definitive production-ready pipeline by integrating the high-performing modules from all previous experiments. It was designed as a "Best-of-Breed" synthesis, combining LUFS normalization, DeepFilterNet denoising, Soft-grounded VAD, Oracle MSDD, and Contextual ASR.

## Historical Motivation
After identifying Oracle MSDD as the best diarization engine (Phase 5) and DeepFilterNet as a useful diarization aid (Phase 4), the team needed to build a single, unified orchestrator that could handle the entire flow in a stable, repeatable manner.

## Hypothesis being Tested
A multi-stage pipeline that uses **Conditioned Audio** (Normalized + DFN) for all stages—including VAD, Diarization, and ASR—will produce a more stable and contextually coherent conversational transcript than using raw audio.

## What Changed from Previous Phase
*   **Integrated Orchestration:** Moved from individual step scripts to a single master orchestrator (`run_phase7_optimal.py`).
*   **Soft VAD:** Introduced temporal padding (0.1s) to prevent word clipping at segment boundaries.
*   **Contextual ASR:** Enabled `condition_on_previous_text` in WhisperX to help the model maintain speaker-specific terminology and conversational flow.
*   **Context Guard:** Implemented a prompt buffer that feeds the previous 50 words into the current segment's transcription to maintain state.

## Outcomes
*   **Stability Success:** The pipeline is highly robust and handles various noise levels without crashing.
*   **Metric Insight:** Confirmation that **Global DFN is destructive**. Even with context guards, the ASR stage's WER remained high (0.38) on conditioned audio.
*   **Baseline Establishment:** Phase 7 established the upper bound for a "Single-Path" architecture, proving that the pipeline must eventually move to a "Dual-Path" model (Raw for ASR, DFN for Diarization).
