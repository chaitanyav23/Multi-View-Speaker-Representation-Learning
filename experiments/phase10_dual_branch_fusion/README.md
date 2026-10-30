# Phase 10: Dual-Branch Fusion Architecture

## Phase Objective
Phase 10 aimed to definitively resolve the "Audio Paradox" discovered in earlier phases (where denoised audio helps Diarization but hurts ASR). The objective was to build a true "Dual-Branch" pipeline that feeds raw audio to the ASR engine and enhanced audio to the Diarization engine, then merges the results using a sophisticated custom fusion algorithm.

## Historical Motivation
Phases 7 and 8 confirmed that a "Single-Path" architecture using enhanced audio imposes a hard cap on ASR accuracy (WER ~0.37). Phase 9 proved that a single path using raw audio is stable but still suffers from minor Identity Swaps. Phase 10 hypothesized that separating the streams would provide the "Best of Both Worlds."

## Hypothesis being Tested
A pipeline utilizing **Raw Audio for ASR** (to maintain word clarity) and **DeepFilterNet Audio for Diarization** (to improve speaker separation), merged via a custom word-overlap voting algorithm, will achieve a lower cpWER than any single-path architecture.

## What Changed from Previous Phase
*   **True Forking:** The orchestrator actively runs WhisperX on `bwc_normalized.wav` and NeMo MSDD on a separate `oracle_rttm` stream.
*   **Custom Fusion Logic (`custom_fusion.py`):** Abandoned simple midpoint interpolation in favor of a complex word-duration overlap algorithm.
*   **Segment-Level Majority Voting:** Added a post-processing step that assigns a single speaker ID to an entire WhisperX segment based on the majority vote of its constituent words.

## Outcomes (The "Fusion Failure")
*   **Hypothesis Disproven:** The dual-branch approach (Phase 10B) achieved a **cpWER of 0.53**, which is worse than the Phase 5 baseline (0.45).
*   **Logic Regression:** The `custom_fusion` script's "Majority Vote" logic actively destroyed sub-segment cross-talk, forcing entire overlapping sentences to be attributed to a single speaker.
*   **DFN Diarization Penalty:** Phase 10C (Raw/Raw) outperformed Phase 10B (DFN/Raw), proving that DFN actually *hurts* MSDD diarization when used in this specific configuration.
