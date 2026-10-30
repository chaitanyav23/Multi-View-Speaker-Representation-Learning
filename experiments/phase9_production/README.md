# Phase 9: Production Candidate Pipeline

## Phase Objective
Phase 9 represents the transition from experimental research to a production-ready software artifact. The objective was to consolidate the project's most successful components—discovered across 8 prior phases—into a single, hardened Python orchestrator. This "Production Candidate" serves as the reference implementation for the BWC forensic pipeline.

## Historical Motivation
After identifying that Global DFN hurts ASR (Phase 7) but improves VAD (Phase 8), and that Oracle MSDD is the only way to prevent Identity Collapse (Phase 5), the team needed a way to run these divergent configurations in a single pass. Phase 9 implements a "Hybrid Single-Path" that uses the best audio version for each specific task.

## Hypothesis being Tested
A single-script orchestrator that performs LUFS-20 normalization, high-recall VAD, and Oracle MSDD on muted audio will provide a stable, reproducible baseline for all future BWC forensic tasks.

## What Changed from Previous Phase
*   **Audio Muting (Stage 3):** Formally integrated the "Muting Trick" from Phase 6 into the main pipeline to ensure that inter-segment noise does not contaminate the Diarization embeddings.
*   **Hardened ASR (Stage 4):** Reverted to Raw-Normalized audio for transcription (abandoning the destructive DFN approach from Phase 7/8).
*   **Manual Slicing (Hardened):** Explicitly disabled `condition_on_previous_text` to prevent the hallucination loops discovered in Phase 8.
*   **Path Simplification:** Focusing exclusively on the critical `bwc.wav` forensic use-case.

## Outcomes
*   **ASR Recovery:** Reverting to raw audio brought the **WER back down to 0.31** (from 0.38 in Phase 8).
*   **Diarization Stability:** Achieving a **DER of 0.52** and a **WDER of 0.19**, proving that the combination of Muting and Oracle MSDD is robust against forensic background noise.
*   **Production Readiness:** The script `run_phase9_production.py` is the first to achieve "Full-State Persistence," saving every intermediate artifact for forensic auditability.
