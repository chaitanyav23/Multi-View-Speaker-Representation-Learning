# Phase 9 Script Analysis

## Primary Orchestrator: `run_phase9_production.py`
This script serves as the project's production entry point.
*   **`run_stage3_grounded_audio`:**
    *   **Logic:** Implements the Binary Muting algorithm. It reads the WAV file, creates a zero-filled array of the same length, and then copies over the audio only for the indices specified in the Grounding JSON.
    *   **Purpose:** To eliminate "embedding poisoning" in MSDD caused by sirens and wind noise during non-speech intervals.
*   **`run_stage4_hardened_asr`:**
    *   **Parameter:** `condition_on_previous_text=False`.
    *   **Rationale:** In high-noise environments, previous context can "leak" into current silence, causing hallucinations. Disabling this makes the ASR more independent and forensicly reliable.
*   **`run_stage6_oracle_msdd`:**
    *   **Config:** Hardcodes `num_speakers: 2` (for the BWC officer/suspect scenario).
    *   **Parameters:** Uses `multiscale_weights: [1, 2, 3, 3, 3]` to prioritize small-window interjections.

## Forensic Evaluation: `eval_phase9_metrics.py`
*   **Comprehensive Logging:** Unlike previous eval scripts, this one logs 19 distinct metrics—from VAD Miss Rate to SA-WER—providing a complete picture of pipeline health.
*   **Diagnostic Mapping:** Automatically correlates ASR errors with Diarization confusion to identify the exact cause of cpWER spikes.

## Artifact Manifest
*   `audio_normalized/`: Standardized input.
*   `grounding_master/`: The "Master Manifest" defining where speech lives.
*   `audio_grounded/`: The "Muted" signal used for Diarization.
*   `outputs_asr/`: Raw word timestamps.
*   `outputs_msdd/`: Raw speaker turns.
*   `outputs_final/`: The integrated conversational timeline.
