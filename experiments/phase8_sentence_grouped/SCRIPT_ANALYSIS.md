# Phase 8 Script Analysis

## Main Orchestrator: `run_phase8_main.py`
This script introduces a "Context-Heavy" approach to processing.
*   **Modified `run_soft_vad`:**
    *   **Logic:** Implements a two-pass cleanup.
    *   Pass 1: Deletes segments shorter than `min_duration` (0.05s).
    *   Pass 2: Merges any segments where `start(next) - end(curr) <= 1.0s`.
    *   **Rationale:** To provide the ASR with a continuous "sentence" instead of fragmented words.
*   **Modified `run_contextual_asr`:**
    *   **Buffer Management:** Uses `buffer_s=0.5`. 
    *   **Mechanism:** Extracts a larger audio chunk for the Whisper `large-v3` encoder, but only preserves the text for the specific VAD segment.
*   **Shared Utils:** Direct reuse of `prepare_conditioned_audio` (Normalization + DFN) from Phase 7.

## Diagnostic Scripts

### `eval_phase8_metrics.py`
*   **Logic:** Standard project evaluation.
*   **Metric Shift:** Highlights the **VAD F1-Score (0.84)** as a key achievement of this phase.

### `extract_phase8_audit.py`
*   **Forensic Verification:** Analyzes whether the 1.0s merge gap caused distinct speakers to be merged into a single segment (The "Turn-Merge Error").

## Key Hyperparameters
*   `vad-merge-gap`: 1.0s (Critical for sentence formation).
*   `asr-buffer`: 0.5s (Critical for encoder synchronization).
*   `min-duration`: 0.05s (Allows for extremely short interjections like "mm-hmm").
