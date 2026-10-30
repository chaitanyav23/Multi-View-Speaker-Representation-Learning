# Phase 5 Script Analysis

## Core breakthrough: `pipeline_msdd_oracle.py`
This script implements the "Forced Grounding" logic.
*   **`oracle_vad` parameter:** When set to `True`, the `NeuralDiarizer` skips its internal VAD stage and directly uses the `rttm_filepath` provided in the manifest.
*   **Runtime Logging:** Includes timing logic to prove the efficiency of the oracle approach (averaging ~40s per file).
*   **Clustering Parameters:**
    *   `max_num_speakers: 20`: Allows the model to find the optimal number of people in a crowd.
    *   `sparse_search_volume: 30`: Optimizes the multiscale search space.

## Conversion Utility: `convert_grounding_to_rttm.py`
*   **Logic:** Reads a standard project Grounding JSON and formats it into the space-delimited RTTM standard.
*   **Structure:** Every VAD segment is given a dummy speaker label (`SPEAKER_00`), which MSDD subsequently overwrites with its predicted labels.

## Diagnostics: `diarization_diagnostics.py`
*   **Purpose:** Forensic verification of speaker attribution.
*   **Logic:** Compares the final prediction to ground truth and generates a CSV breakdown of errors.
*   **Output:** `diarization_diagnostics.csv`, which identifies the exact timestamps of Misses, False Alarms, and Confusion.

## Orchestrator: `run_phase5.py`
*   **Sequence:**
    1. Prepare RTTMs from Master Grounding.
    2. Run Baselines (Pyannote Auto/Fixed).
    3. Run advanced MSDD path.
    4. Run `run_hardened_with_custom_diar.py` to merge WhisperX text with MSDD speakers.
    5. Trigger `evaluate_conversational_timeline.py` for comparative metrics.
