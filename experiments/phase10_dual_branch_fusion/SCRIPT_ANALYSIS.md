# Phase 10 Script Analysis

## Main Orchestrator: `step3_run_pipelines.py`
This script executes three distinct pipeline variants:
*   **`run_pipeline_10A`:** Re-evaluates Pyannote 3.1 using the new base `vad_segments`.
*   **`run_pipeline_10B`:** The core "Dual Branch" test. It calls `run_msdd_fixed2` on `NORM_WAV`, but calls `run_asr_and_align` on `DFN_WAV`. 
*   **`run_pipeline_10C`:** The "Raw Control". It calls both MSDD and ASR on `NORM_WAV`.

## The Fusion Logic: `custom_fusion.py`
This script was a major departure from previous phases.
*   **Input:** Requires an `aligned_res` dictionary (from WhisperX) where every word has a start/end time.
*   **Overlap Calculation:** For every word, it iterates through all MSDD segments to calculate `overlap_dur`. 
    *   **Case A (Single Candidate):** Word falls entirely within one speaker's turn.
    *   **Case B (Multiple Candidates):** Word spans a speaker boundary. It assigns the speaker with the maximum overlap and logs a `speaker_transition` diagnostic.
    *   **Case C (No Overlap):** Word falls outside MSDD segments. It finds the nearest speaker within a `tolerance` (0.5s).
*   **The "Majority Vote" Overwrite:**
    ```python
    for seg in aligned_res["segments"]:
        # ... count speakers ...
        seg["speaker"] = Counter(speakers).most_common(1)[0][0]
    ```
    *   **Impact:** This loop overrides the careful word-level assignments. If a segment has 4 words from Speaker A and 2 words from Speaker B, all 6 words are attributed to Speaker A in the final output. This single block of code caused the cpWER regression.
