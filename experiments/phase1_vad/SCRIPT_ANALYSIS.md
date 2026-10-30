# Phase 1 Script Analysis

## Core Orchestrator: `run_phase1.py`
This script manages the execution flow of the benchmarking process.
*   **Logic:** Iterates through a list of VAD models (`silero`, `pyannote`), runs the specific generator, and then invokes the `run_final_hardened_pipeline.py` using the resulting grounding directory.
*   **Evaluation:** Automatically calls `evaluate_conversational_timeline.py` after each run to generate metrics named `phase1_silero` and `phase1_pyannote`.

## VAD Engines

### `pipeline_vad_silero.py`
*   **Model:** Loaded locally from `external/silero-vad` (version specified in `torch.hub.load`).
*   **Function `apply_min_cut`:** 
    *   **Logic:** If a segment is longer than `max_len`, it searches for the frame with the minimum VAD score within a central window and recursively splits there.
    *   **Purpose:** Prevents over-long segments while minimizing the chance of splitting in the middle of a word.
*   **Function `merge_segments`:**
    *   **Logic:** Aggregates short detected segments into larger chunks (up to 28 seconds).
    *   **Rationale:** Whisper performs better with more context; very short segments (e.g., 0.5s) increase ASR overhead and reduce accuracy.

### `pipeline_vad_pyannote.py`
*   **Model:** `pyannote/segmentation-3.0` via HuggingFace.
*   **Pipeline:** `VoiceActivityDetection` wrapper.
*   **Hyperparameters:**
    *   `min_duration_on: 0.1s`: Minimum speech duration to be considered valid.
    *   `min_duration_off: 0.1s`: Minimum silence duration to trigger a split.
*   **Logic:** Similar to Silero, it applies the `merge_segments` logic to the raw pyannote output to ensure consistent segment lengths across benchmarks.

## Support Utilities
*   **`scripts.utils.audio_utils`:** Used for standardized audio loading at 16kHz.
*   **`scripts.utils.device`:** Abstracts CUDA/CPU selection.

## I/O Specifications
*   **Input:** `raw_audio/*.wav`
*   **Output (Intermediate):** `experiments/phase1_vad/grounding_<engine>/<audio_name>_grounding.json`
*   **Output (Final):** `experiments/phase1_vad/outputs_<engine>/*.json`
