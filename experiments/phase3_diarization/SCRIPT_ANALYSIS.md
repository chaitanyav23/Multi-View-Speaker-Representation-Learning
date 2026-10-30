# Phase 3 Script Analysis

## Core Model: `pipeline_msdd.py`
This script encapsulates the integration with NVIDIA NeMo.
*   **NeuralDiarizer:** The primary class from `nemo.collections.asr.models`. It handles the entire multi-stage process.
*   **Manifest Logic:** Automatically generates the NeMo-required JSON manifest for each input audio file.
*   **Configuration (`OmegaConf`):**
    *   **VAD Settings:** Uses `vad_multilingual_marblenet`. Parameters like `window_length_in_sec: 0.15` and `threshold: 0.5` were tuned but failed to handle siren interference.
    *   **Speaker Embeddings:** Uses `titanet_large`. It extracts multiscale embeddings using a set of sliding windows (1.5s, 1.2s, 1.0s, 0.8s, 0.5s).
    *   **MSDD Decoder:** Uses `diar_msdd_telephonic`. This is the core sequence model that predicts speaker activity for each frame while considering the surrounding multiscale context.
*   **Output Parsing:** Parses the predicted `.rttm` file and converts it into a project-standard JSON list of segments.

## Orchestrator: `run_phase3.py`
*   **Logic:** Attempts to run the MSDD pipeline and the Pyannote 3.1 baseline in parallel.
*   **Evaluation:** Triggers `evaluate_conversational_timeline.py` on the outputs. Note that if `pipeline_msdd.py` fails or times out, the evaluation step receives empty directories.

## Merger: `run_hardened_with_custom_diar.py`
*   **Purpose:** This script was developed to facilitate "Modular Diarization." 
*   **Logic:** It loads the word-level JSON from the ASR stage and the speaker-segment JSON from the MSDD stage. It then performs a temporal join, assigning each word to the speaker active at that word's midpoint.
*   **Significance:** This script allowed the team to continue using the high-quality WhisperX ASR while swapping out the diarization engine, a pattern that became central to the project's later success.
