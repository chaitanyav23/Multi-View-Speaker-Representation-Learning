# Phase 7 Script Analysis

## Master Orchestrator: `run_phase7_optimal.py`
This script is the most complex in the project, managing five independent models.
*   **`prepare_conditioned_audio`:**
    *   **Logic:** Normalizes to LUFS-20 then runs DeepFilterNet.
    *   **Podcast Exception:** Automatically skips DFN for files with "podcast" in the name to avoid crashing GPUs on long-form audio.
*   **`run_soft_vad`:**
    *   **Function:** Extends detected speech boundaries by a `padding` value (default 0.1s).
    *   **Rationale:** Reduces the chance of word-clipping during the Manual Slicing stage.
*   **`run_msdd_fixed2`:**
    *   **Config:** Uses `split_infer: True` and `infer_batch_size: 16`. 
    *   **Optimization:** The multiscale weights `[1, 2, 3, 3, 3]` bias the model toward finer temporal resolutions, which is better for fast-paced turns in noisy audio.
*   **`run_contextual_asr`:**
    *   **Mechanism:** Implements a sliding state machine where `prev_text` from segment $N$ is used as the `initial_prompt` for segment $N+1$.
    *   **Repetition Guard:** Detects and logs if the model starts repeating the same string across multiple segments.

## Evaluation Suite
### `eval_phase7_metrics.py`
*   **Logic:** Aggregates WER, DER, SA-WER, and cpWER.
*   **Focus:** Direct comparison of the "Optimal" pipeline against the Phase 5 "Breakthrough" metrics.

### `extract_phase7_audit.py`
*   **Logic:** Similar to Phase 6, it extracts forensic timestamps for every speaker turn to verify attribution accuracy.
