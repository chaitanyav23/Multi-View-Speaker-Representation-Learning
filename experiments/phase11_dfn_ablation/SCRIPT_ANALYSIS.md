# Phase 11 Script Analysis

## Main Orchestrator: `step3_run_pipelines.py`
This script executes the three control variants in a single pass to eliminate environment variables as a source of error.
*   **Pipeline 11A Execution:** 
    *   Reads `11A_grounding.json` (generated from `bwc_normalized.wav`).
    *   Executes `run_msdd` on `bwc_normalized.wav`.
    *   Executes `run_asr_and_align` on `bwc_normalized.wav`.
*   **Pipeline 11B Execution:**
    *   Reads `11B_grounding.json` (generated from `bwc_deepfilternet.wav`).
    *   Executes `run_msdd` on `bwc_normalized.wav` (MSDD always uses NORM to avoid total failure).
    *   Executes `run_asr_and_align` on `bwc_deepfilternet.wav`.
*   **Pipeline 11C Execution:**
    *   Reuses `11A_grounding.json`.
    *   Reuses the `msdd_11A` output directly from memory.
    *   Executes `run_asr_and_align` on `bwc_deepfilternet.wav`.
    *   **Scientific Value:** This is a perfect A/B test. The temporal boundaries and speaker assignments are identical to 11A. Any change in the output text is solely due to the ASR engine's interpretation of the DFN audio.

## Evaluation Generation: `step5_compile_report.py`
*   **Logic:** Reads all generated CSVs and automatically answers the three core research questions.
*   **Automation:** By automating the interpretation of the metrics, the team ensured that the conclusion (to ban DFN) was derived mathematically, free from confirmation bias.
