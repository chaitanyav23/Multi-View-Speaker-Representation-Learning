# Phase 11 Pipeline Flow

## Sequential Processing Steps

1.  **Audio Generation** (`step1_generate_audio.py`)
    *   Produces `bwc_normalized.wav` and `bwc_deepfilternet.wav`.

2.  **Grounding Generation** (`step2_generate_grounding.py`)
    *   Runs Pyannote VAD on Normal audio -> `11A_grounding.json`.
    *   Runs Pyannote VAD on DFN audio -> `11B_grounding.json`.

3.  **Matrix Execution** (`step3_run_pipelines.py`)
    *   Executes Pipeline 11A (Norm/Norm/Norm).
    *   Executes Pipeline 11B (DFN/DFN/Norm).
    *   Executes Pipeline 11C (Norm/DFN/Norm).

4.  **Evaluation Mapping** (`step4_evaluate.py`)
    *   Generates base metric CSVs for all three pipelines using `evaluate_conversational_timeline.py`.

5.  **Report Compilation** (`step5_compile_report.py`)
    *   Aggregates the individual CSVs into `phase11_dfn_ablation_summary.md`.

## Data Artifacts
*   `audio_variants/`: Holds the two fundamental test signals.
*   `grounding/`: Holds the two fundamental VAD maps.
*   `reports/`: Contains the definitive answers to the DFN hypothesis.
