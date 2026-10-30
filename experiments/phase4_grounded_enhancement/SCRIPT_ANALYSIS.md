# Phase 4 Script Analysis

## Orchestrator: `run_phase4.py`
This script manages a multi-variant benchmark.
*   **Normalization Logic:** (Implicitly handled by the presence of `audio_normalized` directory or pre-run librosa scripts).
*   **Reuse of Phase 2 Pipelines:** It directly calls `experiments/phase2_enhancement/pipeline_enhancement_*.py`, demonstrating the reusability of the modular enhancement blocks.
*   **Master Grounding Integration:** Explicitly passes `--grounding-dir experiments/phase4_grounded_enhancement/grounding_master` to `run_final_hardened_pipeline.py`.
*   **Comparison Loop:** Iterates through `baseline`, `speechbrain`, and `deepfilternet` to produce three distinct output sets.

## Key Sub-Processes
*   **`run_final_hardened_pipeline.py`:** The stable production entry point. In Phase 4, it is used as a benchmarking tool to run the ASR and Diarization stages on the pre-processed audio.
*   **`evaluate_conversational_timeline.py`:** Used to generate the `phase4_baseline`, `phase4_speechbrain`, and `phase4_deepfilternet` reports.

## Artifact Management
*   **`audio_normalized/`:** Audio files with standardized peak amplitudes.
*   **`audio_deepfilternet/`:** Audio that has been both normalized AND denoised.
*   **`outputs_*/`:** Final conversational JSONs for each experiment variant.
