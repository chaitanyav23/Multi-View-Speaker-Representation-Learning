# Phase 2 Script Analysis

## Enhancement Pipelines

### `pipeline_enhancement_deepfilternet.py`
*   **Engine:** `DeepFilterNet3` (DFN).
*   **Implementation:** Invokes `python3 -m df.enhance` as a subprocess.
*   **Logic:**
    *   Walks the `audio_dir`.
    *   Enhances each `.wav` file.
    *   Renames the DFN-standard output (`*_DeepFilterNet3.wav`) back to the original filename to maintain compatibility with the downstream pipeline.
*   **Mechanism:** DFN uses a two-stage approach: a low-latency linear filter followed by a deep neural network to suppress non-linear noise.

### `pipeline_enhancement_speechbrain.py`
*   **Engine:** `SpectralMaskEnhancement` (MetricGAN+).
*   **Model:** `speechbrain/metricgan-plus-voicebank`.
*   **Logic:**
    *   Loads audio into memory using `torchaudio`.
    *   Applies `enhance_batch` to generate the clean mask.
    *   Saves result at 16kHz.
*   **Mechanism:** MetricGAN+ uses a GAN where the generator is trained to minimize the PESQ (Perceptual Evaluation of Speech Quality) loss directly.

## Orchestrator: `run_phase2.py`
*   **Flow:**
    1. Runs the enhancement script.
    2. Runs `custom_pipeline_stage1_vad.py` (Silero) on the enhanced audio.
    3. Runs the `run_final_hardened_pipeline.py`.
    4. Evaluates and generates reports named `phase2_deepfilternet` and `phase2_speechbrain`.

## Support Scripts
*   **`custom_pipeline_stage1_vad.py`:** A modular version of the Silero VAD script used to generate grounding on the fly for enhanced audio.
*   **`evaluate_conversational_timeline.py`:** The project-standard evaluation tool used to compute metrics against ground truth.
