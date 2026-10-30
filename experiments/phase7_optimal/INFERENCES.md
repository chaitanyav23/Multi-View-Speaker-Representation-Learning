# Phase 7 Inferences

## Scientific Findings
1.  **Recall/Precision Seesaw:** Improving VAD Recall via DFN and Padding is possible, but it comes at the direct expense of ASR and Diarization Precision if the same audio version is used throughout.
2.  **Model Preference for Raw Noise:** Foundation models (Whisper) and robust Diarizers (MSDD) have an inherent preference for the "raw noise distribution" they were trained on. Neural denoising moves the signal into an "uncanny valley" that these models cannot navigate.

## Engineering Findings
1.  **Complexity vs. Performance:** An integrated orchestrator like `run_phase7_optimal.py` is easier to run but harder to optimize than individual step-scripts because it hides the intermediate degradations.
2.  **Context-on-Noise Risk:** Contextual prompting is a dangerous feature for noisy data. It turns random noise into coherent (but hallucinated) text.

## Architectural Bottlenecks
*   **The Single-File Bottleneck:** The current architecture forces a choice between "Raw" and "Enhanced." It cannot yet handle multiple audio streams for different stages.
*   **Sequential Latency:** The five-stage pipeline is slow. Running DFN + Pyannote + MSDD + WhisperX + Aligner takes significant wall-clock time per minute of audio.
