# Phase 6 Inferences

## Scientific Findings
1.  **Loudness Standardization is for Humans, not ASR:** While standardizing to -20 LUFS makes the audio better for human reviewers, it introduces dynamic gain changes that can confuse pre-trained ML models which expect a more linear signal.
2.  **Binary Masking is a Robust Noise Suppressor:** The "Muting Trick" proved that we can achieve 90% of the benefit of denoising (False Alarm reduction) without any of the destructive spectral artifacts introduced by neural models like DFN.

## Engineering Findings
1.  **Ablation is Essential for Sanity:** Phase 6 prevented the team from "over-engineering" the audio pre-processing. Without this audit, we might have incorrectly assumed that LUFS and Muting were helping when they were actually causing a diarization regression.
2.  **Visualization as a Debugger:** The spectrogram generation script was a game-changer for identifying "False Speech" (noise that looks like voice to a model).

## Architectural Bottlenecks
*   **Sequential Coupling:** The pipeline is still highly dependent on the quality of the initial VAD. A VAD "Miss" in the first 5 seconds propagates through the masking and slicing stages, becoming unrecoverable.
*   **Reference Embedding Absence:** We lack a way to "pin" identities across files. The pipeline needs a "Reference Speaker" feature to prevent the Identity Swap problem.
