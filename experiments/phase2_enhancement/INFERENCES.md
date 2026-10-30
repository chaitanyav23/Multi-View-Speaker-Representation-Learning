# Phase 2 Inferences

## Scientific Findings
1.  **Denoising is Destructive for Foundation Models:** Large-scale transformer models like Whisper are trained on hundreds of thousands of hours of noisy audio. They learn to internally "denoise" and focus on speech. External neural denoising likely removes the very features these models use for robust decoding.
2.  **SNR vs. Artifact Ratio:** There is a "tipping point" where the benefit of noise reduction is outweighed by the cost of artifact introduction. For current models, this point is very low, making raw audio almost always preferable for ASR.

## Engineering Findings
1.  **Stage-Isolation is Critical:** The fact that VAD improved slightly while ASR degraded significantly suggests that we should use different audio "versions" for different stages (e.g., Use DFN for VAD, but Raw for ASR).
2.  **Standardization of Filenames:** Renaming DFN output files back to original names is a simple but vital engineering pattern to prevent "filepath rot" in complex pipelines.

## Architectural Bottlenecks
*   **The "Clean Audio" Trap:** Engineers often assume clean audio = better results. Phase 2 proved that "natural noise" is often safer than "neural silence."
*   **Lack of Dual-Pathing:** The current pipeline only allows one audio file per run. An ideal architecture would allow the VAD to see one version of audio while the ASR sees another.
