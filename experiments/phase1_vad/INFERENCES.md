# Phase 1 Inferences

## Scientific Findings
1.  **VAD is the Bottleneck for Forensics:** In low-SNR environments (BWC), the VAD stage accounts for ~30% of the total error. Improving ASR models has diminishing returns if the VAD fails to ground the speech segments correctly.
2.  **Segmentation Frequency:** Pyannote's frame-wise segmentation provides a more "human-like" breakdown of conversation than Silero's window-based scoring, especially for fast-paced turn-taking.

## Engineering Findings
1.  **Min-Cut Split Point Selection:** Using the minimum probability score as a split point is significantly more effective than splitting at fixed intervals (e.g., every 30s). Fixed splitting often cleaves words in half, leading to "word fragments" in ASR.
2.  **GPU Memory vs. Accuracy:** Pyannote v3.0 requires significantly more VRAM and compute time than Silero. For real-time applications, Silero is superior, but for high-accuracy research/offline processing, Pyannote is the clear choice.

## Architectural Bottlenecks
*   **Sequential Dependence:** The strict dependence of the entire pipeline on the Grounding JSON means a VAD failure is irrecoverable. Later phases should explore "soft grounding" or multi-pass VAD.
*   **Normalization Absence:** Phase 1 ran on raw audio. High-variance loudness in BWC likely confused the VAD thresholds.
