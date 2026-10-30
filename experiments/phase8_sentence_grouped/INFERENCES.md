# Phase 8 Inferences

## Scientific Findings
1.  **Linguistic vs. Acoustic Context:** Providing WhisperX with a "sentence" (linguistic context) is beneficial, but the "acoustic context" provided by a 0.5s buffer is even more critical for encoder stability.
2.  **Recall/Precision Tradeoff in Forensics:** A 1.0s merge gap maximizes **Recall** (finding all speech) but minimizes **Identity Precision** (correctly assigning turns). In a forensic context, high Recall is generally preferred to ensure no evidence is missed.

## Engineering Findings
1.  **The Context Guard is Mandatory:** Without the repetition guard and word-limit on the `initial_prompt`, contextual ASR quickly devolves into infinite loops in noisy BWC audio.
2.  **Modular Pipeline Reuse:** The ability to swap only the VAD merging logic while keeping the rest of the Phase 7 architecture intact proved the value of the modular project structure.

## Architectural Bottlenecks
*   **The Artifact Wall (Final Proof):** Phase 8 definitively proved that even with optimal context and optimal diarization, the project cannot achieve a low cpWER on DFN-enhanced audio. **The ASR branch MUST use raw audio.**
