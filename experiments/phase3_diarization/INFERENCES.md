# Phase 3 Inferences

## Scientific Findings
1.  **Clustering vs. Decoding:** Agglomerative clustering is a "global" operation that is vulnerable to the "mean" noise profile of a file. Sequence-to-sequence decoding (like MSDD) is a "local" context-aware operation that is potentially much more robust to drift, provided it isn't tripped up by its own VAD.
2.  **Identity Collapse is a Forensics Dealbreaker:** Any pipeline that collapses to one speaker is a complete failure for conversational intelligence, even if its DER looks superficially "acceptable."

## Engineering Findings
1.  **The "Oracle" Requirement:** Neural models like MSDD are too complex to be run "unconstrained" on noisy forensic data. They require **External Grounding** to constrain their search space and prevent computation loops.
2.  **Modular Fusion Works:** The success of the `run_hardened_with_custom_diar.py` script proved that we can effectively decouple ASR and Diarization, allowing us to use the best model for each specific task without needing a single end-to-end model.

## Architectural Bottlenecks
*   **VAD Coupling:** NeMo's monolithic `NeuralDiarizer` architecture makes it difficult to inject external VAD results without deep configuration hacks.
*   **Real-time Constraint:** MSDD is too slow for real-time use in its multiscale configuration, but essential for forensic "Ground Truth" generation.
