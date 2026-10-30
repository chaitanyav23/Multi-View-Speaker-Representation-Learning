# Phase 9 Inferences

## Scientific Findings
1.  **Diarization requires Isolation:** The success of the "Muting Trick" in maintaining a 0.52 DER proves that neural diarizers perform best when the signal is physically cleaned of inter-turn noise.
2.  **ASR requires Linearity:** The failure of DFN and the regression of LUFS-20 (compared to Peak Norm) confirm that large transformer models prefer the most linear representation of the original signal possible.

## Engineering Findings
1.  **Orchestration over Optimization:** In a production setting, the ability to repeat an experiment exactly (using Phase 9's single-script flow) is more valuable than achieving a slightly better metric via manual "one-off" tuning.
2.  **The Metadata Trail:** Saving the manifest, the RTTM, and the intermediate JSONs is essential for forensic accountability.

## Architectural Bottlenecks
*   **The VRAM Wall:** Stage 6 (MSDD) and Stage 4 (ASR) are run sequentially in the production script but could be run in parallel on a dual-GPU system to halve the runtime.
*   **Static Number of Speakers:** The `num_speakers: 2` constraint is a bottleneck for generalization. The pipeline needs a "Speaker Discovery" pass.
