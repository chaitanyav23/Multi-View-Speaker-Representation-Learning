# Phase 10 Future Direction

## Remaining Limitations
*   **Fusion Fragility:** We lack a mathematically robust way to assign speaker identities to word-level timestamps that accounts for micro-overlaps.
*   **Dataset Generality:** All advanced testing (Phases 4-10) has been exclusively on `bwc.wav`. We do not know if the Hardened Architecture will generalize to multi-speaker, high-quality audio like the "Karen" dataset.

## Next Phase: Phase 11 DFN Ablation
Phase 11 will revisit the DeepFilterNet component. While Phase 10 proved it hurts Diarization when Oracle VAD is used, we need to definitively ablate its impact on the *VAD stage itself* before completely removing it from the pipeline.

## Long-term Work
*   **The "Karen" Evaluation:** We must test the Phase 9 Production baseline against the `karen.wav` dataset (4 speakers, clean audio, rapid turn-taking) to ensure the `num_speakers: 2` constraint and the VAD parameters haven't overfit the BWC problem space.
*   **Probabilistic Fusion:** Developing a fusion algorithm that uses the WhisperX confidence score and the MSDD probability matrix to make a statistically informed assignment rather than a simple geometric overlap.
