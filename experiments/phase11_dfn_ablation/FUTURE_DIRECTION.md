# Phase 11 Future Direction

## Remaining Limitations
*   **The "Clean" Baseline:** Now that DFN is removed, we must accept the base WER of ~0.30 on BWC data. We cannot artificially lower it without harming cpWER.
*   **Dataset Lock-in:** All of this optimization (including the rejection of DFN) was derived strictly from BWC audio. 

## Next Phase: Phase 12 Karen Evaluation
Phase 12 will finally introduce a new dataset to the pipeline: `karen.wav`. This is a high-quality, 4-speaker podcast recording with rapid turn-taking and minimal background noise. The goal is to see if the "Hardened" production pipeline (which was deeply over-fitted to 2-speaker BWC noise) can generalize to a completely different acoustic environment.

## Long-term Work
*   **Generalization Framework:** Modifying the hardcoded `num_speakers: 2` logic to dynamically infer the number of speakers based on the dataset.
*   **Pipeline Simplification:** Stripping out all the branching/ablation logic from the codebase to leave only the lean, production-ready "Raw Oracle" path.
