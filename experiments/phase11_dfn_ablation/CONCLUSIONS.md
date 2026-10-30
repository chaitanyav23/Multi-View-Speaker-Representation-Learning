# Phase 11 Conclusions

## Summary of Results
Phase 11 definitively proved that Neural Denoising (specifically DeepFilterNet) is a net negative for the forensic conversational pipeline. The ablation study demonstrated that DFN shatters VAD bounding boxes and significantly increases ASR deletions by suppressing high-frequency consonants.

## Hypothesis Success
The hypothesis was **confirmed**. Complete removal of DFN from the pipeline (Pipeline 11A) resulted in the best performance across every metric, yielding a cpWER of 0.5032 compared to the DFN-inclusive cpWER of 0.6528.

## Key Takeaways
*   **DFN is Officially Retired:** Neural denoising will no longer be used as a preprocessing step for either ASR or VAD in this project.
*   **Normalization is Sufficient:** LUFS or Peak normalization provides enough gain stability for the models to function optimally without introducing spectral artifacts.
*   **The Baseline is Restored:** The project must revert to the pure "Raw-Normalized" path established in Phase 5 and Phase 9.

## Preservation
The ablation framework from `step3_run_pipelines.py` is a valuable diagnostic methodology, but the actual DFN implementations will be excised from future production architecture.
