# Phase 4 Conclusions

## Summary of Results
Phase 4 provided a breakthrough by identifying the **WDER vs. WER tradeoff**. While DeepFilterNet (DFN) degrades raw transcription accuracy, it significantly improves speaker separation. The net result is a **6.8% relative improvement in cpWER** on the BWC dataset.

## Hypothesis Success
The hypothesis was **confirmed**. Normalization + DFN achieved a lower overall cpWER than the baseline, proving that diarization gains can outweigh ASR losses in high-noise environments.

## Key Takeaways
*   **DeepFilterNet** is beneficial for Diarization but detrimental to ASR.
*   **Peak Normalization** should be a mandatory first step in the pipeline.
*   **cpWER** is the superior metric for evaluating conversational pipelines.

## Preservation
The **Normalization logic** (-20dB peak) will be moved into the production `run_final_hardened_pipeline.py`. The **DFN enhancement** will be reserved for the Diarization branch only.
