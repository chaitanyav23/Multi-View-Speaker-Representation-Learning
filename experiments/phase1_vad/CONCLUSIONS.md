# Phase 1 Conclusions

## Summary of Results
Phase 1 successfully benchmarked two major VAD models. While Silero VAD is faster and more computationally efficient, **Pyannote v3.0** demonstrated superior recovery of speech in difficult BWC conditions, specifically reducing the Miss rate by 35%.

## Hypothesis Success
The hypothesis that Pyannote would outperform Silero in complex environments was **partially confirmed**. While it improved DER and Miss rates, it introduced more False Alarms (hallucinations), indicating a need for better denoising or pre-processing.

## Key Takeaways
*   **Pyannote v3.0** is the recommended grounding engine for the remainder of the project.
*   The **Min-Cut logic** is a vital stability feature for long-form audio processing.
*   Standardizing the **Grounding JSON** format allowed for seamless iteration across subsequent phases.

## Preservation
The `pipeline_vad_pyannote.py` script and its merging logic will be preserved as the core "Stage 1" of the production pipeline.
