# Phase 15: Conclusions

## Final Assessment
Phase 15 successfully demonstrated that the "Hardened Pipeline" architecture is robust enough to produce high-quality transcripts even in the absence of audio normalization. The baseline results on the `karen.wav` dataset (WER: 14.35%, DER: 40.93%) provide a clear benchmark for future ablation studies.

## Key Takeaways
1.  **Feasibility of Raw Processing:** Loudness normalization is not strictly required for the pipeline to function, but its absence increases certain types of errors (notably ASR deletions and Diarization confusion).
2.  **Model Resilience:** Whisper `large-v3` is the strongest component of the pipeline, showing remarkable accuracy (85.6% word-level accuracy) on non-normalized input.
3.  **Diarization Bottleneck:** Speaker diarization (MSDD) remains the weakest link in the chain. The high DER (40.9%) is the primary factor preventing the pipeline from achieving human-level conversational transcription.
4.  **VAD Robustness:** Pyannote 3.1 VAD is a reliable "grounding" mechanism. Even without normalization, it achieves an F1 score of 0.8967, ensuring that the ASR and MSDD models are provided with high-recall speech segments.

## Summary Verdict
The removal of Step 1 (LoudNorm) results in a "purer" but slightly more error-prone signal. For datasets with consistent recording quality like `karen.wav`, the impact is manageable. However, for noisier or more dynamic datasets (like BWC), normalization likely remains a mandatory preprocessing step to prevent catastrophic degradation of the diarization embedding space.
