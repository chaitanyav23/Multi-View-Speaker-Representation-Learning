# Conclusions: Phase 17

## Final Verdict
The "No-Normalization" approach for BWC audio is **superior for structural capture and diarization** but **inferior for lexical transcription accuracy**.

## Summary of Findings
1. **Normalization is detrimental to VAD Recall:** The use of `LoudNorm` in previous phases was a primary driver of the high Miss Rate (~29%). By removing it, we recovered ~8% of speech that was previously "invisible" to the pipeline.
2. **ASR Requires Consistency:** WhisperX (large-v2) is sensitive to the loudness fluctuations present in raw BWC audio. The 3.7% regression in WER is a significant trade-off for the improved recall.
3. **Diarization is cleaner without compression:** NeMo MSDD performed better with original signal dynamics, reducing speaker confusion by 26% (relative reduction from 10.35% to 7.70%).

## Recommendation
For the BWC pipeline, we should **move away from LoudNorm**. While it improves WER, the risk of missing critical low-volume speech is too high for the target domain (Body Worn Cameras). 

The "Phase 17" configuration (No-Norm + Pyannote 3.1) should be considered the new baseline for high-noise field audio, provided that we can implement secondary noise-gating to combat the increased False Alarm rate.

## Key Decision Drivers
- **Mission Criticality:** BWC audio requires high recall of all speakers, even if transcription quality is slightly reduced.
- **Architectural Simplification:** Removing the normalization step reduces compute latency and pipeline complexity without a catastrophic loss in performance.
- **Metric Weighting:** The improvements in cpWER and DER outweigh the regression in WER in the context of multi-speaker "conversational" analysis.
