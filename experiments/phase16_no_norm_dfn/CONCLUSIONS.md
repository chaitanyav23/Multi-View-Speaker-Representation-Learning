# Phase 16 Conclusions: Implementation Verdict

## 1. Experimental Failure
Phase 16 failed to achieve its primary objective of improving pipeline accuracy on high-noise audio. Every major metric (WER, DER, WDER) showed a regression compared to the Phase 11A baseline. While some speech was "recovered" (lower Miss Rate), the resulting data was too noisy and distorted for reliable transcription or diarization.

## 2. Incompatibility of No-Norm and DFN
The core hypothesis—that omitting normalization would allow DFN to work more purely on the signal—was disproven. The lack of a stable reference level led to inconsistent VAD triggers and hindered the ASR decoder. Normalization is a critical "signal conditioning" step that cannot be replaced by neural enhancement alone.

## 3. DFN Configuration Issues
DeepFilterNet, in its default "DeepFilterNet3" configuration, is too aggressive for the target audio. The "musical noise" and spectral artifacts it leaves behind are more damaging to modern VAD/Diarization models than the original broadband noise. The pipeline is currently optimized for "natural" noise, and DFN's "synthetic" noise reduction breaks these optimizations.

## 4. Bottleneck Identification
The VAD stage is the primary bottleneck in this configuration. The 56% False Alarm rate polluted all downstream stages. If neural enhancement is to be used in the future, the VAD model must be either re-trained or significantly recalibrated to ignore the specific artifacts produced by that enhancement.

## 5. Architectural Recommendation
We recommend **against** using the Phase 16 "No-Norm + DFN" configuration for production or further refinement in its current state. The pipeline should revert to a configuration that includes robust normalization (e.g., LoudNorm) and explores more conservative or hybrid noise reduction strategies.
