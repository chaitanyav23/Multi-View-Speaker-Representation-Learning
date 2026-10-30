# Phase 16 Inferences: Technical Takeaways

## 1. The Normalization Necessity
The omission of volume normalization (the "No-Norm" hypothesis) was invalidated. While normalization can amplify noise, its absence creates a high-variance signal that degrades the performance of models trained on normalized datasets (Whisper, Pyannote). Consistent signal energy is more important for these models than preserving the raw dynamic range, especially when combined with neural enhancement.

## 2. Neural Enhancement Artifacts
DeepFilterNet (DFN) provides superior noise suppression to traditional spectral subtraction, but its "synthetic" output profile is poorly matched to the expectations of downstream Voice Activity Detection. The high False Alarm rate (56.08%) suggests that VAD models perceive DFN's residual artifacts as speech. This implies that if DFN is used, a custom VAD calibration or a more conservative enhancement threshold is required.

## 3. Embedding Distortion in MSDD
The increase in Speaker Confusion (21.51%) indicates that DFN modifies the acoustic features used for speaker identification. This inference suggests that speaker embeddings (X-vectors or similar) are sensitive to the non-linear phase shifts and spectral smoothing introduced by deep-learning-based denoisers.

## 4. The Miss/FA Trade-off
Phase 16 succeeded in reducing the Diarization Miss Rate (from 28.7% to 17.0%) compared to Phase 11A. This confirms that DFN is effective at "recovering" speech that was previously lost in the noise. However, the accompanying surge in False Alarms and Confusion makes this a net negative. The inference is that we have moved from an "under-detection" regime to an "over-detection" regime.

## 5. WhisperX Resilience
WhisperX demonstrated a degree of resilience by ignoring many of the noise-induced VAD segments (reflected in the high deletion count). However, the failure to improve WER suggests that DFN did not make the actual speech segments more intelligible to the model, and may have even introduced distortions that hindered decoding.
