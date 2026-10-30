# Inferences: Phase 17

## Technical Inferences

### 1. LoudNorm is a "Double-Edged Sword"
The experiment confirms that `LoudNorm` acts as a compressor that improves ASR performance (lower WER) by bringing speech into a "sweet spot" for neural acoustic models. However, it simultaneously acts as a high-pass filter for presence, "crushing" quieter speech segments and increasing the Miss Rate. 

### 2. VAD vs. ASR Divergence
VAD models (specifically Pyannote 3.1) appear to be more robust to raw signals than ASR models (WhisperX). The VAD's ability to "see" speech in the noise improved significantly without normalization, but the ASR's ability to "understand" that speech degraded. This suggests that the VAD-grounded approach in Phase 17 is better at *locating* conversational events, even if it's worse at *transcribing* them.

### 3. Diarization Thrives on Original Dynamics
The decrease in Speaker Confusion (10.35% -> 7.70%) is a strong signal. It implies that the features used for speaker identification (d-vectors/embeddings) are more representative when they haven't been subjected to the non-linear gain changes of LoudNorm. LoudNorm may be introducing artifacts that make different speakers sound more similar to the embedding extractor.

### 4. DER is a Deceptive Metric for BWC
While DER improved (46.79% -> 44.93%), the components shifted dramatically. A lower DER in this case hides a much higher False Alarm rate. In a production BWC environment, a high Miss Rate is usually more critical than a high False Alarm rate (it's better to have a bit of noise in the transcript than to miss a suspect's confession). Therefore, Phase 17's configuration might be "better" for law enforcement use cases despite the lexical WER regression.

## Operational Inferences
- **BWC Specificity:** The "No-Norm" strategy is likely only viable for high-dynamic-range field recordings (BWC, dashcam). For studio-quality podcasts or interviews, normalization is still likely preferred.
- **Normalization Strategy:** Instead of global LoudNorm, a more targeted approach like Peak Normalization or Automatic Gain Control (AGC) that preserves relative speaker volumes might be the optimal middle ground.
