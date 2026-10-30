# Future Direction: Phase 18 (Tich Karen) and Beyond

Based on the empirical evidence gathered during Phase 18, the following roadmap is proposed for the continued evolution of the Conversational Intelligence pipeline.

## 1. Immediate Next Step: Neural Speech Enhancement
The primary objective is to solve the "False Alarm" problem in the VAD.
- **Action**: Integrate **DeepFilterNet3** or **ClearerVoice** as a mandatory preprocessing step before Step 3 (VAD).
- **Hypothesis**: By suppressing sirens and non-speech noise, we can improve VAD Precision from 0.71 to >0.85, which should subsequently drop the DER from 62% to <35%.

## 2. Dynamic Thresholding (Adaptive VAD)
- **Action**: Implement a "Noise Floor Analysis" step that calculates the Signal-to-Noise Ratio (SNR) of the audio and dynamically adjusts the Pyannote `onset` and `offset` thresholds.
- **Hypothesis**: Hard-coded VAD thresholds (0.5) are suboptimal for varying environments (Quiet Room vs. City Street).

## 3. Multi-Modal Diarization
- **Action**: Explore using the *content* of the speech to assist in speaker identification. For example, if a speaker says "Officer, please," the system should increase the likelihood of the preceding speaker being labeled as "POLICE."
- **Technology**: Large Language Model (LLM) based post-processing for speaker role assignment.

## 4. Resource Optimization
- **Action**: Phase 18 used Whisper `large-v3`. We should evaluate if `distil-whisper` or `large-v2` can maintain the ~22% WER while reducing inference time by 2-3x.

## 5. Expansion of Stress Testing
- **Action**: Apply the Phase 18 diagnostic framework (including PCA/UMAP) to other "problematic" samples in the `raw_audio/` directory to see if the VAD-as-bottleneck finding generalizes across all BWC data.

## 6. Real-Time Feasibility
- **Action**: Investigate "Streaming MSDD" or "Buffering VAD" to determine if this architecture can be adapted for low-latency (<5 seconds) applications, or if it must remain an offline batch process.
