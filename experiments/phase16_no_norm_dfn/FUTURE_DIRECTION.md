# Phase 16 Future Direction: Roadmap for Noise Mitigation

## 1. Hybrid Enhancement (DFN + Normalization)
The immediate next step should be a "Phase 16B" that combines DeepFilterNet with standard volume normalization (LoudNorm). This will test if the stability of a normalized signal can mitigate the artifacts introduced by DFN.

## 2. VAD Recalibration
If neural enhancement is retained, we must explore custom VAD thresholds. Specifically:
- **Hysteresis Tuning:** Increasing the `min_speech_duration` and `min_silence_duration` to filter out short, rhythmic DFN artifacts.
- **Sensitivity Adjustment:** Lowering the VAD threshold to account for the cleaner (but potentially more artifact-prone) signal.

## 3. Alternative Denoisers
DFN3 may be too aggressive for the low-quality "Karen" audio. Future phases should evaluate:
- **CleanUNet:** Known for better preservation of speech harmonics.
- **Demucs (Denoiser):** A simpler, potentially more conservative approach that might introduce fewer spectral artifacts.
- **Spectral Subtraction (Traditional):** Re-evaluating traditional methods as a baseline for comparison against neural methods.

## 4. Multi-Path VAD Fusion
Implement a "Consensus VAD" architecture:
- Path A: VAD on raw (normalized) audio.
- Path B: VAD on DFN-enhanced audio.
- Logic: Only trigger a speech segment if both paths agree, or use the raw path to "gate" the enhanced path. This could eliminate the 56% False Alarm rate while still benefiting from the lower Miss Rate of the enhanced signal.

## 5. Embedding-Aware Enhancement
Investigate enhancement techniques that specifically optimize for speaker embeddings (e.g., MetricGAN+), rather than just perceived intelligibility or SNR. This could address the high Confusion Rate (21.51%) observed in Phase 16.
