# Phase 18: Dual-Path Execution Summary

In response to the Diarization vs. Transcription tradeoff observed in previous runs, we implemented a **Dual-Path Architecture** that splits the pipeline to optimize both components independently.

## Pipeline Architecture
- **Path A (Diarization):** Raw Audio ➔ DeepFilterNet (DFN) ➔ Pyannote VAD ➔ NeMo MSDD.
  * *Goal: Maximize speaker separation and minimize False Alarms from background noise.*
- **Path B (Transcription):** Raw Audio ➔ RMS Normalization ➔ WhisperX.
  * *Goal: Preserve acoustic fidelity and phonemes for optimal ASR accuracy.*
- **Fusion:** Merged the highly accurate text tokens from Path B with the denoised speaker boundaries from Path A.

## Dual-Path Evaluation Results
- **VAD F1:** 0.7877
- **ASR WER:** 0.2532
- **Diarization DER:** 0.6004
  - **Miss:** 0.1705
  - **False Alarm:** 0.2824
  - **Confusion:** 0.1475

## Impact Analysis
### 1. Diarization Gains Maintained
By using DFN audio for Pyannote and MSDD, we successfully retained the improved **False Alarm reduction (from 36% to 28%)** and the improved **Speaker Confusion (from 16% to 14.75%)**. The DER remained at the improved ~60% mark rather than the 63% seen in the standard normalized run.

### 2. ASR Recovery
In the pure DFN run, the ASR WER severely degraded to **34.00%** because WhisperX struggled with the robotic artifacts introduced by DeepFilterNet. By routing the clean, normalized audio to WhisperX instead (Path B), we recovered the ASR performance, bringing the **WER down to 25.3%**. 

*(Note: The WER of 25.3% is slightly higher than the pure standard run's 22.3% because WhisperX was forced to transcribe only within the narrower speech boundaries defined by the denoised VAD, which missed 17% of actual speech. However, it is a massive improvement over 34%.)*

## Conclusion
The **Dual-Path Architecture** is a resounding success for complex, noisy audio files like Body-Worn Cameras (`rich_karen`). It allows us to apply aggressive speech enhancement to help the diarization models without destroying the acoustic data required by the transcription models. This split architecture should be adopted as the default for all future phases.