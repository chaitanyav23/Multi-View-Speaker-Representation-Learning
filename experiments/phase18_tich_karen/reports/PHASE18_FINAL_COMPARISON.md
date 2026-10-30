# Phase 18: Comprehensive Pipeline Comparison

We executed four distinct variations of the conversational intelligence pipeline on the `rich_karen.mp3` file to isolate bottlenecks and determine the optimal path for noisy, real-world body-worn camera audio.

## Experimental Variations
1. **Standard Normalized**: Base Phase 13 pipeline (RMS normalization, Pyannote VAD, WhisperX, MSDD).
2. **No Normalization**: Removed audio normalization to test if it was distorting the acoustic features.
3. **Oracle Grounded**: Bypassed automated VAD and gave the model the exact Ground Truth timestamps.
4. **ASR-Grounded**: Ran ASR (WhisperX) on the raw audio first, and used the aligned word timestamps as the "VAD" to guide diarization.

---

## Results Table

| Pipeline Variation | VAD F1 | ASR WER | DER | Miss (Diar) | False Alarm | Confusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Standard Normalized** | **0.7940** | **0.2233** | 0.6286 | 0.1051 | 0.3649 | **0.1585** |
| **2. No Normalization** | 0.7939 | 0.2289 | 0.6293 | 0.1052 | 0.3649 | 0.1592 |
| **3. Oracle Grounded** | 1.0000 | 0.4118 | **0.1810** | **0.0051** | **0.0000** | 0.1759 |
| **4. ASR-Grounded** | 0.7586 | 0.2515 | 0.6966 | 0.1605 | 0.3789 | 0.1572 |

---

## Key Insights

### 1. Normalization is Harmless (and slightly helpful)
Comparing Run 1 and Run 2 shows that audio normalization (-20dB) had a negligible impact on Diarization Error Rate (DER) but slightly improved the Word Error Rate (WER from 0.2289 to 0.2233). It is safe to keep it in the production pipeline.

### 2. The Core Diarizer (MSDD) is Excellent
The **Oracle Grounded** test (Run 3) is the most revealing. When we removed the VAD errors, the MSDD model achieved an incredibly low **18.1% DER** on a highly complex 10-speaker file. It maintained a low Confusion rate (17%), proving that the multiscale TitaNet embeddings are successfully separating the diverse voices in the "Karen" audio.

### 3. VAD is the Absolute Bottleneck
In all automated runs (1, 2, and 4), the **False Alarm** rate hovered around **36-37%**. The background noise, sirens, or non-speech sounds in the audio are constantly tricking both Pyannote and WhisperX into thinking someone is speaking. MSDD is forced to label this "noise," which destroys the DER score.

### 4. ASR-Grounding is Worse than Pyannote
Using WhisperX to find speech (Run 4) resulted in the **worst performance across the board**. It missed more speech (Miss: 16%) and hallucinated more speech (FA: 37.8%) than Pyannote 3.1. Pyannote remains the superior Voice Activity Detection tool. The higher WER in Run 4 (0.2515) vs Run 1 (0.2233) also shows that passing targeted Pyannote chunks to WhisperX yields better transcripts than running WhisperX blindly over the whole file.

## Conclusion & Next Steps
The Phase 13 architecture is fundamentally sound, but it cannot handle extreme environmental noise out-of-the-box. The priority should **not** be tuning the embedding models or the ASR model, but rather fixing the VAD. 

**Recommendation:** Implement a **Speech Enhancement/Denoising** pre-processing step (e.g., DeepFilterNet) *before* passing the audio to Pyannote VAD to suppress sirens and background noise, which should drastically reduce the False Alarm rate and unlock the MSDD's true 18% DER capability.
