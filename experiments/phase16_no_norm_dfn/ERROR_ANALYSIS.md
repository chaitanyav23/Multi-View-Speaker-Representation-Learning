# Phase 16 Error Analysis: No-Normalization + DeepFilterNet (DFN)

## Primary Error Drivers

### 1. VAD Spectral Artifacts and False Alarms
The most significant error driver in Phase 16 was the high **False Alarm Rate (56.08%)** in the VAD stage. 
- **Mechanism:** DeepFilterNet (DFN) uses a deep neural network to estimate and subtract noise. At low SNR, this process can introduce "musical noise" or rhythmic spectral artifacts.
- **Impact:** Pyannote 3.1 VAD, while robust, appears sensitive to these neural artifacts, interpreting them as high-energy speech events. This led to a massive over-segmentation of the audio, flagging 56% of non-speech regions as active speech.

### 2. ASR Deletion Clusters
The ASR phase showed a high number of deletions (334).
- **Observation:** Large blocks of "speech" segments identified by VAD resulted in empty or nonsensical transcriptions.
- **Root Cause:** When the VAD passes noise artifacts to WhisperX, the model's internal decoder often fails to find a high-probability path and outputs nothing (deletion) or hallucinated tokens. The "No-Norm" approach meant that some legitimate speech was too low-volume for WhisperX to decode reliably after DFN processing, contributing to the deletion count.

### 3. Diarization Confusion (MSDD)
The **Confusion Rate of 21.51%** is a significant regression.
- **Analysis:** MSDD (Multi-scale Speaker Diarization) relies on clean speaker embeddings. DFN processing, while removing noise, also modifies the phase and spectral envelope of the target speech.
- **Impact:** These subtle distortions in the speech signal likely blurred the boundaries between speaker embeddings in the latent space. Without volume normalization, the variance in speaker energy levels further complicated the clustering process, leading to segments being attributed to the wrong speaker or "UNK" (Unknown).

### 4. Word-Level Diarization Errors (WDER)
The WDER rose to **22.54%**.
- **Correlation:** This error is directly downstream of both VAD False Alarms and MSDD Confusion. 
- **Cause:** Because many "words" were attributed to noise-induced VAD segments, the alignment process (WhisperX `align` module) struggled to map transcribed words to the correct speaker turns. The lack of normalization meant that quieter speaker turns were frequently overshadowed by noise-induced "ghost" speakers.

## Diagnostic Summary
The combination of **No-Normalization** and **Neural Enhancement** created a "hostile" environment for the downstream pipeline. DFN improved recall by bringing some low-level signals above the VAD threshold, but at the cost of precision and signal integrity. The pipeline's components (VAD, ASR, MSDD) are tuned for relatively "natural" noise profiles; the synthetic noise profile produced by DFN's aggressive filtering caused these components to fail in non-linear ways.
