# Phase 2 Error Analysis

## Artifact Characterization
*   **Robotic Voice (DFN/MetricGAN):** The most common error was the introduction of "phasiness" or "underwater" sound quality.
*   **Consonant Erasure:** Sibilants (/s/, /sh/) and plosives (/p/, /b/) were often smoothed over, causing WhisperX to mis-transcribe words (e.g., "stop" becoming "shop" or "top").

## VAD/ASR Mismatch
*   **False Transparency:** Sometimes the VAD correctly identified a segment in the enhanced audio, but the ASR stage (running on the same enhanced audio) produced complete gibberish or empty strings because the speech had been "over-cleaned" into silence.

## Diarization "Blurring"
*   In multi-speaker BWC scenarios, enhancement models tended to normalize all voices toward a "mean" spectral profile. This reduced the inter-speaker variance that Pyannote's embedding model relies on, leading to increased **Confusion** errors.

## Root Cause: Non-Stationary Noise
*   BWC noise is often non-stationary (clanking, wind gusts). Neural enhancers trained on stationary noise (fan hum, white noise) struggled to adapt, often leaving behind "musical noise" artifacts that are more disruptive to ML models than the original noise.
