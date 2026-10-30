# GLOBAL RESULTS ANALYSIS

This document synthesizes the numerical findings from 18 phases of experimentation on the BWC (Body-Worn Camera) Conversational Intelligence Pipeline. It provides a data-driven look at what improved, what degraded, and where the fundamental limits of the current technology lie.

## 1. Transcription Performance (ASR / WER)

The project successfully reduced Word Error Rate (WER) by abandoning aggressive pre-processing and optimizing the model configuration.

| Epoch | Description | Avg. WER (BWC) | Key Driver |
| :--- | :--- | :--- | :--- |
| **Baseline** | Raw Audio + WhisperX | 0.4074 | High noise sensitivity. |
| **Enhancement** | DeepFilterNet (DFN) | 0.4980 | **Decline:** DFN distorted acoustic onsets. |
| **Optimized** | Faster-Whisper + Norm | 0.3158 | **Improvement:** Stable timestamps & RMS Norm. |
| **Hardened** | Muted Path (Phase 11+) | **0.3002** | **Peak:** Hallmark of the "Muting Trick." |

### Key Insight: The Enhancement Trap
One of the most significant findings was that neural audio enhancement (DFN, SpeechBrain) which makes audio clearer to human ears actually **degrades** Transformer-based ASR models. The models are trained on noisy data and learn to "see through" the noise; the artifacts introduced by denoising filters disrupt this learned pattern.

## 2. Diarization Performance (DER / cpWER)

Diarization saw the most dramatic fluctuations as we transitioned between algorithms.

| Strategy | Phases | DER (Avg) | cpWER (Avg) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **AHC Clustering** | 1 - 3 | 0.85+ | 0.90+ | **Failure:** Collapsed to single speaker. |
| **Oracle MSDD** | 5 | 0.4492 | 0.4582 | **Success:** Solved speaker separation. |
| **Production MSDD** | 9 | 0.5296 | 0.5974 | **Stability:** Robust but higher error. |
| **Hardened MSDD** | 18 | **0.1810*** | **0.3649** | **SOTA:** Peak performance with Oracle VAD. |

*\*Phase 18 achieved 18.1% DER when given perfect "Oracle" VAD segments, proving the Diarizer itself is highly capable.*

## 3. The "VAD Ceiling" Discovery

The analysis of Phase 18 (Tich Karen) revealed the project's primary technical bottleneck: **Voice Activity Detection (VAD).**

*   **VAD Impact on DER:** When using Standard VAD, DER was 62.86%. When using Oracle VAD (perfect human-labeled boundaries), DER dropped to 18.10%.
*   **The Problem:** Current VAD models (Pyannote, Silero) cannot perfectly distinguish between high-decibel shouting and high-decibel sirens/radio chatter. 
*   **The Consequence:** False Alarm speech segments are passed to the Diarizer, which then tries to assign a human "Speaker ID" to a siren, causing massive "Confusion" and "False Alarm" errors in the final metrics.

## 4. Normalization and Acoustic Conditioning

Extensive testing in Phases 14-17 proved that **RMS Normalization (-20dB LUFS)** is the optimal conditioning step.
*   **Stability:** It prevents clipping in high-gain regions (officer shouting near mic).
*   **Embedding Consistency:** It ensures that the TitaNet embedding model receives a consistent signal-to-noise ratio, improving speaker identification.
*   **Safety:** Unlike neural denoising, it is a linear operation that does not introduce temporal artifacts.

## 5. Stability vs. Peak Metrics (The Phase 9 Paradox)

Phase 9 (Production) showed a higher cpWER (0.59) than Phase 5 (0.45). This was a deliberate design choice:
*   **Phase 5 (Research):** Used narrow VAD windows that captured "clean" speech but missed "quiet" interjections.
*   **Phase 9 (Production):** Used high-recall VAD (0.80 recall) to ensure forensic completeness. This captured more noise, which slightly "dirtied" the metrics but ensured no critical speaker turn was lost.

## Final Project Conclusions

1.  **Muting is Essential:** The "Muting Trick" (Epoch 4) is the only reliable way to prevent Whisper hallucinations in the presence of sirens.
2.  **MSDD is Superior:** Multi-scale Diarization is required for BWC audio; clustering is not viable.
3.  **ASR is Robust:** Faster-Whisper Large-v3 can handle BWC noise effectively if protected from hallucination loops.
4.  **VAD is the Frontier:** Future improvements in BWC transcription will come from better "Forensic VAD" models that can distinguish sirens from human shouting.
