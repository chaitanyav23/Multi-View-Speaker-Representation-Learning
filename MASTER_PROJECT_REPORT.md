# MASTER PROJECT REPORT: BWC CONVERSATIONAL INTELLIGENCE PIPELINE

## 1. Executive Summary
The Body-Worn Camera (BWC) Conversational Intelligence Pipeline project is a comprehensive research and engineering effort to solve the challenges of speaker-attributed transcription in hostile acoustic environments. Over 18 phases of development, the project evolved from a standard linear ASR approach to a sophisticated, dual-path "Hardened" architecture. The final system achieves a significant balance between transcription accuracy (WER) and speaker attribution (DER/cpWER), identifying Voice Activity Detection (VAD) as the primary bottleneck for future improvements.

## 2. Project Timeline & Experimental Chronology

The project followed a hypothesis-driven evolution across 18 distinct phases:

### Phase 1-5: The Foundational Research
*   **Phase 1 (VAD Baseline):** Compared Silero vs. Pyannote 3.1. Found Pyannote's high-recall approach safer for forensic applications.
*   **Phase 2 (The Enhancement Trap):** Tested DeepFilterNet and SpeechBrain. Discovered that neural denoising catastrophicially degrades Whisper ASR (WER +24%).
*   **Phase 3 (Diarization Failure):** Standard clustering collapsed into single speakers due to embedding drift in noisy environments.
*   **Phase 4 (Grounded Enhancement):** Confirmed that even with perfect speech boundaries, denoising hurts transcription fidelity.
*   **Phase 5 (The MSDD Breakthrough):** Introduced NeMo's Multi-scale Diarization Decoder (MSDD) with Oracle VAD injection, solving the clustering collapse and reducing cpWER to 0.45.

### Phase 6-11: Refinement and Ablation
*   **Phase 6 (Diagnostic Audit):** Uncovered and fixed a critical data contamination issue (DFN-leaked baseline).
*   **Phase 7-9 (Hardening):** Developed the "Muting Trick" and manual slicing to stabilize WhisperX performance.
*   **Phase 10 (Dual-Branch Fusion):** Attempted to use Denoised audio for Diarization and Raw for ASR. Failed due to temporal alignment drift.
*   **Phase 11 (DFN Ablation):** Proved conclusively that DFN doubles word deletions in Whisper.

### Phase 12-18: Generalization and Real-World Stress Testing
*   **Phase 12-13 (Karen Benchmark):** Validated the pipeline on a complex 10-speaker dataset.
*   **Phase 14-17 (Normalization Study):** Proved RMS Normalization (-20dB) is the optimal and safest acoustic conditioning.
*   **Phase 18 (Tich Karen - State of the Art):** Stress-tested the "Hardened" architecture on real-world noisy audio. Confirmed that while MSDD is capable of 18% DER with Oracle VAD, actual performance is limited by VAD sensitivity to environmental noise (sirens).

## 3. MSDD Findings: Sequence-to-Sequence vs. Clustering
One of the project's most significant technical findings was the superiority of **Multi-scale Diarization Decoder (MSDD)** over traditional Agglomerative Hierarchical Clustering (AHC).
*   **The Problem:** Traditional AHC (used in Pyannote) fails when speaker embeddings drift as subjects move relative to the camera.
*   **The Solution:** MSDD uses a sequence-to-sequence approach with multi-scale TitaNet embeddings. It doesn't just look at a "voiceprint" but considers the temporal context of the conversation.
*   **Oracle Injection:** We discovered that MSDD is extremely robust if bypassed of its internal VAD (MarbleNet) and fed "Oracle" boundaries from a high-recall VAD like Pyannote 3.1.

## 4. Fusion Evolution: The Path to Attribution
The pipeline's method for assigning a speaker to a word evolved through three stages:
1.  **Linear Fusion:** Simple overlap check between Whisper segments and Diarization segments.
2.  **Dual-Branch Fusion (Failed):** Attempted to fuse metadata from two different audio representations (Denoised vs. Raw). Abandoned due to sub-millisecond drift causing misattribution.
3.  **Word-Level Midpoint Fusion:** The current "Hardened" logic. It calculates the temporal midpoint of every word-token from the ASR branch and maps it to the corresponding speaker label in the Diarization RTTM. This is more robust to slight boundary jitter.

## 5. Final Best Pipeline: The "Hardened" Architecture
The culmination of this research is the **Hardened Master Grounding Pipeline**:
1.  **Frontend:** RMS Normalization (-20dB LUFS) using `librosa`.
2.  **VAD Grounding:** Pyannote 3.1 generates master speech segments.
3.  **Muting Trick:** All non-speech regions are zeroed out in the NumPy array to prevent Whisper hallucinations.
4.  **ASR Path:** Faster-Whisper (Large-v3) transcribes manual slices; Wav2Vec2.0 performs forced alignment for word-level precision.
5.  **Diarization Path:** NeMo MSDD (Oracle-mode) processes the same segments using TitaNet-Large embeddings.
6.  **Fusion:** Word-level midpoint intersection maps speakers to the final transcript.

## 6. Remaining Limitations
*   **VAD Noise Sensitivity:** Even the best VADs (Pyannote) struggle to distinguish sirens and radio chatter from human shouting, leading to "False Alarm" speech segments that confuse the diarizer.
*   **Acoustic Clipping:** Extreme proximity of the officer to the microphone can destroy the harmonic structure, making speaker identification difficult.
*   **Overlap Resolution:** While MSDD handles some overlap, simultaneous shouting in high-stress BWC encounters still leads to "Confusion" errors in the diarization output.
