# MASTER PHASE COMPARISON

This document provides a comprehensive comparison of all 18 phases of the BWC Pipeline project. It evaluates the progression of technology, performance, and stability across the project's lifespan.

## Phase Overview & Performance Tiers

| Tier | Phases | Key Characteristics | Status |
| :--- | :--- | :--- | :--- |
| **Foundational** | 1 - 4 | Baseline testing, VAD comparisons, and initial "Enhancement Trap" discovery. | **Obsolesced** |
| **Breakthrough** | 5 | Introduction of NeMo MSDD with Oracle VAD Injection. Peak theoretical performance. | **Research SOTA** |
| **Diagnostic** | 6 - 11 | Audit of contamination, Ablation studies on DFN, Dual-Branch failures. | **Validated** |
| **Production** | 9 | Stabilized, scriptable version of the Phase 5/7 architecture. | **Stable Baseline** |
| **Hardened** | 12 - 18 | Multi-speaker generalization (Karen), RMS Normalization, and "Muting" logic. | **Current SOTA** |

## Head-to-Head Comparison

| Feature | Weakest Phase | Strongest Phase | Reason |
| :--- | :--- | :--- | :--- |
| **ASR Accuracy (WER)** | Phase 2 (Denoised) | Phase 11 (Ablated) | Denoising destroyed word onsets; Raw/Norm preserved them. |
| **Diarization (DER)** | Phase 3 (Clustering) | Phase 18 (Oracle-Grounded) | Clustering collapsed to 1 speaker; Oracle MSDD achieved 18% DER. |
| **Fusion Integrity** | Phase 10 (Dual-Branch) | Phase 9 (Midpoint Fusion) | Phase 10 had temporal drift; Phase 9 used robust word-midpoints. |
| **Noise Resilience** | Phase 1 (Baseline) | Phase 18 (Muted Path) | Phase 1 hallucinated in sirens; Phase 18 muted them. |

## Top Performers by Component

### 1. Best Voice Activity Detection (VAD)
*   **Winner:** **Pyannote 3.1** (Phases 5-18)
*   **Rationale:** Superior recall in noisy environments. While it has higher False Alarm rates than Silero, it ensures critical speech is never skipped—a requirement for forensic integrity.

### 2. Best Automatic Speech Recognition (ASR)
*   **Winner:** **Faster-Whisper (Large-v3)** with **Wav2Vec2 Alignment** (Phases 7+)
*   **Rationale:** The transition to `faster-whisper` improved inference speed, and `Wav2Vec2` provided the sub-segment timestamps necessary for accurate speaker attribution.

### 3. Best Diarization Strategy
*   **Winner:** **NeMo MSDD with Oracle Grounding** (Phases 5, 12, 18)
*   **Rationale:** Sequence-to-sequence modeling (MSDD) is significantly more robust to the non-stationary noise of BWC audio than traditional clustering (AHC).

### 4. Best Fusion Logic
*   **Winner:** **Word-Level Midpoint Fusion** (Phases 9-18)
*   **Rationale:** By mapping the mathematical midpoint of each word-token to the diarization timeline, the system remains immune to slight boundary jitter at the start or end of segments.

## Summary of Key Phase Failures (Lessons Learned)

*   **Phase 2 (The Enhancement Trap):** Proved that "cleaning" audio for humans often makes it worse for AI models. DFN increased WER from 0.40 to 0.49.
*   **Phase 3 (Clustering Collapse):** Confirmed that standard i-vector/x-vector clustering cannot handle the dynamic acoustic environment of a moving body camera.
*   **Phase 10 (The Synchronization Wall):** Taught us that ASR and Diarization MUST use the exact same audio representation, or sub-millisecond drift will destroy the transcript's attribution.

