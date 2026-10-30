# Architecture: Phase 13 Karen Generalization

The Phase 13 architecture is a modular, multi-stage pipeline designed for high-fidelity conversational transcription and diarization. It builds upon the "Hardened" architecture validated in Phase 11A.

## 1. Audio Preprocessing Layer
To ensure consistent input for subsequent models, all audio undergoes normalization.
-   **Tool:** FFmpeg `loudnorm` filter.
-   **Parameters:**
    -   Integrated Loudness (`I`): -20 LUFS.
    -   True Peak (`TP`): -1.5 dB.
    -   Loudness Range (`LRA`): 11 LU.
-   **Target Format:** 16kHz Mono WAV (PCM s16le).

## 2. Voice Activity Detection (VAD) Layer
The VAD layer identifies regions of interest, reducing the search space for ASR and providing a "grounding" for diarization.
-   **Model:** Pyannote 3.1 VAD.
-   **Post-processing:**
    -   `merge_gap`: 0.15s (segments closer than this are merged).
    -   `min_duration`: 0.1s (segments shorter than this are discarded).
-   **Output:** A grounding JSON containing speech segments.

## 3. Automated Speech Recognition (ASR) & Alignment Layer
This layer handles the conversion of audio to text and provides precise word-level timestamps.
-   **ASR Model:** WhisperX (Large-v3).
-   **Language:** English (`en`).
-   **Alignment Model:** Wav2Vec2-based alignment (standard in WhisperX).
-   **Processing:** Segments identified by VAD are transcribed individually to prevent hallucination during long silences. The resulting words are then aligned against the full audio to refine timestamps.

## 4. Diarization Layer (MSDD)
Phase 13 utilizes the NeMo Multi-scale Diarization Decoder (MSDD) for speaker separation.
-   **Embedding Model:** TitaNet_Large.
-   **Clustering:** Hierarchical Agglomerative Clustering (HAC).
-   - **Max Speakers:** 10 (Auto-discovery enabled).
-   **Multiscale Configuration:**
    -   Window lengths: [1.5, 1.2, 1.0, 0.8, 0.5] seconds.
    -   Shift lengths: [0.75, 0.6, 0.5, 0.4, 0.25] seconds.
    -   Weights: [1, 2, 3, 3, 3].
-   **Oracle VAD:** The RTTM generated from the Pyannote VAD layer is passed as "Oracle VAD" to MSDD to focus its decoding on speech regions.

## 5. Fusion & Attribution Layer
The fusion layer merges the outputs of ASR and Diarization.
-   **Mechanism:** Maximum-Overlap Fusion.
-   **Logic:**
    -   Each word from ASR is assigned to a speaker turn from MSDD that shares the greatest temporal overlap.
    -   **Fallback:** If no overlap is found, the word is assigned to the nearest speaker turn within a 0.5s tolerance.
    -   **Grouping:** Consecutive words by the same speaker are aggregated into "turns" or "segments" for readability.

## 6. Evaluation Framework
The architecture includes an integrated evaluation suite:
-   **ASR:** WER via `jiwer`.
-   **Diarization:** DER via `pyannote.metrics`.
-   **Conversational:** cpWER (Concatenated Permutation WER), WDER (Word Diarization Error Rate), and SA-WER (Speaker-Attributed WER).
