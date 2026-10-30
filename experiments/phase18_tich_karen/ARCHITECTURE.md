# Architecture Design: Phase 18 (Tich Karen)

## System Overview
The Phase 18 architecture follows a modular, sequential design optimized for high-precision conversational analysis. It integrates state-of-the-art models for VAD, ASR, and Diarization, connected through a custom manifest-based data exchange layer.

## Component Stack

### 1. Audio Frontend
- **Library**: `librosa`, `soundfile`
- **Processing**: 
    - Resampling to 16,000 Hz (Standard for Whisper and NeMo).
    - RMS Normalization to -20.0 dB.
    - Mono-channel conversion.

### 2. Voice Activity Detection (VAD)
- **Model**: `pyannote/voice-activity-detection` (via Hugging Face)
- **Framework**: `pyannote.audio` 3.1
- **Role**: Generates candidate speech segments to be passed to both ASR and Diarization. This stage acts as a "gatekeeper," significantly impacting the downstream computational load and accuracy.

### 3. Automatic Speech Recognition (ASR)
- **Model**: Whisper `large-v3`
- **Framework**: `whisperx`
- **Role**: Performs transcription and forced alignment.
- **Alignment Model**: PyTorch-based phoneme alignment for precise word-level timestamps.
- **Optimization**: Processes VAD-gated chunks to minimize hallucination in silent or noisy regions.

### 4. Neural Diarization (MSDD)
- **Framework**: NVIDIA NeMo (ASR Collection)
- **Speaker Embedding**: `titanet_large` (Multi-scale)
- **Diarization Decoder**: `diar_msdd_telephonic` (Multi-scale Diarization Decoder)
- **Configuration**:
    - `sigmoid_threshold`: 0.7
    - `max_rp_threshold`: 0.25
    - `infer_batch_size`: 16
- **Role**: Analyzes the multi-scale embeddings within the VAD-defined segments to assign speaker labels. MSDD is chosen for its ability to handle overlapping speech better than traditional clustering methods.

### 5. Fusion & Post-Processing
- **Logic**: Custom overlap-aware assignment.
- **Role**: Assigns speaker labels from the MSDD RTTM to the word-level segments from WhisperX based on temporal intersection.

## Data Flow Architecture
1.  **Raw Audio** (.mp3) -> **Preprocessed Audio** (.wav)
2.  **Preprocessed Audio** -> **Pyannote VAD** -> **Segments.json**
3.  **Segments.json** + **Audio** -> **WhisperX** -> **Aligned_ASR.json**
4.  **Segments.json** + **Audio** -> **NeMo MSDD** -> **Labels.rttm**
5.  **Aligned_ASR.json** + **Labels.rttm** -> **Final_Transcript.json**

## Diagnostic Layer
The architecture includes an "Embedding Analysis" sub-module that extracts TitaNet embeddings and applies:
- **PCA**: Principal Component Analysis for global variance mapping.
- **UMAP**: Uniform Manifold Approximation and Projection for local cluster density analysis.
- **Interactive Visualization**: Matplotlib/Plotly-based tools for manual cluster inspection.
