# Architecture: Phase 16 (No-Norm + DFN)

The Phase 16 pipeline is a multi-stage modular architecture designed for robust conversational AI processing in high-noise environments. It prioritizes signal purity over loudness consistency by employing DeepFilterNet3 and skipping traditional normalization.

## 1. Preprocessing Layer
- **FFmpeg Downsampling:** Ensures 16,000 Hz sample rate and mono channel configuration, satisfying the requirements of most downstream neural models.
- **DeepFilterNet3 (DFN):** Replaces the traditional normalization/denoising stack. DFN utilizes a deep filter network to suppress non-stationary noise while preserving speech components.

## 2. Voice Activity Detection (VAD) Layer
- **Model:** Pyannote 3.1 `voice-activity-detection`.
- **Function:** Generates binary speech/non-speech masks.
- **Post-processing:** Implements a `merge_gap` of 0.15s and a `min_duration` of 0.1s to smooth detections and remove flickers.

## 3. Transcription & Alignment Layer (ASR)
- **Model:** Whisper v3 Large.
- **Strategy:** Segment-conditioned transcription. Audio is sliced based on VAD boundaries and passed to Whisper.
- **Word-level Alignment:** WhisperX alignment model (`Wav2Vec2`) is used to refine word boundaries and provide high-resolution timestamps.

## 4. Diarization Layer
- **Model:** NeMo NeuralDiarizer with Multi-scale Speaker Diarization Decoder (MSDD).
- **Embeddings:** TitaNet-L (Large) for robust speaker representations.
- **Multiscale parameters:** 
  - Window lengths: [1.5, 1.2, 1.0, 0.8, 0.5] seconds.
  - Multi-scale weights: [1, 2, 3, 3, 3].
- **Thresholding:** Sigmoid threshold set to 0.7 for conservative speaker assignment.

## 5. Fusion Layer
- **Logic:** Custom Python implementation (`step6_fusion.py`).
- **Mechanism:** Assigns speakers from the MSDD RTTM to word segments from WhisperX based on maximum temporal overlap.
- **Fallback:** Nearest-neighbor assignment within a 0.5s tolerance for words falling outside strict RTTM boundaries.

## 6. Evaluation Framework
- **VAD Eval:** Temporal mask comparison at 10ms resolution.
- **ASR Eval:** JIWER-based Word Error Rate (WER).
- **Diarization Eval:** Pyannote.metrics implementation for DER, Miss, FA, and Confusion.
- **Conversational Eval:** cpWER, WDER, and SA-WER for joint ASR/Diarization assessment.
