# Phase 14 Pipeline Flow

1. **Audio Prep**: ffmpeg -> 16kHz Mono -> LoudNorm (-20dB).
2. **Segmentation**: Pyannote VAD -> JSON Segments.
3. **Transcription**: WhisperX Large-v3 (Segmented) -> Phoneme Alignment (Wav2Vec2) -> Word Timestamps.
4. **Diarization**: Oracle RTTM + Manifest -> NeMo MSDD -> Speaker Boundaries.
5. **Reconstruction**: Word-to-Speaker Mapping -> Segment Grouping -> Forensic Transcript.