# Phase 6 Architecture: The "Hardened Audit"

## Overview
Phase 6 implements a "Defensive" architecture designed to protect the diarizer and ASR from out-of-segment noise using a binary masking strategy.

## ASCII Pipeline Diagram

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     LUFS Normalization (-20)    │
│       (ffmpeg loudnorm)         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    VAD Segment Detection        │
│      (Pyannote 3.1)             │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Audio Muting Mask           │
│ (Zero-out non-speech samples)   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Diarization (Muted)        │
│   (Pyannote 3.1 Fixed-2)        │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Manual Slice ASR            │
│ (Transcribe each VAD segment)   │
└─────────────────────────────────┘
```

## Data Flow
1.  **Loudnorm Preprocessing:** Audio is normalized to a target Integrated Loudness (I) of -20 LUFS.
2.  **Detection:** Pyannote VAD identifies speech regions.
3.  **Binary Muting:** The audio waveform is multiplied by a binary mask (1 for speech, 0 for silence). This prevents the diarizer from "hallucinating" speakers in the background noise between segments.
4.  **Sliced ASR:** Instead of feeding the whole file, the pipeline feeds individual 16kHz slices to the WhisperX transcription engine.
5.  **Diagnostic Feedback:** The output of this phase is fed into `extract_phase6_audit.py` to compare against ground truth on a per-second basis.

## Engineering Decisions
*   **Muting over Denoising:** The "Muting Trick" was hypothesized to be a cleaner, non-destructive alternative to neural denoising (DFN), as it leaves speech samples untouched while completely eliminating inter-segment noise.
*   **Fixed-2 Speaker Constraint:** Constraining the diarizer to exactly 2 speakers was used as a diagnostic control to see how the model handles overlapping noise when forced to choose between known speakers.
