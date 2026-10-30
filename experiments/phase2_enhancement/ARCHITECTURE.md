# Phase 2 Architecture

## Overview
Phase 2 modifies the base pipeline by inserting a Denoising/Enhancement block at the very beginning of the processing chain.

## ASCII Pipeline Diagram

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Audio Enhancement Block     │
│ (DeepFilterNet3 / SpeechBrain)  │
└─────────────────────────────────┘
         │
         ▼
[ Enhanced Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│       Stage 1: VAD              │
│      (Silero VAD)               │
└─────────────────────────────────┘
         │
         ▼
[ Grounding JSON (segments.json) ]
         │
         ▼
┌─────────────────────────────────┐
│     Hardened Pipeline Run       │
│   (ASR + Diarization + Eval)    │
└─────────────────────────────────┘
```

## Data Flow
1.  **Enhancement:**
    *   **DeepFilterNet3:** Performs spectral masking and deep filtering to remove ambient noise.
    *   **SpeechBrain (MetricGAN+):** Uses a Generator to synthesize a "cleaner" version of the audio mask.
2.  **VAD:** The enhanced audio is passed to Silero VAD to generate grounding segments.
3.  **Inference:** The enhanced audio and its corresponding segments are passed to WhisperX for transcription and alignment.

## Engineering Decisions
*   **VAD Choice (Silero):** Phase 2 reverted to Silero VAD for the benchmark to ensure that any improvements (or degradations) were strictly attributable to the audio enhancement and not Pyannote's permissive nature.
*   **Overwrite Mode:** All enhancement scripts default to overwriting previous runs to ensure that processed audio files correspond exactly to the current model configuration.
