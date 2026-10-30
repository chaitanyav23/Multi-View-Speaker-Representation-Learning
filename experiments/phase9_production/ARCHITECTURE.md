# Phase 9 Architecture: The "Hardened Master Grounding"

## Overview
Phase 9 implements a "Clean Signal" architecture. It prioritizes grounding boundaries and identity precision by physically silencing non-speech regions of the audio before the primary AI engines process it.

## ASCII Pipeline Diagram

```
[ Raw Audio (bwc.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Stage 1: Normalization      │
│   (LUFS -20, 16kHz, Mono)       │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Stage 2: Master VAD         │
│   (Pyannote 3.1 High-Recall)    │
└─────────────────────────────────┘
         │
         ▼
[ Grounding JSON ] ───▶ [ Muting Algorithm ]
         │                    │
         │                    ▼
         │             [ Muted Audio ]
         ▼                    │
┌──────────────────────┐      ▼
│ Stage 4: Hardened ASR│ ┌─────────────────────┐
│ (WhisperX + Slicing) │ │ Stage 6: Oracle MSDD│
└──────────────────────┘ └─────────────────────┘
         │                          │
         ▼                          ▼
[ Aligned Words ]           [ Speaker IDs ]
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────────┐
│           Stage 7: Speaker Attribution           │
│         (Identity + Text Midpoint Fusion)        │
└──────────────────────────────────────────────────┘
```

## Data Flow
1.  **Normalization:** Standardizes loudness to ensure VAD consistency.
2.  **Master Grounding:** Uses a very permissive VAD config (0.05s min) to capture all potential speech.
3.  **Muting:** Generates a version of the audio where every non-speech sample is exactly zero.
4.  **Inference:**
    *   **ASR Branch:** Transcribes the *raw-normalized* audio using the VAD segments (to avoid artifacts).
    *   **Diarization Branch:** Performs MSDD on the *muted* audio (to focus solely on speaker embeddings).
5.  **Fusion:** The word-level timestamps from the ASR branch are assigned identities from the Diarization branch.

## Engineering Decisions
*   **High-Recall VAD:** Explicitly optimized for forensic safety; it is better to have a False Alarm (noise as speech) than to miss a critical suspect statement.
*   **Disabled Context (ASR):** Disabling `condition_on_previous_text` was a defensive decision to prevent the pipeline from "making up" stories in noisy sections.
*   **Full Intermediate Persistence:** Every file—from the normalized WAV to the RTTM—is saved in its own directory for legal/forensic audit.
