# Phase 7 Architecture: The "Optimal Single-Path"

## Overview
Phase 7 implements a highly modular pipeline where each stage is "soft-grounded" by the previous one. It uses a single version of "Conditioned" audio for the entire execution thread.

## ASCII Pipeline Diagram

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Conditioning Stage          │
│   (LUFS Norm + DeepFilterNet)   │
└─────────────────────────────────┘
         │
         ▼
[ Conditioned Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Soft-Grounded VAD           │
│ (Pyannote 3.1 + 0.1s Padding)   │
└─────────────────────────────────┘
         │
         ▼
[ Grounding segments ] ───▶ [ RTTM Generator ]
         │                        │
         │                        ▼
         ▼                [ Oracle RTTM ]
┌──────────────────┐              │
│  Contextual ASR  │ ◀────────────┘
│ (Buffered Slices)│      ▼
└──────────────────┘  ┌─────────────────────┐
         │            │   Oracle MSDD       │
         │            │ (Sequence Decoder)  │
         ▼            └─────────────────────┘
[ Aligned Words ]                 │
         │                        ▼
         ▼                [ Speaker IDs ]
┌───────────────────────────────────────────┐
│             Midpoint Fusion               │
└───────────────────────────────────────────┘
```

## Data Flow
1.  **Conditioning:** Full-spectrum denoising and LUFS-20 normalization.
2.  **Soft VAD:** Detects speech and expands boundaries by 100ms to capture onset/offset transients.
3.  **Diarization:** MSDD runs in Fixed-2 mode using the padded VAD boundaries.
4.  **ASR:** WhisperX Large-v3 transcribes buffered slices (0.2s context window) using the previous segment's text as a stateful prompt.
5.  **Merge:** Word-level midpoint identity assignment.

## Engineering Decisions
*   **Multiscale Weights `[1, 2, 3, 3, 3]`:** Specifically tuned for BWC to prioritize smaller temporal windows, helping to capture short interjections in noisy environments.
*   **Context Guard (50 words):** Prevents the `initial_prompt` from becoming too long, which can cause Whisper to enter repetition loops.
*   **0.1s VAD Padding:** Essential for forensic integrity; prevents the first letter of a word (like "P" or "T") from being cut off by the VAD gate.
