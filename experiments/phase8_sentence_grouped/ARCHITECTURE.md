# Phase 8 Architecture: The "Sentence-Grouped" Pipeline

## Overview
Phase 8 modifies the temporal logic of the pipeline to focus on logically complete utterances rather than individual speech activities.

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
┌─────────────────────────────────┐
│      Soft VAD Detection         │
│  (Pyannote 3.1 + 0.1s Padding)  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Sentence Grouping Logic    │
│  (Merge any segments < 1.0s gap)│
└─────────────────────────────────┘
         │
         ▼
[ Sentence-Level Grounding JSON ] ───▶ [ Oracle RTTM ]
         │                                  │
         ▼                                  ▼
┌──────────────────┐               ┌─────────────────────┐
│ Contextual ASR   │               │   Oracle MSDD       │
│ (0.5s Buffer)    │               │ (Sentence-Mode)     │
└──────────────────┘               └─────────────────────┘
         │                                  │
         ▼                                  ▼
[ Aligned Words ]                  [ Speaker IDs ]
         │                                  │
         ▼                                  ▼
┌──────────────────────────────────────────────────────────┐
│                   Midpoint Fusion                        │
└──────────────────────────────────────────────────────────┘
```

## Data Flow
1.  **VAD Detection:** Standard Pyannote 3.1 VAD.
2.  **Utterance Merging:** A "greedy" merging algorithm combines all segments separated by less than 1000ms.
3.  **Diarization:** MSDD processes these larger "sentences," extracting embeddings over longer time windows.
4.  **ASR:** WhisperX transcribes the sentence-level chunks using a 500ms pre/post-buffer for acoustic context.
5.  **Alignment:** Forced alignment is performed on the whole sentence to recover word-level timings.

## Engineering Decisions
*   **1.0s Threshold:** Chosen to reflect the standard linguistic pause between related sentences in a single speaker's turn.
*   **0.5s ASR Buffer:** Increased from 0.2s to prevent "acoustic shock" at the start of segments, where the Whisper encoder fails to sync with the speaker's pitch.
