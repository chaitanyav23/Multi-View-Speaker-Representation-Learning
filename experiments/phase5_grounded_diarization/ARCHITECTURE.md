# Phase 5 Architecture: The "Grounded MSDD"

## Overview
Phase 5 implements a hybrid architecture where one model (Pyannote) provides the "where" (speech boundaries) and another model (NeMo MSDD) provides the "who" (speaker identities).

## ASCII Pipeline Diagram

```
[ Normalized Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Stage 1: Master VAD         │
│      (Pyannote 3.1)             │
└─────────────────────────────────┘
         │
         ▼
[ Grounding JSON ] ───▶ [ RTTM Converter ]
         │                   │
         │                   ▼
         │           [ Oracle RTTM ]
         ▼                   │
┌─────────────────────────────────┐   ▼
│   Branch 1: WhisperX ASR        │ ┌──────────────────────┐
│ (Transcription + Alignment)     │ │ Branch 2: NeMo MSDD  │
└─────────────────────────────────┘ │ (Oracle-VAD Mode)    │
         │                          └──────────────────────┘
         ▼                                  │
[ Aligned Words JSON ]                      ▼
         │                          [ MSDD Segments ]
         ▼                                  │
┌──────────────────────────────────────────────────────────┐
│             Stage 4: Midpoint Fusion                     │
│    (Assigning MSDD speaker IDs to WhisperX words)        │
└──────────────────────────────────────────────────────────┘
```

## Data Flow
1.  **VAD Grounding:** Pyannote 3.1 generates a Master Grounding JSON.
2.  **Oracle Conversion:** The JSON is converted to an RTTM file. This file "lies" to MSDD, telling it that these segments are perfectly labeled speech regions.
3.  **MSDD Decoding:** MSDD reads the Oracle RTTM and only computes speaker embeddings and sequence decoding within those boundaries. This bypasses the noisy `MarbleNet` VAD.
4.  **Parallel ASR:** WhisperX transcribes the raw normalized audio using the same grounding segments.
5.  **Word-Level Fusion:** Each word is assigned to the speaker active at its temporal midpoint.

## Engineering Decisions
*   **Oracle VAD Injection:** Setting `oracle_vad: True` in the NeMo config is the single most important decision of the project. It reduced runtime by 90% and eliminated infinite loops.
*   **Decoupled Paths:** Separating ASR from Diarization allows the pipeline to use the raw audio for transcription (to avoid DFN artifacts) while potentially using enhanced audio for the diarizer (to improve embeddings).
