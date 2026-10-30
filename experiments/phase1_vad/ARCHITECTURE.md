# Phase 1 Architecture

## Overview
Phase 1 utilizes a dual-branch architecture to benchmark two different VAD engines. The architecture follows a "Grounding-First" approach, where VAD is performed independently of transcription to create a structured manifest of speech activity.

## ASCII Pipeline Diagram

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     VAD Engine Selection        │
│ (Silero VAD vs Pyannote v3.0)   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Segment Post-Processing     │
│  (Min-Cut Splitting & Merging)  │
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
1.  **Ingestion:** Raw 16kHz mono audio is loaded.
2.  **Inference:**
    *   **Silero Branch:** Uses a window-based score thresholding approach.
    *   **Pyannote Branch:** Uses a frame-wise segmentation model (segmentation-3.0) wrapped in a VAD pipeline.
3.  **Refinement:**
    *   Long segments (>30s) are recursively split using a **Min-Cut** algorithm (finding the lowest probability frame to minimize word cutoff).
    *   Short adjacent segments are merged up to 28s to maximize context for ASR (WhisperX).
4.  **Serialization:** Results are saved in a standardized JSON format containing `start`, `end`, and `vad_type`.

## Engineering Decisions
*   **Target Segment Length (28-30s):** Chosen to align with Whisper's 30-second context window.
*   **Recursive Min-Cut:** Implemented specifically for Silero to handle its tendency to create very long continuous segments in noisy backgrounds.
*   **Standardized Grounding Interface:** Decoupling VAD from the main pipeline allows for "Oracle VAD" testing and modular replacement of VAD engines.
