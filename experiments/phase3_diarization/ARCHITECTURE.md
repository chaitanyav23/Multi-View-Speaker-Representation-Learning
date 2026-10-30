# Phase 3 Architecture

## Overview
Phase 3 attempted to implement a "Black-Box" Diarization architecture where a single neural model (MSDD) handles VAD, Embedding extraction, and Clustering in one integrated process.

## ASCII Pipeline Diagram (The Attempted Flow)

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     NeMo NeuralDiarizer         │
│  (diar_msdd_telephonic config)  │
│                                 │
│  1. MarbleNet VAD [CRITICAL FAIL]│
│  2. TitaNet Embeddings          │
│  3. MSDD Sequence Decoder       │
└─────────────────────────────────┘
         │
         ▼
[ RTTM Speaker Segments ]
         │
         ▼
┌─────────────────────────────────┐
│     Hardened Merger Script      │
│ (ASR Text + MSDD Timestamps)    │
└─────────────────────────────────┘
         │
         ▼
[ Final Conversational JSON ]
```

## Data Flow
1.  **Manifest Creation:** A JSON manifest is generated to point NeMo to the audio file.
2.  **Configuration:** A massive YAML-style config (`OmegaConf`) defines the windows, shifts, and multiscale weights for the TitaNet and MSDD models.
3.  **VAD Stage (Internal):** MSDD uses its own internal VAD (`vad_multilingual_marblenet`) by default.
4.  **Inference:** The model outputs an RTTM (Rich Transcription Time Marked) file.
5.  **Merging:** `run_hardened_with_custom_diar.py` attempts to align the words from the ASR stage with the speaker segments from the RTTM.

## Engineering Decisions
*   **Multiscale Weights:** Set to `[1, 1, 1, 1, 1]` to give equal importance to different temporal resolutions (0.5s to 1.5s windows).
*   **Thresholding:** A high sigmoid threshold (`0.7`) was set for MSDD to minimize False Alarms in noisy backgrounds.
*   **Decoupled ASR:** In this phase, ASR was run independently, and the diarization was intended to be "swapped in" after the fact.
