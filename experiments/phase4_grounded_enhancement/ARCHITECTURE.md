# Phase 4 Architecture

## Overview
Phase 4 implements a "Normalized-Enhanced" pipeline where gain control is applied before neural denoising to ensure the enhancement models operate within their optimal dynamic range.

## ASCII Pipeline Diagram

```
[ Raw Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Peak Normalization (-20dB)  │
│         (Librosa)               │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Audio Enhancement Block     │
│         (DeepFilterNet3)        │
└─────────────────────────────────┘
         │
         ▼
[ Processed Audio (.wav) ]
         │
         ▼
┌─────────────────────────────────┐
│     Hardened Pipeline Stage     │
│   (VAD -> ASR -> Diarization)   │
│   Using Master Grounding JSON   │
└─────────────────────────────────┘
```

## Data Flow
1.  **Normalization:** Input audio is peaked to -20dB to prevent clipping and ensure consistent volume across files.
2.  **Enhancement:** The normalized audio is processed via DFN or SB.
3.  **Grounding (Fixed):** Instead of running VAD on the fly, this phase uses `grounding_master/`, a set of pre-validated VAD segments. This ensures that any change in metrics is caused by the *audio quality* and not by changes in *segment boundaries*.
4.  **Inference:** Standard WhisperX + Pyannote 3.1 run.

## Engineering Decisions
*   **Decoupling VAD from Enhancement Study:** By using `grounding_master`, the team isolated the impact of audio quality on ASR/Diarization. This revealed the "WDER vs. WER" tradeoff that was hidden in Phase 2.
*   **-20dB Target:** Chosen to provide sufficient headroom for the spectral masking operations in DeepFilterNet.
