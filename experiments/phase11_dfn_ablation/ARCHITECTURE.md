# Phase 11 Architecture: The Ablation Matrix

## Overview
Phase 11 utilizes a Matrix Architecture to test the exact points of failure introduced by neural enhancement. It holds the fusion logic constant while swapping the audio streams fed to the VAD and ASR stages.

## ASCII Pipeline Diagram (The Ablation Matrix)

```
                       [ Raw Audio (.wav) ]
                               │
                               ▼
                ┌──────────────────────────────┐
                │        Normalization         │
                └──────────────┬───────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
        [ NORM Audio ]                [ DFN Audio ]
          │        │                    │       │
          ▼        │                    ▼       │
   [ NORM VAD ]    │             [ DFN VAD ]    │
        │          │                  │         │
        │          │                  │         │
┌───────┼──────────┼──────────────────┼─────────┼──────────┐
│       │          ▼                  │         ▼          │
│       │   ┌─────────────┐           │  ┌─────────────┐   │
│       ├──▶│   11A ASR   │           │  │   11B ASR   │   │
│       │   │  (Norm)     │           │  │   (DFN)     │   │
│       │   └─────────────┘           │  └─────────────┘   │
│       │                             │                    │
│       │   ┌─────────────┐           │                    │
│       ├──▶│   11C ASR   │◀──────────┘                    │
│       │   │  (DFN)      │                                │
│       │   └─────────────┘                                │
└───────┼──────────────────────────────────────────────────┘
        │
        ▼
[ Oracle MSDD ]
(Always on NORM Audio)
```

## Data Flow (The Three Pipelines)
1.  **Pipeline 11A (The Control):** VAD, ASR, and MSDD all process the LUFS-Normalized audio. This is the theoretical "cleanest" baseline.
2.  **Pipeline 11B (Full DFN):** VAD and ASR process the DFN-Enhanced audio. MSDD processes Normalized audio (since Phase 4 already proved DFN breaks MSDD completely). This simulates the "Enhanced Path" from previous phases.
3.  **Pipeline 11C (Isolated ASR Degradation):** Uses the exact same VAD bounding boxes and MSDD speaker assignments as 11A. The *only* difference is that WhisperX is fed the DFN audio. This isolates the acoustic impact of DFN on the transformer encoder.

## Engineering Decisions
*   **Constant Diarization:** Because diarization (especially the `custom_fusion.py` script) is so volatile, it was held strictly constant across 11A and 11C to ensure that any change in cpWER was mathematically guaranteed to be the fault of the ASR engine's interaction with the DFN audio.
