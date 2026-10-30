# Phase 10 Architecture: "Dual-Branch Evaluator"

## Overview
Phase 10 is designed as a comparative orchestrator that runs three distinct pipeline variations (10A, 10B, 10C) to test the interaction between Audio type (Raw vs. DFN), Diarization engine (Pyannote vs. MSDD), and a new Custom Fusion script.

## ASCII Pipeline Diagram (Branch 10B/10C focus)

```
[ Master VAD Segments ]
         │
         ▼
[ Oracle RTTM Generator ]
         │
         ▼
  ┌──────┴─────────────────────────────────┐
  │                                        │
  ▼                                        ▼
[ Branch 1: ASR ]                  [ Branch 2: Diarization ]
(Uses Raw Audio)                   (Uses Raw OR DFN Audio)
  │                                        │
  ▼                                        ▼
┌──────────────────┐               ┌─────────────────────┐
│ WhisperX         │               │ NeMo MSDD           │
│ (Text + Align)   │               │ (Speaker Segments)  │
└──────────────────┘               └─────────────────────┘
  │                                        │
  ▼                                        ▼
[ Aligned Words JSON ]             [ MSDD Timestamps JSON]
  │                                        │
  └──────────────────┬─────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│             Stage 3: Custom Fusion                       │
│  1. Compute Word-to-Diarization temporal overlap.        │
│  2. Assign speaker to word based on max overlap.         │
│  3. Segment-Level Majority Vote overwrite.               │
└──────────────────────────────────────────────────────────┘
```

## Data Flow
1.  **Preparation:** Pre-calculated `bwc_normalized.wav`, `bwc_deepfilternet.wav`, and `bwc_vad.json` are loaded.
2.  **Variant 10A (Control):** Runs the "Muting Trick" with Pyannote 3.1 directly on Normalized audio.
3.  **Variant 10B (The Hypothesis):** Runs MSDD on Normalized Audio, but runs ASR on DFN audio.
4.  **Variant 10C (The Raw Baseline):** Runs both MSDD and ASR on Normalized Audio.
5.  **Custom Fusion:** In 10B and 10C, the two independent streams of data (Text vs Identity) are joined using `custom_fusion.py`.

## Engineering Decisions
*   **The Overlap Algorithm:** Replaced the "midpoint" logic (which only checks a single millisecond) with an "overlap duration" calculation to handle words that span across a speaker turn boundary.
*   **The Majority Vote Flaw:** The architecture forces an entire WhisperX segment (which can be several seconds long) to inherit the speaker ID that won the majority of the words. This was intended to "smooth out" short hallucinated Identity Swaps but instead masked genuine cross-talk.
