# Phase 11: DeepFilterNet Ablation Study

## Phase Objective
Phase 11 was designed to answer a single, critical question: Does DeepFilterNet (DFN) provide any net positive value to the forensic audio pipeline, or is it fundamentally destructive? By conducting a rigorous ablation study, the team isolated the impact of DFN on the VAD, ASR, and Diarization stages independently.

## Historical Motivation
Throughout Phases 2, 4, 7, and 10, DFN was observed to "clean" the audio to the human ear and lower False Alarm rates, but it consistently correlated with higher Word Error Rates (WER) and conversational errors (cpWER). Because the pipeline is highly interdependent, it was difficult to prove exactly *where* the DFN damage was occurring. Phase 11 isolates these variables.

## Hypothesis being Tested
DeepFilterNet actively degrades the acoustic features required by Transformer-based ASR models (Whisper) and sequence-to-sequence diarizers (MSDD), and its complete removal from the pipeline will result in strictly better performance across all primary metrics.

## What Changed from Previous Phase
*   **True Isolation:** Created three distinct pipelines (11A, 11B, 11C) that mix-and-match Normalized and DFN audio across the Grounding and ASR stages while keeping the Diarization stage constant.
*   **VAD Comparison:** Directly compared the VAD bounding boxes generated from Raw audio vs DFN audio.

## Outcomes (The Definitive Rejection)
*   **Hypothesis Confirmed:** DFN was proven to be fundamentally destructive to modern ML models in this context.
*   **VAD Degradation:** DFN caused the VAD to "shatter" continuous speech into 70 disconnected micro-segments (compared to 41 on raw audio), mistaking natural speech pauses for hard silence.
*   **ASR Destruction:** When isolating just the ASR stage (11C vs 11A), DFN caused Deletions to double, proving it 'eats' trailing consonants.
*   **Conclusion:** DFN is officially retired from the primary BWC pipeline.
