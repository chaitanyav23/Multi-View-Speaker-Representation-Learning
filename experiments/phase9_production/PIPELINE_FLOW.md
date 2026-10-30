# Phase 9 Pipeline Flow

## Sequential Processing Steps

1.  **Normalization**
    *   LUFS-20 target via FFmpeg.
    *   Artifact: `audio_normalized/bwc_normalized.wav`.

2.  **Master VAD Grounding**
    *   Pyannote 3.1 with 0.1s padding and 0.05s min-duration.
    *   Artifact: `grounding_master/bwc_grounding.json`.

3.  **The Muting Fork**
    *   Audio is physically muted in all regions not listed in the VAD grounding.
    *   Artifact: `audio_grounded/bwc_grounded.wav`.

4.  **Parallel Execution Branch**
    *   **Path A (ASR):** WhisperX transcribes the *Normalized* audio using the VAD segments.
    *   **Path B (Diar):** NeMo MSDD diarizes the *Grounded (Muted)* audio using the VAD segments as Oracle boundaries.

5.  **Final Timeline Synthesis**
    *   Alignment and identity fusion.
    *   Artifact: `outputs_final/bwc_final.json`.

6.  **Forensic Metrics**
    *   Comprehensive evaluation against BWC ground truth.

## Intermediate Artifacts
*   `outputs_alignment/`: Word-level timestamps before speaker ID is applied.
*   `outputs_msdd/`: Raw MSDD predictions in RTTM/JSON format.
*   `reports/`: The definitive `phase9_comprehensive_metrics.csv`.
