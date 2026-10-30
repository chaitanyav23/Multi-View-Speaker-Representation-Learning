# Phase 10 Pipeline Flow

## Sequential Processing Steps

1.  **Preparation**
    *   Pre-generate `bwc_normalized.wav` and `bwc_deepfilternet.wav`.
    *   Pre-generate `bwc_vad.json` (Master VAD).

2.  **Branch Execution (Variant 10B/10C focus)**
    *   **Diarization Path:** 
        *   Convert VAD JSON to `oracle.rttm`.
        *   Run NeMo MSDD (`diar_msdd_telephonic`) on the assigned audio variant.
        *   Result: `outputs_phase10X/bwc/pred_rttms/bwc.rttm`.
    *   **ASR Path:**
        *   Slice assigned audio variant based on VAD.
        *   Run WhisperX Large-v3.
        *   Run WhisperX Wav2Vec2 alignment.
        *   Result: `aligned_res` dict in memory.

3.  **Custom Fusion**
    *   Compute temporal overlap matrix between ASR words and MSDD speaker blocks.
    *   Perform greedy assignment.
    *   Execute Segment-Level Majority Overwrite.

4.  **Serialization**
    *   Write `bwc_final.json` and `bwc_final.txt`.
    *   Write `bwc_word_metadata.json` (for diagnostics).

## Intermediate Artifacts
*   `outputs_phase10B/bwc_word_metadata.json`: Contains the sub-segment diagnostic data proving the `multiple_candidates` logic was working before the overwrite.
