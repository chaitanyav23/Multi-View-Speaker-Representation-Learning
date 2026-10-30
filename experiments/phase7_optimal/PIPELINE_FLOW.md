# Phase 7 Pipeline Flow

## Sequential Processing Steps

1.  **Audio Conditioning**
    *   Input: `raw_audio/*.wav`
    *   Result: `audio/*.wav` (LUFS Normalized + DFN Enhanced).

2.  **Soft-Grounded VAD**
    *   Input: Conditioned Audio.
    *   Operation: Detect + 0.1s Padding.
    *   Result: `grounding/*_soft_vad.json`.

3.  **Parallel Branching**
    *   **Branch A (ASR):** Sliced transcription with Context Guard.
    *   **Branch B (Diar):** Oracle MSDD with Fixed-2 speaker constraint.

4.  **Final Fusion**
    *   Operation: Word-level temporal assignment.
    *   Result: `outputs/*_optimal_final.json`.

5.  **Forensic Audit**
    *   Extraction of turn-level segments for human verification in `error_audit/`.

## Intermediate Artifacts
*   `diarization/`: RTTM and raw JSON outputs from NeMo MSDD.
*   `outputs/asr/`: Individual JSONs containing the "Contextual" transcriptions.
*   `reports/`: Consolidated CSVs for each model variant.
