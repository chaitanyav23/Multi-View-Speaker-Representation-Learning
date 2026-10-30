# Phase 5 Pipeline Flow

## Sequential Processing Steps

1.  **VAD Grounding (Pyannote 3.1)**
    *   Operation: Run high-recall VAD on normalized audio.
    *   Result: `grounding_master/*.json`

2.  **RTTM Preparation**
    *   Operation: Convert JSON segments to RTTM format.
    *   Result: `phase5_audit/oracle_rttms/*.rttm`

3.  **Branch 1: ASR Transcription**
    *   Model: WhisperX Large-v3.
    *   Input: `grounding_master/` + `audio_normalized/`.
    *   Result: `outputs_msdd/bwc_words.json` (unlabeled words).

4.  **Branch 2: Oracle MSDD Diarization**
    *   Model: NeMo MSDD (`diar_msdd_telephonic`).
    *   Input: `oracle_rttms/` + `audio_normalized/`.
    *   Mechanism: Bypasses MarbleNet VAD; performs TitaNet embedding and multiscale decoding.
    *   Result: `diar_msdd_raw/bwc_diarization.json` (labeled time-segments).

5.  **Stage 4: Timeline Fusion**
    *   Script: `run_hardened_with_custom_diar.py`.
    *   Operation: Midpoint timestamp assignment.
    *   Result: `outputs_msdd/bwc_hardened.json`.

6.  **Stage 5: Evaluation & Diagnostics**
    *   Comparison against ground truth.

## Breakthrough Metrics
The total runtime of the Diarization stage dropped from **>300s (timeout)** to **~37s** per file due to the bypassing of the internal VAD.
