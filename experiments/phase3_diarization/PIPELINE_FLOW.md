# Phase 3 Pipeline Flow

## Sequential Processing Steps

1.  **Diarization Path (MSDD)**
    *   **Preprocessing:** Create manifest.
    *   **Inference:** Run NeMo NeuralDiarizer (VAD + Embedding + MSDD Decoding).
    *   **Artifact:** `outputs_msdd/<audio>_diarization.json`

2.  **ASR Path (Hardened)**
    *   **Operation:** WhisperX runs on the audio to generate word-level timestamps.
    *   **Artifact:** `asr_outputs/*.json`

3.  **Fusion (The "Custom Diar" Run)**
    *   **Script:** `run_hardened_with_custom_diar.py`
    *   **Logic:**
        ```
        FOR each word IN asr_json:
          word_mid = (word.start + word.end) / 2
          speaker = FIND active speaker IN diar_json AT word_mid
          ASSIGN speaker TO word
        ```
    *   **Artifact:** `outputs_msdd_final/*.json`

4.  **Evaluation**
    *   Comparison of MSDD-derived conversational JSON against Ground Truth.

## Technical Blockers
*   The **MarbleNet VAD** stage within Step 1 frequently hung on BWC files, preventing the pipeline from reaching the Fusion stage.
*   **Identity Collapse** in the Pyannote Baseline Path occurred between Step 2 and Step 3, resulting in all words being assigned to `SPEAKER_00`.
