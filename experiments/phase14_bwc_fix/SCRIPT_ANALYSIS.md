# Phase 14 Script Analysis

- **step2_vad_grounding.py**: Utilizes Pyannote 3.1 VAD. Critical hyper-parameters: `merge_gap=0.15` and `min_duration=0.1`. It implements a custom config download and patching mechanism to ensure compatibility with local environments.
- **step4_whisperx.py**: Bypasses the standard WhisperX pipeline to perform manual segment-based transcription using `faster-whisper` (CTranslate2). It disables `condition_on_previous_text` to prevent semantic hallucinations between disconnected BWC segments.
- **step5_msdd.py**: The "BWC Fix" core. It injects a Pyannote-generated RTTM as "Oracle VAD" into NeMo's MSDD. It uses TitaNet-Large with multiscale windows ([1.5, 1.2, 1.0, 0.8, 0.5]) and the `diar_msdd_telephonic` model.
- **step6_fusion.py**: Performs speaker attribution. It prioritizes temporal overlap but includes a 500ms fallback tolerance for nearest-neighbor speaker turns.