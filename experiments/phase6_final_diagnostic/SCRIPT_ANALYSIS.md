# Phase 6 Script Analysis

## Main Orchestrator: `run_phase6_main.py`
This script integrates several advanced features:
*   **`normalize_audio`:** Utilizes `ffmpeg` with the `loudnorm` filter. This is a two-pass algorithm that ensures the audio meets the EBU R128 loudness standard.
*   **`enhance_audio`:** Wraps the pipelines from Phase 2, allowing for a comparative study of DFN vs. Baseline within the same script.
*   **Muting Logic:** 
    ```python
    mask = np.zeros_like(audio)
    for seg in vad_segments:
        mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
    muted_audio = audio * mask
    ```
    This ensures that the `diarizer` only "sees" signal during known speech regions.
*   **Manual ASR Slicing:** Instead of using Whisper's built-in VAD, it iterates over `vad_segments` and transcribes each `audio_slice` independently.

## Diagnostic Tools

### `extract_phase6_audit.py`
*   **Logic:** Performs a high-resolution comparison between the Ground Truth RTTM and the Predicted JSON.
*   **Purpose:** Identifying "Identity Swaps" (where the officer and suspect identities are flipped) which are common in Pyannote 3.1.

### `eval_phase6_metrics.py`
*   **Logic:** Computes standard metrics (WER, DER, cpWER) but adds a focus on "Per-Enhancer" performance, allowing for a direct comparison of how DFN affects the "Muting" architecture.

### `generate_spectrograms.py`
*   **Logic:** Generates Mel-spectrograms of the raw vs. enhanced audio with VAD segments overlaid as boxes.
*   **Purpose:** visual debugging of False Alarms (e.g., seeing a VAD box over a siren squiggle in the spectrogram).
