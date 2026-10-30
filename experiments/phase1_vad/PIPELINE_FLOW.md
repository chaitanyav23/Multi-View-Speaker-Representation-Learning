# Phase 1 Pipeline Flow

## Sequential Processing Steps

1.  **Audio Ingestion**
    *   Path: `raw_audio/*.wav`
    *   Preprocessing: Loaded as 16kHz mono.

2.  **VAD Inference**
    *   **Engine A (Silero):** Recursive splitting via `apply_min_cut` on raw probability scores.
    *   **Engine B (Pyannote):** Neural segmentation at 10ms frame resolution.

3.  **Grounding Generation**
    *   Artifact: `grounding_<engine>/<audio>_grounding.json`
    *   Structure:
        ```json
        {
          "file": "bwc.wav",
          "segments": [
            {"start": 0.5, "end": 15.2},
            {"start": 16.0, "end": 29.5}
          ]
        }
        ```

4.  **Downstream Processing (Hardened Pipeline)**
    *   Input: `Grounding JSON`
    *   Stage 1: **WhisperX ASR** on each segment defined in the JSON.
    *   Stage 2: **Alignment** of transcribed words to audio timestamps.
    *   Stage 3: **Diarization** within the speech segments using Pyannote.
    *   Stage 4: **Merging** and Formatting.

5.  **Evaluation**
    *   Comparison of the final output against ground truth using `evaluate_conversational_timeline.py`.

## Intermediate Artifacts
*   `grounding_silero/`: High-level segmentation based on energy/probability.
*   `outputs_silero/`: Final conversational JSONs derived from Silero's grounding.
*   `outputs_pyannote/`: Final conversational JSONs derived from Pyannote's grounding.
