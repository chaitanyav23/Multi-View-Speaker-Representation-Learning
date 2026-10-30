# Phase 2 Pipeline Flow

## Sequential Processing Steps

1.  **Audio Enhancement (Preprocessing)**
    *   Input: `raw_audio/*.wav`
    *   Operation: Denoising via DFN or MetricGAN+.
    *   Output: `experiments/phase2_enhancement/audio_<enhancer>/*.wav`

2.  **Stage 1: VAD (Grounding)**
    *   Input: **Enhanced** Audio.
    *   Operation: Silero VAD.
    *   Output: `experiments/phase2_enhancement/grounding_<enhancer>/*.json`

3.  **Stage 2: ASR (WhisperX)**
    *   Input: **Enhanced** Audio + Grounding JSON.
    *   Operation: Word-level transcription.

4.  **Stage 3: Alignment & Diarization**
    *   Input: Transcribed segments + Enhanced Audio.
    *   Operation: Mapping words to timestamps and speakers.

5.  **Final Evaluation**
    *   Comparison of the "Enhanced" output against the raw ground truth.

## Data Artifacts
*   `audio_deepfilternet/`: Intermediate processed audio files.
*   `grounding_deepfilternet/`: Segments detected *after* denoising.
*   `outputs_deepfilternet/`: Final conversational JSONs.
