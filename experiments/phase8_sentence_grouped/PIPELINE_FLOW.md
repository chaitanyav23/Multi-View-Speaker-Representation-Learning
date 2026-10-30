# Phase 8 Pipeline Flow

## Sequential Processing Steps

1.  **Normalization & Enhancement**
    *   Target: LUFS-20 + DeepFilterNet3.

2.  **Sentence-Level Grounding**
    *   Step A: Pyannote VAD Detection.
    *   Step B: Padding (0.1s).
    *   Step C: **Greedy Gap Merging** (1.0s window).
    *   Result: `grounding/*_soft_vad.json`.

3.  **Advanced Diarization**
    *   Inference: MSDD (Oracle Mode) using sentence segments.
    *   Result: `diarization/*.json`.

4.  **Buffered Contextual ASR**
    *   Inference: WhisperX with 0.5s audio buffers and textual state prompting.
    *   Result: `outputs/asr/*.json`.

5.  **Alignment & Identity Fusion**
    *   Word-level assignment.
    *   Result: `outputs/*_final.json`.

## Data Artifacts
*   `grounding/`: Standardized JSON containing sentence-level boundaries.
*   `diarization/`: RTTMs reflecting speaker identities over whole sentences.
*   `outputs/`: Final conversational JSONs and TXT summaries.
