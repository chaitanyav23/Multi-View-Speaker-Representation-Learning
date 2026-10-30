# Pipeline Flow: Phase 13 Karen Generalization

The Phase 13 pipeline follows a linear data flow with a bifurcated processing stage for ASR and Diarization, followed by a final fusion step.

## Data Flow Diagram

```text
+-----------------------+
|  Raw Audio (Karen)    |
+-----------+-----------+
            |
            v
+-----------------------+
| FFmpeg Normalization  |
+-----------+-----------+
            |
            v
+-----------------------+
|  Pyannote 3.1 VAD     |
+-----------+-----------+
            |
            +---------------------------------------+
            |                                       |
            v                                       v
+-----------------------+               +-----------------------+
|    Grounding JSON     |               |   Oracle VAD RTTM     |
+-----------+-----------+               +-----------+-----------+
            |                                       |
            v                                       v
+-----------------------+               +-----------------------+
|  WhisperX Large-v3    |               |      NeMo MSDD        |
|  (Transcription)      |               |    (Diarization)      |
+-----------+-----------+               +-----------+-----------+
            |                                       |
            v                                       v
+-----------------------+               +-----------------------+
|   Wav2Vec2 Alignment  |               |  Speaker Turns RTTM   |
+-----------+-----------+               +-----------+-----------+
            |                                       |
            +-------------------+-------------------+
                                |
                                v
                    +-----------------------+
                    | Maximum-Overlap Fusion|
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Final Transcript JSON |
                    |      & Final TXT      |
                    +-----------------------+
                                |
                                v
                    +-----------------------+
                    | Evaluation Suite      |
                    | (WER, DER, cpWER...)  |
                    +-----------------------+
```

## Stage Descriptions

### 1. Pre-Processing & Grounding
The pipeline begins by normalizing the audio to a standard LUFS level. The Pyannote VAD model then identifies all speech segments. These segments serve two purposes:
1.  **ASR Guidance:** WhisperX transcribes only the audio within these segments, significantly reducing hallucinations during silence.
2.  **Diarization Guidance:** The segments are converted to an RTTM format that acts as "Oracle VAD" for the MSDD model, focusing the diarization effort on known speech regions.

### 2. Parallel Feature Extraction
-   **ASR Branch:** WhisperX performs transcription followed by a second pass using Wav2Vec2 to align words at a character level, ensuring precise start and end times for every word.
-   **Diarization Branch:** NeMo MSDD uses a multi-scale approach, extracting embeddings at multiple temporal resolutions. It then uses the provided VAD grounding to cluster these embeddings into distinct speaker identities.

### 3. Maximum-Overlap Fusion
This is the critical "joining" stage. The pipeline iterates through every word produced by the ASR branch and looks for a corresponding speaker turn in the Diarization branch.
-   If a word's time interval overlaps with a speaker turn, the speaker ID is assigned.
-   If multiple overlaps exist, the one with the maximum duration is chosen.
-   If no overlap exists, a 0.5s search window is applied to find the nearest turn.

### 4. Categorized Reporting
Finally, the results are compared against ground truth using various metrics. The pipeline generates reports for each subsystem (VAD, ASR, Diarization) as well as the end-to-end system (Conversational Metrics).
