# Pipeline Flow: Phase 18 (Tich Karen)

## Execution Sequence
The Phase 18 pipeline is orchestrated by `run_pipeline.py`, which executes a linear sequence of 10 steps. However, internally, the phase explores multiple "Flow Paths" to compare performance under different conditions.

### Standard Sequential Flow
1.  **Ingestion**: `rich_karen.mp3` loaded.
2.  **Normalization**: Volume leveled to -20dB RMS.
3.  **VAD**: Pyannote 3.1 identifies "Who speaks when."
4.  **ASR**: WhisperX transcribes the identified VAD segments.
5.  **Diarization**: MSDD assigns speaker identities to the VAD segments.
6.  **Fusion**: Text and Speaker Labels are aligned by timestamp.

---

## Experimental Variations (Ablation Paths)

The phase evaluated four distinct flow configurations to isolate the impact of specific components:

### Path A: Standard Normalized (Production Candidate)
- **Flow**: Preprocess -> Pyannote VAD -> WhisperX -> MSDD.
- **Goal**: Establish a baseline for the existing Phase 13 architecture on noisy data.

### Path B: No Normalization
- **Flow**: Raw Audio (16kHz) -> Pyannote VAD -> WhisperX -> MSDD.
- **Goal**: Determine if RMS normalization distorts the acoustic features used by TitaNet embeddings.

### Path C: Oracle Grounded (Theoretical Upper Bound)
- **Flow**: Audio -> Ground Truth Timestamps (Oracle VAD) -> WhisperX -> MSDD.
- **Goal**: Isolate the diarizer (MSDD) performance. By removing VAD errors, we can see if the speaker clustering itself is failing.

### Path D: ASR-Grounded
- **Flow**: Audio -> WhisperX (Blind) -> Word Timestamps as VAD -> MSDD.
- **Goal**: Test if WhisperX's internal VAD (VAD-filter) is more robust than Pyannote for noisy audio.

---

## Decision Gates & Logic
- **VAD Thresholding**: The pipeline uses a 0.5 threshold for Pyannote segment detection.
- **MSDD Batching**: The flow utilizes dynamic batching based on the number of detected segments to optimize GPU memory.
- **Alignment Strategy**: If a word spans two speaker segments, it is assigned to the speaker with the maximum overlap (>50%).
