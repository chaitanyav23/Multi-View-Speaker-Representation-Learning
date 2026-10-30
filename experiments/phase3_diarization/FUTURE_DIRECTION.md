# Phase 3 Future Direction

## Remaining Limitations
*   **Total System Timeout:** The pipeline cannot currently process the BWC dataset using MSDD without crashing or hanging.
*   **Clustering Drift:** Pyannote's inability to separate speakers in noise remains a major forensic risk.

## Next Phase: Phase 4 Grounded Enhancement
Phase 4 will attempt to combine the findings from Phase 2 (Enhancement) and Phase 3 (Diarization). We will test if **selective denoising** (DeepFilterNet) can clean up the audio just enough for the VAD and Diarization models to work without destroying the ASR clarity.

## Long-term Work: The "Oracle" Breakthrough
The ultimate solution identified in this phase is the development of an **Oracle-guided MSDD**. This involves using the reliable Pyannote 3.1 VAD to "ground" the audio, and then forcing MSDD to only perform clustering within those known speech boundaries, effectively bypassing its internal MarbleNet VAD.
