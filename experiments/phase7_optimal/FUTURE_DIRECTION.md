# Phase 7 Future Direction

## Remaining Limitations
*   **The Artifact Wall:** We cannot improve ASR further if we continue to use DFN-enhanced audio for transcription.
*   **Sequential Bottleneck:** The pipeline is too slow for production-scale forensic processing.

## Next Phase: Phase 8 Sentence Grouped
Phase 8 will explore a different way to use "context." Instead of just prompting WhisperX, we will attempt to **Group VAD Segments into Sentences** before processing. This will test if providing Whisper with larger, logically-complete chunks of audio can reduce the WER without needing destructive enhancement.

## Long-term Work
*   **Dual-Branch Fusion v1:** Implementing the "Forked Architecture" where VAD/Diarization runs on DFN-enhanced audio, but ASR runs on Raw-Normalized audio.
*   **Parallel Orchestrator:** Converting the master orchestrator to a task-graph based system (like Dask or Prefect) to run stages in parallel on multiple GPUs.
