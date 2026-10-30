# Phase 9 Future Direction

## Remaining Limitations
*   **Sequential Latency:** The production script is slow because it runs stages in series.
*   **BWC Specificity:** The current pipeline is highly tuned for 2-speaker BWC data and may struggle with the "Karen" dataset (4+ speakers).

## Next Phase: Phase 10 Dual-Branch Fusion
Phase 10 will attempt to optimize the **Fusion Logic**. Instead of simple midpoint assignment, we will test the **"Dual-Branch Fusion"** strategy:
1.  Transcribe using raw audio.
2.  Diarize using DFN-enhanced audio.
3.  Perform a "Voter-Based" fusion where the speaker assignment is weighted by the confidence scores of the ASR and Diarization models.

## Long-term Work
*   **Containerization:** Packaging the Phase 9 pipeline into a Docker container with all dependencies (FFmpeg, CUDA, NeMo) pre-installed.
*   **Automatic Role Attribution:** Moving beyond "SPEAKER_00" to "OFFICER" and "SUSPECT" using NLP-based role detection.
