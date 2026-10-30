# Phase 5 Inferences

## Scientific Findings
1.  **Decoding Trumps Clustering:** Sequence-to-sequence decoding (MSDD) is fundamentally more robust for forensic audio than distance-based clustering. It can "track" a speaker through a noisy environment where a global cluster would otherwise collapse.
2.  **The "VAD is the Bottleneck" Proof:** This phase proved that the failures in Phase 3 were 100% attributable to VAD. Once VAD was removed as a variable, the neural diarization model performed excellently.

## Engineering Findings
1.  **Subprocess Stability:** Ripping out the internal VAD of a monolithic library (NeMo) is often necessary for stability in specialized domains.
2.  **Fusion Logic is the Weak Link:** The "midpoint assignment" fusion is simple and effective but lacks the nuance to handle sub-100ms overlaps. A more sophisticated fusion strategy (e.g., probability-weighted assignment) might be needed.

## Architectural Bottlenecks
*   **The RTTM Handshake:** Converting between JSON and RTTM is an extra "hop" that introduces overhead and potential formatting errors. A native internal manifest merger would be more efficient.
*   **Parallelism Overhead:** Running WhisperX and MSDD in parallel uses significant VRAM (~12GB+ total), potentially exceeding the limits of mid-range consumer GPUs.
