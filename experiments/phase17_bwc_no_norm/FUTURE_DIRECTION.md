# Future Direction: Beyond Phase 17

## Proposed Enhancements

### 1. Hybrid Normalization
- **Concept:** Instead of global `LoudNorm`, apply a light **Automatic Gain Control (AGC)** or **Peak Normalization** to -1dB.
- **Goal:** Provide enough consistency for WhisperX to lower WER while preserving enough dynamic range for Pyannote VAD to maintain low Miss Rates.

### 2. Post-VAD Noise Gating
- **Concept:** Since Phase 17 increased False Alarms (16.7%), a secondary "noise classifier" could be run on every VAD-detected segment.
- **Goal:** Filter out segments that are clearly background noise (sirens, wind, car doors) before they reach the ASR/Diarization stages.

### 3. Hyper-parameter Tuning for No-Norm VAD
- **Concept:** Re-tune `merge_gap` and `min_duration` specifically for the un-normalized signal.
- **Goal:** A larger `merge_gap` (e.g., 0.4s) might help bridge the "energy valleys" in un-normalized speech, reducing fragmentation and potentially lowering False Alarms.

### 4. Segment-Level Normalization
- **Concept:** Detect speech segments using the Phase 17 "No-Norm" VAD, then normalize *only those segments* before passing them to WhisperX.
- **Goal:** This "Just-In-Time Normalization" would combine the best of both worlds: high recall from raw signal VAD and high lexical accuracy from normalized segment ASR.

### 5. Multi-Source Fusion
- **Concept:** Compare the outputs of a normalized branch and a non-normalized branch, and fuse the results using a confidence-weighted consensus (e.g., ROVER).
- **Goal:** Use the normalized branch for "clear" speech and the non-normalized branch for "distant/quiet" speech.

## Next Experimental Phase
The immediate next step (Phase 18) should focus on **Just-In-Time Normalization (JITN)** to see if we can recover the 3.7% WER loss while keeping the 8.2% Miss Rate gain achieved in Phase 17.
