# Error Analysis: Phase 17

## Root Cause Diagnostics
The removal of the `LoudNorm` stage introduced specific error modalities that were not as prevalent in Phase 11A.

### 1. High-Frequency False Alarms (VAD)
- **Symptom:** The False Alarm rate rose from 7.66% to 16.74%.
- **Analysis:** Without normalization, the signal contains "raw" noise floors. Background sounds (wind, clothing rustle, vehicle engines) often have higher peak amplitudes than distant speech in un-normalized audio. The Pyannote VAD, while robust, appears to trigger more easily on these raw noise transients.
- **Impact:** Leads to "hallucinated" segments where WhisperX attempts to find words in noise, increasing transcription overhead and potentially polluting the final transcript with low-confidence garbage text.

### 2. WhisperX Sensitivity to Dynamic Range
- **Symptom:** WER increased by ~3.7%.
- **Analysis:** Neural ASR models like Whisper are typically trained on data that is peak-normalized or has consistent loudness. The BWC audio has high dynamic range (loud sirens vs. quiet speech). Without normalization, some speech segments may be too quiet for the acoustic model's attention mechanism to process effectively, leading to deletions or incorrect substitutions.

### 3. Boundary Fragmentation
- **Symptom:** A higher number of raw segments (indicated in `step2` logs).
- **Analysis:** Un-normalized audio has more fluctuating amplitudes. If the signal drops below the VAD threshold momentarily due to a lack of gain control, a single speaker turn might be fragmented into several small segments.
- **Mitigation Check:** The `merge_gap` of 0.15s in `step2_vad_grounding.py` helps, but may need to be increased to 0.3s or 0.5s for un-normalized BWC audio to combat this fragmentation.

### 4. Over-Sensitivity to Overlap
- **Symptom:** Confusion decreased, but Miss rate is still ~20%.
- **Analysis:** In BWC scenarios, "near-field" speech (the officer) is much louder than "far-field" speech (the suspect). LoudNorm usually equalizes these. Without it, the far-field speech remains significantly quieter. While the VAD is detecting more of it (lower miss rate), the diarization model might still struggle to separate these vastly different energy levels when they overlap.
