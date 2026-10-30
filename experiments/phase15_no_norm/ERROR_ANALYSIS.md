# Phase 15: Error Analysis

## ASR Error: Deletion Bias
The most prominent ASR error in Phase 15 is the high deletion rate (154 deletions vs. 49 substitutions).
- **Cause:** Without normalization, low-amplitude speech at the beginning or end of VAD segments may fall below the ASR model's attention threshold or be classified as noise by the Whisper encoder.
- **Impact:** Significant loss of short words or the final syllables of sentences.
- **Mitigation Strategy:** Re-introduce a subtle normalization or implement a "VAD Padding" technique to give the ASR model more context around the speech regions.

## Diarization Error: High Confusion
Confusion (17.2%) is the dominant component of the 40.9% DER.
- **Cause:** Speaker ID swapping. MSDD (Multiscale Diarization Decoder) appears to struggle with maintaining consistent speaker identities in the `karen_repaired.wav` file, possibly due to the lack of loudness consistency which TitaNet might rely on for robust embedding extraction.
- **Impact:** Results in a WDER of 17.8%, meaning nearly 1 in 5 words is assigned to the wrong person.
- **Mitigation Strategy:** Evaluate if multiscale weights in Step 5 need adjustment or if a global clustering step should precede MSDD to stabilize speaker centroids.

## VAD Error: Frame-Level False Alarms
The frame-level False Alarm rate is exceptionally high (68.9%).
- **Cause:** Pyannote 3.1 is highly sensitive. Without normalization, the signal-to-noise ratio (SNR) might be interpreted such that background noise or "room tone" is labeled as speech.
- **Impact:** The ASR stage (WhisperX) is forced to process many "empty" segments. While Whisper is generally good at ignoring silence (resulting in few insertions), it increases total compute time and the risk of hallucination.
- **Mitigation Strategy:** Increase the VAD threshold or apply a noise-reduction filter in Step 0.

## Fusion Error: Timing Misalignment
- **Cause:** WhisperX alignment can occasionally drift, especially if the underlying audio is not normalized, leading to "mushy" phoneme boundaries.
- **Impact:** Words that occur near speaker transitions might be attributed to the previous or next speaker.
- **Observation:** The fallback mechanism in `step6_fusion.py` (0.5s tolerance) mitigates some of this, but it cannot fix cases where the alignment puts a word entirely inside the wrong speaker's RTTM segment.
