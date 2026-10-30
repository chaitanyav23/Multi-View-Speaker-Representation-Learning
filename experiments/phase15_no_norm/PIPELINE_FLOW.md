# Phase 15: Pipeline Flow

## Data Flow Diagram
The pipeline follows a sequential "Grounding" pattern where the output of the VAD stage constrains the execution of all subsequent stages.

```
[Raw Audio (karen.wav)]
      |
      v (Step 0: FFmpeg)
[Repaired Audio (karen_repaired.wav)]
      |
      +------------------------------------------+
      |                                          |
      v (Step 2: Pyannote VAD)                   |
[VAD Grounding (karen_grounding.json)]           |
      |                                          |
      +--------------+---------------------------|
      |              |                           |
      v (Step 3)     v (Step 4: WhisperX)        |
[Oracle RTTM]  [Aligned Words (aligned_words.json)]
      |              |                           |
      v (Step 5: NeMo MSDD)                      |
[Diarized RTTM (karen_msdd.rttm)]                |
      |                                          |
      +--------------+---------------------------+
                     |
                     v (Step 6: Fusion)
           [Final JSON/TXT Transcript]
                     |
                     v (Steps 8-14: Evaluation)
                [Metrics Reports]
```

## Stage Interdependencies
1.  **VAD -> ASR:** Step 4 uses the `karen_grounding.json` to slice the audio for the Whisper model. This prevents the ASR from processing silence or background noise as speech.
2.  **VAD -> MSDD:** Step 5 uses the Oracle RTTM generated from VAD segments. This ensures that MSDD speaker clustering is only performed on regions identified as speech by Pyannote.
3.  **ASR + MSDD -> Fusion:** Step 6 requires the exact timestamps of words (from Step 4) and the speaker labels (from Step 5) to align them.
4.  **Ground Truth -> Eval:** Step 7 processes the raw ground truth text into a format comparable with the system outputs (JSON/RTTM), enabling automated evaluation.

## Error Propagation Path
- If **VAD** misses a segment, that speech is lost for both ASR and Diarization.
- If **ASR** mis-aligns a word, the **Fusion** stage might attribute it to the wrong speaker.
- If **MSDD** produces high **Confusion** (swapping speakers), the **cpWER** and **WDER** metrics will degrade significantly even if the ASR was perfect.
