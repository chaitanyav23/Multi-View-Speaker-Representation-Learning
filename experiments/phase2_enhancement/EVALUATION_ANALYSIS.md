# Phase 2 Evaluation Analysis

## Metrics Focus

### Primary Evaluator: cpWER
*   In Phase 2, **cpWER** became the critical metric for deciding whether enhancement helped. Because cpWER penalizes both transcription errors (WER) and speaker mis-assignments (DER), it captures the "total cost" of artifact introduction.

### Secondary Evaluator: WDER (Word Diarization Error Rate)
*   Used to see if cleaner audio helped the diarization model (Pyannote) correctly identify speakers.

## Impact of Enhancement on Metrics

### 1. WER Degradation
*   Neural enhancement models often perform "over-subtraction," where frequencies essential for consonant recognition (e.g., /s/, /t/, /f/) are suppressed along with background noise. This leads to high **Substitutions** in the WER report.

### 2. FA/Miss Tradeoff
*   Ideally, enhancement should reduce **False Alarms** (by removing noise that triggers VAD).
*   In reality, it often increased **Miss rates** because low-volume speech was misidentified as noise and entirely removed by the spectral mask.

### 3. Diarization Artifacts
*   Diarization models rely on speaker-specific embeddings. Aggressive denoising can "smear" these embeddings or make two distinct speakers sound similarly "robotic," increasing the **Confusion** rate in the DER.
