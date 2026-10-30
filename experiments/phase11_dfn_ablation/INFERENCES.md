# Phase 11 Inferences

## Scientific Findings
1.  **Neural Filters are Task-Specific:** DeepFilterNet is exceptional at making audio intelligible for *human* listeners on a phone call. It is terrible at preserving the mathematical distribution of mel-frequencies required by *machine* listeners like Whisper.
2.  **Silence is Not Golden:** The VAD model (Pyannote) relies on the natural background noise floor to provide context for what constitutes "speech." When that noise floor is artificially dropped to zero, the VAD's internal thresholds fail, leading to over-pruning.

## Engineering Findings
1.  **The Matrix Approach:** Using a multi-variant matrix (11A, 11B, 11C) was the only way to scientifically isolate the DFN problem. It proved that the team was correct to question DFN's utility in Phase 4.
2.  **Metrics Auto-Compilation:** The `step5_compile_report.py` script represents a leap in maturity for the project. By automating the derivation of answers, the research pipeline became self-documenting.

## Architectural Bottlenecks
*   **The Pursuit of "Clean":** The project spent several phases trying to "clean" the audio, only to discover that raw, normalized audio was the best path all along. This underscores the risk of relying on perceptual human audio quality when designing ML pipelines.
