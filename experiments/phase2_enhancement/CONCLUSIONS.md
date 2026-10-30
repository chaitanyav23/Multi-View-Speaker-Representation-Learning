# Phase 2 Conclusions

## Summary of Results
Phase 2 tested the hypothesis that neural audio enhancement would improve conversational intelligence pipeline performance. The results were **overwhelmingly negative**. Both DeepFilterNet3 and SpeechBrain MetricGAN+ increased the Word Error Rate (WER) and Diarization Error Rate (DER) on the critical BWC dataset.

## Hypothesis Success
The hypothesis was **disproven**. While background noise was reduced visually (in spectrograms) and audibly, the downstream ML models (WhisperX and Pyannote) performed significantly worse on the processed audio.

## Key Takeaways
*   **Do not use full-spectrum neural denoising** as a global preprocessing step for this pipeline.
*   **WhisperX is resilient to raw noise** but vulnerable to neural artifacts.
*   **DeepFilterNet3** is the only model that remained even remotely competitive, making it a candidate for *selective* use or VAD-only grounding in future iterations.

## Preservation
The enhancement scripts will be kept for specialized forensic cases where human listenability is more important than machine transcription, but they will be **disabled** in the primary production pipeline.
