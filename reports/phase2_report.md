# Phase 2: Speech Enhancement Study

## Objective
Reduce Whisper substitutions caused by low-SNR BWC audio.

## BWC Results Comparison

| Enhancer        |    WER |   Subs |   Dels |   Ins |    DER |   Miss |     FA |   Conf |   WDER |
|:----------------|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|-------:|
| None (Baseline) | 0.3037 |     68 |     48 |    31 | 0.4234 | 0.1784 | 0.1746 | 0.0704 | 0.0992 |
| speechbrain     | 0.4504 |     86 |    111 |    21 | 0.5603 | 0.2889 | 0.2282 | 0.0432 | 0.0702 |

## Statistical Interpretation
- **SpeechBrain (MetricGAN+) significantly DEGRADED performance.** WER increased from 0.3037 to 0.4504.
- Analysis: Enhancement artifacts are likely more damaging to Whisper's transformer-based attention than the original noise.
