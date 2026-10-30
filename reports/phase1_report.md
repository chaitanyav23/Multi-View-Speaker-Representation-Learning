# Phase 1: VAD Benchmarking Study

## Objective
Determine whether the current Silero-based Grounded VAD is responsible for the high Miss Rate and False Alarm Rate.

## BWC Results Comparison

| VAD      |    WER |    DER |   Miss |     FA |   Conf |   WDER |   SA-WER |
|:---------|-------:|-------:|-------:|-------:|-------:|-------:|---------:|
| silero   | 0.3037 | 0.4234 | 0.1784 | 0.1746 | 0.0704 | 0.0992 |   0.4029 |
| pyannote | 0.3079 | 0.4147 | 0.1164 | 0.2299 | 0.0683 | 0.1033 |   0.4112 |

## Statistical Interpretation
- **Best overall DER:** 1      silero
1    pyannote
Name: VAD, dtype: object
- **Pyannote** reduced the Miss Rate by 6.2% compared to Silero.
- **Silero** is more conservative, with 5.53% fewer False Alarms.
