# Phase 16 Results Analysis: No-Normalization + DeepFilterNet (DFN)

## Overview
Phase 16 investigated the impact of omitting standard volume normalization in favor of a specialized neural enhancement pass using **DeepFilterNet (DFN)**. This phase aimed to determine if preserving the raw signal's dynamic range while aggressively removing noise would improve ASR and Diarization performance on challenging audio (specifically the "Karen" dataset).

## Core Metrics

| Metric | Value | Baseline (BWC 11A) | Delta |
| :--- | :--- | :--- | :--- |
| **WER (Word Error Rate)** | 33.55% | 30.02% | +3.53% |
| **DER (Diarization Error Rate)** | 52.68% | 46.79% | +5.89% |
| **VAD F1-Score** | 85.98% | - | - |
| **Attribution Accuracy** | 77.46% | - | - |
| **WDER (Word Diarization Error Rate)** | 22.54% | 11.36% | +11.18% |

## Detailed Breakdown

### 1. Voice Activity Detection (VAD)
The VAD performance in Phase 16 showed a high F1-score (85.98%) but revealed a critical weakness in precision:
- **Recall:** 85.19% (Relatively strong capturing of speech)
- **False Alarm Rate:** 56.08%
- **Miss Rate:** 14.81%

The extremely high False Alarm rate (56.08%) indicates that the DeepFilterNet enhancement may be introducing artifacts or failing to suppress non-speech signals in a way that Pyannote 3.1 VAD can handle. Over half of the actual silence/noise was flagged as speech.

### 2. ASR Performance
The WER of 33.55% represents a regression compared to Phase 11A. 
- **Substitutions:** 158
- **Deletions:** 334
- **Insertions:** 34
The high deletion count (334) suggests that while VAD is flagging "speech," WhisperX is failing to transcribe it, likely because the flagged segments are actually noise artifacts or highly distorted speech.

### 3. Diarization and Attribution
- **Confusion:** 21.51%
- **Miss:** 17.00%
- **False Alarm:** 14.17%
- **Attribution Accuracy:** 77.46%

The confusion rate doubled compared to baseline, indicating that the MSDD (Multi-scale Speaker Diarization) module struggled to differentiate between speakers in the DFN-enhanced, non-normalized audio. The WDER increase of +11.18% further confirms that speaker-to-word alignment was severely impacted.

## Summary of Findings
While DeepFilterNet was effective at reducing the Miss Rate in diarization (down from 28.7% in 11A to 17% in 16), the trade-off was a massive increase in False Alarms and Confusion. The "No-Norm" approach appears to allow low-level noise artifacts to persist or be amplified by the DFN process, leading to the observed VAD and Diarization regressions.
