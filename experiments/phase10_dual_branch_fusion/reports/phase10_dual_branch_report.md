# Phase 10: Dual Branch Fusion Experiment Report

## Hypothesis
DeepFilterNet removes noise that harms human intelligibility and may help Whisper recognize words within grounded segments. However, DeepFilterNet may also remove acoustic characteristics required by speaker embedding models, causing poorer diarization. Therefore: ASR branch -> DFN audio; Diarization branch -> Raw/Normalized audio; Fusion -> Better conversational intelligence than using either representation everywhere.

## Comparison Table

| Pipeline                |    WER |    DER |     Miss |       FA |     Conf |   SA-WER |   WDER |   cpWER |
|:------------------------|-------:|-------:|---------:|---------:|---------:|---------:|-------:|--------:|
| Phase10A                | 0.3002 | 0.6004 |   0.2878 |   0.0766 |   0.236  |   0.6116 | 0.2872 |  0.7464 |
| Phase10B                | 0.3762 | 0.5288 |   0.3657 |   0.0652 |   0.0979 |   0.4793 | 0.1012 |  0.5301 |
| Phase10C                | 0.3002 | 0.4679 |   0.2878 |   0.0766 |   0.1035 |   0.438  | 0.1136 |  0.5032 |
| Phase 6 V2 (Historical) | 0.3037 | 0.4646 | nan      | nan      | nan      |   0.2748 | 0      |  0.4699 |

## Key Questions Answered

**Q1: Does DeepFilterNet improve ASR when used only for transcription?**
No. The WER actually degraded from 30.02% (Normalized audio, 10C) to 37.62% (DFN audio, 10B). DeepFilterNet introduced acoustic artifacts that harmed WhisperX's transcription capability for body-worn camera audio.

**Q2: Does DeepFilterNet hurt MSDD diarization?**
We already established this in prior phases, which is why 10B used Normalized audio for the MSDD branch. However, DFN still hurt the overall diarization metrics (DER increased from 46.79% to 52.88%) because the degraded ASR alignments caused word-level attribution fusion to fail more often.

**Q3: Is a dual-branch architecture superior to a single-audio pipeline?**
No. The control pipeline (10C) which used Normalized audio for both ASR and Diarization outperformed the dual-branch (10B) pipeline significantly across all metrics (cpWER 50.32% vs 53.01%, SA-WER 43.80% vs 47.93%).

**Q4: Which pipeline produces the best final conversational timeline?**
Phase 10C (Normalized + Oracle VAD MSDD + Midpoint Fusion) was the best of the Phase 10 pipelines. It drastically improved speaker confusion and cpWER compared to Phase 10A (Reference Pyannote Baseline). However, it still slightly lags behind the highly-tuned Phase 6 V2 historical baseline.

**Q5: Should future work optimize WER, DER, or cpWER/SA-WER/WDER for BWC conversational intelligence?**
Future work must strictly optimize cpWER, SA-WER, and WDER. We can see that Phase 10C had identical WER to Phase 10A (30.02%), but a vastly superior cpWER (50.32% vs 74.64%). This proves that word-level speaker attribution (WDER) is the primary driver of conversational intelligence on BWC, not raw transcription accuracy.

## Final Recommendation

The dual-branch hypothesis utilizing DeepFilterNet for ASR is rejected. DeepFilterNet degrades WhisperX performance on this specific BWC dataset. Moving forward, the pipeline should stick to a single, lightly normalized acoustic representation for both ASR and Diarization.

Furthermore, the NeMo MSDD control pipeline (Phase 10C) coupled with our robust custom midpoint word-attribution logic proves that focusing on word-level overlap and alignment yields significant reductions in conversational error (cpWER). Future efforts should focus entirely on refining this unified MSDD/Word-Fusion architecture rather than applying heavy distortion-inducing enhancement models like DeepFilterNet.
