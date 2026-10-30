# Appendix B

## B.1 Experimental Progression

This appendix isolates the ablation evidence that led to the final Phase20 architecture. It does not enumerate every experiment in the repository. The evidence base is restricted to experiments that test a component later used, rejected, or constrained in Phase20: neural enhancement, loudness normalization, dual-path processing, Pyannote VAD post-processing, oracle-grounded MSDD, and word-level fusion.

The final Phase20 pipeline, as documented in `phase20_standalone/results_to_be_published.md`, is a dual-path system. Path A applies DeepFilterNet before Pyannote VAD and NeMo MSDD diarization with TitaNet embeddings. Path B applies loudness normalization before Pyannote VAD and WhisperX large-v3 transcription/alignment. Fusion assigns aligned words to diarization speakers by maximum temporal overlap.

The relevant progression is:

```text
Enhancement tests
  -> Phase 2: global DeepFilterNet and SpeechBrain degrade BWC ASR/DER.
  -> Phase 4: DeepFilterNet improves word attribution/cpWER under grounded conditions but worsens WER.
  -> Phase 11: DeepFilterNet is destructive for ASR/VAD when used broadly.

Diarization tests
  -> Phase 3/5: standard clustering/Pyannote diarization is not sufficient for BWC speaker attribution.
  -> Phase 5: oracle-grounded NeMo MSDD lowers cpWER despite higher DER.
  -> Phase 18: oracle VAD sharply lowers DER, identifying VAD as the diarization bottleneck.

Signal-path tests
  -> Phase 15/17: no-normalization variants expose recall/ASR trade-offs.
  -> Phase 10: dual-branch infrastructure is mechanically viable, but segment-level majority fusion fails.
  -> Phase 19: publication-grade dual-path architecture is stabilized.
  -> Phase 20: probability-guided Pyannote VAD post-processing is added; final standalone release adds validated fusion metrics.
```

## B.2 Ablation Results

### Evidence Matrix

| Experiment | Purpose | Pipeline configuration | Audio preprocessing | VAD model | ASR model | Diarization model | Fusion method | Key reported metrics |
|---|---|---|---|---|---|---|---|---|
| Phase 2 enhancement | Test global neural enhancement | Raw baseline vs DeepFilterNet3 vs SpeechBrain MetricGAN+ | Enhancement before the full pipeline | Existing grounded VAD path; exact per-row VAD metrics unavailable | WhisperX | Pyannote-era diarization path | Conversational timeline evaluator | BWC DFN: WER 0.3781, DER 0.4261, cpWER 0.5203. SpeechBrain: WER 0.4504, DER 0.5603, cpWER 0.5482. |
| Phase 4 grounded enhancement | Isolate denoising under normalized/grounded conditions | Normalized baseline vs normalized + DFN vs normalized + SpeechBrain | Peak normalization to -20 dB; optional denoising | Master grounding; frame VAD metrics unavailable | WhisperX | Pyannote diarization | Conversational timeline evaluator | DFN worsened WER 0.2975 -> 0.3698, improved WDER 0.1033 -> 0.0558, improved cpWER 0.5054 -> 0.4711. |
| Phase 5 grounded diarization | Compare standard Pyannote diarization with oracle-grounded MSDD | Pyannote 3.1 baseline vs NeMo MSDD with injected VAD RTTM | Same normalized/grounded audio as Phase 4 baseline | Pyannote VAD converted to RTTM for MSDD | WhisperX | Pyannote 3.1 vs NeMo MSDD with `oracle_vad=True` | Conversational timeline evaluator | MSDD cpWER 0.4582 vs Pyannote 0.5054; DER 0.4492 vs 0.4114. |
| Phase 10 dual-branch fusion | Test whether separated ASR/diarization streams improve conversational output | 10A Pyannote control; 10B DFN ASR + normalized diarization; 10C normalized ASR + normalized diarization | Normalized and DFN branch variants | Grounding reused from preceding architecture | WhisperX | Pyannote control or oracle MSDD | Custom word-overlap plus segment majority vote | 10B cpWER 0.5301; 10C cpWER 0.5032; Phase 5 midpoint baseline cpWER 0.4582. |
| Phase 11 DFN ablation | Isolate DFN effects on VAD/ASR/diarization | 11A normalized control; 11B full DFN; 11C DFN only for ASR | Normalized vs DFN branch inputs | VAD compared on normalized vs DFN audio | WhisperX | MSDD held constant across ablations | Same conversational evaluator | 11A WER 0.3244, DER 0.4703, cpWER 0.5032; 11B WER 0.4215, DER 0.6267, cpWER 0.6528; 11C WER 0.3781, DER 0.5318, cpWER 0.5301. |
| Phase 15 no-normalization | Test removal of loudnorm on Karen data | Repair-only, no loudness normalization | 16 kHz mono repair only | Pyannote 3.1 | Whisper large-v3/WhisperX | NeMo MSDD | Time-based fusion | WER 0.1435, DER 0.4093, VAD F1 0.8967, attribution accuracy 0.8218. Dataset differs from BWC. |
| Phase 16 no-normalization + DFN | Test DFN as replacement for normalization on Karen data | Repair + DeepFilterNet, no loudnorm | 16 kHz mono repair plus DFN | Pyannote 3.1 | WhisperX | NeMo MSDD/TitaNet | Dual-branch fusion | WER 0.3355, DER 0.5268, VAD F1 0.8598, attribution accuracy 0.7746. |
| Phase 17 BWC no-normalization | Isolate loudnorm on BWC | Repair-only BWC pipeline, no loudnorm | 16 kHz mono repair only | Pyannote 3.1 | WhisperX large-v2 | NeMo MSDD | Time-based word/speaker fusion | WER 0.3368, DER 0.4493, VAD F1 0.8130, attribution accuracy 0.8934. |
| Phase 18 Tich Karen stress test | Identify front-end bottleneck under high noise | Normalized, no-normalization, oracle-grounded, ASR-grounded variations | RMS normalization or no normalization depending on variant | Pyannote 3.1, oracle VAD, or ASR-grounded variants | WhisperX | NeMo MSDD/TitaNet | Fusion after ASR/diarization | Standard normalized DER 0.6286; oracle-grounded DER 0.1810 reported in analysis; VAD identified as bottleneck. |
| Phase 19 dual-path evaluation | Stabilize final dual-path framework before Phase20 VAD change | Path A DFN -> Pyannote VAD -> MSDD; Path B loudnorm -> Pyannote VAD -> WhisperX | Path A DFN; Path B -20 LUFS loudnorm | Pyannote VAD | WhisperX large-v3 | NeMo MSDD/TitaNet | Maximum-overlap fusion | WER 0.3244, DER 0.4471, speaker accuracy 0.9027, timeline accuracy 0.7623. |
| Phase 20 improved VAD / standalone | Add probability-guided VAD and final validation metrics | Same dual path as Phase19 plus improved Pyannote VAD post-processing | Path A DFN; Path B loudnorm | `pyannote/voice-activity-detection` with probability-guided recursive min-cut | WhisperX large-v3 | NeMo MSDD/TitaNet | Maximum-overlap fusion with validated cpWER/SA-WER/speaker accuracy/timeline accuracy | Final standalone: WER 0.3244, DER 0.4471, speaker accuracy 0.9027, WAE 0.0973, timeline accuracy 0.7053. |

### Table 1. Ablation Summary

| Component | Baseline | Modified | Main Effect | Final Decision |
|---|---|---|---|---|
| Neural enhancement | Raw/normalized full-pipeline audio | Global DeepFilterNet or SpeechBrain | Phase 2 and Phase 11 show WER and cpWER degradation; Phase 4 shows DFN can improve WDER/cpWER under grounded diarization-sensitive conditions. | Do not use enhancement globally or for ASR. Phase20 confines DeepFilterNet to Path A, the diarization path. |
| Loudness normalization | No-normalization repair-only audio | Loudnorm/normalization before ASR path | Phase 17 shows no-normalization improves some structural metrics but increases BWC WER from 0.3002/0.3244-class baselines to 0.3368; Phase 18 reports normalization slightly improves WER and does not degrade diarization. | Keep loudness normalization on the transcription path. Do not treat normalization as a substitute for VAD improvement. |
| Dual-path processing | Single acoustic representation | Separate diarization and ASR paths | Phase 10 initial dual-branch attempt failed because of DFN/ASR and fusion logic; Phase19 stabilizes the dual-path design with DFN Path A, loudnorm Path B, and auditable per-stage metrics. | Keep dual-path architecture in Phase20, but keep ASR off DFN audio and preserve word-level fusion. |
| Pyannote VAD | Silero or earlier VAD variants | Pyannote VAD with Phase20 probability-guided post-processing | Phase 1 identified Pyannote’s recall advantage over Silero; Phase18 identifies VAD as the limiting bottleneck; Phase20 adds probability-guided splitting and records chunk constraints. On the BWC Phase19/20 comparison, VAD metrics do not change. | Use Pyannote VAD. Keep probability-guided post-processing as an auditable segmentation safeguard, not as a demonstrated BWC metric improvement in the committed comparison. |
| Oracle-grounded MSDD | Standard Pyannote diarization/clustering | NeMo MSDD with injected VAD RTTM (`oracle_vad=True`) | Phase 5 lowers cpWER from 0.5054 to 0.4582 even though DER increases; Phase18 oracle grounding reduces DER from 0.6286 to 0.1810 on Tich Karen. | Use oracle-grounded MSDD/TitaNet for speaker separation. |
| Fusion | Segment-level majority or wrapper speaker assignment | Word-level maximum-overlap attribution | Phase 10 shows segment-level majority voting increases cpWER; Phase20 validates maximum-overlap fusion with speaker accuracy, WAE, and timeline accuracy. | Use word-level maximum-overlap fusion and evaluate it separately. |

### Table 2. Component Contribution

`Metric unavailable.` means the repository artifact reviewed for that experiment did not report the metric under that name. Speaker Accuracy is reported only where an attribution accuracy or final fusion metric exists; older WDER-only rows are not converted unless the file explicitly reports attribution accuracy.

| Experiment | DER | WER | Speaker Accuracy | Timeline Accuracy |
|---|---:|---:|---:|---:|
| Phase 2 DeepFilterNet global enhancement | 0.4261 | 0.3781 | Metric unavailable. | Metric unavailable. |
| Phase 2 SpeechBrain global enhancement | 0.5603 | 0.4504 | Metric unavailable. | Metric unavailable. |
| Phase 4 normalized baseline | 0.4114 | 0.2975 | Metric unavailable. | Metric unavailable. |
| Phase 4 DeepFilterNet grounded | 0.4408 | 0.3698 | Metric unavailable. | Metric unavailable. |
| Phase 5 Pyannote 3.1 baseline | 0.4114 | 0.2975 | Metric unavailable. | Metric unavailable. |
| Phase 5 oracle-grounded MSDD | 0.4492 | 0.2975 | Metric unavailable. | Metric unavailable. |
| Phase 10A Pyannote control | 0.6064 | 0.3244 | Metric unavailable. | Metric unavailable. |
| Phase 10B DFN/dual-branch attempt | 0.5318 | 0.3781 | Metric unavailable. | Metric unavailable. |
| Phase 10C normalized/MSDD control | 0.4703 | 0.3244 | Metric unavailable. | Metric unavailable. |
| Phase 11A normalized control | 0.4703 | 0.3244 | Metric unavailable. | Metric unavailable. |
| Phase 11B full DFN | 0.6267 | 0.4215 | Metric unavailable. | Metric unavailable. |
| Phase 11C DFN ASR only | 0.5318 | 0.3781 | Metric unavailable. | Metric unavailable. |
| Phase 15 no-normalization, Karen | 0.4093 | 0.1435 | 0.8218 | Metric unavailable. |
| Phase 16 no-normalization + DFN, Karen | 0.5268 | 0.3355 | 0.7746 | Metric unavailable. |
| Phase 17 no-normalization, BWC | 0.4493 | 0.3368 | 0.8934 | Metric unavailable. |
| Phase 18 standard normalized, Tich Karen | 0.6286 | 0.2233 | Metric unavailable. | Metric unavailable. |
| Phase 18 oracle-grounded, Tich Karen | 0.1810 | 0.4118 | Metric unavailable. | Metric unavailable. |
| Phase 19 dual-path evaluation | 0.4471 | 0.3244 | 0.9027 | 0.7623 |
| Final Phase20 standalone | 0.4471 | 0.3244 | 0.9027 | 0.7053 |

### Extracted Stage Metrics

| Experiment | VAD Precision | VAD Recall | VAD F1 | VAD Miss | VAD False Alarm | ASR Subs | ASR Ins | ASR Dels | Diar Miss | Diar FA | Diar Conf | JER | WAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 11A normalized control | 0.8512 | 0.7667 | 0.8067 | 0.2333 | 0.2978 | 84 | 27 | 46 | 0.2905 | 0.0800 | 0.0998 | 0.1407 | Metric unavailable. |
| Phase 11B full DFN | 0.8649 | 0.7003 | 0.7739 | 0.2998 | 0.2431 | 86 | 31 | 87 | 0.3947 | 0.0666 | 0.1654 | 0.2733 | Metric unavailable. |
| Phase 11C DFN ASR only | Metric unavailable. | Metric unavailable. | Metric unavailable. | Metric unavailable. | Metric unavailable. | 73 | 17 | 93 | 0.3691 | 0.0685 | 0.0942 | 0.1492 | Metric unavailable. |
| Phase 15 no-normalization, Karen | 0.8554 | 0.9422 | 0.8967 | 0.0578 | 0.6889 | 49 | 22 | 154 | 0.0820 | 0.1554 | 0.1720 | Metric unavailable. | Metric unavailable. |
| Phase 16 no-normalization + DFN, Karen | 0.8679 | 0.8519 | 0.8598 | 0.1481 | 0.5608 | 158 | 34 | 334 | 0.1700 | 0.1417 | 0.2151 | Metric unavailable. | Metric unavailable. |
| Phase 17 no-normalization, BWC | 0.8314 | 0.7954 | 0.8130 | 0.2046 | 0.0625 | 71 | 19 | 73 | 0.2050 | 0.1674 | 0.0770 | Metric unavailable. | Metric unavailable. |
| Phase 19 Path A / final diarization path | 0.8448 | 0.7721 | 0.8068 | 0.2279 | 0.2991 | 84 | 27 | 46 | 0.2300 | 0.1399 | 0.0771 | 0.8424 | 0.0973 |
| Phase 19 Path B / final ASR path | 0.8417 | 0.7670 | 0.8026 | 0.2330 | 0.3042 | 84 | 27 | 46 | 0.2300 | 0.1399 | 0.0771 | 0.8424 | 0.0973 |
| Final Phase20 Path A / diarization path | 0.8448 | 0.7721 | 0.8068 | 0.2279 | 0.2991 | 86 | 26 | 45 | 0.2300 | 0.1399 | 0.0771 | 0.8424 | 0.0973 |
| Final Phase20 Path B / ASR path | 0.8417 | 0.7670 | 0.8026 | 0.2330 | 0.3042 | 86 | 26 | 45 | 0.2300 | 0.1399 | 0.0771 | 0.8424 | 0.0973 |

## B.3 Lessons Learned

### Enhancement Trap

The repository repeatedly shows that neural enhancement can make audio sound cleaner while making machine transcription worse. Phase 2 rejects global enhancement: DeepFilterNet raises BWC WER to 0.3781 and SpeechBrain raises it to 0.4504. Phase 11 is stricter: 11A without DFN has WER 0.3244 and cpWER 0.5032, while full DFN has WER 0.4215 and cpWER 0.6528. The Phase 11 report states that DFN “shattered” VAD segments and increased deletions.

The exception is Phase 4, where grounded DeepFilterNet reduces WDER from 0.1033 to 0.0558 and cpWER from 0.5054 to 0.4711 despite worsening WER from 0.2975 to 0.3698. This is the component-level evidence for separating enhancement from ASR. The final Phase20 architecture follows that separation by placing DeepFilterNet only on Path A, the diarization path, while Path B uses loudness normalization for ASR.

### Why Dual-path Emerged

A single acoustic representation could not satisfy both objectives. The enhancement experiments show that ASR prefers minimally distorted audio, while diarization and attribution can benefit from noise reduction or different grounding assumptions. Phase 10 tested a dual-branch implementation but rejected its first execution because DFN-ASR and segment-level majority fusion degraded cpWER. The lesson was not that branching was impossible; it was that the branches must preserve word-level timing and avoid DFN in the transcription path.

Phase19 is the first stabilized publication-grade dual-path artifact: Path A uses DeepFilterNet, Pyannote VAD, and NeMo MSDD/TitaNet; Path B uses loudnorm, Pyannote VAD, and WhisperX large-v3. Phase20 inherits that structure.

### Why Probability-guided VAD

Phase18 identifies VAD as the bottleneck: the standard normalized Tich Karen run has DER 0.6286, while oracle grounding is reported at DER 0.1810. Phase20 therefore modifies VAD post-processing rather than ASR or MSDD. The Phase20 VAD script retains Pyannote frame probabilities and searches for probability minima when recursively splitting segments longer than 30 s, with RMS and duration fallbacks.

The BWC Phase19-vs-Phase20 comparison reports no VAD metric change in the committed run: Path A and Path B precision, recall, F1, false alarm, and miss all have zero deltas, and no recursive splits occur for this BWC file. The repository evidence therefore supports probability-guided VAD as an auditable segmentation safeguard and a response to the identified VAD bottleneck, but not as a demonstrated numerical improvement on the committed BWC Phase19/20 comparison.

### Why MSDD

Phase5 is the decisive oracle-grounded diarization experiment. It compares Pyannote 3.1 against NeMo MSDD with injected VAD RTTM. Pyannote reports lower DER (0.4114), but MSDD lowers cpWER from 0.5054 to 0.4582 by better separating speaker streams. The Phase5 analysis explicitly states that Pyannote’s lower DER can reflect identity collapse rather than useful speaker-attributed transcription.

Phase18 reinforces this conclusion at the front-end level: when MSDD is given oracle-quality VAD segments, DER drops to 0.1810 in the reported analysis. This supports the final choice of oracle-grounded MSDD/TitaNet and explains why Phase20 converts Path A VAD to an oracle RTTM for NeMo.

### Why Fusion

Phase10 shows that fusion can damage an otherwise viable diarization/ASR pairing. The segment-level majority vote in `custom_fusion.py` raises cpWER relative to the Phase5 midpoint baseline because WhisperX segments can contain cross-talk from multiple speakers. The conclusion is explicit: word-level attribution is mandatory.

Phase20 uses maximum temporal overlap at the word level and validates fusion separately. The final standalone metrics report speaker accuracy 0.9027, word attribution error 0.0973, and timeline accuracy 0.7053. The fusion validation report defines word attribution error as `1 - speaker_accuracy` and timeline accuracy as exact active-speaker-set agreement at 10 ms resolution.

## B.4 Final Design Rationale

The final architecture is not the result of a single winning ablation. It is a synthesis of partial and sometimes conflicting findings:

| Design Decision | Experimental Evidence |
|---|---|
| Use DeepFilterNet only in the diarization branch | Phase 2 and Phase 11 reject global/ASR DFN because WER and cpWER degrade. Phase 4 shows DFN can improve WDER and cpWER under grounded diarization-sensitive conditions. Phase20 confines DFN to Path A. |
| Use loudness normalization on the ASR branch | Phase 4 identifies normalized audio as strong for raw WER. Phase 17 shows no-normalization improves some structural metrics but worsens BWC WER to 0.3368. Phase18 reports normalization slightly improves WER and does not degrade diarization. Phase20 uses loudnorm on Path B. |
| Use dual-path processing | Phase 10 demonstrates the need to separate ASR and diarization assumptions but also exposes the failure of DFN-ASR and segment-level fusion. Phase19 stabilizes the dual-path architecture later inherited by Phase20. |
| Use Pyannote VAD with probability-guided post-processing | Phase1 and later reports motivate Pyannote over Silero for coverage; Phase18 identifies VAD as the bottleneck. Phase20 implements probability-guided recursive min-cut, although the committed BWC comparison reports no metric improvement. |
| Use oracle-grounded NeMo MSDD/TitaNet | Phase5 shows cpWER improvement from 0.5054 to 0.4582 over Pyannote despite higher DER. Phase18 shows oracle grounding can reduce DER to 0.1810 on a stress test. |
| Use maximum-overlap word fusion | Phase10 shows segment-level majority fusion degrades conversational accuracy. Phase20 uses word-level maximum overlap and reports speaker accuracy, WAE, and timeline accuracy. |

The final Phase20 framework is therefore best understood as an engineered compromise. ASR is protected from neural denoising by Path B loudness normalization. Diarization receives a separately conditioned Path A and oracle VAD grounding. The VAD stage is instrumented with probability traces and recursive min-cut safeguards because prior stress tests identified front-end segmentation as the dominant bottleneck. Fusion is kept at word level because segment-level smoothing erased short speaker turns. The resulting system combines the successful constraints from multiple ablations rather than adopting any single experiment wholesale.

