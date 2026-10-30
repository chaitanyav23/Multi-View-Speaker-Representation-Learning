# Appendix A

## Baseline Pipeline

This appendix compares the original complete BWC baseline against the final Phase20 pipeline using repository artifacts only. The baseline selected here is the Phase 1 Silero run because `experiments/phase1_vad/SCRIPT_ANALYSIS.md` states that `run_phase1.py` generated `phase1_silero` and `phase1_pyannote` metrics, and `experiments/phase1_vad/RESULTS_ANALYSIS.md` labels Silero as the baseline. The metrics used for the baseline BWC experiment are the BWC row in `reports/conversational_eval_phase1_silero.csv` and `experiments/phase1_vad/METRICS_TABLES.md`. Missing baseline VAD frame metrics were derived from the Phase 1 Silero grounding file, `ground_truth/bwc.txt`, and `raw_audio/bwc.wav` using the same 10 ms frame convention used elsewhere in the repository.

```text
raw_audio/bwc.wav
  -> Phase 1 Silero grounding
     -> experiments/phase1_vad/pipeline_vad_silero.py
     -> Silero VAD, threshold_on=0.30, threshold_off=0.25
     -> recursive min-cut for >30 s regions
     -> merge segments up to 28 s
  -> run_final_hardened_pipeline.py
     -> Pyannote speaker-diarization-3.1 on VAD-muted audio
     -> WhisperX large-v3 on manually sliced VAD regions
     -> WhisperX Wav2Vec2 alignment
     -> whisperx.assign_word_speakers fusion
  -> evaluate_conversational_timeline.py
     -> WER, DER, SA-WER, cpWER, WDER, JER, IER
```

**Baseline stage inventory**

| Stage | Repository evidence | Baseline implementation |
|---|---|---|
| Preprocessing | `scripts/extract_audio.py`; `scripts/utils/audio_utils.py`; Phase 1 script analysis | Mono 16 kHz WAV input. No Phase 1 enhancement or loudness normalization is applied before the Silero baseline grounding in `pipeline_vad_silero.py`; audio is loaded at `SILERO_SR = 16000`. |
| VAD | `experiments/phase1_vad/pipeline_vad_silero.py` | Local `external/silero-vad` model loaded by `torch.hub.load(..., model="silero_vad", source="local")`. Postprocessing uses onset threshold 0.30, offset threshold 0.25, recursive minimum-score cuts for segments longer than 30 s, and merge-to-28 s context windows. |
| ASR | `run_final_hardened_pipeline.py`; `experiments/phase1_vad/run_phase1.py` | WhisperX `large-v3`, `float16`, English. The model transcribes VAD-bounded manual audio slices with `condition_on_previous_text=False`. |
| Alignment | `run_final_hardened_pipeline.py` | `whisperx.load_align_model(language_code=args.language)` and `whisperx.align(...)`, i.e. WhisperX-selected Wav2Vec2 alignment. |
| Diarization | `run_final_hardened_pipeline.py` | `pyannote/speaker-diarization-3.1` on audio muted outside the VAD grounding. Speaker count is passed through command-line `min_speakers`/`max_speakers`; Phase 1 runner does not set them. |
| Embeddings and clustering | `run_final_hardened_pipeline.py`; `CODE_DERIVED_PIPELINE_AUDIT.md` | Pyannote diarization pipeline, not NeMo MSDD. The audit characterizes this path as Pyannote 3.1 clustering with automatic speaker-count inference. |
| Fusion | `run_final_hardened_pipeline.py` | Speaker attribution exists. Fusion uses `whisperx.assign_word_speakers` after WhisperX alignment and Pyannote diarization. |
| Evaluation | `experiments/phase1_vad/run_phase1.py`; `evaluate_conversational_timeline.py`; `experiments/phase1_vad/METRICS_TABLES.md` | `evaluate_conversational_timeline.py` produces WER, substitutions, deletions, insertions, DER, miss, false alarm, confusion, SA-WER, cpWER, WDER, deltaCP, deltaSA, JER, IER, and 5-gram duplicate counts. |

## Proposed Pipeline

The proposed system is the final Phase20 standalone artifact. The repository identifies `phase20_standalone/results_to_be_published.md` as the publication-ready final result; its metrics match `phase20_standalone/metrics/all_metrics.json`.

```text
raw_audio/bwc.wav
  -> scripts/step01_preprocess.py
     -> audio/bwc_repaired.wav, mono 16 kHz
  -> dual path
     -> Path A: DeepFilterNet -> Pyannote VAD -> NeMo MSDD/TitaNet diarization
     -> Path B: -20 LUFS loudnorm -> Pyannote VAD -> WhisperX large-v3 ASR/alignment
  -> scripts/step07_fusion.py
     -> maximum-overlap word-to-speaker attribution
  -> scripts/step04_vad_eval.py
  -> scripts/step06b_asr_eval.py
  -> scripts/step06a_diar_eval.py
  -> scripts/step08_fusion_eval.py
```

**Phase20 stage inventory**

| Stage | Repository evidence | Phase20 implementation |
|---|---|---|
| Preprocessing | `phase20_standalone/scripts/step01_preprocess.py` | `ffmpeg -ar 16000 -ac 1` repairs/resamples to mono 16 kHz. |
| Enhancement | `step02a_deepfilternet.py`; `results_to_be_published.md` | Path A applies DeepFilterNet3 through `deepFilter` to produce `audio/bwc_dfn.wav`. Path B applies loudness normalization, not DeepFilterNet. |
| Normalization | `step02b_loudnorm.py` | Path B uses FFmpeg `loudnorm=I=-20.0:TP=-1.5:LRA=11`, resampled to mono 16 kHz. |
| VAD | `step03_vad.py`; `models.md`; `results_to_be_published.md` | `pyannote/voice-activity-detection`. Postprocessing merges gaps below 0.15 s, removes segments shorter than 0.10 s, recursively splits segments longer than 30 s at probability minima or RMS fallback, and merges final chunks up to 28 s. |
| ASR | `step05b_whisperx.py`; `models.md` | WhisperX `large-v3` / faster-whisper. Path B VAD segments are manually sliced; transcription uses `condition_on_previous_text=False`; alignment uses WhisperX-selected English alignment model. |
| Diarization | `step05a_msdd.py`; `models.md` | NeMo `diar_msdd_telephonic` with `oracle_vad=True`; VAD JSON is written to `pathA_oracle.rttm`. |
| Embeddings and clustering | `step05a_msdd.py`; `models.md` | TitaNet Large speaker embeddings (`titanet_large`) with multiscale windows and NeMo clustering parameters, followed by MSDD inference. |
| Fusion | `step07_fusion.py`; `common.py` | Speaker attribution exists. Each aligned word is assigned to the diarization speaker with maximum temporal overlap; nearest-speaker fallback is allowed within 0.5 s. |
| Evaluation | `step04_vad_eval.py`; `step06b_asr_eval.py`; `step06a_diar_eval.py`; `step08_fusion_eval.py` | VAD precision/recall/F1/miss/false alarm; ASR WER and edit counts; diarization DER/JER/components; fusion cpWER, SA-WER, speaker accuracy, word attribution error, and timeline accuracy. |

## Component-by-component comparison table

| Component | Original baseline | Final Phase20 | Change |
|---|---|---|---|
| Audio topology | Single path | Dual path | Single audio representation changed to separate diarization and transcription paths. |
| Preprocessing | Mono 16 kHz load for Silero; no Phase 1 enhancement documented | Mono 16 kHz repair; Path A DeepFilterNet; Path B -20 LUFS loudnorm | Added explicit repair plus path-specific enhancement/normalization. |
| VAD model | Silero VAD from `external/silero-vad` | `pyannote/voice-activity-detection` | Silero VAD -> Pyannote VAD. |
| VAD postprocessing | Threshold 0.30/0.25; recursive min-score split; merge to 28 s | Merge/filter plus probability-guided recursive splitting; RMS/duration fallback; merge to 28 s | Score-threshold Silero segmentation changed to probability-guided Pyannote segmentation. |
| ASR model | WhisperX `large-v3` | WhisperX `large-v3` | Model family unchanged. |
| ASR segmentation | Manual slicing from Silero grounding | Manual slicing from Path B Pyannote VAD | Silero-grounded slicing -> Path B Pyannote probability-guided slicing. |
| Alignment | WhisperX Wav2Vec2 alignment | WhisperX English alignment model | Alignment strategy unchanged at the wrapper level. |
| Diarization model | `pyannote/speaker-diarization-3.1` on muted audio | NeMo `diar_msdd_telephonic` with oracle VAD | Pyannote diarization -> oracle-grounded MSDD. |
| Embeddings/clustering | Pyannote diarization pipeline clustering | TitaNet Large multiscale embeddings plus NeMo clustering/MSDD | Pyannote clustering -> TitaNet/MSDD diarization stack. |
| Fusion | `whisperx.assign_word_speakers` | Maximum-overlap word attribution | Wrapper fusion -> explicit maximum-overlap fusion. |
| Evaluation | Conversational timeline CSV metrics | Stage-specific JSON/Markdown metrics and aggregate report | Single conversational evaluator -> per-stage plus fusion evaluation suite. |

## Metrics comparison table

Absolute difference is `Phase20 - Baseline`. Relative difference is `(Phase20 - Baseline) / Baseline * 100`. For error metrics, negative values indicate lower error in Phase20. For VAD, Phase20 has separate Path A and Path B metrics; Path B is the direct ASR segmentation path.

| Metric | Baseline | Phase20 | Absolute Difference | Relative Difference |
|---|---:|---:|---:|---:|
| VAD Precision, Path A | 0.755216 | 0.844796 | +0.089580 | +11.86% |
| VAD Recall, Path A | 0.952377 | 0.772101 | -0.180276 | -18.93% |
| VAD F1, Path A | 0.842414 | 0.806814 | -0.035600 | -4.23% |
| VAD Miss, Path A | 0.047623 | 0.227899 | +0.180276 | +378.55% |
| VAD False Alarm, Path A | 0.704979 | 0.299079 | -0.405901 | -57.58% |
| VAD Precision, Path B | 0.755216 | 0.841664 | +0.086447 | +11.45% |
| VAD Recall, Path B | 0.952377 | 0.766975 | -0.185402 | -19.47% |
| VAD F1, Path B | 0.842414 | 0.802585 | -0.039829 | -4.73% |
| VAD Miss, Path B | 0.047623 | 0.233025 | +0.185402 | +389.31% |
| VAD False Alarm, Path B | 0.704979 | 0.304217 | -0.400763 | -56.85% |
| ASR WER | 0.303700 | 0.324380 | +0.020680 | +6.81% |
| ASR Substitutions | 68 | 86 | +18 | +26.47% |
| ASR Insertions | 31 | 26 | -5 | -16.13% |
| ASR Deletions | 48 | 45 | -3 | -6.25% |
| ASR Reference Words | 484 | 484 | 0 | 0.00% |
| Diarization DER | 0.423400 | 0.447059 | +0.023659 | +5.59% |
| Diarization Miss | 0.178400 | 0.230000 | +0.051600 | +28.92% |
| Diarization False Alarm | 0.174600 | 0.139916 | -0.034684 | -19.86% |
| Diarization Confusion | 0.070400 | 0.077143 | +0.006743 | +9.58% |
| Diarization JER | 0.085700 | 0.842448 | +0.756748 | +883.02% |
| Fusion Speaker Accuracy | 0.900800 | 0.902743 | +0.001943 | +0.22% |
| Fusion WAE / WDER | 0.099200 | 0.097257 | -0.001943 | -1.96% |
| Fusion Timeline Accuracy | Not reported | 0.705312 | Not computable | Not computable |

Baseline ASR reference words are not present as a separate column in `reports/conversational_eval_phase1_silero.csv`; the value of 484 is the denominator implied by the reported BWC WER and edit counts and matches the Phase20 reference-word count in `phase20_standalone/metrics/asr_metrics.json`. Baseline speaker accuracy is reported here as `1 - WDER`, following the Phase20 validation definition that word attribution error/WDER is `1 - speaker_accuracy`. Baseline timeline accuracy was not found in the Phase 1 baseline artifacts. The baseline `JER` column is generated by `evaluate_conversational_timeline.py` as `Conf / (1 - Miss)`, whereas Phase20 uses `pyannote.metrics.diarization.JaccardErrorRate`; the JER row is therefore included for completeness but should not be interpreted as a like-for-like metric definition.

## Improvements

Phase20 substantially reduces VAD false alarms relative to the Silero baseline. Path B false alarm rate decreases from 0.704979 to 0.304217, an absolute reduction of 0.400763 and a relative reduction of 56.85%. Path A shows a similar reduction of 57.58%.

Phase20 also reduces ASR insertions and deletions. Insertions decline from 31 to 26, and deletions decline from 48 to 45. Fusion word attribution error declines from 0.099200 to 0.097257, corresponding to a small increase in speaker accuracy from 0.900800 to 0.902743.

Phase20 adds metrics that were not available in the original Phase 1 baseline artifacts, especially timeline accuracy and explicit word attribution error in `phase20_standalone/metrics/fusion_metrics.json`.

## Trade-offs

The reduction in VAD false alarms comes with lower VAD recall. Baseline Silero grounding has recall 0.952377 after long-context merging, whereas Phase20 Path B recall is 0.766975 and Path A recall is 0.772101. The corresponding VAD miss rates are higher in Phase20.

ASR WER is higher in Phase20: 0.324380 versus 0.303700. The increase is driven by more substitutions, partially offset by fewer insertions and deletions.

Diarization DER is also higher in Phase20: 0.447059 versus 0.423400. Phase20 reduces diarization false alarm but increases diarization miss and confusion in the published BWC run.

## Discussion

The repository shows a clear architectural shift from an early single-path Silero-grounded WhisperX/Pyannote pipeline to a final dual-path system. The final system separates diarization and transcription audio preparation, applies DeepFilterNet only to the diarization path, applies loudness normalization to the ASR path, replaces Silero VAD with Pyannote VAD, replaces Pyannote diarization with oracle-grounded NeMo MSDD using TitaNet embeddings, and replaces wrapper-based speaker assignment with explicit maximum-overlap fusion.

The numerical comparison does not show uniform improvement across all metrics. Phase20 improves VAD precision, VAD false alarm, ASR insertions, ASR deletions, and word-level speaker attribution error. The original baseline remains better on VAD recall, VAD F1, ASR WER, and DER in the available BWC metrics. The Phase20 contribution documented in the repository is therefore best characterized as a more auditable, modular, dual-path evaluation pipeline with improved false-alarm control and explicit fusion metrics, rather than a uniformly lower-error replacement across every measured axis.
