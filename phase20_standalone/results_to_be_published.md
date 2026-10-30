# Phase20 Final Published Results

=================================================

## Pipeline Overview

Phase20 is a dual-path speech-processing pipeline. Path A applies DeepFilterNet before Pyannote VAD and NeMo MSDD diarization. Path B applies loudness normalization before Pyannote VAD and WhisperX large-v3 transcription/alignment. Fusion assigns aligned words to diarization speakers by maximum temporal overlap and evaluates speaker-attributed output.

## Environment Summary

| Component | Value |
|---|---:|
| Python | 3.10.12 |
| OS | Ubuntu 22.04.5 LTS |
| GPU | Tesla T4, 15360 MiB |
| Driver | 580.159.04 |
| NVIDIA-SMI CUDA | 13.0 |
| Torch | 2.8.0+cu128 |
| TorchAudio | 2.8.0+cu128 |
| WhisperX | 3.8.5 |
| Faster-Whisper | 1.2.1 |
| ctranslate2 | 4.7.2 |
| Pyannote Audio | 4.0.4 |
| NeMo | 2.7.3 |
| NumPy | 2.2.6 |

## Hardware

Validated on NVIDIA Tesla T4 with CUDA-capable Torch execution. DeepFilterNet, Pyannote VAD, WhisperX, and NeMo MSDD all executed on CUDA in the final validation logs.

## Software

The exact software environment is frozen in `environment/requirements.lock`, `environment/requirements_frozen.txt`, `environment/pip_freeze.txt`, and `environment/environment_snapshot.json`.

## Model Versions

See `environment/model_manifest.md` for model repositories, revisions, cache paths, and hashes.

## Dataset

| Item | File |
|---|---|
| Audio | `raw_audio/bwc.wav` |
| Transcript/speaker reference | `ground_truth_json/bwc.json` |
| Diarization reference | `ground_truth_rttm/bwc.rttm` |

## Metrics

### VAD

| Metric | Path A | Path B |
|---|---:|---:|
| Precision | 0.844796 | 0.841664 |
| Recall | 0.772101 | 0.766975 |
| F1 | 0.806814 | 0.802585 |
| False Alarm | 0.299079 | 0.304217 |
| Miss | 0.227899 | 0.233025 |

### ASR

| Metric | Value |
|---|---:|
| WER | 0.324380 |
| Substitutions | 86 |
| Insertions | 26 |
| Deletions | 45 |
| Reference Words | 484 |
| Hypothesis Words | 465 |
| Aligned Words | 465 |
| Whisper Segments | 41 |

### Diarization

| Metric | Value |
|---|---:|
| DER | 0.447059 |
| JER | 0.842448 |
| Miss | 0.230000 |
| False Alarm | 0.139916 |
| Confusion | 0.077143 |

### Fusion

| Metric | Value |
|---|---:|
| cpWER | 0.533058 |
| SA-WER | 0.533058 |
| Speaker Accuracy | 0.902743 |
| Word Attribution Error | 0.097257 |
| Timeline Accuracy | 0.705312 |
| Words Evaluated | 401 |
| Correct Attributions | 362 |
| Speaker Mapping | `{"speaker_0": "Speaker_1"}` |

=================================================

## Comparison

| Metric | Phase19 | Original Phase20 | Final Published Phase20 |
|---|---:|---:|---:|
| WER | 0.324380 | 0.324380 | 0.324380 |
| Aligned Words | 465 | 465 | 465 |
| Whisper Segments | 41 | 41 | 41 |
| Path B VAD Segments | 41 | 41 | 41 |
| DER | 0.447059 | 0.447059 | 0.447059 |
| Speaker Accuracy | 0.902743 | 0.902743 | 0.902743 |

=================================================

## Tables

### Table 1. VAD Metrics

| Metric | Path A | Path B |
|---|---:|---:|
| Precision | 0.844796 | 0.841664 |
| Recall | 0.772101 | 0.766975 |
| F1 | 0.806814 | 0.802585 |
| False Alarm | 0.299079 | 0.304217 |
| Miss | 0.227899 | 0.233025 |

### Table 2. ASR Metrics

| Metric | Value |
|---|---:|
| WER | 0.324380 |
| Substitutions | 86 |
| Insertions | 26 |
| Deletions | 45 |
| Reference Words | 484 |
| Hypothesis Words | 465 |

### Table 3. Speaker Diarization Metrics

| Metric | Value |
|---|---:|
| DER | 0.447059 |
| JER | 0.842448 |
| Miss | 0.230000 |
| False Alarm | 0.139916 |
| Confusion | 0.077143 |

### Table 4. Fusion Metrics

| Metric | Value |
|---|---:|
| cpWER | 0.533058 |
| SA-WER | 0.533058 |
| Speaker Accuracy | 0.902743 |
| Word Attribution Error | 0.097257 |
| Timeline Accuracy | 0.705312 |

### Table 5. Pipeline Summary

| Stage | Model | Output | Status |
|---|---|---|---|
| Enhancement | DeepFilterNet3 | `audio/bwc_dfn.wav` | PASS |
| VAD | Pyannote VAD | `grounding/pathB_vad.json` | PASS |
| ASR | WhisperX large-v3 | `transcription/aligned_words.json` | PASS |
| Diarization | NeMo MSDD | `diarization/pathA_msdd.rttm` | PASS |
| Fusion | Temporal overlap attribution | `fusion/attributed_words.json` | PASS |

### Table 6. Metric Definitions

| Metric | Definition |
|---|---|
| WER | `(Substitutions + Deletions + Insertions) / Reference Words` |
| DER | Diarization error rate from miss, false alarm, and confusion components |
| JER | Jaccard error rate over diarization speaker regions |
| cpWER | Concatenated minimum-permutation word error rate |
| SA-WER | Speaker-attributed WER after speaker mapping |
| Speaker Accuracy | Correct speaker attributions divided by evaluated words |
| Word Attribution Error | `1 - Speaker Accuracy` |
| Timeline Accuracy | Temporal agreement between speaker-attributed words and reference timeline |

=================================================

## Figures

- `plots/missing_word_heatmap.png`
- `plots/reference_vs_vad.png`
- `plots/reference_vs_whisperx.png`
- `plots/segment_word_histogram.png`
- `plots/timeline_comparison.png`
- `plots/timeline_reference_vs_whisperx.png`
- `plots/word_density.png`

=================================================

## Reproducibility Statement

This repository reproduces the original committed Phase20 ASR/VAD checkpoints using the archived frozen environment: WER `0.324380`, 465 aligned words, 41 Whisper segments, and 41 Path B VAD segments. The final validation run executed DeepFilterNet, Pyannote, WhisperX, NeMo MSDD, evaluation, and fusion without unintended fallback use.

=================================================

## Known Limitations

- The frozen historical environment requires `pip install --no-deps -r environment/requirements.lock`.
- The lock intentionally preserves the DeepFilterNet/NumPy and wheel/packaging metadata conflicts documented in `README.md`.
- Pyannote models require HuggingFace authentication and accepted gated model terms on first download.
- CUDA-capable hardware is the validated configuration.
- `TRANSFORMERS_CACHE` emits a deprecation warning under Transformers 4.57.6; it is retained for historical cache compatibility.

=================================================

## Citation

```bibtex
@software{phase20_standalone_2026,
  title = {Phase20 Standalone Reproducible Speech Processing Artifact},
  author = {Phase20 Research Team},
  year = {2026},
  url = {REPOSITORY_URL},
  note = {Frozen environment reproducing commit 44e9bb7 Phase20 results}
}
```

Software citation: cite this repository, the frozen environment manifests in `environment/`, and the model providers listed in `environment/model_manifest.md`.

Repository citation: include the repository URL, release tag, commit hash, and `results_to_be_published.md` in the manuscript artifact appendix.
