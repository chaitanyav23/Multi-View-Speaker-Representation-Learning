# Phase 13 — Karen End-to-End Generalization Benchmark Report

## Objective
This report evaluates the generalization performance of the current best-validated conversational intelligence pipeline (Phase 11A / Phase 10C architecture) on the unseen "Karen" dataset.

## Architecture
```text
Karen.wav (Truncated 426s)
      │
      ▼
FFmpeg LoudNorm (I=-20 TP=-1.5 LRA=11)
      │
      ▼
Karen_Normalized.wav
      │
      ▼
Pyannote 3.1 VAD (Merge-Gap=0.15s)
      │
      ▼
Grounding JSON -> Oracle-VAD RTTM
      │
      ├─────────────────────┐
      │                     │
      ▼                     ▼
NeMo MSDD             WhisperX Large-v3
Speaker Discovery     language="en"
(max_spk=10, auto)    Wav2Vec2 Alignment
      │                     │
      ▼                     ▼
Speaker Turns     Word-level Timeline
      │                     │
      └──────────┬──────────┘
                 ▼
      Maximum-Overlap Fusion
                 ▼
      Speaker Attributed Timeline
                 ▼
            Evaluation
```

## Directory Structure
```text
experiments/phase13_karen_generalization/
├── audio/
│   └── karen_normalized.wav
├── grounding/
│   ├── karen_grounding.json
│   └── karen_oracle.rttm
├── manifests/
│   └── karen_manifest.json
├── diarization/
│   ├── karen_msdd.rttm
│   └── msdd_outputs/
├── outputs/
│   ├── aligned_words.json
│   ├── karen_final.json
│   └── karen_final.txt
├── reports/
│   ├── audio_audit.csv
│   ├── vad_metrics.csv
│   ├── asr_metrics.csv
│   ├── diarization_metrics.csv
│   ├── attribution_metrics.csv
│   └── conversational_metrics.csv
└── comparisons/
    └── karen_vs_bwc.csv
```

## Audio Audit
| Metric | Value |
| --- | --- |
| Duration | 426.0s |
| Sample Rate | 16000 Hz |
| LUFS | -20.16 |
| RMS | 0.091 |
| Peak | 0.664 |

## VAD Analysis
| Metric | Value |
| --- | --- |
| Accuracy | 0.8277 |
| Precision | 0.8588 |
| Recall | 0.9427 |
| F1 | 0.8988 |
| Miss Rate | 0.0573 |
| False Alarm Rate | 0.6700 |

*Observations:* Decent F1 score, but high False Alarm rate suggests the VAD is overly sensitive or noise in the background is being caught.

## ASR Analysis
| Metric | Value |
| --- | --- |
| WER | 0.1467 |
| Substitutions | 41 |
| Deletions | 156 |
| Insertions | 33 |

*Observations:* WER is significantly lower than BWC (30%), indicating the Karen recording has higher intelligibility or less masking noise.

## Diarization Analysis
| Metric | Value |
| --- | --- |
| DER | 0.4070 |
| Miss | 0.0814 |
| False Alarm | 0.1512 |
| Confusion | 0.1745 |

*Observations:* Confusion and False Alarm are the primary components of DER.

## Attribution Analysis
| Metric | Value |
| --- | --- |
| Attribution Accuracy | 0.8151 |

## Overlap Analysis
* **Overlap Duration:** 9.00s (2.11%)
* **Observation:** Overlap is minimal in this recording compared to BWC.

## Conversational Metrics
| Metric | Value |
| --- | --- |
| cpWER | 0.2685 |
| WDER | 0.1849 |
| SA-WER | 0.2685 |

## Karen vs BWC Comparison
| Metric | BWC Phase 11A | Karen | Delta |
| --- | --- | --- | --- |
| WER | 0.3002 | 0.1467 | -0.1535 |
| DER | 0.4679 | 0.4070 | -0.0609 |
| Miss | 0.2878 | 0.0814 | -0.2064 |
| FA | 0.0766 | 0.1512 | +0.0746 |
| Conf | 0.1035 | 0.1745 | +0.0710 |
| cpWER | 0.5032 | 0.2685 | -0.2347 |
| WDER | 0.1136 | 0.1849 | +0.0713 |

## Final Questions Answered

1. **Does the pipeline generalize beyond BWC?**
   Yes, the architecture successfully processed the Karen dataset and achieved significantly lower cpWER than on BWC.

2. **How many speakers did MSDD discover?**
   MSDD discovered **6 speakers**.

3. **Is MSDD under-clustering or over-clustering?**
   MSDD is **over-clustering**. The ground truth has 4 speakers (Speaker_1, Speaker_2, Speaker_3, Speaker_4). MSDD found 6, leading to increased Confusion.

4. **What is the largest contributor to cpWER?**
   The largest contributor to cpWER is **ASR deletions** and **Speaker Confusion**. However, the absolute cpWER is much better than BWC.

5. **Is overlap a major failure mode?**
   No, overlap in this recording is only 2.11%, so it is not a primary driver of errors.

6. **Is attribution a larger problem than ASR?**
   In terms of relative error contribution, WDER (18.49%) is comparable to WER (14.67%). Both are significant.

7. **What should be improved in Phase 14?**
   Speaker clustering stability (reducing over-clustering) and handling False Alarms in VAD appear to be the next logical frontiers for generalization.

## Bottleneck Identification
The primary bottleneck remains **Speaker Confusion** in diarization (specifically over-clustering) and **Deletions** in ASR. The pipeline generalizes well in terms of structure, but parameter tuning for clustering and VAD sensitivity is needed for different acoustic environments.
