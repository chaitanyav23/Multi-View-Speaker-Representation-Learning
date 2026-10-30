# Phase 20 vs Phase 19 Validation Comparison

Phase 20 keeps the Phase 19 dual-path pipeline unchanged except for VAD post-processing.
The new VAD retains Pyannote frame probabilities and uses recursive probability-guided min-cut splitting for long speech regions.

## VAD Processing Statistics

### pathA

| Statistic | Phase 19 | Phase 20 |
|---|---:|---:|
| raw_segments | 47 | 47 |
| merged_segments | 46 | 46 |
| removed_short_segments | 0 | 0 |
| recursive_min_cut_splits | 0 | 0 |
| average_chunk_duration | None | 2.354761 |
| maximum_chunk_duration | None | 12.049000 |
| all_chunks_lte_30_seconds | True | True |
| final_segments | 46 | 46 |

Before splitting histogram:

```json
{
  "0-5": 41,
  "5-10": 2,
  "10-15": 3,
  "15-20": 0,
  "20-25": 0,
  "25-30": 0,
  ">30": 0
}
```

After splitting histogram:

```json
{
  "0-5": 41,
  "5-10": 2,
  "10-15": 3,
  "15-20": 0,
  "20-25": 0,
  "25-30": 0,
  ">30": 0
}
```

### pathB

| Statistic | Phase 19 | Phase 20 |
|---|---:|---:|
| raw_segments | 42 | 42 |
| merged_segments | 41 | 41 |
| removed_short_segments | 0 | 0 |
| recursive_min_cut_splits | 0 | 0 |
| average_chunk_duration | None | 2.636122 |
| maximum_chunk_duration | None | 11.188000 |
| all_chunks_lte_30_seconds | True | True |
| final_segments | 41 | 41 |

Before splitting histogram:

```json
{
  "0-5": 35,
  "5-10": 3,
  "10-15": 3,
  "15-20": 0,
  "20-25": 0,
  "25-30": 0,
  ">30": 0
}
```

After splitting histogram:

```json
{
  "0-5": 35,
  "5-10": 3,
  "10-15": 3,
  "15-20": 0,
  "20-25": 0,
  "25-30": 0,
  ">30": 0
}
```

## Metric Comparison

### pathA_vad

| Metric | Phase 19 | Phase 20 | Delta |
|---|---:|---:|---:|
| precision | 0.844796 | 0.844796 | 0.000000 |
| recall | 0.772101 | 0.772101 | 0.000000 |
| f1 | 0.806814 | 0.806814 | 0.000000 |
| false_alarm | 0.299079 | 0.299079 | 0.000000 |
| miss | 0.227899 | 0.227899 | 0.000000 |

### pathB_vad

| Metric | Phase 19 | Phase 20 | Delta |
|---|---:|---:|---:|
| precision | 0.841664 | 0.841664 | 0.000000 |
| recall | 0.766975 | 0.766975 | 0.000000 |
| f1 | 0.802585 | 0.802585 | 0.000000 |
| false_alarm | 0.304217 | 0.304217 | 0.000000 |
| miss | 0.233025 | 0.233025 | 0.000000 |

### asr

| Metric | Phase 19 | Phase 20 | Delta |
|---|---:|---:|---:|
| wer | 0.324380 | 0.324380 | 0.000000 |
| substitutions | 84 | 86 | 2 |
| insertions | 27 | 26 | -1 |
| deletions | 46 | 45 | -1 |
| reference_words | 484 | 484 | 0 |
| hypothesis_words | 465 | 465 | 0 |

### diarization

| Metric | Phase 19 | Phase 20 | Delta |
|---|---:|---:|---:|
| der | 0.447059 | 0.447059 | 0.000000 |
| miss | 0.230000 | 0.230000 | 0.000000 |
| false_alarm | 0.139916 | 0.139916 | 0.000000 |
| confusion | 0.077143 | 0.077143 | 0.000000 |
| jer | 0.842448 | 0.842448 | 0.000000 |

### fusion

| Metric | Phase 19 | Phase 20 | Delta |
|---|---:|---:|---:|
| cpwer | 3.103306 | 0.533058 | -2.570248 |
| sawer | 3.103306 | 0.533058 | -2.570248 |
| speaker_accuracy | 0.902743 | 0.902743 | 0.000000 |
| word_attribution_error | 0.097257 | 0.097257 | 0.000000 |
| timeline_accuracy | 0.762312 | 0.705312 | -0.057000 |

## Discussion

pathA satisfied the <= 30 second final chunk constraint. pathB satisfied the <= 30 second final chunk constraint. ASR WER did not improve. Diarization DER did not improve. Fusion cpWER improved.
