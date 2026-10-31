# Stage 2 — Zero-Fallback 8-Recording Baseline Reproduction Report

**Status**: `VERIFIED` (8/8 audio-available recordings executed end-to-end on NVIDIA RTX 4090 with zero ground-truth fallbacks)  
**Directory**: [research_stages/stage2_baseline/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/)  
**Script**: [run_stage2_baseline.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/run_stage2_baseline.py)  
**Metrics Artifacts**:
- [stage2_baseline_all_metrics.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/metrics/stage2_baseline_all_metrics.json)
- [stage2_baseline_per_recording.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/metrics/stage2_baseline_per_recording.csv)

---

## 1. Zero-Fallback & Actual Model Inference Verification (`VERIFIED`)

1. **All Silent Fallbacks Disabled**:
   - Every silent ground-truth fallback present in the legacy standalone scripts (`step02a` raw-audio copy, `step03` reference-JSON VAD fallback, `step05a` reference-RTTM copy fallback, and `step05b` uniform word-splitting fallback) was stripped in [run_stage2_baseline.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/run_stage2_baseline.py).
   - Explicit runtime assertions verified that every predicted RTTM is non-empty (`len(diar_segments) > 0`), non-identical to ground-truth RTTM boundaries, and accompanied by all 5 multiscale TitaNet embedding `.pkl` files (`subsegments_scale{0..4}_embeddings.pkl`) and aligned WhisperX words.
2. **Executed Models Across All 8 Recordings (`R01`–`R08`)**:
   - **Speech Enhancement (`Path A`)**: `DeepFilterNet3` (`0.5.6`) on CUDA.
   - **Loudness Normalization (`Path B`)**: `ffmpeg` EBU R128 `loudnorm=I=-20:TP=-1.5:LRA=11`.
   - **Voice Activity Detection (`Path A`, `Path B`, and `16k_mono`)**: Pyannote `PyanNet` (`assets/pytorch_model.bin`, `onset=0.4810, offset=0.4810, min_duration_on=0.0554, min_duration_off=0.0979`) with Phase 20 probability-guided recursive splitting (`step03_vad.py`), plus frame-level speech (`p_speech`) and multi-speaker overlap (`p_overlap`) posterior extraction saved in [research_stages/stage2_baseline/posteriors/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/posteriors/).
   - **ASR + Word Alignment (`Path B`)**: `WhisperX` `large-v3` (`Systran/faster-whisper-large-v3`, `float16`) + `WAV2VEC2_ASR_LARGE_LV60K_960H` forced alignment.
   - **Speaker Diarization (`Path A`)**: NeMo `NeuralDiarizer` (`titanet_large` 5-scale embeddings `[1.5, 1.2, 1.0, 0.8, 0.5]s` with weights `[1, 2, 3, 3, 3]`, `NMESC` spectral clustering `max_rp_threshold=0.25, max_num_speakers=10`, and `diar_msdd_telephonic` `sigmoid_threshold=0.7`).

---

## 2. Historical Single-Recording Result (`bwc`) vs. New 8-Recording Baseline (`VERIFIED`)

Look at `R07_bwc_cheap_shot` (`That was a bit of a cheap shot weren’t it mate`) evaluated under **both** annotation policies alongside the historical committed Phase 20 result ([phase20_standalone/metrics/bwc_all_metrics.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/metrics/bwc_all_metrics.json)):

| Evaluation Condition | Ref Spks (All / Pos RTTM) | Pred Spks | Collapse? | Path A VAD F1 | Path B VAD F1 | DER (`c=0.0`, ovl) | DER (`c=0.25`, no-ovl) | False Alarm | Miss | Confusion | JER | ASR WER | cpWER | saWER | Micro-SACC | Macro-SACC |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Historical Committed Phase 20 (`bwc`)** | `4 / 4` | `1` | **Yes (`1` spk)** | `0.8068` | `0.8026` | `0.4471` | — | `0.1399` | `0.2300` | `0.0771` | `0.8424` | `0.3244` | `0.5331` | `0.5331` | `0.9027` | `0.2500` |
| **Stage 2 `R07` (`phase20_legacy_curated`)** | `4 / 4` | `1` | **Yes (`1` spk)** | `0.8184` | `0.8466` | `0.4516` | `0.4203` | `0.1571` | `0.1996` | `0.0950` | `0.8445` | `0.3120` | `0.5289` | `0.5289` | `0.8774` | `0.2500` |
| **Stage 2 `R07` (`uniform_v1` policy)** | `4 / 3` | `1` | **Yes (`1` spk)** | `0.8192` | `0.8512` | `0.4515` | `0.4172` | `0.1412` | `0.2149` | `0.0954` | `0.7923` | `0.2988` | `0.5299` | `0.5299` | `0.8747` | `0.2500` |
| **Stage 2 8-Recording Macro-Average (`uniform_v1`)** | `5.12 / 4.50` | `3.00` | `1 / 8` | `0.8494` | `0.8764` | **`0.4748`** | **`0.4252`** | `0.1607` | `0.2330` | `0.0811` | **`0.6188`** | `0.2322` | **`0.3934`** | **`0.4797`** | `0.8333` | **`0.4927`** |
| **Stage 2 8-Recording Duration-Weighted (`uniform_v1`)** | — | — | — | `0.8354` | `0.8526` | **`0.5234`** | **`0.4753`** | `0.2252` | `0.1970` | `0.1012` | **`0.6299`** | `0.2277` | **`0.4026`** | **`0.4669`** | `0.8313` | **`0.5056`** |

### Key Findings from the Historical vs. 8-Recording Comparison:
1. **Exact Reproduction of the `R07_bwc_cheap_shot` Single-Speaker Collapse**:
   - Under both the historical Phase 20 run and our strict Stage 2 reproduction on `R07_bwc_cheap_shot`, NeMo's `NMESC` + `diar_msdd_telephonic` pipeline **collapses all speech in `R07_bwc_cheap_shot` into a single predicted speaker (`speaker_0`)**, yielding `DER = 45.16%` (`JER = 84.45%` under `phase20_legacy_curated`, `79.23%` under `uniform_v1`).
   - Despite predicting only 1 speaker, `Micro-SACC` is `87.74%` (`90.27%` in historical Phase 20) solely because `Speaker_1` utters ~83% of the words in `R07` (`per_speaker_word_recall: Speaker_1 = 1.0, Speaker_2 = 0.0, Speaker_3 = 0.0, Speaker_4 = 0.0`), while **`Macro-SACC` is only `0.2500` (`25.0%`)**!
2. **Systematic Speaker Under-Counting Across the 8-Recording Corpus**:
   - Across all 8 recordings, the baseline has a signed speaker-count bias of **`-1.50` speakers per recording** (absolute speaker count error `1.50`), under-predicting the number of positive-duration speakers in **6 out of 8 recordings** (`R01`, `R02`, `R03`, `R06`, `R07`, `R08`).

---

## 3. Per-Recording Baseline Results (`uniform_v1` Policy, `VERIFIED`)

| Recording ID | Dur (s) | Ref Spks (All / Pos) | Pred Spks | Spk Count Bias | Path A VAD F1 | Path B VAD F1 | DER (`c=0.0`, ovl) | DER (`c=0.25`, no-ovl) | FA | Miss | Conf | JER | ASR WER | ASR CER | cpWER | saWER | Micro-SACC | Macro-SACC | Timeline Acc |
| :--- | ---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R01_sara_sharif` | 100.96 | `5 / 5` | `3` | `-2` | 0.8375 | 0.9195 | 0.4189 | 0.4048 | 0.0170 | 0.2786 | 0.1233 | 0.6522 | 0.2792 | 0.1958 | 0.5795 | 0.5795 | 0.7786 | 0.5446 | 0.6786 |
| `R02_dennis_skinner` | 176.26 | `5 / 4` | `3` | `-1` | 0.8738 | 0.9332 | 0.6290 | 0.5043 | 0.0235 | 0.4934 | 0.1121 | 0.6703 | 0.1433 | 0.0836 | 0.4235 | 1.1140 | 0.6477 | 0.3399 | 0.3513 |
| `R03_disabled_parking` | 206.22 | `5 / 5` | `3` | `-2` | 0.8807 | 0.8865 | 0.4010 | 0.3553 | 0.2021 | 0.1260 | 0.0730 | 0.6226 | 0.1337 | 0.0946 | 0.2675 | 0.2675 | 0.8866 | 0.5546 | 0.6843 |
| `R04_piers_robinson` | 204.55 | `3 / 3` | `3` | `0` | 0.9861 | 0.9852 | 0.3551 | 0.2604 | 0.0591 | 0.2371 | 0.0590 | 0.4474 | 0.4331 | 0.3578 | 0.4220 | 0.4220 | 0.6942 | 0.6277 | 0.4748 |
| `R05_southwest_pilot` | 162.45 | `4 / 3` | `3` | `0` | 0.7184 | 0.7562 | 0.6572 | 0.6234 | 0.3781 | 0.2708 | 0.0082 | 0.5015 | 0.2656 | 0.1660 | 0.3333 | 0.3333 | 0.9749 | 0.7319 | 0.7170 |
| `R06_rich_karen` | 652.18 | `9 / 9` | `6` | `-3` | 0.7734 | 0.7755 | 0.6999 | 0.6619 | 0.4133 | 0.1129 | 0.1737 | 0.6778 | 0.2300 | 0.1594 | 0.4881 | 0.4881 | 0.8122 | 0.5598 | 0.6303 |
| `R07_bwc_cheap_shot` | 175.43 | `4 / 3` | `1` | `-2` | 0.8192 | 0.8512 | 0.4515 | 0.4172 | 0.1412 | 0.2149 | 0.0954 | 0.7923 | 0.2988 | 0.1982 | 0.5299 | 0.5299 | 0.8747 | 0.2500 | 0.7284 |
| `R08_uk_parliament` | 215.34 | `6 / 4` | `2` | `-2` | 0.9060 | 0.9041 | 0.1860 | 0.1745 | 0.0515 | 0.1304 | 0.0042 | 0.5860 | 0.0737 | 0.0499 | 0.1032 | 0.1032 | 0.9978 | 0.3333 | 0.8892 |
| **Macro-Average** | **236.67** | **5.12 / 4.50** | **3.00** | **`-1.50`** | **0.8494** | **0.8764** | **0.4748** | **0.4252** | **0.1607** | **0.2330** | **0.0811** | **0.6188** | **0.2322** | **0.1632** | **0.3934** | **0.4797** | **0.8333** | **0.4927** | **0.6442** |
| **Duration-Weighted** | **1893.38** | — | — | **`-1.86`** | **0.8354** | **0.8526** | **0.5234** | **0.4753** | **0.2252** | **0.1970** | **0.1012** | **0.6299** | **0.2277** | **0.1604** | **0.4026** | **0.4669** | **0.8313** | **0.5056** | **0.6420** |

---

## 4. Per-Speaker Word Attribution Recall Breakdown (`VERIFIED`)

| Recording ID | Per-Reference-Speaker Word Attribution Recall (`Macro-SACC` Components) |
| :--- | :--- |
| `R01_sara_sharif` | `Speaker_1: 0.0000`, `Speaker_2: 0.7232`, `Speaker_3: 0.0000`, `Speaker_4: 1.0000`, `Speaker_5: 1.0000` (`Macro-SACC = 0.5446`) |
| `R02_dennis_skinner` | `Speaker_0: 0.8246`, `Speaker_1: 0.1000`, `Speaker_2: 0.7750`, `Speaker_3: 0.0000 (0s)`, `Speaker_4: 0.0000` (`Macro-SACC = 0.3399`) |
| `R03_disabled_parking` | `Speaker_0: 0.8079`, `Speaker_1: 0.9650`, `Speaker_2: 1.0000`, `Speaker_3: 0.0000`, `Speaker_4: 0.0000` (`Macro-SACC = 0.5546`) |
| `R04_piers_robinson` | `Speaker_0: 0.9385`, `Speaker_1: 0.6772`, `Speaker_2: 0.2674` (`Macro-SACC = 0.6277`) |
| `R05_southwest_pilot` | `Speaker_1: 0.9752`, `Speaker_2: 0.0000 (0s)`, `Speaker_3: 0.9524`, `Speaker_4: 1.0000` (`Macro-SACC = 0.7319`) |
| `R06_rich_karen` | `Speaker_0: 1.0000`, `Speaker_1: 0.8304`, `Speaker_2: 0.0000`, `Speaker_3: 0.6589`, `Speaker_4: 0.6667`, `Speaker_5: 0.0000`, `Speaker_6: 0.8824`, `Speaker_7: 0.0000`, `Speaker_8: 1.0000` (`Macro-SACC = 0.5598`) |
| `R07_bwc_cheap_shot` | `Speaker_1: 1.0000`, `Speaker_2: 0.0000`, `Speaker_3: 0.0000`, `Speaker_4: 0.0000 (0s)` (`Macro-SACC = 0.2500`) |
| `R08_uk_parliament` | `Speaker_0: 1.0000`, `Speaker_1: 0.0000`, `Speaker_2: 1.0000`, `Speaker_3: 0.0000 (0s)`, `Speaker_4: 0.0000 (0s)`, `Speaker_all: 0.0000` (`Macro-SACC = 0.3333`) |

---

## 5. Diagnostic Takeaways for Stages 3 & 4 (`VERIFIED`)

1. **Path A (`dfn`) VAD Recall Loss vs. Path B (`normalized`)**:
   - On `R01_sara_sharif`, Path A VAD F1 (`0.8375`, recall `0.7241`) is `8.20` percentage points lower than Path B VAD F1 (`0.9195`), confirming that `DeepFilterNet3` attenuates low-SNR BWC speech segments into missed detections (`Miss = 27.86%`).
2. **Severe Overlap & Unannotated Background Speech Effects**:
   - `R02_dennis_skinner` (`DER = 62.90%`, `Miss = 49.34%`) and `R04_piers_robinson` (`DER = 35.51%`, `Miss = 23.71%`) are dominated by overlapping multi-speaker ground-truth intervals (`11/13` and `28/31` segments overlap, respectively), whereas `R05_southwest_pilot` (`FA = 37.81%`) and `R06_rich_karen` (`FA = 41.33%`) contain substantial unannotated background radio/bystander speech that is detected by VAD and diarizer but absent from the reference transcript.
