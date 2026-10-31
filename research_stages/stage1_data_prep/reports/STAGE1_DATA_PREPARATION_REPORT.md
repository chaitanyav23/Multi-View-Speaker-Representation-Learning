# Stage 1 — Non-Destructive Data Preparation & Annotation Quality Report

**Status**: `VERIFIED` (24/24 processed WAV files validated; 12/12 ground-truth transcripts parsed non-destructively; 0 source files modified)  
**Directory**: [research_stages/stage1_data_prep/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/)  
**Scripts**:
- [run_stage1_prep.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/run_stage1_prep.py)
- [run_stage1b_dfn.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/run_stage1b_dfn.py)

---

## 1. Non-Destructive Preservation Verification (`VERIFIED`)

1. **Source File Integrity**:
   - All 8 source MP3 files in [raw_audio/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/raw_audio/), all 12 source `.txt` transcripts in [ground_truth/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth/), and all frozen reference files in [phase20_standalone/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/) remain untouched (`git diff` on tracked files is empty).
   - SHA-256 checksums for every source `.mp3` and `.txt` file are recorded in [recording_manifest.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/manifests/recording_manifest.json).
2. **Canonical Recording Identifiers**:
   - To eliminate failure modes caused by spaces, ampersands (`&`), and Unicode right single quotation marks (`’`, `U+2019`) in shell invocations and NeMo manifests, each audio-available recording is assigned a deterministic canonical identifier `R01`–`R08` while preserving full provenance to the original stem:

| Canonical ID | Original Filename Stem | Source Audio Present | Duration (s) |
| :--- | :--- | :---: | ---: |
| `R01_sara_sharif` | `Bodycam footage show arrests of Sara Sharif’s alleged killers` | Yes (`mp3`) | `100.9604` |
| `R02_dennis_skinner` | `Dennis Skinner kicked out of Commons` | Yes (`mp3`) | `176.2627` |
| `R03_disabled_parking` | `Karen Has A Problem With Disabled Parking` | Yes (`mp3`) | `206.2164` |
| `R04_piers_robinson` | `Piers Confronts Tommy Robinson` | Yes (`mp3`) | `204.5446` |
| `R05_southwest_pilot` | `Police bodycam shows arrest of Southwest Airlines pilot` | Yes (`mp3`) | `162.4468` |
| `R06_rich_karen` | `Rich Karen Messes Around & Finds Out` | Yes (`mp3`) | `652.1789` |
| `R07_bwc_cheap_shot` | `That was a bit of a cheap shot weren’t it mate` | Yes (`mp3`) | `175.4267` |
| `R08_uk_parliament` | `UK Parliament Debate` | Yes (`mp3`) | `215.3419` |
| `M01_drunk_karen` | `Drunk Karen Gets a Huge Reality Check` | **No** (`.txt` only) | N/A |
| `M02_entitled_restaurant` | `Entitled Karen Refuses to Pay Restaurant Bill Then Attacks the Manager` | **No** (`.txt` only) | N/A |
| `M03_entitled_girl` | `When an Entitled Girl Gets Instantly Humbled` | **No** (`.txt` only) | N/A |
| `M04_traffic_stop` | `Woman's Attitude Makes Traffic Stop 10x Worse` | **No** (`.txt` only) | N/A |

---

## 2. Audio Generation & Validation Summary (`VERIFIED`)

Three controlled 16 kHz mono 16-bit PCM WAV variants (`24` files total across `R01`–`R08`) were generated in [research_stages/stage1_data_prep/audio/](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/audio/) and validated in [audio_validation_report.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/reports/audio_validation_report.csv):
- `16k_mono`: Unenhanced 16 kHz mono PCM waveform (`ffmpeg -af aresample=async=1:first_pts=0 -ar 16000 -ac 1 -c:a pcm_s16le`) — serves as the **common waveform** for Stage 3 primary dual-encoder comparison.
- `dfn`: DeepFilterNet3 (`0.5.6`) enhanced 16 kHz mono PCM waveform (`Path A` input).
- `normalized`: EBU R128 loudness-normalized (`loudnorm=I=-20:TP=-1.5:LRA=11`) 16 kHz mono PCM waveform (`Path B` input).

| Recording ID | Variant | Sample Rate | Channels | Samples | Duration (s) | Diff vs Source (s) | NaNs / Infs | Peak Amp | Clipped Samples | RMS (dBFS) | Status |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| `R01_sara_sharif` | `16k_mono` | 16000 | 1 | 1,615,366 | 100.9604 | 0.0290 | 0 / 0 | 0.9969 | 0 | -23.820 | `PASS` |
| `R01_sara_sharif` | `dfn` | 16000 | 1 | 1,615,366 | 100.9604 | 0.0290 | 0 / 0 | 0.9367 | 0 | -25.752 | `PASS` |
| `R01_sara_sharif` | `normalized` | 16000 | 1 | 1,615,366 | 100.9604 | 0.0290 | 0 / 0 | 0.9017 | 0 | -17.725 | `PASS` |
| `R02_dennis_skinner` | `16k_mono` | 16000 | 1 | 2,820,203 | 176.2627 | 0.0377 | 0 / 0 | 0.2287 | 0 | -26.012 | `PASS` |
| `R02_dennis_skinner` | `dfn` | 16000 | 1 | 2,820,203 | 176.2627 | 0.0377 | 0 / 0 | 0.2154 | 0 | -29.877 | `PASS` |
| `R02_dennis_skinner` | `normalized` | 16000 | 1 | 2,820,203 | 176.2627 | 0.0377 | 0 / 0 | 0.8380 | 0 | -16.561 | `PASS` |
| `R03_disabled_parking` | `16k_mono` | 16000 | 1 | 3,299,463 | 206.2164 | 0.0464 | 0 / 0 | 1.0000 | 13 (`3.9e-6`) | -21.758 | `PASS` |
| `R03_disabled_parking` | `dfn` | 16000 | 1 | 3,299,463 | 206.2164 | 0.0464 | 0 / 0 | 0.7628 | 0 | -24.167 | `PASS` |
| `R03_disabled_parking` | `normalized` | 16000 | 1 | 3,299,463 | 206.2164 | 0.0464 | 0 / 0 | 0.8480 | 0 | -17.139 | `PASS` |
| `R04_piers_robinson` | `16k_mono` | 16000 | 1 | 3,272,713 | 204.5446 | 0.0465 | 0 / 0 | 0.2982 | 0 | -23.884 | `PASS` |
| `R04_piers_robinson` | `dfn` | 16000 | 1 | 3,272,713 | 204.5446 | 0.0465 | 0 / 0 | 0.2804 | 0 | -25.742 | `PASS` |
| `R04_piers_robinson` | `normalized` | 16000 | 1 | 3,272,713 | 204.5446 | 0.0465 | 0 / 0 | 0.8419 | 0 | -16.047 | `PASS` |
| `R05_southwest_pilot` | `16k_mono` | 16000 | 1 | 2,599,149 | 162.4468 | 0.0348 | 0 / 0 | 0.7754 | 0 | -24.962 | `PASS` |
| `R05_southwest_pilot` | `dfn` | 16000 | 1 | 2,599,149 | 162.4468 | 0.0348 | 0 / 0 | 0.6128 | 0 | -28.294 | `PASS` |
| `R05_southwest_pilot` | `normalized` | 16000 | 1 | 2,599,149 | 162.4468 | 0.0348 | 0 / 0 | 0.8435 | 0 | -17.481 | `PASS` |
| `R06_rich_karen` | `16k_mono` | 16000 | 1 | 10,434,862 | 652.1789 | 0.0464 | 0 / 0 | 0.8464 | 0 | -17.540 | `PASS` |
| `R06_rich_karen` | `dfn` | 16000 | 1 | 10,434,862 | 652.1789 | 0.0464 | 0 / 0 | 0.7610 | 0 | -23.451 | `PASS` |
| `R06_rich_karen` | `normalized` | 16000 | 1 | 10,434,862 | 652.1789 | 0.0464 | 0 / 0 | 0.8430 | 0 | -15.795 | `PASS` |
| `R07_bwc_cheap_shot` | `16k_mono` | 16000 | 1 | 2,806,828 | 175.4267 | 0.0377 | 0 / 0 | 0.9985 | 0 | -21.173 | `PASS` |
| `R07_bwc_cheap_shot` | `dfn` | 16000 | 1 | 2,806,828 | 175.4267 | 0.0377 | 0 / 0 | 0.7318 | 0 | -24.518 | `PASS` |
| `R07_bwc_cheap_shot` | `normalized` | 16000 | 1 | 2,806,828 | 175.4267 | 0.0377 | 0 / 0 | 0.8604 | 0 | -16.797 | `PASS` |
| `R08_uk_parliament` | `16k_mono` | 16000 | 1 | 3,445,470 | 215.3419 | 0.0377 | 0 / 0 | 0.4494 | 0 | -20.560 | `PASS` |
| `R08_uk_parliament` | `dfn` | 16000 | 1 | 3,445,470 | 215.3419 | 0.0377 | 0 / 0 | 0.4227 | 0 | -23.005 | `PASS` |
| `R08_uk_parliament` | `normalized` | 16000 | 1 | 3,445,470 | 215.3419 | 0.0377 | 0 / 0 | 0.8446 | 0 | -16.206 | `PASS` |

---

## 3. Recording-Level & Speaker-Level Annotation Quality (`VERIFIED`)

Full details are exported in:
- [stage1_recording_quality.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/reports/stage1_recording_quality.csv)
- [stage1_speaker_quality.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/reports/stage1_speaker_quality.csv)
- [stage1_segment_ledger.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/reports/stage1_segment_ledger.csv) (`1,362` segments)
- [stage1_anomalies_and_exclusions.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/reports/stage1_anomalies_and_exclusions.json) (`331` logged anomalies/exclusions)

### 3.1 Recording-Level Annotation Summary (`uniform_v1` Policy)

| Recording ID | Audio? | Total Segs | Pos-Dur Segs (`>0s`) | Zero-Dur Segs (`0s`) | Neg-Dur Segs (`<0s`) | Overlapping Pos Segs | Clean Single-Spk Segs | Annotated Spks (All) | Pos-Dur Spks (`RTTM`) | Zero-Dur-Only Spks | Clean Spks | Anomalies Logged |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R01_sara_sharif` | Yes | 7 | 7 | 0 | 0 | 0 | 7 | 5 | 5 | 0 | 5 | 2 |
| `R02_dennis_skinner` | Yes | 15 | 13 | 2 | 0 | 11 | 2 | 5 | 4 | 1 (`Speaker_3`) | 1 (`Speaker_0`) | 3 |
| `R03_disabled_parking` | Yes | 59 | 54 | 5 | 0 | 2 | 52 | 5 | 5 | 0 | 5 | 5 |
| `R04_piers_robinson` | Yes | 33 | 31 | 2 | 0 | 28 | 3 | 3 | 3 | 0 | 3 | 19 |
| `R05_southwest_pilot` | Yes | 61 | 26 | 35 | 0 | 0 | 26 | 4 | 3 | 1 (`Speaker_2`) | 3 | 35 |
| `R06_rich_karen` | Yes | 187 | 131 | 56 | 0 | 4 | 127 | 9 | 9 | 0 | 9 | 56 |
| `R07_bwc_cheap_shot` | Yes | 54 | 39 | 15 | 0 | 2 | 37 | 4 | 3 | 1 (`Speaker_4`) | 3 | 15 |
| `R08_uk_parliament` | Yes | 16 | 11 | 5 | 0 | 0 | 10 | 6 (`5+all`) | 4 (`3+all`) | 2 (`Speaker_3,4`) | 3 | 8 |
| **Audio Subset (`R01–R08`)** | **8** | **432** | **312** | **120** | **0** | **47** | **264** | **41 (`40+all`)** | **36 (`35+all`)** | **5** | **32** | **143** |
| `M01_drunk_karen` | No | 290 | 210 | 78 | 2 | 31 | 0 | 17 | 15 | 2 | 0 | 80 |
| `M02_entitled_restaurant` | No | 251 | 229 | 22 | 0 | 227 | 0 | 10 | 10 | 0 | 0 | 42 |
| `M03_entitled_girl` | No | 119 | 111 | 8 | 0 | 6 | 0 | 11 | 11 | 0 | 0 | 9 |
| `M04_traffic_stop` | No | 270 | 215 | 55 | 0 | 37 | 0 | 9 | 9 | 0 | 0 | 57 |
| **Total (`12` Files)** | **8 / 12** | **1,362** | **1,077** | **283** | **2** | **348** | **264** | **88 (`87+all`)** | **81 (`80+all`)** | **7** | **32** | **331** |

### 3.2 Explicit Accounting of Excluded, Zero-Duration, Malformed & Ambiguous Segments

1. **Zero-Duration (`start == end`) Segments (`283` total; `120` in `R01–R08`)**:
   - Caused by 1-second timestamp quantization in the source `.txt` annotations.
   - **Policy (`uniform_v1`)**: Retained in `annotations_json/<rec_id>.json` (so word-level ASR/cpWER/saWER evaluation never loses transcribed words) and in `stage1_segment_ledger.csv` with flag `ZERO_DURATION_1S_ROUNDING`, and omitted from `annotations_rttm/<rec_id>.rttm` (since acoustic diarization requires $\text{duration} > 0$).
   - **5 Zero-Duration-Only Speakers in `R01–R08`**:
     - `R02_dennis_skinner::Speaker_3` (`1` seg at `[142.0 - 142.0]`, 3 words)
     - `R05_southwest_pilot::Speaker_2` (`3` segs at `103s, 114s, 117s`, 7 words)
     - `R07_bwc_cheap_shot::Speaker_4` (`1` seg in raw `.txt` at `[149.0 - 149.0]`, 2 words; note that under `phase20_legacy_curated`, `[134.0 - 135.0]` is reassigned from `Speaker_2` to `Speaker_4` and `[149.0 - 149.0]` is reassigned to `Speaker_3`)
     - `R08_uk_parliament::Speaker_3` (`1` seg at `[54.0 - 54.0]`, 4 words)
     - `R08_uk_parliament::Speaker_4` (`1` seg at `[96.0 - 96.0]`, 1 word)
2. **Non-Monotonic & Overlapping Segments (`47` positive-duration segments in `R01–R08`)**:
   - `R02_dennis_skinner` (`11` of `13` positive segments overlap; `Speaker_1`, `Speaker_2`, and `Speaker_4` have `0` non-overlapping segments) and `R04_piers_robinson` (`28` of `31` positive segments overlap; `17` non-monotonic start-time regressions).
   - **Policy**: Retained in RTTM for full-overlap diarization scoring (`collar=0.0, skip_overlap=False`), and explicitly flagged (`HAS_GT_OVERLAP_N_PARTNERS`, `usable_for_clean_speaker_eval=False`) so they are excluded from clean single-speaker verification trials in Stage 4 while evaluated separately in the overlap stratum.
3. **Collective Label `Speaker_all` (`R08_uk_parliament`, lines 2, 15, 29)**:
   - Represents chorus/crowd parliamentary reactions (`"Order"`, `"Hear, hear"`, `"Yeah"`).
   - **Policy**: Logged with `COLLECTIVE_SPEAKER_ALL` and excluded from single-speaker verification pairs (`usable_for_clean_speaker_eval=False`).
4. **Unicode En-Dash (`–`, `U+2013`), Single-Digit Hour (`0:02:25`), and Negative Duration (`<0s`)**:
   -Normalized in-memory without altering source `.txt` files (`M04_traffic_stop` line 325 `–`; `M03_entitled_girl` line 86 `0:02:25`; `M01_drunk_karen` lines 532 & 538 negative durations `-42s` and `-58s`).
