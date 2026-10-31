# 05 — Dataset Inventory & Corpus Statistics (`Phase D1 & D2`)

**Audit Status:** `VERIFIED`  
**Structured CSV Companions:** [`dataset_inventory.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/dataset_inventory.csv), [`recording_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/recording_statistics.csv), [`speaker_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/speaker_statistics.csv)

---

## 1. Executive Summary of Available Audio & Ground Truth (`VERIFIED`)

A complete inspection of [`raw_audio/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/raw_audio), [`ground_truth/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth), [`ground_truth_json/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_json), [`ground_truth_rttm/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_rttm), and [`reports/dataset_comparison_report.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/reports/dataset_comparison_report.json) establishes:

1. **Physical Audio Files Present (`VERIFIED`):**
   - **8 `.mp3` files** exist in [`raw_audio/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/raw_audio) (all `44,100 Hz`, `2-channel stereo`, MP3 encoded, totaling **`1,893.694 seconds` = `31 min 33.69 s`** of audio).
   - **Zero `.wav` files** exist anywhere in the repository (`*.wav` is ignored in [`.gitignore`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/.gitignore#L2)).
2. **Ground-Truth Transcript Files Present (`VERIFIED`):**
   - **12 `.txt` files** exist in [`ground_truth/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth).
   - **8 of the 12 `.txt` files** have matching `.mp3` audio in `raw_audio/`.
   - **4 of the 12 `.txt` files** have **NO audio file** in the repository (`Drunk Karen Gets a Huge Reality Check`, `Entitled Karen Refuses to Pay Restaurant Bill Then Attacks the Manager`, `When an Entitled Girl Gets Instantly Humbled`, and `Woman's Attitude Makes Traffic Stop 10x Worse`). Historical durations for these 4 missing recordings (`1055.43s`, `857.00s`, `1011.91s`, `1058.09s`, totaling `3,982.43s` = `66 min 22.43 s`) are recorded in [`reports/dataset_comparison_report.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/reports/dataset_comparison_report.json#L1-L120).
3. **Pre-Parsed JSON / RTTM Files (`VERIFIED`):**
   - [`ground_truth_json/bwc.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_json/bwc.json) & [`ground_truth_rttm/bwc.rttm`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_rttm/bwc.rttm): Curated version of `That was a bit of a cheap shot weren’t it mate.txt` (48 JSON segments, 37 positive-duration RTTM segments, 4 speakers).
   - [`ground_truth_json/karen.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_json/karen.json) & [`ground_truth_rttm/karen.rttm`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth_rttm/karen.rttm): Legacy annotation of `karen` (`652.17s` in historical Phase 5–18 runs, 239 JSON segments, 204 positive-duration RTTM segments, 6 speakers `Speaker_1..6`), which was replaced in commit `09428e8` by `Rich Karen Messes Around & Finds Out.txt` (`652.225s`, 187 segments, 9 speakers `Speaker_0..8`).

---

## 2. Complete Recording-Level Inventory (`Phase D1`)

| # | Recording ID (Stem) | Audio Exists? | Format / SR / Ch | Audio Duration (s) | GT Speakers | Total GT Segs | Positive-Dur Segs (`>0s`) | Zero-Dur Segs (`0s`) | Neg-Dur Segs (`<0s`) | Sum Pos Seg Dur (s) | Union Speech Coverage (s) | Dominant Speaker (Share) | GT Words | `step00` Parser Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `Bodycam footage show arrests of Sara Sharif’s alleged killers` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `100.989` | `5` (`Speaker_1..5`) | `7` | `7` | `0` | `0` | `93.0` | `93.0` (92.1%) | `Speaker_2` (`70.97%`) | `163` | **PASS** (merges 5-line block) |
| **2** | `Dennis Skinner kicked out of Commons` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `176.300` | `5` (`Speaker_0..4`) | `15` | `13` | `2` | `0` | `284.0` | `162.0` (91.9%) | `Speaker_1` (`41.90%`) | `309` | **PASS** (has non-monotonic jump) |
| **3** | `Karen Has A Problem With Disabled Parking` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `206.263` | `5` (`Speaker_0..4`) | `59` | `54` | `5` | `0` | `178.0` | `176.0` (85.3%) | `Speaker_1` (`57.30%`) | `658` | **PASS** |
| **4** | `Piers Confronts Tommy Robinson` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `204.591` | `3` (`Speaker_0..2`) | `33` | `31` | `2` | `0` | `284.0` | `186.0` (90.9%) | `Speaker_1` (`49.30%`) | `853` | **CRASHES** (16 Unicode `–` + `0:02:25`) |
| **5** | `Police bodycam shows arrest of Southwest Airlines pilot` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `162.482` | `4` (`Speaker_1..4`) | `61` | `26` | **`35`** (57.4%!) | `0` | `97.0` | `97.0` (59.7%) | `Speaker_1` (`70.10%`) | `384` | **PASS** (drops 35/61 segs in RTTM!) |
| **6** | `Rich Karen Messes Around & Finds Out` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `652.225` | `9` (`Speaker_0..8`) | `187` | `131` | **`56`** (29.9%!) | `0` | `371.0` | `369.0` (56.6%) | `Speaker_6` (`32.08%`) | `1,809` | **PASS** (drops 56/187 segs in RTTM!) |
| **7** | `That was a bit of a cheap shot weren’t it mate` (`bwc`) | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `175.464` | `4` (`Speaker_1..4`) | `54` (`48` curated) | `39` (`37` curated) | `15` (`11` curated) | `0` | `123.0` (`119.0`) | `122.0` (69.5%) | `Speaker_1` (`86.99%`) | `503` (`484`) | **PASS** (inline format) |
| **8** | `UK Parliament Debate` | **YES** (`raw_audio/*.mp3`) | MP3, 44.1kHz, 2ch | `215.380` | `6` (`Speaker_0..4` + `Speaker_all`) | `16` | `11` | `5` | `0` | `195.0` | `195.0` (90.5%) | `Speaker_2` (`49.74%`) | `478` | **PASS** (includes `Speaker_all`) |
| **Subtotal (8 Audio-Available Recordings)** | — | **8 / 8 MP3** | — | **`1,893.694` s (`31.56 min`)** | **`41` local IDs** (`40` excl. `Speaker_all`) | **`432`** | **`312`** | **`120`** (27.8%) | **`0`** | **`1,625.0` s** | **`1,400.0` s (`23.33 min`)** | — | **`5,157`** | **7 PASS / 1 CRASH** |
| **9** | `Drunk Karen Gets a Huge Reality Check` | **NO (`MISSING`)** | — | *(1055.43s hist.)* | `17` (`Speaker_0..16`) | `290` | `210` | `78` | **`2`** | `833.0` | `804.0` | `Speaker_1` (`25.45%`) | `3,677` | **CRASHES** (2 negative durations) |
| **10** | `Entitled Karen Refuses to Pay Restaurant Bill...` | **NO (`MISSING`)** | — | *(857.00s hist.)* | `10` (`Speaker_1..10`) | `251` | `229` | `22` | `0` | **`7,685.0`** (corrupt!) | `334.0` | `Speaker_1` (`47.79%`) | `2,417` | **CORRUPT** (201 duplicate block timestamps) |
| **11** | `When an Entitled Girl Gets Instantly Humbled` | **NO (`MISSING`)** | — | *(1011.91s hist.)* | `11` (`Speaker_0..10`) | `119` | `111` | `8` | `0` | `796.0` | `772.0` | `Speaker_1` (`44.22%`) | `1,877` | **PASS** (but no audio) |
| **12** | `Woman's Attitude Makes Traffic Stop 10x Worse` | **NO (`MISSING`)** | — | *(1058.09s hist.)* | `9` (`Speaker_0..8`) | `270` | `215` | `55` | `0` | `896.0` | `861.0` | `Speaker_0` (`31.36%`) | `4,106` | **PASS** (but no audio) |
| **Total (All 12 GT `.txt` Files)** | — | **8 MP3 / 4 Missing** | — | `1,893.69s` local (`5,876.12s` w/ hist.) | **`88` local IDs** | **`1,362`** | **`1,077`** | **`283`** | **`2`** | — | `4,171.0` s | — | **`17,234`** | — |

---

## 3. Speaker-Level Distribution & Imbalance Analysis (`Phase D2`)

Full per-speaker statistics across all 88 speaker entries are saved in [`research_audit/speaker_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/speaker_statistics.csv). Focusing on the **41 speaker entries across the 8 recordings that actually have audio**:

### 3.1 Critical Speaker Sufficiency Breakdown (`VERIFIED`)
Out of the **41 recording-local speaker IDs** in the 8 audio-available recordings:
1. **1 Pseudo-Speaker (`Speaker_all` in `UK Parliament Debate`):** Represents simultaneous crowd/MPs shouting (`[00:00:10 - 00:00:10]`, `[00:00:25 - 00:00:25]`, `[00:00:36 - 00:00:38]`), not an individual speaker identity.
2. **5 Speakers Have ZERO Positive-Duration Segments (`0.0s` of usable audio!):**
   - Because all timestamps in `ground_truth/*.txt` are rounded to whole seconds (`HH:MM:SS`), sub-second utterances are recorded with `start == end` (`duration = 0.0s`) and **omitted from RTTM files**:
     - `Police bodycam shows arrest of Southwest Airlines pilot`: **`Speaker_3`** (1 segment, `[00:02:02 - 00:02:02]`, `0.0s`) and **`Speaker_4`** (1 segment, `[00:02:41 - 00:02:41]`, `0.0s`). Thus, in positive-duration RTTM intervals, this "4-speaker" recording only contains **2 speakers** (`Speaker_1` and `Speaker_2`)!
     - `Rich Karen Messes Around & Finds Out`: **`Speaker_4`** (1 segment, `0.0s`), **`Speaker_5`** (3 segments, all `0.0s`), and **`Speaker_8`** (3 segments, all `0.0s`). Thus, 3 of the 9 annotated speakers in `Rich Karen` have **zero seconds of positive duration**!
3. **11 Speakers Have Only ONE Positive-Duration Segment (`1` turn `>0s`):**
   - `Bodycam footage show arrests of Sara Sharif’s alleged killers`: `Speaker_3` (`1` seg, `7.0s`), `Speaker_4` (`1` seg, `2.0s`), `Speaker_5` (`1` seg, `3.0s`).
   - `Dennis Skinner kicked out of Commons`: `Speaker_0` (`1` seg, `18.0s`), `Speaker_3` (`1` seg, `51.0s`).
   - `Karen Has A Problem With Disabled Parking`: `Speaker_0` (`1` seg, `14.0s`), `Speaker_4` (`1` seg, `8.0s`).
   - `Rich Karen Messes Around & Finds Out`: `Speaker_3` (`1` pos seg of `2.0s` + 6 zero-dur segs).
   - `UK Parliament Debate`: `Speaker_0` (`1` pos seg, `10.0s`), `Speaker_3` (`1` pos seg, `5.0s`), `Speaker_4` (`1` pos seg, `2.0s`), plus `Speaker_all` (`1` pos seg, `2.0s`).
4. **Only 25 Speakers Have $\ge 2$ Positive-Duration Segments (`viable_for_contrastive_pairs = True`):**
   - And only **21 speakers** have $\ge 2$ segments of length $\ge 1.5\text{s}$ (`viable_for_multiscale_1_5s = True`)!

### 3.2 Severe Dominant-Speaker Skew (`VERIFIED`)
Every BWC recording exhibits extreme speaker dominance (where the primary officer or subject accounts for 50%–87% of total speech duration):
- `That was a bit of a cheap shot weren’t it mate` (`bwc`): `Speaker_1` = `107.0s / 123.0s` (**`86.99%`** of speech duration; `397 / 503` words = **`78.9%`** of raw words; `362 / 401` evaluated words = **`90.27%`**).
- `Bodycam footage show arrests of Sara Sharif’s alleged killers`: `Speaker_2` = `66.0s / 93.0s` (**`70.97%`**).
- `Police bodycam shows arrest of Southwest Airlines pilot`: `Speaker_1` = `68.0s / 97.0s` (**`70.10%`**).
- `Karen Has A Problem With Disabled Parking`: `Speaker_1` = `102.0s / 178.0s` (**`57.30%`**).

### 3.3 Recording-Local vs. Global Speaker IDs (`VERIFIED`)
- Speaker labels (`Speaker_0`, `Speaker_1`, ...) are **recording-local indices** reset in each `.txt` file (`Speaker_1` in `bwc` is a Peterborough police constable; `Speaker_1` in `Southwest Airlines` is a US police officer; `Speaker_1` in `Dennis Skinner` is the Speaker of the UK House of Commons, John Bercow).
- **Important Consequence for Supervised/Contrastive Learning:** Treating `Speaker_1` across different files as the same class is **invalid**; speaker labels must be namespace-prefixed by recording ID (e.g., `bwc::Speaker_1`, `southwest::Speaker_1`), and cross-recording speaker pairs are almost certainly negative pairs (except potentially the two UK Parliament recordings or narrator voiceovers).
