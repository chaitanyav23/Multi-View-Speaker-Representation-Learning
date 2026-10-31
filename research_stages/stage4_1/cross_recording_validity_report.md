# Stage 4.1 — Critical Issue A: Speaker Identity & Cross-Recording Verification Validity Report

**Audit Status**: `INVALID` (Pooled Cross-Recording Verification) / `CORRECTED` (Restricted to Within-Recording Verification)  
**Supporting Artifacts**:
- [speaker_identity_audit.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/speaker_identity_audit.csv) (`88` speaker rows across all `12` ground-truth files)
- [verification_pair_audit.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/verification_pair_audit.csv) (Turn-level and subsegment-level pair populations by recording and scope)

---

## 1. Executive Conclusion on Cross-Recording Verification

**Genuine cross-recording speaker verification CANNOT be performed on this dataset.**  
In Stage 4 ([run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L648-L654)), the reported `pooled_all_pairs` metrics (`32,131` pairs: `2,828` positive, `29,303` negative) combined **`2,828` strictly within-recording positive pairs** with **`6,989` within-recording negative pairs + `22,314` cross-recording negative pairs** (`0` cross-recording positive pairs).

Because **zero (`0`) verified cross-recording positive speaker pairs exist**, any evaluation that pools cross-recording pairs with within-recording pairs is **scientifically invalid as a speaker verification benchmark**:
1. Recording/session identity is 100% negatively correlated with positive speaker labels across recordings ($P(\text{same speaker} \mid \text{different recording}) = 0$).
2. Adding `22,314` cross-recording pairs to the negative pool conflates **speaker discrimination** with **recording/session/channel discrimination**. An encoder that merely embeds channel/background acoustics into its representation will assign low similarity to cross-recording pairs and appear to achieve lower pooled EER (`22.63%` vs `24.33%` for TitaNet on `16k_mono`), even though it has not been tested on a single cross-recording same-speaker trial.

---

## 2. Ground-Truth Speaker Identity Audit (`R01–R08` and `M01–M04`)

We inspected every raw transcript in `ground_truth/*.txt`, `ground_truth_json/`, `ground_truth_rttm/`, and the Stage 1 parser ([run_stage1_prep.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/run_stage1_prep.py#L313-L353)):

1. **Anonymous Recording-Local Labels Only**:
   - Every ground-truth transcript labels speakers strictly using anonymous recording-local indices (`Speaker_0`, `Speaker_1`, ..., `Speaker_16`, plus `Speaker_all` in `R08_uk_parliament`).
   - No speaker biometric metadata, person names, badge numbers, or cross-recording speaker correspondence tables exist in the repository.
   - Although two filenames reference UK Parliament (`R02_dennis_skinner` — *Dennis Skinner Gets Booted From Parliament*, and `R08_uk_parliament` — * Prime Minister's Questions*), the transcripts contain only `Speaker_0..Speaker_4` and `Speaker_all` without mapping which local speaker index corresponds to the Speaker of the House or any Member of Parliament across the two clips. Guessing or heuristically linking speaker identities across recordings would violate research integrity.
2. **Stage 1 Namespacing**:
   - In [run_stage1_prep.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/run_stage1_prep.py#L344), Stage 1 explicitly namespaces every local label as `global_speaker_id = f"{rec_id}::{spk_id}"`.
   - Across all 12 ground-truth files, there are **`88` unique `rec_id::Speaker_X` labels**:
     - **`41` speaker labels** across the 8 audio-available recordings (`R01–R08`).
     - **`47` speaker labels** across the 4 audio-missing recordings (`M01–M04`).
3. **Breakdown of the 41 Speaker Labels in Audio-Available Recordings (`R01–R08`)**:
   - **`5` speakers** have **zero positive-duration segments** (`0.0s` duration due to 1-second timestamp rounding `start == end`): `R02::Speaker_3`, `R05::Speaker_2`, `R07::Speaker_4`, `R08::Speaker_3`, `R08::Speaker_4`.
   - **`1` label** is a **collective crowd label** (`R08_uk_parliament::Speaker_all`: `[00:00:00 - 00:00:02] "Yeah!"`), which is not an individual speaker.
   - **`3` speakers** appear **exclusively inside multi-speaker overlapping segments** (`0` clean non-overlapping segments): `R02::Speaker_1` (`6` overlapping segments), `R02::Speaker_2` (`1` overlapping segment), `R02::Speaker_4` (`2` overlapping segments).
   - **`12` speakers** have **only `1` clean non-overlapping segment** (cannot form any within-recording positive turn pairs): `R01::Speaker_3`, `R01::Speaker_4`, `R01::Speaker_5`, `R03::Speaker_2`, `R03::Speaker_3`, `R03::Speaker_4`, `R04::Speaker_0`, `R04::Speaker_1`, `R04::Speaker_2`, `R06::Speaker_8`, `R07::Speaker_3`, `R08::Speaker_1`.
   - **`1` speaker** (`R01::Speaker_1`) has `2` clean ground-truth segments (`[00:00:06-00:00:10]` and `[00:01:36-00:01:37]`), but the second 1.0s segment has zero Scale-4 (`0.5s`) VAD subsegments with $\ge 75\%$ coverage, leaving only `1` usable clean turn at Scale 4 (`0` positive turn pairs).
   - **`19` speakers** across 7 recordings (`R01, R02, R03, R05, R06, R07, R08`) have $\ge 2$ usable clean turns at Scale 4 and can form within-recording positive turn pairs.
   - **`18` speakers** across 6 recordings (`R01, R03, R05, R06, R07, R08`) have $\ge 2$ usable clean turns in a recording that also has at least one other clean speaker (making them eligible for Turn-Disjoint Leave-One-Turn-Out evaluation).

---

## 3. Recording-by-Recording Pair Population & Eligibility Audit

From [verification_pair_audit.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/verification_pair_audit.csv), the exact distribution of clean turns and pairwise verification trials across `R01–R08` is:

| Recording | Stage 1 Clean GT Turns | Usable Clean Turns (Scale 4) | Clean Speakers Present | Speakers w/ $\ge 2$ Clean Turns | Within-Rec Pos Pairs | Within-Rec Neg Pairs | Total Within-Rec Pairs | Share of All Within-Rec Pairs | Turn EER Eligible? | Reason if Ineligible / Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R01_sara_sharif` | 7 | 6 | 5 | 1 (`Speaker_2`) | 1 | 14 | 15 | 0.15% | **Yes** (Fragile) | Only `1` positive pair (`15` pairs total) |
| `R02_dennis_skinner` | 2 | 2 | 1 | 1 (`Speaker_0`) | 1 | **0** | 1 | 0.01% | **No** | **`0` negative pairs** (`Speaker_1,2,4` only in overlap) |
| `R03_disabled_parking` | 52 | 51 | 5 | 2 (`Speaker_0,1`) | 556 | 719 | 1,275 | 12.99% | **Yes** | Well-powered (`1,275` pairs) |
| `R04_piers_robinson` | 3 | 3 | 3 | **0** | **0** | 3 | 3 | 0.03% | **No** | **`0` positive pairs** (each of 3 speakers has 1 clean turn) |
| `R05_southwest_pilot` | 26 | 25 | 3 | 3 (`Speaker_1,3,4`) | 134 | 166 | 300 | 3.06% | **Yes** | Well-powered (`300` pairs) |
| `R06_rich_karen` | 127 | 124 | 9 | 8 (`Speaker_0..7`) | 1,834 | 5,792 | 7,626 | **77.68%** | **Yes** | **Dominates 77.68% of all within-rec pairs** (`7,503/9,817` = `76.43%` in Stage 4) |
| `R07_bwc_cheap_shot` | 37 | 34 | 3 | 2 (`Speaker_1,2`) | 286 | 275 | 561 | 5.71% | **Yes** | Well-powered (`561` pairs) |
| `R08_uk_parliament` | 10 | 9 | 2 | 2 (`Speaker_0,2`) | 16 | 20 | 36 | 0.37% | **Yes** | Small sample (`36` pairs; 2 speakers well-separated) |
| **All Within-Recording** | **264** | **254** | **30** | **19** | **2,828** | **6,989** | **9,817** | **100.0%** | **Yes (6/8 recs)** | Valid within-recording verification scope |
| **Cross-Recording Only** | — | — | — | **0** | **0** | **22,314** | **22,314** | — | **INVALID** | **`0` positive pairs exist across recordings** |

*(Note on `R06_rich_karen` pair count: `R06` has `123` usable clean turns in `build_non_overlapping_turn_dataset` (`123 * 122 / 2 = 7,503` pairs, `76.43%` of `9,817`) when filtering strictly on Scale 4 `gt_is_overlap == False`, or `124` turns (`7,626` pairs) when counting by Stage 1 segment overlap flags; in both cases `R06` accounts for $>76\%$ of all within-recording pairs.)*

---

## 4. Methodological Consequences for Stage 4.1 & Stage 5

1. **Retire `pooled_all_pairs` (`32,131` pairs)**: All Stage 4 numbers derived from `pooled_all_pairs` (`32,131` pairs) are **retired as invalid** for speaker verification claims.
2. **Restrict Verification to Within-Recording Pairs (`9,817` pairs)**: All pairwise verification evaluations in Stage 4.1 are restricted strictly to within-recording pairs.
3. **Report Both Pooled OOF and Macro-Averaged 6-Recording Metrics**:
   - Because `R02_dennis_skinner` (`0` negative pairs) and `R04_piers_robinson` (`0` positive pairs) cannot define a within-recording EER/HTER/ROC-AUC at the turn level, only **6 recordings (`R01, R03, R05, R06, R07, R08`)** are eligible for per-recording turn verification metrics, and only **4 recordings (`R03, R05, R06, R07`)** have $>100$ turn pairs.
   - Because `R06_rich_karen` accounts for `76.43%` (`7,503 / 9,817`) of pooled within-recording pairs, any pooled within-recording metric is heavily weighted toward `R06`. Therefore, Stage 4.1 reports both **Pooled Out-of-Fold (OOF)** metrics and **Equal-Recording Macro-Averaged (6-recording)** metrics.
