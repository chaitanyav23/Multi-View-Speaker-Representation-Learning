# Stage 4.1 — Critical Issue F: Dataset, Cache & Audio Normalization Reconciliation Report

**Audit Status**: `CORRECTED`  
**Supporting Artifacts**:
- [population_flow.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/population_flow.csv) (`11`-step hierarchical population flow from `1,362` raw GT segments to `254` clean turns and `15,994` / `47,982` VAD subsegments)
- [cache_accounting.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/cache_accounting.csv) (`120` rows: exact subsegment and vector counts for every `(recording, condition, scale)` tuple)
- [audio_condition_audit.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/audio_condition_audit.csv) (`24` rows: SHA-256 hashes, durations, RMS dBFS, active-frame dBFS, peak dBFS, and exact commands for every generated WAV file)

---

## 1. Reconciliation of All Apparent Numerical Discrepancies Across Stages 1–4

An independent review noted three apparent numerical discrepancies across the Stage 1–4 reports. Our source-code and artifact audit resolved all three with 100% mathematical precision:

### Discrepancy 1: Clean Non-Overlapping Turns (`264` vs `254` vs `388`) and Turn Pairs (`9,817` vs `4,061 / 5,756`)
- **What is in the actual data and `stage4_master_results.json`**:
  1. Across all 12 ground-truth files (`R01–R08` and `M01–M04`), Stage 1 parsed **`1,362` total annotated segments** (`432` in `R01–R08` with audio; `930` in `M01–M04` without audio).
  2. Of the `432` segments in `R01–R08`, `120` have zero duration (`0.0s`, `start == end` due to 1-second timestamp rounding), leaving **`312` positive-duration RTTM segments**.
  3. Of those `312` positive-duration segments, `47` are multi-speaker overlapping segments (`has_gt_overlap == True`) and `1` is a collective crowd label (`R08::Speaker_all`), leaving **`264` Stage 1 clean single-speaker ground-truth turns** (`usable_for_clean_speaker_eval == True`).
  4. When [run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L289-L320) (`build_non_overlapping_turn_dataset`) maps those `264` clean turns to Scale-4 (`0.5s`) VAD subsegments (requiring turn duration $\ge 0.5\text{s}$ and $\ge 1$ clean Scale-4 VAD subsegment with $\ge 75\%$ temporal overlap inside the turn), exactly **`10` short/unvoiced turns have zero matching VAD subsegments**, leaving **`254` usable clean turns** per audio condition (`762` across the 3 audio conditions).
  5. Those **`254` usable clean turns** generate exactly **`9,817` within-recording pairs** (`2,828` positive, `6,989` negative) and **`22,314` cross-recording pairs** (`0` positive, `22,314` negative), totaling **`32,131` all-pairs**. These exact integers (`254`, `2,828`, `6,989`, `9,817`, `32,131`) are stored in [stage4_master_results.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/metrics/stage4_master_results.json#L223-L250).
- **Where `388` turns (`4,061` pos / `5,756` neg = `9,817` pairs) in `STAGE4_DECISION_GATE_REPORT.md` came from**:
  - In an early draft of `STAGE4_DECISION_GATE_REPORT.md` (before `R01_sara_sharif`'s multi-paragraph blocks were merged in Stage 1), prose text in Section 1.2 cited `388` turns and `4,061 / 5,756` pairs, and was not updated when `run_stage4_complementarity.py` was re-executed on the final Stage 1 ledger (`254` turns, `2,828` positive, `6,989` negative). The authoritative numbers from the actual code and JSON/CSV artifacts are **`264` Stage 1 clean turns $\to$ `254` VAD-covered clean turns $\to$ `9,817` within-recording pairs (`2,828` pos, `6,989` neg)**.

### Discrepancy 2: Stage 3 Cache Subsegment & Vector Counts (`15,994` vs `47,982` vs `95,964` vs `40,386`)
- **What is physically stored in `research_stages/stage3_dual_encoder/cache/`**:
  - As verified in [cache_accounting.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/cache_accounting.csv) and `stage3_cache_validation.csv`, every audio condition (`16k_mono`, `dfn`, `normalized`) uses the **exact same NeMo Path A VAD segmentation boundaries** across the 8 recordings and 5 temporal scales:
    - **Scale 0 (`1.5s` window, `0.75s` shift)**: `1,840` subsegments (`1,090` clean, `349` overlap, `258` partial-boundary, `143` non-speech)
    - **Scale 1 (`1.2s` window, `0.60s` shift)**: `2,297` subsegments (`1,418` clean, `385` overlap, `312` partial-boundary, `182` non-speech)
    - **Scale 2 (`1.0s` window, `0.50s` shift)**: `2,761` subsegments (`1,769` clean, `417` overlap, `346` partial-boundary, `229` non-speech)
    - **Scale 3 (`0.8s` window, `0.40s` shift)**: `3,474` subsegments (`2,286` clean, `496` overlap, `376` partial-boundary, `316` non-speech)
    - **Scale 4 (`0.5s` window, `0.25s` shift)**: `5,622` subsegments (`3,870` strictly clean `not gt_is_overlap`, or `4,234` clean speaker label; `703` `OVERLAP::`, `353` `PARTIAL_BOUNDARY::`, `332` `NON_SPEECH_OR_ROUNDING_GAP`)
  - **Total subsegments per audio condition across all 5 scales**:
    $$1,840 + 2,297 + 2,761 + 3,474 + 5,622 = \mathbf{15,994\text{ subsegments}}$$
  - **Total dual-encoder subsegment pairs across all 3 audio conditions (`16k_mono`, `dfn`, `normalized`)**:
    $$3 \times 15,994 = \mathbf{47,982\text{ subsegment pairs}}$$
  - **Total single-encoder embedding vectors across 2 encoders (`TitaNet-Large` + `ResNet293-LM`) and 3 conditions**:
    $$2 \times 47,982 = \mathbf{95,964\text{ embedding vectors}}$$
  - **Total physical cache files on disk**:
    - `120` TitaNet `.npz` files + `120` ResNet293 `.npz` files = **`240` `.npz` files**
    - **`120` `.json` metadata files** (one per `(recording, condition, scale)` tuple).
  - **Where `40,386` (`3 × 13,462`) came from**: An obsolete pre-fix draft number before all 8 recordings' VAD caches were finalized. The exact count on disk is **`15,994` per condition (`47,982` across 3 conditions; `95,964` single-encoder vectors)**.

### Discrepancy 3: Audio Loudness Normalization Target (`I=-16 LUFS` in Code vs `-20 LUFS` in Report Text)
- **What the code actually executed**:
  - In [run_stage1_prep.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/run_stage1_prep.py#L471), the FFmpeg loudness normalization command executed for all 8 recordings was:
    ```bash
    ffmpeg -y -i <mono_wav> -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 16000 -ac 1 -acodec pcm_s16le <norm_wav>
    ```
  - As verified in [audio_condition_audit.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/audio_condition_audit.csv), the resulting `*_normalized.wav` files have overall RMS levels between **`-15.795 dBFS` (`R06_rich_karen`) and `-17.725 dBFS` (`R01_sara_sharif`)**, and active-frame mean levels between **`-17.882 dBFS` and `-22.323 dBFS`**, with true peaks strictly bounded between **`-0.898 dBFS` and `-1.536 dBFS`**.
- **Why `-20 LUFS` appeared in prose**:
  - The legacy `phase20_standalone` pipeline used `-20 LUFS` (`normalize_audio.py`), and the Stage 1–3 Markdown prose mistakenly referred to the `normalized` condition as `-20 LUFS` instead of the `-16 LUFS` (`loudnorm=I=-16:TP=-1.5:LRA=11`) actually executed in `run_stage1_prep.py` line 471.

---

## 2. Complete 11-Step Population Flow Table

From [population_flow.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/population_flow.csv):

| Step | Population Name | Unit | Recs | Speakers | Count / Condition | Total Across 3 Conditions | Excluded From Prev | Exact Exclusion Reason |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | All parsed GT segments (`R01..R08` + `M01..M04`) | `annotated_segments` | 12 | 88 | 1,362 | 1,362 | 0 | Full non-destructive Stage 1 ledger |
| **2** | Audio-available GT segments (`R01..R08`) | `annotated_segments` | 8 | 41 | 432 | 432 | 930 | `930` segments belong to `M01..M04` (no MP3 in `raw_audio/`) |
| **3** | Positive-duration RTTM reference segments (`R01..R08`) | `rttm_segments` | 8 | 36 | 312 | 312 | 120 | `120` segments have `start == end` (`0.0s` due to 1s rounding) |
| **4** | Clean single-speaker positive-duration GT turns | `clean_gt_turns` | 8 | 32 | 264 | 264 | 48 | `47` overlapping multi-speaker segments + `1` `Speaker_all` segment |
| **5** | Usable clean GT turns w/ $\ge 1$ Scale-4 (`0.5s`) VAD subsegment | `verified_clean_turns` | 8 | 30 | **254** | **762** | 10 | `10` short/unvoiced turns missed by Path A VAD at $\ge 75\%$ coverage |
| **6** | Clean turns of speakers w/ $\ge 2$ clean turns in multi-speaker rec | `loto_clean_turns` | 6 | 18 | **240** (`237` in 5 recs) | **720** | 14 | `12` single-turn speakers + `2` turns of `R02::Speaker_0` (1 clean spk) |
| **7** | All multiscale VAD subsegments across Scales 0..4 (`R01..R08`) | `subsegments` | 8 | 36 | **15,994** | **47,982** | 0 | Full multiscale VAD cache (`95,964` single-encoder vectors) |
| **8** | Scale-4 (`0.5s` window, `0.25s` shift) base subsegments | `scale4_subsegments` | 8 | 36 | **5,622** | **16,866** | 10,372 | Excludes Scales 0..3 (`1,840 + 2,297 + 2,761 + 3,474 = 10,372`) |
| **9** | Scale-4 subsegments matched to a multi-subsegment speaker (Stage 4 leaked set) | `scale4_matched_subsegments` | 8 | 31 | **5,290** (`4,937` in 7 recs) | **15,870** | 332 | `331` `NON_SPEECH_OR_ROUNDING_GAP` + `1` single-subseg speaker clip |
| **10** | Scale-4 clean single-speaker subsegments ($\ge 75\%$ purity, no overlap) | `scale4_clean_subsegments` | 8 | 31 | **4,234** (`3,869` strictly non-ovl) | **12,702** | 1,056 | `703` `OVERLAP::` subsegments + `353` `PARTIAL_BOUNDARY::` subsegments |
| **11** | Scale-4 clean subsegments eligible for Turn-Disjoint LOTO | `scale4_loto_clean_subsegments` | 6 (5 w/ $\ge 2$ spks) | 18 | **3,437** (`2,771` strictly non-ovl in 5 recs) | **10,311** | 797 | `323` in `R02` + `331` in `R04` + `143` from single-turn speakers |
