# 12 — Evaluation Framework Audit (`Phase K`)

**Audit Status:** `VERIFIED`  
**Executed Metric Implementation:** [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L465-L780)  
**Unused Standalone Metric Modules Audited:** [`speaker_mapping.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/speaker_mapping.py#L1-L599), [`cpwer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/cpwer.py#L1-L170), [`sawer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/sawer.py#L1-L145), [`attribution_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/attribution_metrics.py#L1-L231), [`timeline_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/timeline_metrics.py#L1-L158), [`fusion_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/fusion_metrics.py#L1-L244)

---

## 1. Audit of Metrics Actually Executed by `run_phase20.py` (`VERIFIED`)

In `phase20_standalone/`, the step scripts (`step04_vad_eval.py`, `step06a_diar_eval.py`, `step06b_asr_eval.py`, `step08_fusion_eval.py`) call evaluation functions defined inside [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L465-L780):

| Metric | Function & Lines in `common.py` | Library / Algorithm | Exact Formula & Configuration | Verified Value on `bwc` (`bwc_all_metrics.json`) |
|---|---|---|---|---|
| **VAD Precision / Recall / F1 / Accuracy** | `evaluate_vad_segments()` (`L465–516`) | `pyannote.metrics.detection.DetectionPrecisionRecallFMeasure(collar=0.0)` + `DetectionAccuracy(collar=0.0)` | Evaluates binary speech timeline against union of positive-duration GT RTTM intervals (`37` segs, `118.0s` union speech on `bwc`). | Path A: `F1=0.8068` (`P=0.8427, R=0.7738`)<br>Path B: `F1=0.8026` (`P=0.8392, R=0.7690`) |
| **DER (`collar=0.0, skip_overlap=False`)** | `evaluate_diarization()` (`L519–567`) | `pyannote.metrics.diarization.DiarizationErrorRate(collar=0.0, skip_overlap=False)` | $\text{DER} = \frac{T_{\text{FA}} + T_{\text{Miss}} + T_{\text{Conf}}}{T_{\text{Total Ref}}}$ using Hungarian optimal 1-to-1 speaker mapping (`metric.optimal_mapping`). | **`0.4471`** (`44.71%`: `FA=16.58s`, `Miss=20.98s`, `Conf=15.65s`, `Total=119.0s`) |
| **DER (`collar=0.25, skip_overlap=True`)** | `evaluate_diarization()` (`L519–567`) | `DiarizationErrorRate(collar=0.25, skip_overlap=True)` | Ignores $\pm 0.25\text{s}$ around every reference boundary and excludes multi-speaker overlap intervals. | **`0.3241`** (`32.41%`) |
| **JER (`collar=0.0, skip_overlap=False`)** | `evaluate_diarization()` (`L533–539`) | `pyannote.metrics.diarization.JaccardErrorRate(collar=0.0, skip_overlap=False)` | Equal-weighted average over all $K_{\text{ref}}$ reference speakers of $1 - \frac{|R_s \cap H_{\pi(s)}|}{|R_s \cup H_{\pi(s)}|}$ (1-to-1 Hungarian mapping $\pi$). | **`0.8424`** (`84.24%` — exposes 1-speaker collapse!) |
| **ASR WER / MER / WIL / WIP / CER** | `evaluate_asr()` (`L570–599`) | `jiwer.process_words()`, `jiwer.cer()` | Concatenates all GT segment texts and all WhisperX segment texts, normalizes via `normalize_text_for_wer()` (lowercase, strip `[^\w\s']`), computes global edit distance. | `WER=0.3244` (`32.44%`), `CER=0.1976`, `Ref=484`, `Hyp=465` |
| **Word Speaker Accuracy & WAE** | `evaluate_fusion()` (`L677–778`) | Custom greedy co-occurrence mapping in `common.py` | Pairs each GT word with nearest hypothesis word within $\pm 2.0\text{s}$ (preferring exact token match within $\pm 0.75\text{s}$), builds `(pred_spk, gt_spk)` count table, maps greedily (`1-to-1`), computes $\text{Acc} = N_{\text{correct}} / N_{\text{matched}}$. | `Speaker Acc = 0.9027` (`90.27%`), `WAE = 0.0973` (`362 / 401` matched words) |

---

## 2. Critical Methodological Flaws Identified in the Current Evaluation (`VERIFIED`)

### Flaw 1: Dominant-Speaker Illusion in Word Speaker Accuracy (`WAE` / `Speaker Accuracy`)
- On `bwc`, NeMo MSDD predicts **only 1 speaker (`speaker_0`)** for the entire recording (`outputs/msdd/bwc.rttm` has 30 segments, all `speaker_0`).
- In `evaluate_fusion()`, `speaker_0` is mapped to `Speaker_1` (`{"speaker_0": "Speaker_1"}`).
- Because `Speaker_1` (the police officer) speaks `362` of the `401` temporally matched words (`90.27%`), **a degenerate diarizer that predicts 1 speaker for the whole file achieves `90.27%` Speaker Attribution Accuracy (`9.73%` WAE)!**
- Meanwhile, `Speaker_2`, `Speaker_3`, and `Speaker_4` have **`0.0%` recall**, which is completely hidden by `Speaker Accuracy = 90.27%` and `DER_0.25 = 32.41%`, and is only visible in **`JER = 0.8424`** (since JER weights each of the 4 reference speakers equally: $\frac{1}{4}(0.3696 + 1.0 + 1.0 + 1.0) = 0.8424$)!
- **Mandatory Fix:** Always report **Macro-Averaged Per-Speaker Attribution F1 / Recall (`Macro-SACC`)**, **Speaker Count Error ($\Delta K = |\hat{K} - K_{\text{ref}}|$)**, and **JER** alongside DER and micro-averaged WAE!

### Flaw 2: Dead / Unexecuted Metric Modules in `phase20_standalone/scripts/` (`VERIFIED`)
Six well-engineered metric modules exist in `phase20_standalone/scripts/`:
- [`speaker_mapping.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/speaker_mapping.py#L1-L599) (Hungarian 1-to-1 optimal assignment via `scipy.optimize.linear_sum_assignment` vs greedy mapping)
- [`cpwer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/cpwer.py#L1-L170) (Concatenated Minimum-Permutation Word Error Rate, `cpWER`)
- [`sawer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/sawer.py#L1-L145) (Speaker-Attributed Word Error Rate, `saWER`, using Needleman-Wunsch sequence alignment of `(word, speaker)` tuples)
- [`attribution_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/attribution_metrics.py#L1-L231) (Needleman-Wunsch aligned `WAE` and `SACC`)
- [`timeline_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/timeline_metrics.py#L1-L158) (Decomposed `FA`, `Miss`, `Confusion`, `DER`, `JER`)
- [`fusion_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/fusion_metrics.py#L1-L244) (Unified orchestrator calling `cpwer`, `sawer`, `attribution_metrics`, and `timeline_metrics`)

**However, none of these 6 files is called by `run_phase20.py` or `step08_fusion_eval.py` (`VERIFIED`)!** Instead, `step08_fusion_eval.py` calls `common.evaluate_fusion()`, which uses a naive timestamp proximity heuristic (`±2.0s`) instead of Needleman-Wunsch text alignment (`attribution_metrics.py`) and never computes `cpWER` or `saWER`!

### Flaw 3: Greedy vs. Hungarian Speaker Mapping in `common.evaluate_fusion()`
In [`common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L728-L742), `evaluate_fusion()` sorts `(count, pred_speaker, gt_speaker)` triples descending and assigns pairs greedily, whereas [`speaker_mapping.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/speaker_mapping.py#L220-L280) properly solves the global linear sum assignment (Hungarian algorithm). While identical when $\hat{K}=1$, greedy assignment can be suboptimal when $\hat{K} \ge 3$.

### Flaw 4: Integer Ground-Truth Timestamp Bias on `collar=0.0` DER & Zero-Duration Turns
As verified in [`06_GROUND_TRUTH_VALIDATION.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/06_GROUND_TRUTH_VALIDATION.md), `120 / 432` (`27.8%`) of ground-truth segments across the 8 audio recordings have `start == end` (`0.0s` duration) due to 1-second integer timestamp rounding and are dropped from `ground_truth_rttm/*.rttm`.
- Consequently, `DER (collar=0.0)` includes up to `±0.5s` boundary rounding noise on every turn AND penalizes detected sub-second turns as False Alarms.
- Therefore, **both `DER (collar=0.0, skip_overlap=False)` and `DER (collar=0.25, skip_overlap=True)`** must be reported, supplemented by **embedding-level metrics** and **word-level `cpWER` / `saWER`** (which do not depend on RTTM boundary quantization!).

---

## 3. Recommended Three-Tier Evaluation Protocol for Multi-View Speaker Representation Learning (`Phase K`)

To rigorously evaluate multi-view speaker representation learning without being misled by dominant-speaker collapse or 1-second timestamp rounding, every experiment must report metrics across three tiers:

1. **Tier 1 — Intrinsic Embedding Geometry Metrics (Before Clustering/MSDD):**
   - Pairwise Verification **EER (%)** and **minDCF ($p_{\text{target}}=0.05$)** on non-overlapping ground-truth subsegments across all 5 scales (`Scale 0..4`).
   - **Intra- vs. Inter-Speaker Cosine Separation ($\Delta\cos$)** and **Fisher Discriminant Ratio (FDR)**.
   - **Cosine Silhouette Score** and **Davies-Bouldin Index (DBI)** per recording.
2. **Tier 2 — Diarization & Clustering Metrics (Across All 8 Audio Recordings + Per-Recording):**
   - **JER (`collar=0.0`)** — primary metric sensitive to minority-speaker recall and cluster count collapse.
   - **DER (`collar=0.0, skip_overlap=False`)** and **DER (`collar=0.25, skip_overlap=True`)** with decomposed **False Alarm (FA)**, **Missed Speech (Miss)**, and **Speaker Confusion (Conf)** rates.
   - **Speaker Count Accuracy / Mean Absolute Speaker Count Error ($|\hat{K} - K_{\text{ref}}|$)** and **Cluster Purity / NMI (Normalized Mutual Information)** at base scale (`Scale 4`).
3. **Tier 3 — Downstream Speaker-Attributed ASR Metrics (Activating `fusion_metrics.py`):**
   - **Concatenated Minimum-Permutation WER (`cpWER`)** ([`cpwer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/cpwer.py#L1-L170)).
   - **Speaker-Attributed WER (`saWER`)** ([`sawer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/sawer.py#L1-L145)).
   - **Micro WAE / SACC** AND **Macro-Averaged Per-Speaker Attribution F1 (`Macro-SACC`)**.
   - **Paired Bootstrap / Wilcoxon Signed-Rank Significance Tests** across the 8 recordings (and across all 312 positive-duration turns).
