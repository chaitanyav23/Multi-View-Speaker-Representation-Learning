# Metric Definitions & Evaluation Protocols

This document defines the mathematical formulations, units, averaging conventions, and exact code provenance for every metric reported across the **Multi-View Speaker Representation Learning** research stages and the **ConvoFuse** CLI deliverable.

---

## 1. Provenance Classification of Metrics

To maintain strict scientific transparency, metrics are categorized into three provenance tiers:

1. **Tier 1 — Computed Directly by the ConvoFuse CLI Evaluator ([`convofuse/evaluator.py`](convofuse/evaluator.py))**:
   - Computed post-inference on every CLI run (`<rec_id>__evaluation_report.json` and `CORPUS_SUMMARY.json`).
   - Includes: **Concatenated Minimum-Permutation WER (`cpWER`)**, **Lexical Word Diarization Error Rate (`Lex-WDER`)**, **Timestamp Word Diarization Error Rate (`TS-WDER`, `assigned` and `all_ref`)**, **Oracle Correction Audit (`Beneficial`, `Harmful`, `Neutral`, `Correction Precision`)**, and **Cryptographic & Structural Integrity Metrics** (`word_sequence_intact`, `timestamp_sequence_intact`, `unk_invariance_verified`, `frozen_inputs_sha256_verified`).
2. **Tier 2 — Computed by Audited Stage 2–5B Research Scripts (`research_stages/`)**:
   - Computed and frozen in [`research_stages/stage2_baseline/`](research_stages/stage2_baseline/), [`research_stages/stage4_1/`](research_stages/stage4_1/), [`research_stages/stage5a_1_final_audit/`](research_stages/stage5a_1_final_audit/), [`research_stages/stage5a_1_2_freeze_closure/`](research_stages/stage5a_1_2_freeze_closure/), and [`research_stages/stage5b/`](research_stages/stage5b/).
   - Includes: **Diarization Error Rate (`DER`)** (`c=0.0s, ovl` and `c=0.25s, no-ovl`), **Missed Speech (`Miss`)**, **False Alarm (`FA`)**, **Speaker Confusion (`Conf`)**, **Jaccard Error Rate (`JER`)**, **Speaker Count MAE & Bias**, **Turn-Level Verification `ROC-AUC` & `HTER`**, **Pairwise Error Complementarity & Error Jaccard**, **ASR Word Error Rate (`WER`)**, **Speaker-Attributed WER (`SA-WER`)**, and **Speaker Attribution Accuracy (`Macro-SACC`, `Micro-SACC`)**.
3. **Tier 3 — Inherited Historical Phase20 Standalone Metrics ([`phase20_standalone/`](phase20_standalone/))**:
   - Reported only in **Part C (Appendix)** of [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md) for the historical single-recording `bwc.wav` benchmark.
   - Includes: **VAD Precision / Recall / F1**, **Historical Timeline Accuracy**, and **Historical Word Attribution Error**.

---

## 2. Speaker Verification & Complementarity Metrics (Stage 5A.1 / 5A.1.2)

Evaluated on clean, non-overlapping ground-truth speaker turns of duration $\ge 1.0\text{s}$ (`250` turns yielding `9,489` within-recording pairs across `R01`–`R08`; `6` recordings with both positive and negative pairs).

### 2.1 Within-Recording Turn-Pair Cosine Similarity
For any two clean turns $u, v$ within recording $r$, each turn is represented by the mean of its constituent multiscale subsegment affinities from the normalized or raw affinity matrix $S$:

$$s(u, v) = \frac{1}{|I_u| |I_v|} \sum_{i \in I_u} \sum_{j \in I_v} S_{ij}$$

where $I_u, I_v$ are the subsegment indices overlapping turn $u$ and turn $v$.

### 2.2 Area Under the ROC Curve (`ROC-AUC`)
- **Definition**: Probability that a randomly chosen same-speaker turn pair $(y_{uv}=1)$ receives a higher affinity score $s(u,v)$ than a randomly chosen different-speaker turn pair $(y_{uv}=0)$.
- **Range & Direction**: $[0, 1]$, higher is better ($\uparrow$).
- **Averaging Conventions**:
  - **Macro `ROC-AUC`**: Unweighted arithmetic mean of per-recording `ROC-AUC` across the 6 valid recordings (`R01`, `R02`, `R03`, `R04`, `R06`, `R08`) or 5 strict recordings (excluding `R08`, which has 366 positive and only 12 negative pairs).
  - **Pooled Out-of-Fold (`OOF`) `ROC-AUC`**: Computed by pooling Leave-One-Recording-Out (LORO) calibrated scores across all pairs (`N = 9,488` for 6-Rec; `N = 9,110` for 5-Rec).

### 2.3 Half Total Error Rate (`HTER`) under LORO Calibration
- **Definition**: For each held-out recording $r$, the decision threshold $\theta_{-r}^*$ is calibrated on the remaining recordings to achieve Equal Error Rate ($\text{FAR}(\theta_{-r}^*) \approx \text{FRR}(\theta_{-r}^*)$). On the held-out recording $r$:

$$\text{FAR}_r(\theta_{-r}^*) = \frac{\text{FP}_r}{\text{FP}_r + \text{TN}_r}, \qquad \text{FRR}_r(\theta_{-r}^*) = \frac{\text{FN}_r}{\text{FN}_r + \text{TP}_r}$$

$$\text{HTER}_r = \frac{\text{FAR}_r(\theta_{-r}^*) + \text{FRR}_r(\theta_{-r}^*)}{2} \times 100\%$$

- **Units & Direction**: Percentage (`%`), lower is better ($\downarrow$).

### 2.4 Pairwise Error Complementarity & Error Jaccard Index
Let $E_T$ and $E_W$ denote the sets of verification pairs misclassified by `TitaNet-Large (DFN)` and `WeSpeaker ResNet293-LM (Mono)`, respectively, under LORO threshold calibration:

$$\text{Error Jaccard} = \frac{|E_T \cap E_W|}{|E_T \cup E_W|}, \qquad \text{Complementary Error Fraction} = 1 - \text{Error Jaccard} = \frac{|E_T \triangle E_W|}{|E_T \cup E_W|}$$

---

## 3. Acoustic Speaker Diarization Metrics (Stages 2–5B)

Computed via `pyannote.metrics.diarization` (`GreedyMapper` / Hungarian optimal one-to-one mapping between reference speaker set $\mathcal{R}$ and hypothesis speaker set $\mathcal{H}$).

### 3.1 Diarization Error Rate (`DER`) and Additive Decomposition
- **Formula**:

$$\text{DER} = \frac{T_{\text{Miss}} + T_{\text{FA}} + T_{\text{Conf}}}{T_{\text{Ref}}} \times 100\%$$

where:
- **$T_{\text{Ref}}$**: Total reference speech duration (summed over all active reference speakers at each instant).
- **Missed Speech (`Miss` = $T_{\text{Miss}} / T_{\text{Ref}}$)**: Reference speaker time not covered by any hypothesis speaker (including unrecovered second speakers during simultaneous overlap).
- **False Alarm (`FA` = $T_{\text{FA}} / T_{\text{Ref}}$)**: Hypothesis speaker time where no reference speaker is active (or where hypothesis predicts more simultaneous speakers than reference).
- **Speaker Confusion (`Conf` = $T_{\text{Conf}} / T_{\text{Ref}}$)**: Speech time assigned to the wrong speaker after optimal one-to-one Hungarian mapping $\phi^*: \mathcal{H} \to \mathcal{R}$.
- **Evaluation Regimes**:
  1. **Primary Strict Regime (`c=0.0s, ovl`)**: `collar = 0.0s`, `skip_overlap = False` (evaluates all speech including simultaneous multi-speaker overlap and exact boundaries).
  2. **Secondary Forgiving Regime (`c=0.25s, no-ovl`)**: `collar = 0.25s` ($\pm 125\text{ ms}$ around reference boundaries), `skip_overlap = True`.
- **Averaging Conventions**:
  - **Macro DER**: Arithmetic mean of per-recording DER across `R01`–`R08`.
  - **Duration-Weighted (Pooled) DER**: $\sum_r (T_{\text{Miss},r} + T_{\text{FA},r} + T_{\text{Conf},r}) / \sum_r T_{\text{Ref},r}$.

### 3.2 Jaccard Error Rate (`JER`)
- **Formula**: For each reference speaker $k \in \mathcal{R}$ and its optimally mapped hypothesis speaker $\phi^*(k) \in \mathcal{H}$:

$$\text{JER} = \frac{1}{|\mathcal{R}|} \sum_{k \in \mathcal{R}} \frac{\text{FA}_k + \text{Miss}_k}{\text{Total}_k} \times 100\%$$

where $\text{Total}_k$ is the union duration of reference speaker $k$ and mapped hypothesis speaker $\phi^*(k)$, $\text{FA}_k$ is hypothesis-only duration for $\phi^*(k)$, and $\text{Miss}_k$ is reference-only duration for $k$ (equal weight per reference speaker regardless of talk time).
- **Units & Direction**: Percentage (`%`), lower is better ($\downarrow$).

### 3.3 Speaker Count Estimation Metrics (`Spk Count MAE` & `Bias`)
Let $K_{\text{ref}, r}$ and $K_{\text{pred}, r}$ denote the reference and predicted speaker counts on recording $r$:

$$\text{Spk Count MAE} = \frac{1}{R} \sum_{r=1}^{R} |K_{\text{pred}, r} - K_{\text{ref}, r}|, \qquad \text{Spk Count Bias} = \frac{1}{R} \sum_{r=1}^{R} (K_{\text{pred}, r} - K_{\text{ref}, r})$$

---

## 4. Lexical ASR & Speaker-Attributed Transcription Metrics

### 4.1 Word Error Rate (`WER`)
Computed after standard lexical normalization (lowercasing, punctuation stripping, whitespace normalization) using Levenshtein edit distance:

$$\text{WER} = \frac{S + D + I}{N_{\text{ref}}} \times 100\%$$

where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N_{\text{ref}}$ is total reference words.

### 4.2 Concatenated Minimum-Permutation Word Error Rate (`cpWER`)
- **Provenance**: Directly computed in [`convofuse/evaluator.py`](convofuse/evaluator.py) and [`research_stages/stage6_generative_refinement/evaluator.py`](research_stages/stage6_generative_refinement/evaluator.py).
- **Definition**: For each reference speaker $k \in \mathcal{R}$, all reference words spoken by $k$ are concatenated chronologically into $W^{\text{ref}}_k$. For each hypothesis speaker $h \in \mathcal{H}$ (including unmapped `UNK` streams when present), all hypothesis words assigned to $h$ are concatenated chronologically into $W^{\text{hyp}}_h$. Over all one-to-one permutations/mappings $\pi: \mathcal{H} \to \mathcal{R}$ (solved exactly via the Hungarian algorithm on edit-distance cost matrix $C_{h,k} = \text{EditDistance}(W^{\text{hyp}}_h, W^{\text{ref}}_k)$):

$$\text{cpWER} = \min_{\pi} \frac{\sum_{(h, k) \in \pi} \text{EditDistance}(W^{\text{hyp}}_h, W^{\text{ref}}_k) + \sum_{h \notin \text{dom}(\pi)} |W^{\text{hyp}}_h| + \sum_{k \notin \text{im}(\pi)} |W^{\text{ref}}_k|}{\sum_{k \in \mathcal{R}} |W^{\text{ref}}_k|} \times 100\%$$

- **Interpretation**: Jointly penalizes ASR lexical errors ($S, D, I$) and speaker attribution errors (a word attributed to the wrong speaker counts as a deletion for the true speaker and an insertion for the wrong speaker).
- **Units & Direction**: Percentage (`%`), lower is better ($\downarrow$).

### 4.3 Lexical Word Diarization Error Rate (`Lex-WDER`)
- **Provenance**: Directly computed in [`convofuse/evaluator.py`](convofuse/evaluator.py).
- **Definition**: Aligns the global hypothesis word sequence $W^{\text{hyp}}$ to the global reference word sequence $W^{\text{ref}}$ via Levenshtein word alignment (yielding $M$ matched/substituted word pairs $(w^{\text{hyp}}_m, w^{\text{ref}}_m)$ with speaker labels $(s^{\text{hyp}}_m, s^{\text{ref}}_m)$). Using the optimal Hungarian speaker mapping $\pi^*$:

$$\text{Lex-WDER} = \frac{1}{M} \sum_{m=1}^{M} \mathbb{I}\!\left(\pi^*(s^{\text{hyp}}_m) \neq s^{\text{ref}}_m\right) \times 100\%$$

- **Interpretation**: Measures the fraction of aligned ASR words assigned to the wrong speaker against the lexical ground-truth transcript.
- **Units & Direction**: Percentage (`%`), lower is better ($\downarrow$).

### 4.4 Timestamp Word Diarization Error Rate (`TS-WDER`)
- **Provenance**: Directly computed in [`convofuse/evaluator.py`](convofuse/evaluator.py).
- **Definition**: Assigns each ASR word $w_i$ (`[start_i, end_i]`) a reference speaker label $s^{\text{rttm}}_i$ from the gold reference RTTM using maximum positive interval intersection (or nearest RTTM interval within a `0.5s` midpoint collar). Under the optimal Hungarian mapping $\pi^*$ between hypothesis speakers and reference RTTM speakers:
  - **`TS-WDER (assigned)` (Primary)**: Evaluated over all words $i \in \mathcal{A}_{\text{assigned}}$ that have both a non-`UNK` reference RTTM speaker and a non-`UNK` hypothesis speaker:

$$\text{TS-WDER}_{\text{assigned}} = \frac{1}{|\mathcal{A}_{\text{assigned}}|} \sum_{i \in \mathcal{A}_{\text{assigned}}} \mathbb{I}\!\left(\pi^*(s^{\text{hyp}}_i) \neq s^{\text{rttm}}_i\right) \times 100\%$$

  - **`TS-WDER (all_ref)`**: Treats hypothesis `UNK` on a reference-assigned word as an attribution error.

### 4.5 Speaker-Attributed WER (`SA-WER`) and Speaker Attribution Accuracy (`Macro-SACC` / `Micro-SACC`)
- **Provenance**: Computed in Stage 2–5A.1 downstream evaluation scripts ([`research_stages/stage2_baseline/`](research_stages/stage2_baseline/)).
- **`SA-WER`**: Word error rate computed when reference and hypothesis words must match both lexically and in their mapped speaker label under optimal global alignment.
- **`Micro-SACC`**: Percentage of aligned words across the recording attributed to the correct reference speaker after optimal Hungarian speaker mapping (`100% - WDER`).
- **`Macro-SACC`**: Unweighted arithmetic mean of per-reference-speaker word attribution recall across all reference speakers in the recording.

---

## 5. Generative Refinement Audit & Integrity Metrics (`convofuse/evaluator.py`)

### 5.1 Oracle Correction Audit (`Beneficial`, `Harmful`, `Neutral`, `Correction Precision`)
For each word $i$ where `LG-TPST` accepts a speaker change ($s^{\text{refn}}_i \neq s^{\text{base}}_i$):
- **Beneficial Correction**: $\pi^*(s^{\text{base}}_i) \neq s^{\text{rttm}}_i$ and $\pi^*(s^{\text{refn}}_i) = s^{\text{rttm}}_i$ (fixes a baseline attribution error).
- **Harmful Correction**: $\pi^*(s^{\text{base}}_i) = s^{\text{rttm}}_i$ and $\pi^*(s^{\text{refn}}_i) \neq s^{\text{rttm}}_i$ (introduces an attribution error on a previously correct word).
- **Neutral Correction**: Either word $i$ has no reference RTTM speaker (`UNK`), or neither baseline nor refined label matches $s^{\text{rttm}}_i$ (e.g., shifting between two non-target speakers, or improving lexical turn alignment without changing RTTM overlap).
- **Correction Precision**:

$$\text{Correction Precision} = \frac{N_{\text{Beneficial}}}{N_{\text{Beneficial}} + N_{\text{Harmful}}}$$

### 5.2 Deterministic Integrity & Cryptographic Invariance Metrics
- **`word_sequence_intact` (Boolean)**: `True` iff $[w^{\text{refn}}_1.\text{word}, \dots, w^{\text{refn}}_N.\text{word}] == [w^{\text{base}}_1.\text{word}, \dots, w^{\text{base}}_N.\text{word}]$.
- **`timestamp_sequence_intact` (Boolean)**: `True` iff $[(w^{\text{refn}}_i.\text{start}, w^{\text{refn}}_i.\text{end})]_{i=1}^N == [(w^{\text{base}}_i.\text{start}, w^{\text{base}}_i.\text{end})]_{i=1}^N$.
- **`unk_invariance_verified` (Boolean)**: `True` iff for all $i \in \{1, \dots, N\}$, $(w^{\text{base}}_i.\text{speaker} == \text{"UNK"}) \iff (w^{\text{refn}}_i.\text{speaker} == \text{"UNK"})$.
- **`frozen_inputs_sha256_verified` (Boolean)**: `True` iff the SHA-256 digests of all frozen input files (`asr_words_json`, `rttm_file`, `overlap_npz`) match before and after pipeline execution.

---

## 6. Historical Phase20 Standalone Appendix Metrics (`phase20_standalone`)

Reported only in **Part C (Appendix)** of [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md):
- **VAD Precision / Recall / F1**: Frame-level speech detection precision $\text{TP}/(\text{TP}+\text{FP})$, recall $\text{TP}/(\text{TP}+\text{FN})$, and harmonic mean $2 \cdot \text{Prec} \cdot \text{Rec} / (\text{Prec} + \text{Rec})$.
- **Historical Timeline Accuracy**: Proportion of recording timeline (`10 ms` frames) where both speech activity and mapped speaker identity agree with the 2-speaker historical reference on `bwc.wav`.
