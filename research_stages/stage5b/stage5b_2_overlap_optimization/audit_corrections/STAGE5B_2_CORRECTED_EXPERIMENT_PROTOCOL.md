# Stage 5B.2: Corrected Experiment Protocol & Re-evaluation Suite

## 1. Experimental Scope & Objectives

This protocol defines the re-evaluation of Stage 5B.2 under corrected algorithms:
1. Physical support binarization (`binarize_overlap_corrected`) replacing midpoint interval conversion.
2. Track-aware pyannote reference loading (`ann[Segment(s, e), f"track_{i}"] = spk`) preventing silent track overwriting.
3. Genuine dual-encoder prototype affinity attribution (`M2-R1`) with four acoustic ablations.
4. Locally bounded temporal-context attribution (`M1-R1`) with minimum evidence threshold and strict abstention.
5. Locally bounded conversational turn alternation (`M3-R1`) and bidirectional conservative abstention (`M4-R1`).
6. Hungarian-aligned oracle diagnostics (`M5A`, `M5B`, `M5C`).
7. Integrated VAD + Overlap evaluations (`O40-R1`, `O41-R1`, `O42-R1`).

---

## 2. Experimental Configurations

The following configurations are evaluated across all 8 canonical benchmark recordings:

### Baseline Configurations (Single-Label Reference)
- **O00**: C00 Baseline Single-Label (No Overlap, DFN VAD)
- **O01**: C50 Baseline Single-Label (No Overlap, Mono VAD)
- **O02**: C62 Baseline Single-Label (No Overlap, Consensus VAD)

### Overlap Detection Threshold Sweeps (C00 Backbone, Mono Stream, M1-R1)
- **O10-R1**: Threshold $\theta = 0.30$
- **O11-R1**: Threshold $\theta = 0.40$
- **O12-R1**: Threshold $\theta = 0.50$
- **O13-R1**: Threshold $\theta = 0.60$
- **O14-R1**: Threshold $\theta = 0.70$

### Overlap Audio Stream Comparison (C00 Backbone, $\theta = 0.40$, M1-R1)
- **O20-R1**: DFN Audio Stream
- **O21-R1**: Raw 16k Mono Audio Stream

### Attribution Strategy Comparison (C00 Backbone, Mono Stream, $\theta = 0.40$)
- **O30-R1**: Corrected M1 Local Temporal Context (Window 3.0s, Min Evidence 0.20s, Abstain)
- **O31-R1**: Corrected M2 Dual-Encoder Prototype Affinity (Equal Concatenated 448d)
  - *Ablation O31-T*: TitaNet-only (192d)
  - *Ablation O31-W*: WeSpeaker-only (256d)
  - *Ablation O31-NF*: Normalized Cosine Average Fusion
- **O32-R1**: Corrected M3 Local Turn Alternation (Max Gap 5.0s, Min Turn 0.20s, Abstain)
- **O33-R1**: Corrected M4 Conservative Abstention (Bidirectional Agreement within 5.0s, Abstain)
- **O34-R1**: Corrected M5B Predicted Overlap + Hungarian Oracle Attribution
- **O35-R1**: Corrected M5C Full Oracle Overlap + Hungarian Oracle Attribution

### Integrated Upstream Configurations ($\theta = 0.40$)
- **O40-R1**: C50 Mono VAD + Mono Overlap + M1-R1
- **O41-R1**: C62 Consensus VAD + Mono Overlap + M1-R1
- **O42-R1**: C62 Consensus VAD + Mono Overlap + M4-R1 (Conservative Abstention)

---

## 3. Mandatory Evaluation Metrics

For every recording and configuration, the evaluation suite computes:
1. **Diarization Metrics**:
   - Track-aware DER (collar 0.0s)
   - Track-aware DER (collar 0.25s, no overlap exclusion)
   - Missed Detection Rate (%)
   - False Alarm Rate (%)
   - Speaker Confusion Rate (%)
   - Jaccard Error Rate (JER, %)
2. **Overlap Detection Metrics**:
   - Overlap Precision, Recall, F1
   - Ground Truth Overlap Duration (s)
   - Predicted Overlap Duration (s)
   - True Positive, False Positive, False Negative Overlap Duration (s)
3. **Attribution Coverage Metrics**:
   - Secondary Overlap Segments Added
   - Secondary Overlap Duration Added (s)
   - Abstention Rate (%)
4. **Conversational Metrics**:
   - cpWER (%) via WhisperX forced-alignment word tokens
   - saWER (%)
5. **Statistical Aggregations**:
   - Unweighted Macro Mean across 8 recordings
   - Pooled Corpus Aggregations (total error seconds divided by total reference seconds)
   - Paired 10,000-resample bootstrap 95% Confidence Intervals
   - Paired Wilcoxon Signed-Rank tests
