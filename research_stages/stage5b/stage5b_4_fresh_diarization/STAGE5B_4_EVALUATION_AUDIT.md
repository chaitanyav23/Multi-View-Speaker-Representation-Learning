# Stage 5B.4 — Downstream Evaluation & Metrics Protocol Audit

**Audit Status**: Complete  
**Date**: 2026-11-06  
**Auditor**: ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_EVALUATION_AUDIT.md`

---

## 1. Executive Summary & Audit Mandate

Evaluation integrity is paramount in Stage 5B.4. Prior audits in Stage 5B.1 and 5B.2 uncovered subtle but serious evaluation flaws, including:
1. A **cpWER JSON schema defect** where raw lists of segments were passed to downstream evaluation functions expecting dictionary envelopes (`{"segments": [...]}`), silently crashing or yielding invalid error rates.
2. An **unmatched collar comparison** where Stage 2 baseline DER was computed with `collar=0.25s` and overlap excluded, while downstream fusion was evaluated with `collar=0.0s` with overlap included.
3. Confusion between **recording-level macro averages** (unweighted mean across $N=8$ recordings) and **duration-weighted pooled averages** (summing error durations across the entire corpus before dividing by total speech duration).

This audit defines the authoritative evaluation protocol for Stage 5B.4, verifying all scoring harnesses, schema definitions, and statistical aggregation procedures.

---

## 2. Diarization Error Rate (DER) & Component Decomposition

Diarization evaluation must be performed using `pyannote.metrics.diarization.DiarizationErrorRate` and `JaccardErrorRate` against uniform ground-truth reference RTTMs (`uniform_annotations/rttm/`).

### 2.1 Primary & Secondary Evaluation Collars
Stage 5B.4 must report both collar settings side-by-side:
- **DER (Collar 0.0s, Overlap Evaluated)**: The primary strict benchmark. No forgiveness collar is applied; every boundary discrepancy is scored. Because multi-speaker overlap attribution is active in conditions `E01`, `E11`, and `E21`, **overlap evaluation must remain strictly enabled** (`collar=0.0, skip_overlap=False`).
- **DER (Collar 0.25s, Overlap Evaluated)**: The standard NIST/Interspeech benchmark setting. Forgives $\pm 250$ms boundary errors around speech turns to isolate speaker clustering errors from minor acoustic boundary jitter.

### 2.2 Component Error Metrics
For every recording and condition, the three additive components of DER must be logged:
$$\text{DER} = \text{Missed Speech Rate} + \text{False Alarm Rate} + \text{Speaker Confusion Rate}$$
- **Missed Speech (Miss %)**: Reference speech not captured by hypothesis. Directly tracks upstream VAD recall.
- **False Alarm (FA %)**: Hypothesis speech where no reference speech exists. Directly tracks upstream VAD precision and conservative vs aggressive overlap insertion.
- **Speaker Confusion (Conf %)**: Speech detected by both reference and hypothesis but attributed to the wrong speaker cluster. Directly tracks dual-encoder discriminability and NME-SC clustering accuracy.
- **Jaccard Error Rate (JER %)**: Equal-weighted speaker-level overlap metric, unskewed by dominant speakers.

---

## 3. Speaker Count Estimation & MAE

Speaker count error was analyzed in Stage 5B.3. In Stage 5B.4, fresh NME-SC clustering automatically estimates the number of speakers $\hat{K}$ via maximum eigengap:

$$\hat{K} = \arg\max_{k \in [1, K_{\max}-1]} \frac{\lambda_{k+1}}{\lambda_k}$$

Where $\lambda_k$ are the sorted eigenvalues of the normalized Laplacian.

### Tracking Metric:
For each recording $i \in \{1, \dots, 8\}$:
$$\Delta K_i = \hat{K}_i - K_i^*, \quad \text{MAE}(K) = \frac{1}{8} \sum_{i=1}^8 |\hat{K}_i - K_i^*|$$
Where $K_i^*$ is the ground-truth number of speakers.
Under-estimation ($\Delta K < 0$) causes elevated speaker confusion; over-estimation ($\Delta K > 0$) fragments speakers into redundant clusters.

---

## 4. Voice Activity Detection (VAD) Quality Metrics

To isolate upstream segmentation performance from downstream clustering, the speech intervals of each VAD condition (`C00`, `C50`, `C62`) must be scored directly against the unionized reference speech intervals:

1. **VAD Union Reference**: Merged reference intervals where $\ge 1$ speaker is active.
2. **VAD Metrics**:
   - $\text{Precision} = \frac{\text{Speech} \cap \text{Ref Speech}}{\text{Total Hyp Speech}}$
   - $\text{Recall} = \frac{\text{Speech} \cap \text{Ref Speech}}{\text{Total Ref Speech}}$
   - $\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 5. Downstream Conversational cpWER Audit

The ultimate downstream utility of diarization in this project is speaker-attributed transcription.

### 5.1 The JSON Schema Specification
In `phase20_standalone/scripts/fusion_metrics.py`, word attribution maps aligned WhisperX ASR words to diarization hypothesis intervals:

```python
def attributed_words(
    aligned_words: List[Dict[str, Any]], 
    diar_segments: Dict[str, Any] # Must be {"segments": [...]}
) -> List[Dict[str, Any]]: ...
```

> [!CAUTION]
> **Schema Defect Remediation**:
> - Passing a bare Python `list` of segments (e.g., `[{"start": ..., "end": ..., "speaker": ...}]`) causes an immediate `TypeError: list indices must be integers or slices, not str` when the function executes `diar_segments["segments"]`.
> - All Stage 5B.4 attribution and evaluation scripts must strictly wrap segments in the canonical envelope:
>   ```python
>   attribution_envelope = {
>       "segments": [
>           {"start": round(s["start"], 4), "end": round(s["end"], 4), "speaker": s["speaker"]}
>           for s in diar_segments
>       ]
>   }
>   ```

### 5.2 cpWER Scoring
- Concatenated minimum-permutation Word Error Rate (cpWER) is computed using `jiwer` after optimal Hungarian mapping between reference speaker transcriptions and hypothesis speaker transcriptions.
- Reference transcripts: Extracted from `stage1_data_prep/manifests/recording_manifest.json` (`json_uniform`).
- Hypothesis transcripts: Assembled from words attributed to each hypothesis cluster.

---

## 6. Statistical Reporting: Macro vs Pooled Averages

To ensure reproducibility and prevent biased interpretations:

1. **Macro-Average (Primary)**:
   $$\bar{M}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^N M_i$$
   Every recording is weighted equally (1/8), reflecting expected performance across distinct acoustic environments.

2. **Duration-Weighted Pooled Average (Secondary)**:
   $$\bar{M}_{\text{pooled}} = \frac{\sum_{i=1}^N \text{Error Duration}_i}{\sum_{i=1}^N \text{Total Speech Duration}_i}$$
   Longer recordings (e.g., R08 with 583s) contribute proportionally more than shorter recordings (e.g., R07 with 150s).

3. **Statistical Significance Testing**:
   - Non-parametric paired Wilcoxon signed-rank test on matched recording pairs ($n=8$).
   - 10,000-sample clustered bootstrap to construct 95% Confidence Intervals ($\text{CI}_{95}$) for $\Delta\text{DER}$, $\Delta\text{cpWER}$, and $\Delta\text{JER}$.

---

## 7. Audit Verdict on Evaluation

The downstream evaluation protocol is **FULLY SPECIFIED AND VERIFIED**. Incorporating the dictionary schema envelope and reporting both collar settings eliminates the systematic biases identified in earlier audit cycles.
