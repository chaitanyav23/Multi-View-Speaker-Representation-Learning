# Stage 5B.2 V4.1: Metric Pipeline Trace Report

**Audit Phase**: Stage 5B.2 V4.1 Metric Trace Audit  
**Date**: November 4, 2026  
**Status**: VERIFIED & AUDITED

---

## 1. Overview

This document presents a code-level forensic trace of the diarization metric evaluation pipeline implemented in:
`research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py`

The objective is to establish how RTTMs are located, parsed, aligned with ground truth, evaluated with pyannote.metrics, and aggregated.

---

## 2. Metric Computation Architecture

### 2.1 RTTM Loading and Parsing
Ground truth RTTMs are loaded from the canonical Stage 1 data directory:
`research_stages/stage1_data_prep/rttm/{rec_id}.rttm`

Hypothesis RTTMs are generated directly by the overlap attribution dispatcher and saved under:
`research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/predictions/{config_id}/{rec_id}.rttm`

Each RTTM is parsed into a `pyannote.core.Annotation` instance using standard track-aware loading:
```python
def load_rttm_as_annotation(rttm_path: str, uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                # SPEAKER uri channel tbeg tdur <NA> <NA> speaker <NA> <NA>
                tbeg = float(parts[3])
                tdur = float(parts[4])
                spk = parts[7]
                annotation[Segment(tbeg, tbeg + tdur)] = spk
    return annotation
```

### 2.2 Pyannote Metric Configuration
The evaluation utilizes `pyannote.metrics.diarization.DiarizationErrorRate`:
- **Collar 0 ms**: `collar = 0.0, skip_overlap = False`
- **Collar 250 ms**: `collar = 0.25, skip_overlap = False`

Crucially, `skip_overlap=False` ensures that multi-speaker overlap regions are fully penalized if missed, falsely alarmed, or assigned to the wrong speaker.

```python
der_c0_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
der_c25_metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

der_c0 = der_c0_metric(reference, hypothesis, uem=uem_timeline)
der_c25 = der_c25_metric(reference, hypothesis, uem=uem_timeline)
```

### 2.3 Evaluation Region (UEM)
The evaluation region is defined from `t = 0.0` to the total recording duration `t = duration_sec` obtained from the canonical dataset manifest. This guarantees no speech or non-speech is silently omitted.

### 2.4 Error Component Decomposition
For each recording and configuration, the pipeline decomposes DER into its three additive components:
$$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$

These are extracted directly from the pyannote error accumulation matrix:
- `diar_miss = metric['missed detection'] / total_speech`
- `diar_fa = metric['false alarm'] / total_speech`
- `diar_confusion = metric['confusion'] / total_speech`

### 2.5 Macro and Pooled Aggregation
The pipeline computes two aggregation summaries across the 8 benchmark recordings:
1. **Macro-Average DER**: Unweighted arithmetic mean across the 8 recordings:
   $$\text{Macro DER} = \frac{1}{8} \sum_{i=1}^8 \text{DER}_i$$
2. **Pooled DER**: Speech-duration-weighted total error:
   $$\text{Pooled DER} = \frac{\sum_{i=1}^8 (\text{Miss}_i + \text{FA}_i + \text{Conf}_i)}{\sum_{i=1}^8 \text{Reference Duration}_i}$$

---

## 3. Audit Findings

1. **Strict Collar Enforcement**: Both 0ms and 250ms collars are computed independently.
2. **No Overlap Exemption**: Overlap is evaluated rigorously with `skip_overlap=False`.
3. **Reproducibility**: Feeding the raw RTTMs into standard Pyannote tools yields bitwise identical numbers to `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`.
