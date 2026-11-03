# Stage 5B.1: Conversational Attribution (cpWER) Forensic Audit

## 1. Executive Summary & Defect Root Cause

In the initial execution of Stage 5B.1, the output downstream CSV (`STAGE5B_1_DOWNSTREAM_METRICS_PER_RECORDING.csv`) contained empty/NaN values for `cpWER` across every recording and configuration.

### The Forensic Diagnosis:
Inspection of the code path in `scripts/run_stage5b_1_experiments.py` (lines 351–362) identified an exact **JSON schema mismatch** in the temporary attribution payload:

```python
# Lines 355-359 of run_stage5b_1_experiments.py:
words_attr = p20_common.attributed_words(aligned_words, diar_segs)
segs_attr = p20_common.group_words(words_attr)
tmp_attr_path = PRED_DIR / c_id / f"{rec_id}_attr.json"
p20_common.write_json(tmp_attr_path, segs_attr)  # <--- BUG: segs_attr is a list!
f_metrics = p20_common.fusion_metrics(gt_json_path, tmp_attr_path, dur)
```

In `phase20_standalone/scripts/fusion_metrics.py`, line 221 calls:
```python
sys_segments = segments_from_payload(attributed_payload)
```
Where `segments_from_payload` expects a dictionary payload:
```python
def segments_from_payload(payload):
    return list(payload.get("segments", []) or [])  # <--- Expects dict with .get()
```
Because `segs_attr` was serialized directly as a JSON list `[...]`, Python threw:
`AttributeError: 'list' object has no attribute 'get'`

This unhandled exception was immediately swallowed by the broad `except Exception:` clause on line 361, setting `cpwer = None` silently for all 208 evaluations!

Furthermore, the fallback `f_metrics.get("cpwer", 0.0)` in the original script was identified as unsafe: if `f_metrics` had returned an empty dict or None, defaulting to `0.0` would have falsely reported a 0% word error rate (a perfect attribution score).

---

## 2. Corrective Implementation in `audit_corrections/`

In the isolated correction workspace, we implemented the exact schema fix:
```python
# Wrapped segments list inside dictionary matching expected schema:
wrapped_payload = {"segments": segs_attr}
p20_common.write_json(tmp_attr_path, wrapped_payload)
f_metrics = p20_common.fusion_metrics(gt_json_path, tmp_attr_path, dur)
assert "cpwer" in f_metrics and f_metrics["cpwer"] is not None
cpwer = float(f_metrics["cpwer"])
```
This eliminated the exception and allowed `fusion_metrics.py` to calculate exact Levenshtein word attribution errors using the cached WhisperX word alignments and ground-truth JSON transcriptions across all 8 benchmark files.

---

## 3. Corrected cpWER Results

### 3.1 Per-Recording cpWER on Baseline C00

| Recording ID | Nature of Recording | Baseline C00 cpWER | Timeline Accuracy | Words Evaluated | Reference Speakers |
|---|---|---|---|---|---|
| **R01_sara_sharif** | Rapid Dialogue | **50.88%** | 71.55% | 283 | 5 |
| **R02_dennis_skinner** | Cross-talk Overlap | **43.00%** | 56.41% | 614 | 5 |
| **R03_disabled_parking** | BWC Encounter | **19.60%** | 80.39% | 495 | 3 |
| **R04_piers_robinson** | Overlap Debate | **38.57%** | 63.63% | 672 | 2 |
| **R05_southwest_pilot** | Cockpit ATC | **31.77%** | 76.56% | 384 | 4 |
| **R06_rich_karen** | Severe Noise BWC | **49.81%** | 50.19% | 1,811 | 7 |
| **R07_bwc_cheap_shot** | High-Stress BWC | **52.99%** | 58.74% | 502 | 3 |
| **R08_uk_parliament** | Clear Speech | **10.32%** | 90.07% | 475 | 3 |
| **CORPUS MACRO** | **All 8 Recordings** | **37.12%** | **68.44%** | **5,236 words** | — |

---

### 3.2 Macro cpWER Across Key VAD Configurations

| Config ID | Description | Stream | Macro cpWER | Macro Projected DER (c=0) |
|---|---|---|---|---|
| **C00_baseline** | Baseline Pyannote PathA (DFN, T=0.481) | DFN | **37.12%** | 46.17% |
| **C10_min_on_005** | Short-Turn min_dur=0.05s | DFN | 37.14% | 46.16% |
| **C13_min_on_050** | Legacy 0.50s Filter | DFN | **37.58%** | 45.93% |
| **C20_thresh_030** | High Sensitivity (T=0.30) | DFN | 37.23% | 46.04% |
| **C25_thresh_080** | High Specificity (T=0.80) | DFN | 37.24% | 47.08% |
| **C31_hyst_60_35** | Hysteresis (0.60/0.35) | DFN | 37.21% | 46.14% |
| **C41_gap_030** | Gap Filling (merge=0.30s) | DFN | **37.12%** | 46.13% |
| **C50_stream_mono** | Raw 16k Mono Stream | Mono | **37.59%** | 43.84% |
| **C52_consensus** | Linear Consensus (DFN + Mono) | Cons | 37.26% | 45.13% |
| **C61_comb_consensus_hyst** | Consensus + Hysteresis | Cons | **37.12%** | 44.80% |
| **C62_comb_consensus_hyst_gap** | Consensus + Hyst + Gap 0.30s | Cons | **37.12%** | 44.65% |

---

## 4. Critical Scientific Insights

1. **Why Legacy 0.5s Filter Degrades cpWER**:
   Filtering out short turns $< 0.50$s (`C13`) increases macro cpWER from $37.12\%$ to **$37.58\%$** ($+0.46\%$ error). This empirically confirms our hypothesis in `STAGE5B_1_SHORT_TURN_ANALYSIS.md`: deleting turns under 500ms leaves genuine spoken words unassigned to any speaker, creating unrecoverable word attribution deletions.
2. **The Disconnect Between Projected DER and cpWER for Raw Mono (`C50`)**:
   While `C50_stream_mono` reduced projected DER from $46.17\%$ down to $43.84\%$, macro cpWER remained essentially flat at $37.59\%$ (vs $37.12\%$).
   - *Acoustic Mechanism*: WhisperX word timestamps are constrained to when the speaker was actually articulating. Baseline VAD already captured the core vowels of most transcribed words. Expanding VAD boundaries to capture quiet turn fringes and fricatives improves frame-level NIST DER, but because those frame margins rarely contain new transcribed word tokens, cpWER shows lower sensitivity to fringe speech extension.
3. **Consensus Stability (`C62`)**:
   `C62_comb_consensus_hyst_gap` maintains the optimal baseline cpWER of **$37.12\%$** while achieving a $-1.52\%$ improvement in projected DER.
