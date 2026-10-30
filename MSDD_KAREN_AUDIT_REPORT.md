# MSDD Oracle Audit Report: Karen Dataset

## 1. Runtime Configuration Audit

| Parameter | Value | Source |
|-----------|-------|--------|
| `oracle_vad` | `True` | `run_karen_audit.py` config |
| `oracle_num_speakers` | `False` | `run_karen_audit.py` config |
| `num_speakers` | `None` (Automatic) | `karen_manifest.json` |
| `max_num_speakers` | `10` | `run_karen_audit.py` config |
| `language` | `"en"` | `run_karen_audit.py` asr params |

## 2. MSDD Input Trace

**Target RTTM File:** `experiments/karen_11A_audit/grounding/karen_oracle.rttm`

**RTTM Content (First 20 Lines):**
```text
SPEAKER karen 1 0.183 5.889 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 7.338 2.565 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 11.776 0.472 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 13.109 23.574 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 37.156 11.475 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 49.272 2.481 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 52.090 1.587 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 53.879 8.252 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 62.519 1.266 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 64.072 6.615 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 75.007 0.607 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 77.740 9.265 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 87.477 4.540 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 92.287 14.445 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 107.170 9.957 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 118.207 51.705 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 172.342 2.750 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 178.231 1.671 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 180.442 0.658 <NA> <NA> speaker <NA> <NA>
SPEAKER karen 1 182.230 2.110 <NA> <NA> speaker <NA> <NA>
```

**Audit Observations:**
*   **Information Type:** The RTTM contains **only speech activity regions** (Oracle VAD injection). All speaker labels are dummy values ("speaker"), providing **zero identity information** to the model.
*   **Speaker Discovery:** MSDD is in **Inference Mode**. `num_speakers` has been removed from the manifest, and `oracle_num_speakers` is set to `False`. The model must discover the speaker count automatically up to `max_num_speakers=10`.

## 3. Data Leakage Verification

*   **Ground Truth Isolation:** `ground_truth/karen.txt` is **not referenced** in `run_karen_audit.py` or any pipeline stage (VAD, Grounding, WhisperX, MSDD, Fusion). 
*   **Evaluation Only:** The ground truth is used strictly by `evaluate_karen.py` for post-inference metrics.
*   **No Speaker Count Leak:** The previous hardcoded `num_speakers=4` has been removed to ensure the pipeline is truly "blind" to the expected speaker count.

## 4. Final Classification

The current setup represents:
*   **[YES] Oracle VAD only** (Injected external VAD segments to bypass internal MarbleNet).
*   **[NO] Oracle Speaker Count** (Automatic discovery enabled).
*   **[NO] Oracle Diarization** (No identity labels provided).
