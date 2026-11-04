# Stage 5B.2 V4.2: Execution-Scope Deviation Forensic Report

**Audit Phase**: Stage 5B.2 V4.2 Scope Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE — SCOPE DEVIATION FORMALLY RECORDED & ASSESSED

---

## 1. Executive Summary

This report documents an execution-scope deviation that occurred during Stage 5B.2 V4.1.

Under the Stage 5B.2 V4.1 audit mandate, execution authorization was strictly conditional:
> *"Run a minimal M2 benchmark rerun only if the prototype-membership correction changes benchmark membership."*

The dataset scan conducted during V4.1 established that **zero** prototype-membership changes occurred across all 1,840 subsegments in the eight benchmark recordings (1,426 prototypes admitted under both naive summation and 1D interval union). Therefore, under the conditional authorization criteria, no benchmark rerun was authorized.

However, the V4.1 execution log reports that a minimal evaluation script (`run_stage5b_2_corrected_v4_1.py --configs minimal --seed 42`) was executed, evaluating six configurations across eight recordings (48 total benchmark evaluations).

This report transparently records this event as a **protocol-scope deviation**, analyzes exactly what operations took place, confirms that no prohibited model inference or retraining occurred, and demonstrates that the deviation introduced zero metric drift.

---

## 2. Forensic Verification of the Deviation

### 2.1 The Scan Report Finding
As recorded in [`V4_1_M2_DATASET_IMPACT_REPORT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/V4_1_M2_DATASET_IMPACT_REPORT.md):
- Total subsegments scanned: 1,840
- Coverage differences: 0
- Purity classification changes: 0
- Admitted prototypes (naive): 1,426
- Admitted prototypes (union): 1,426
- Result: **Prototype membership was 100% identical**.

### 2.2 The V4.1 Execution Log Record
As recorded in [`V4_1_EXECUTION_LOG.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/V4_1_EXECUTION_LOG.md#L39-L43):
- Command: `python run_stage5b_2_corrected_v4_1.py --configs minimal --seed 42`
- Run count: 6 configurations $\times$ 8 recordings = 48 runs.
- Execution time: 62.37 seconds.

Because the condition for rerun (prototype membership alteration) was not met, running the evaluation script constituted an unauthorized execution beyond the strict conditional boundary.

---

## 3. Scope of Executed Operations: Distinction of Categories

To maintain exact scientific precision, the 48 executed runs must be classified accurately across operational categories:

| Operational Category | Authorized in V4.1? | Executed in V4.1? | Forensic Evidence / Verification |
| :--- | :---: | :---: | :--- |
| **Model Retraining / Fine-Tuning** | **NO** | **NO** | No model weights modified; no gradient updates; no PyTorch training loops. |
| **Model Inference (Neural Forward Pass)** | **NO** | **NO** | No TitaNet, WeSpeaker, Silero VAD, or Pyannote neural inference executed; zero GPU utilization. |
| **Embedding Extraction** | **NO** | **NO** | Cached `.npy` / `.npz` feature embeddings were read-only; no new embeddings computed. |
| **VAD / ASR Inference** | **NO** | **NO** | Precomputed VAD posterior masks and precomputed whisper transcriptions loaded from disk. |
| **Downstream Benchmark Evaluation** | **CONDITIONAL** | **YES (DEVIATION)** | Evaluated 6 configurations across 8 recordings using existing cached representations. |
| **Metric Recomputation / Parsing** | **YES** | **YES** | Read hypothesis RTTMs and reference RTTMs via `pyannote.metrics`. |
| **Documentation / Audit Logging** | **YES** | **YES** | Generated hash tables, unit tests, and audit reports. |

### The Six Configurations Executed:
Inspecting [`run_stage5b_2_corrected_v4_1.py#L1629`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py#L1629):
1. `O00`: Baseline Single-Label (C00 VAD, No Overlap).
2. `O02`: Baseline Single-Label (C62 VAD, No Overlap).
3. `O11-R2`: Overlap Sweep $T=0.40$ (Mono Stream, M1-R2 Proximity Tie-Break).
4. `O34-R2`: Oracle Diagnostic: Predicted Overlap + Hungarian Multi-Label Attribution.
5. `O35-R2`: Oracle Diagnostic: Full Oracle Overlap + Hungarian Multi-Label Attribution.
6. `O42-R2`: Integrated Pipeline: C62 Consensus VAD + Conservative Overlap (M4-R2).

---

## 4. Experimental Impact Assessment

Did the 48-run execution deviation alter any experimental conclusions or benchmark numbers?

1. **Output Cryptographic Verification**:
   As proven in [`V4_1_CSV_HASH_AND_IDENTITY_REPORT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/V4_1_CSV_HASH_AND_IDENTITY_REPORT.md), all four generated metric CSVs are **100% byte-for-byte identical** to the pre-existing V3 and V4 files:
   - `STAGE5B_2_CORRECTED_RESULTS.csv`: `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb`
   - `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`: `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4`
2. **Benchmark Integrity**:
   Because prototype admission was identical, the downstream attribution produced identical RTTMs and identical DER scores.
3. **Upstream / Downstream Safety**:
   No frozen artifacts, model files, or upstream stage caches were modified.

---

## 5. Formal Conclusion

The execution of 48 benchmark runs in V4.1 was an unnecessary protocol-scope deviation that exceeded the conditional authorization. However, because it operated exclusively on frozen cached features and produced bit-identical outputs, **it caused zero data mutation, zero metric drift, and zero compromise of scientific integrity**.
