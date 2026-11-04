# Stage 5B.2 V3 Execution Log

## 1. Environment & Hardware Specifications
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Execution Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10.20, Torch 2.6.0, Pyannote Audio 3.3.2)
- **Hardware Acceleration**: CPU-only execution (zero GPU compute used).
- **Git HEAD**: `44b812bac508843646c9ba51fcfa77e0c0b4ed7b` on branch `main`.

---

## 2. Chronological Record of Actions

1. **Workspace Isolation**:
   - Created `audit_corrections_v3/` with subdirectories `scripts/`, `tests/`, `configs/`, `metrics/`, `predictions/`, `plots/`.
   - Initialized `__init__.py` files across module directories.
2. **Pre-Execution Bitwise Integrity Verification**:
   - Computed SHA-256 for all 1,372 baseline and prior audit artifacts:
     * 298 frozen Stage 1–5A.1.2 artifacts
     * 316 original Stage 5B.2 artifacts
     * 379 Stage 5B.2 V1 audit artifacts
     * 1 source script
     * 378 Stage 5B.2 V2 audit artifacts
   - Results: **1,372 files checked, 1,372 verified bitwise (0 mismatches, 0 missing)**.
   - Recorded in `STAGE5B_2_V3_HASH_AUDIT.csv`.
3. **Forensic Timestamp Audit**:
   - Inspected PyanNet SincNet architecture in `pyannote.audio.models.blocks.sincnet`.
   - Determined receptive field $W = 991\text{ samples} = 61.9375\text{ms}$ and frame stride $S = 270\text{ samples} = 16.875\text{ms}$.
   - Verified that cached `frame_times` array represents window midpoints $t_{\text{mid}}(i) = i \times S + W/2$.
   - Executed independent reference comparison using `pyannote.audio.utils.signal.Binarize`:
     * Pyannote `Binarize`: 28 segments, $68.534\text{s}$ total overlap.
     * V2 Frame-Bin: 28 segments, $68.525\text{s}$ total overlap (diff: $-9\text{ms}$).
     * V1 Dilation: 29 segments, $70.139\text{s}$ total overlap (+1,605ms error).
4. **Deterministic Multi-Label Oracle Implementation**:
   - Implemented `attribute_secondary_oracle_multilabel` in `scripts/run_stage5b_2_corrected_v3.py`.
   - Replaced arbitrary `alts[0]` with 3-tier deterministic ranking (local duration, global co-occurrence, lexicographical tie-break).
   - Added guards for single-speaker false alarms and multi-speaker truncation flags.
5. **Direct Production Unit Test Suite**:
   - Created `tests/test_stage5b_2_production.py` importing directly from `scripts/run_stage5b_2_corrected_v3.py`.
   - Fixed early-return tuple length in `compute_multilabel_hungarian_mapping`.
   - Executed test suite: **19/19 unit tests passed in 0.002s**.
6. **Execution of Minimal Rerun Matrix**:
   - Executed `run_stage5b_2_corrected_v3.py --configs minimal` covering 6 configurations:
     * Oracle: `O34-R2` (M5B), `O35-R2` (M5C)
     * Matched: `O00`, `O02`, `O11-R2`, `O42-R2`
   - Completed 48 evaluations in 62.05s.
   - Merged 16 unchanged configurations from frozen V2 results.
   - Emitted 4 comprehensive metric CSVs and 7 diagnostic plots.
7. **Matched Statistical Reconciliation**:
   - Recomputed matched O02 vs O42 metrics from raw CSVs.
   - Executed 5,000-resample cluster bootstrap on $N=8$ recording clusters.
   - Verified win/loss/tie breakdown (1 win, 3 losses, 4 ties).
