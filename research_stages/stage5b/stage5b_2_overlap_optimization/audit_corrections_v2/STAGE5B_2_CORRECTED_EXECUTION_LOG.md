# Stage 5B.2 Corrected Execution Log (Patch v2)

## 1. Environment & Hardware Specifications
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Execution Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v2/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10.20, Torch 2.6.0, Pyannote Audio 3.3.2)
- **Execution Date**: 2026-11-04

---

## 2. Chronological Record of Actions

1. **Workspace Isolation**:
   Created `audit_corrections_v2/` directory and copied source script:
   `cp .../audit_corrections/scripts/run_stage5b_2_corrected.py .../audit_corrections_v2/scripts/run_stage5b_2_corrected_v2.py`
2. **Pre-Execution Integrity Verification**:
   - Verified 298 frozen Stage 1–5A.1.2 artifacts against `frozen_artifact_hashes_after.csv` (100% bitwise match).
   - Verified 316 original Stage 5B.2 artifacts against `STAGE5B_2_MANIFEST.json` (100% bitwise match).
   - Verified 379 Stage 5B.2 audit_corrections artifacts against `STAGE5B_2_CORRECTIVE_MANIFEST.json` (100% bitwise match).
   - Recorded source script SHA-256: `a5e48223d5a9fec10674aec19932860627be1011c797db1a1527535865c7e893`.
   - Created `STAGE5B_2_CORRECTIVE_V2_HASH_AUDIT.csv` (994 entries).
3. **Unit Test Suite Development & Execution**:
   - Developed `tests/test_stage5b_2_corrections.py` covering all 6 correction requirements.
   - Executed test suite: 21 tests ran, 21 passed (0 failures).
4. **Implementation of Corrected Script**:
   Updated `scripts/run_stage5b_2_corrected_v2.py` incorporating:
   - M1 temporal attribution with proximity tie-breaking and deterministic order.
   - M3/M4 contiguous local turn extraction with gap termination and boundary distances.
   - M2 prototype affinity with overlap purity filtering ($f_{\text{ovl}} \ge 0.40$) and prototype purity ($P_k \ge 0.80$).
   - Overlap timestamp alignment with frame-bin temporal support ($[t_{\text{mid}} - S/2, t_{\text{mid}} + S/2)$).
   - Multi-label Hungarian speaker mapping preserving simultaneous speakers.
   - Strict metric error handling without zero fallbacks.
5. **Execution of Focused Rerun Matrix**:
   - Evaluated 22 configurations across 8 recordings (176 total runs).
   - Runtime: 224.28s. Exit code: 0.
   - Generated per-recording and summary CSV metrics.
   - Executed 5,000-resample cluster bootstrap and Wilcoxon signed-rank tests.
   - Generated 7 analytical diagnostic plots under `STAGE5B_2_CORRECTED_PLOTS/`.
