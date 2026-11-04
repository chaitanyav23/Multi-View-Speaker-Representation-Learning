# Stage 5B.2 V4 — Execution Log

## 1. Environment & Hardware Specifications
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Execution Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10.20, Torch 2.6.0, Pyannote Audio 3.3.2)
- **Hardware Acceleration**: CPU-only execution (zero GPU compute used).
- **Git HEAD**: `1288c1d6e260bf8a575a8e27488fbb03d46022c0` on branch `main`.

---

## 2. Chronological Record of Actions

1. **Workspace Setup & Directory Creation**:
   - Created isolated directory `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4/`.
   - Initialized subdirectories `scripts/`, `tests/`, `configs/`, `metrics/`, `predictions/`, `plots/`.
   - Initialized `__init__.py` files across package modules.

2. **Pre-Execution Bitwise Hash Verification**:
   - Audited 1,496 pre-existing repository artifacts across Stages 1–5A.1.2, 5B.2 original, V1, V2, and V3.
   - Bitwise matches: 1,495 / 1,496. Identified and documented the 1 known documentation update to `MODEL_HANDOFF.md` in V3.
   - Emitted `STAGE5B_2_V4_HASH_AUDIT.csv`.

3. **Forensic Dataset Scan on Ground-Truth RTTMs**:
   - Scanned all 8 reference RTTMs for same-speaker overlaps and duplicates.
   - Identified 3 same-speaker overlaps and 1 duplicate interval in `R02_dennis_skinner.rttm` (total 48.0s reference duration inflation in naive summation).
   - Confirmed 0 same-speaker overlaps in remaining 7 recordings (`R01`, `R03`–`R08`).
   - Verified that secondary speaker attribution in `R02` does not flip between naive and union calculation (0 flips out of 7 overlap events).

4. **Production Code Implementation (`run_stage5b_2_corrected_v4.py`)**:
   - Applied **Audit A**: Implemented deterministic lexicographical tie-breaking for primary-speaker selection:
     `primary_spk = sorted(spk_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]`.
   - Applied **Audit B**: Implemented exact 1D interval union per speaker in `attribute_secondary_oracle_multilabel`, eliminating duration inflation from duplicate/overlapping annotations.
   - Applied **Audit C**: Enhanced `load_dual_encoder_prototypes_pure` with `cache_dir: Optional[Path] = None` to enable direct dependency injection in unit tests with synthetic fixtures.
   - Updated directory paths, summary CSV paths, and added backwards-compatible aliases.

5. **Direct Production Unit Test Suite (`test_stage5b_2_v4_production.py`)**:
   - Implemented direct production imports (zero mocked production code).
   - Covered Audits A, B, C, D and core regression protections.
   - Executed test suite: **21 / 21 unit tests passed in 0.011s**.

6. **Execution of Minimal Rerun Matrix**:
   - Executed `run_stage5b_2_corrected_v4.py --configs minimal --seed 42` covering 6 configurations:
     * `O00` (Baseline single-label C00)
     * `O02` (Baseline single-label C62)
     * `O11-R2` (Sweep T=0.40 Mono M1-R2)
     * `O34-R2` (Oracle Pred Ovl + Hungarian M5B-R2)
     * `O35-R2` (Oracle Full GT Ovl + Hungarian M5C-R2)
     * `O42-R2` (Integrated C62 + Conservative M4-R2)
   - Completed 48 runs (6 configs x 8 recordings) in 62.41s.
   - Merged 16 unchanged configurations from frozen prior audit results.
   - Generated 4 metric CSVs (`RESULTS.csv`, `PER_RECORDING_RESULTS.csv`, `OVERLAP_METRICS.csv`, `WORD_METRICS.csv`) and 7 diagnostic plots.

7. **Statistical Significance & Result Reconciliation**:
   - Conducted 5,000-resample cluster bootstrap across $N=8$ recording clusters.
   - Confirmed numerical results are bitwise identical to V3, confirming complete stability and robust edge-case handling.
