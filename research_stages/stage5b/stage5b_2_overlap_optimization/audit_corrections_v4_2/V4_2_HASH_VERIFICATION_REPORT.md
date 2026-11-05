# Stage 5B.2 V4.2: Hash Verification and Cryptographic Integrity Report

**Audit Phase**: Stage 5B.2 V4.2 Cryptographic Integrity Audit  
**Date**: November 4, 2026 (Updated for V4.2.1 Erratum, November 5, 2026)  
**Auditor**: Independent Forensic Research Engineer  
**Status**: 100% INTEGRITY VERIFIED — ALL BOUNDARIES PRESERVED

---

## 1. Executive Summary

This report establishes the complete cryptographic integrity of the codebase following the creation and erratum reconciliation of the Stage 5B.2 V4.2 deliverables.

All immutable boundaries have been rigorously enforced:
- **Stage 1 through Stage 5A.1.2**: 298/298 frozen artifacts remain bit-for-bit identical to their reference hashes.
- **Stage 5B.1 (VAD Optimization)**: 865 files preserved untouched.
- **Stage 5B.3 (Speaker Counting & Calibration)**: 62 files preserved untouched.
- **Stage 5B.2 Prior Audits**:
  - `audit_corrections_v3`: 125 files intact.
  - `audit_corrections_v4`: 129 files intact.
  - `audit_corrections_v4_1`: 129 files intact.
- **Primary Metric CSVs**: All four authoritative CSVs are bit-identical across V3, V4, and V4.1.
- **Workspace Isolation**: Zero modifications outside `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_2/`.

---

## 2. Frozen Stage 1–5A.1.2 Integrity Audit

- **Reference Manifest**: [`stage5a_1_2_freeze_closure/frozen_artifact_hashes_after.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/frozen_artifact_hashes_after.csv)
- **Total Registered Artifacts**: 298
- **Total Audited**: 298
- **Missing Files**: 0
- **Hash Mismatches**: 0
- **Integrity Compliance**: **100% BIT-IDENTICAL**

### Representative Audit Samples:
| Relative File Path | Registered Reference SHA-256 | Current Workspace SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `research_stages/stage1_data_prep/manifests/recording_manifest.json` | `9b360ce0ee0344d2146f32e604f215dfffe0374e2a14e9f9ae296bf3c72b15aa` | `9b360ce0ee0344d2146f32e604f215dfffe0374e2a14e9f9ae296bf3c72b15aa` | **PASS** |
| `research_stages/stage1_data_prep/manifests/recording_manifest.csv` | `0f074dff312384a2ee6f8eeec9eeaa980070bcfeee11d4d293ffb221087cf9bf` | `0f074dff312384a2ee6f8eeec9eeaa980070bcfeee11d4d293ffb221087cf9bf` | **PASS** |
| `research_stages/stage2_baseline/rttm/R01_sara_sharif_baseline.rttm` | `d121cb07b0bebf54ea3d014eb023e9c8052140bb6f96ba3707fa377f0a1c6e11` | `d121cb07b0bebf54ea3d014eb023e9c8052140bb6f96ba3707fa377f0a1c6e11` | **PASS** |
| `research_stages/stage2_baseline/rttm/R04_piers_robinson_baseline.rttm` | `9cbe49fe2032e2938a9aa0f316bf494a29a0077dd0ec73562e841ba4f74d9e03` | `9cbe49fe2032e2938a9aa0f316bf494a29a0077dd0ec73562e841ba4f74d9e03` | **PASS** |
| `research_stages/stage5a_1_2_freeze_closure/STAGE5A_1_2_AUDIT_REPORT.md` | `a90d97034c563e488185c07b46141c2c366ff2f7fc271dfad6c8d3d92fb0ae33` | `a90d97034c563e488185c07b46141c2c366ff2f7fc271dfad6c8d3d92fb0ae33` | **PASS** |

---

## 3. Four Authoritative Benchmark Metric CSVs

The four benchmark result CSVs remain cryptographically identical across all audit stages:

| Result CSV File | V3 SHA-256 | V4 SHA-256 | V4.1 SHA-256 | Bitwise Identity |
| :--- | :--- | :--- | :--- | :---: |
| `STAGE5B_2_CORRECTED_RESULTS.csv` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | **100% MATCH** |
| `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | **100% MATCH** |
| `STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | **100% MATCH** |
| `STAGE5B_2_CORRECTED_WORD_METRICS.csv` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | **100% MATCH** |

---

## 4. Git Working Tree State

- **Git HEAD Commit**: `ff68fd80b142a12339a7aa6ff75b01a752e26e3f`.
- **Modifications to Prior Stages**: Zero.
- **Git Commit Created**: NONE (execution halted per stop condition).
