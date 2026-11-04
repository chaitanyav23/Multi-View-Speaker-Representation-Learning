# Stage 5B.2 V4.1: CSV Hash and Cryptographic Identity Report

**Audit Phase**: Stage 5B.2 V4.1 Forensic Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE — 100% BIT-FOR-BIT IDENTITY VERIFIED

---

## 1. Executive Summary

An exhaustive forensic comparison was conducted across the result CSV files produced by Stage 5B.2 V3 (`audit_corrections_v3`), Stage 5B.2 V4 (`audit_corrections_v4`), and the newly generated Stage 5B.2 V4.1 (`audit_corrections_v4_1`).

Every metric CSV generated during V4 execution is **100% byte-identical** to the corresponding file in V3 and V4.1. The underlying experimental data never changed.

---

## 2. Cryptographic Hash Table

| File Name | V3 SHA-256 | V4 SHA-256 | V4.1 SHA-256 | Bitwise Identity |
| :--- | :--- | :--- | :--- | :---: |
| `STAGE5B_2_CORRECTED_RESULTS.csv` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | `b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb` | **MATCH (100%)** |
| `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | `4bbfb9fcb36ec285d81851c8082251c1b0c7876daba64d63112c3e3904edb6c4` | **MATCH (100%)** |
| `STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | `84c79047d175566ec43492b518841d7d1428946d743eb30e48e42a71887e4c7f` | **MATCH (100%)** |
| `STAGE5B_2_CORRECTED_WORD_METRICS.csv` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | `28e37d1535fc3d6a7e8ad63a1e0523c7303b156f25a479ecb459e0b6fe98a691` | **MATCH (100%)** |

---

## 3. Physical File Properties

| File Name | File Size (Bytes) | Total Lines | Total Data Rows |
| :--- | :---: | :---: | :---: |
| `STAGE5B_2_CORRECTED_RESULTS.csv` | 11,197 | 62 | 61 |
| `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv` | 71,392 | 489 | 488 |
| `STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv` | 4,211 | 62 | 61 |
| `STAGE5B_2_CORRECTED_WORD_METRICS.csv` | 2,947 | 37 | 36 |

---

## 4. Key Takeaway

Because `audit_corrections_v4` generated bit-identical CSV files to `audit_corrections_v3`, the discrepancy between V3 and V4 was **not caused by a code regression, data mutation, or rerun discrepancy**. The CSV files are and have always been the single source of ground truth.
