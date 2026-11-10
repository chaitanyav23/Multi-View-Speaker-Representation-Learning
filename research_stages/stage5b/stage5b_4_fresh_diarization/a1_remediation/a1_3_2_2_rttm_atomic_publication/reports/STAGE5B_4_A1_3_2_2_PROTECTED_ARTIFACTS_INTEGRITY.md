# Stage 5B.4-A.1.3.2.2 — Protected Historical Artifacts Integrity Report
**Document ID**: `STAGE5B_4_A1_3_2_2_PROTECTED_ARTIFACTS_INTEGRITY.md`  
**Date**: November 10, 2026  
**Status**: APPROVED / VERIFIED  
**Subject**: Verification of Immutable Preservation for All Prior Research Stages and Historical Manifests

---

## 1. Protected Baselines Verification

To ensure strict reproducibility and prevent regression or tampering across the research repository, Stage 5B.4-A.1.3.2.2 tracks and cryptographically verifies 13 protected baseline manifests covering all prior stages:

| # | Manifest Relative Path | SHA-256 Digest | Status |
| :-: | :--- | :--- | :-: |
| 1 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_2_1_final_security_closure/manifests/A1_3_2_1_MANIFEST.json` | `bae8c43429234121c003c7bcb674cec22912476fbf2ebbc83cc54273e5013b8d` | **UNTOUCHED (100%)** |
| 2 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_2_final_corrections/manifests/A1_3_2_MANIFEST.json` | `eada50487dca051432244cf6df671b37802abd19e1f31f248fe9529db3077e80` | **UNTOUCHED (100%)** |
| 3 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_1_corrections/manifests/A1_3_1_MANIFEST.json` | `3125bb97d69f6e5ed04220627654015fc9ecbb64131e0993df6885331a148853` | **UNTOUCHED (100%)** |
| 4 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/manifests/A1_3_MANIFEST.json` | `752c90744b08f6b2325dab0136d2e7e7f50476b664821515aba9bb2fc0aa813e` | **UNTOUCHED (100%)** |
| 5 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_2_corrections/manifests/A1_2_MANIFEST.json` | `2c5b9807e86d0194ac6b478be497bee579aca138daac566f789114469e0b69fa` | **UNTOUCHED (100%)** |
| 6 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_corrections/manifests/A1_1_MANIFEST.json` | `2b6aeef078cee49ee426afd7d8ce0ba08fd08852597601979f963abacec99241` | **UNTOUCHED (100%)** |
| 7 | `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/manifests/A1_MANIFEST.json` | `4a80f20d9a598c88a675bace4fe07909f23e1451390f1033c37331b4d68c17f5` | **UNTOUCHED (100%)** |
| 8 | `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4A_MANIFEST.json` | `5216ff7066d84d440eb3911543f42580e4cae5ba0c3708d5ab128a1f7cf56b43` | **UNTOUCHED (100%)** |
| 9 | `research_stages/stage5b/STAGE5B_BASELINE_MANIFEST.json` | `4ad738d7d67f9af9be2d4d9589fb293931847b05fce8c428f13b929791123a6c` | **UNTOUCHED (100%)** |
| 10 | `research_stages/stage5b/stage5b_1_vad_optimization/STAGE5B_1_MANIFEST.json` | `3fe9becb9a72a24b99548bdfc7629d36309625abb0f164c047f3d564f214ffe9` | **UNTOUCHED (100%)** |
| 11 | `research_stages/stage5b/stage5b_2_overlap_optimization/STAGE5B_2_MANIFEST.json` | `4ad3e03a6ac161836c0fe596e40977173f2f5f005683114ff10df7faccce7abf` | **UNTOUCHED (100%)** |
| 12 | `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_2/V4_2_MANIFEST.json` | `56ba622c83f81fb99b9359af80cdedfd042ced8b905bba09fa728e476d831919` | **UNTOUCHED (100%)** |
| 13 | `research_stages/stage1_data_prep/manifests/recording_manifest.json` | `967ff3fb7aeb14543cd6bd5996724813bbd90669f54277daaf52c3736849be0b` | **UNTOUCHED (100%)** |

Verification Command:
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python \
  research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/scripts/verify_manifest.py \
  --check-baselines-only
```
Output:
```text
======================================================================
STAGE 5B.4-A.1.3.2.2 MANIFEST & BASELINE INTEGRITY VERIFIER
======================================================================

1. Verifying 13 Protected Baseline Manifests...
[PASS] All 13 protected baseline manifests verified (100% untouched).
```

---

## 2. Git Status and Working Tree Verification

- **Repository HEAD**: `2c7c185` ("Stage 5B.4-A.1.3.2.1 — Final Security and Integration Closure")
- **Tracked Files Changed**: `0` (Zero tracked repository files were modified).
- **Working Tree**: The ONLY untracked directory created is:
  `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/`
- Prior remediation directories (`a1_1_3_2_1_final_security_closure/`, `a1_1_3_2_final_corrections/`, `a1_1_3_1_corrections/`, `a1_1_3_corrections/`, `a1_1_2_corrections/`, `a1_1_corrections/`, `stage5b_4_fresh_diarization/`) remain completely untouched.

---

## 3. Integrity Verdict

All 13 historical baseline manifests and earlier remediation workspaces remain 100% cryptographically identical and preserved.
