# Stage 5B.4-A.1.2 — Protected Artifacts Integrity Audit Report

**Author:** Senior ML Systems Engineer & Independent Research Auditor  
**Date:** November 2026  
**Repository:** `Multi-View-Speaker-Representation-Learning`  
**Workspace:** `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_2_corrections/`  
**Status:** **100% IMMUTABLE INTEGRITY VERIFIED**

---

## 1. Executive Summary

A core requirement of Stage 5B.4-A.1.2 is the immutable preservation of all historical research stages, models, benchmark predictions, RTTMs, embeddings, and cryptographic manifests.

Prior to creating any file or executing any test in Stage 5B.4-A.1.2, SHA-256 checksums were recorded for all protected milestone manifests. Upon completion of all implementation and testing activities, these checksums were recomputed and verified.

**Result:** All protected baseline files are 100% byte-for-byte identical to their pre-implementation state. Zero historical files have been modified, overwritten, or relocated.

---

## 2. Protected Baseline Manifest Audit Table

| Protected Manifest File | Pre-Implementation SHA-256 | Post-Implementation SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `stage5b_4_fresh_diarization/a1_remediation/a1_1_corrections/manifests/A1_1_MANIFEST.json` | `2b6aeef078cee49ee426afd7d8ce0ba08fd08852597601979f963abacec99241` | `2b6aeef078cee49ee426afd7d8ce0ba08fd08852597601979f963abacec99241` | **MATCH** |
| `stage5b_4_fresh_diarization/a1_remediation/manifests/A1_MANIFEST.json` | `4a80f20d9a598c88a675bace4fe07909f23e1451390f1033c37331b4d68c17f5` | `4a80f20d9a598c88a675bace4fe07909f23e1451390f1033c37331b4d68c17f5` | **MATCH** |
| `stage5b_4_fresh_diarization/STAGE5B_4A_MANIFEST.json` | `5216ff7066d84d440eb3911543f42580e4cae5ba0c3708d5ab128a1f7cf56b43` | `5216ff7066d84d440eb3911543f42580e4cae5ba0c3708d5ab128a1f7cf56b43` | **MATCH** |
| `stage5b/STAGE5B_BASELINE_MANIFEST.json` | `4ad738d7d67f9af9be2d4d9589fb293931847b05fce8c428f13b929791123a6c` | `4ad738d7d67f9af9be2d4d9589fb293931847b05fce8c428f13b929791123a6c` | **MATCH** |
| `stage5b/stage5b_1_vad_optimization/STAGE5B_1_MANIFEST.json` | `3fe9becb9a72a24b99548bdfc7629d36309625abb0f164c047f3d564f214ffe9` | `3fe9becb9a72a24b99548bdfc7629d36309625abb0f164c047f3d564f214ffe9` | **MATCH** |
| `stage5b/stage5b_2_overlap_optimization/STAGE5B_2_MANIFEST.json` | `4ad3e03a6ac161836c0fe596e40977173f2f5f005683114ff10df7faccce7abf` | `4ad3e03a6ac161836c0fe596e40977173f2f5f005683114ff10df7faccce7abf` | **MATCH** |
| `stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_2/V4_2_MANIFEST.json` | `56ba622c83f81fb99b9359af80cdedfd042ced8b905bba09fa728e476d831919` | `56ba622c83f81fb99b9359af80cdedfd042ced8b905bba09fa728e476d831919` | **MATCH** |
| `recording_manifest.json` | `967ff3fb7aeb14543cd6bd5996724813bbd90669f54277daaf52c3736849be0b` | `967ff3fb7aeb14543cd6bd5996724813bbd90669f54277daaf52c3736849be0b` | **MATCH** |

---

## 3. Git Working Tree Status

A `git status` verification confirms:
1. HEAD commit remains `5898101` (`Stage 5B.4-A.1.1 — a1_1_corrections added`).
2. There are **zero** modified tracked files (`git diff --stat` is empty).
3. The only untracked directory present is the isolated A.1.2 workspace:
   `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_2_corrections/`
4. No commit or push operations were conducted.

---

## 4. Model Checkpoint Storage Verification

Physical model checkpoints remain untouched in their designated storage directories:
- **TitaNet-Large (`diar_msdd_telephonic.nemo`):**
  - Path: `research_stages/stage3_dual_encoder/models/titanet_large/diar_msdd_telephonic.nemo`
  - Computed SHA-256: `450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6`
  - File Size: `98,284,800` bytes
  - State: **VERIFIED UNCHANGED**

- **WeSpeaker ResNet293-LM ONNX (`voxceleb_resnet293_LM.onnx`):**
  - Path: `research_stages/stage3_dual_encoder/models/wespeaker/voxceleb_resnet293_LM.onnx`
  - Computed SHA-256: `dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98`
  - File Size: `100,503,506` bytes
  - State: **VERIFIED UNCHANGED**
