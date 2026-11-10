# Stage 5B.4-A.1.3.2.2 — Execution Log
**Document ID**: `A1_3_2_2_EXECUTION_LOG.md`  
**Date**: November 10, 2026  
**Status**: COMPLETE / VERIFIED  
**Baseline Git HEAD**: `2c7c185`  
**OS**: Linux 6.8.0-52-generic x86_64  
**Python**: 3.10.20 (`/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`)  
**Pytest**: 9.0.3  

---

## 1. Execution Timeline & Activities

### Step 1: Baseline Inspection & Directory Setup
- Git state verified: `cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning && git status --short` -> clean tracked state, HEAD `2c7c185`.
- Workspace verified: `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/`.
- Copied baseline scripts and provenance manifest from `a1_1_3_2_1_final_security_closure/`.

### Step 2: Investigation of Platform Publication Semantics
- Tested `linkat(tmp_fd, "", cond_fd, "link", AT_EMPTY_PATH)` via libc ctypes call:
  - Result: Returned `-1` with `errno: 2 (ENOENT)`. As documented in Linux `linkat(2)`, `AT_EMPTY_PATH` on an open file descriptor requires `CAP_DAC_READ_SEARCH` capability in standard Linux kernels, which unprivileged user-space processes do not have.
  - Furthermore, `linkat` fails with `EEXIST` if destination path already exists, making it unusable for atomic overwrites.
- Tested `/proc/self/fd/{tmp_fd}` linking:
  - Python's `os.link(f"/proc/self/fd/{tmp_fd}", pub_tmp_name, dst_dir_fd=stage_fd)` invokes `linkat(AT_FDCWD, path, dst_dir_fd, name, AT_SYMLINK_FOLLOW)`.
  - Kernel follows `/proc/self/fd/{tmp_fd}` magic link to the open file description, linking the exact inode into `stage_fd`.
  - Inode matches `tmp_fd` exactly (`st_ino`, `st_dev`).

### Step 3: Architecture Refinement for Publication-Link Safety
- Addressed security review finding: creating an ephemeral publication link in `cond_dir` exposes it to potential directory-entry substitution by an adversary with write access to `cond_dir`.
- Resolved by creating `pub_tmp_name` **inside `stage_fd`** (mode `0700`):
  - Because `stage_fd` has mode `0700`, other users with write access to `cond_dir` cannot enter or manipulate `stage_fd`.
  - Cross-directory atomic rename: `os.rename(pub_tmp_name, filename, src_dir_fd=stage_fd, dst_dir_fd=cond_fd)` moves the verified file into destination in a single atomic syscall.
  - Zero ephemeral entries are ever created in `cond_dir`.
  - Post-publication verification asserts `(st_dev, st_ino)` match `tmp_fd`. If a mismatch is detected, the routine fails closed immediately without performing unsafe pathname-based unlinks.

### Step 4: Authoring Comprehensive Adversarial Test Suite
- Authored `tests/test_rttm_descriptor_bound_publication.py` with 18 focused tests:
  - Test 1: Normal successful RTTM publication (0644 mode, clean staging).
  - Test 2: Existing destination replacement (new inode installed).
  - Test 3: Existing hard-linked destination (external sentinel retains 100% byte-identical content).
  - Test 4: Destination symlink (rejected, canary untouched).
  - Test 5: Temporary directory-entry substitution after verification defeated (descriptor linking publishes verified payload; Part B tests link substitution detection).
  - Test 6: Temporary file replacement while descriptor open (fstat link count detects unlink).
  - Test 7: Exact verified inode match (`st_ino`, `st_dev`).
  - Test 8: Concurrent destination replacement attempts (atomic renameat safety).
  - Test 9: Parent condition-directory substitution attempts (O_DIRECTORY | O_NOFOLLOW blocks symlinks).
  - Test 10: Failure during writing (cleaned up, no partial file).
  - Test 11: Failure during verification (cleaned up).
  - Test 12: Failure during publication (cleaned up).
  - Test 13: Cleanup after every failure (0 leftover files).
  - Test 14: No descriptor leaks (0 FD leaks).
  - Test 15: No `O_TRUNC` or `ftruncate` on existing destination inodes.
  - Test 16: Unsupported publication mechanism fails closed.
  - Test 17: Post-rename substitution of destination detected by fstat.
  - Test 18: Cleanup routine does not perform unsafe pathname unlink on foreign destination entries.

---

## 2. Command Execution Log & Test Outputs

### Command 1: Adversarial Test Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/tests/test_rttm_descriptor_bound_publication.py \
  -v
```
Output:
```text
============================== 18 passed in 0.10s ==============================
```

### Command 2: A.1.3.2.2 Full Workspace Test Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/tests/ \
  -v
```
Output:
```text
============================== 69 passed in 1.98s ==============================
```

### Command 3: Stage 5B.4-A.1.3.2.1 Regression Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_2_1_final_security_closure/tests/ \
  -v
```
Output:
```text
============================== 51 passed in 1.65s ==============================
```

### Command 4: Stage 5B.4-A.1.3.2 Regression Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_2_final_corrections/tests/ \
  -v
```
Output:
```text
============================== 50 passed in 1.62s ==============================
```

### Command 5: Stage 5B.4-A.1.3.1 Regression Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_1_corrections/tests/ \
  -v
```
Output:
```text
============================== 64 passed in 1.72s ==============================
```

### Command 6: Stage 5B.4-A.1.3 Regression Suite Execution
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/pytest \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/tests/ \
  -v
```
Output:
```text
============================== 70 passed in 1.69s ==============================
```

### Command 7: 13 Protected Baseline Manifests Verification
```bash
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python \
  /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_3_2_2_rttm_atomic_publication/scripts/verify_manifest.py \
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

## 3. Test Aggregate Summary

| Test Suite | Total Executed | Passed | Failed | Skipped | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A.1.3.2.2 Adversarial Suite** (`test_rttm_descriptor_bound_publication.py`) | 18 | 18 | 0 | 0 | **100% PASS** |
| **A.1.3.2.2 Full Workspace Suite** (all 6 test modules) | 69 | 69 | 0 | 0 | **100% PASS** |
| **Stage 5B.4-A.1.3.2.1 Regression Suite** | 51 | 51 | 0 | 0 | **100% PASS** |
| **Stage 5B.4-A.1.3.2 Regression Suite** | 50 | 50 | 0 | 0 | **100% PASS** |
| **Stage 5B.4-A.1.3.1 Regression Suite** | 64 | 64 | 0 | 0 | **100% PASS** |
| **Stage 5B.4-A.1.3 Regression Suite** | 70 | 70 | 0 | 0 | **100% PASS** |
| **Grand Total** | **304** | **304** | **0** | **0** | **100% PASS** |
