# Stage 5B.4-A.1.3.2.1 RTTM Security & Atomic Publication Report

## 1. Context and Problem Statement

Stage 5B.4 enforces strict path containment: production RTTM prediction files must be written exclusively to canonical subdirectories under `research_stages/stage5b/stage5b_4_fresh_diarization/outputs/{condition_id}/{recording_id}.rttm`.

In Stage 5B.4-A.1.3.2, `revalidate_and_write_rttm()` checked whether the destination file had `st_nlink == 1` and, if so, called `os.ftruncate(dest_fd, 0)` before writing.

An independent audit identified the remaining vulnerability (**Finding B**):
> Checking `st_nlink == 1` followed by calling `os.ftruncate()` introduces a check-to-truncate window. If a concurrent hard link is created to the target inode between the `fstat` check and `ftruncate`, the truncation call would modify the external hard-linked file before any subsequent check could stop it.
> Modifying or truncating an existing destination inode in-place is fundamentally unsafe.

---

## 2. Descriptor-Relative Atomic Publication Architecture

To permanently eliminate in-place truncation and all check-to-truncate race windows, Stage 5B.4-A.1.3.2.1 introduces a **descriptor-relative atomic replacement strategy**.

### 2.1 The Invariant: Never Modify an Existing Destination Inode
Instead of opening and truncating an existing destination file:
1. An existing destination inode is **never opened for writing** and **never truncated**.
2. New RTTM records are written exclusively to a newly created, unlinked temporary file within the same directory.
3. The directory entry is replaced atomically via the POSIX `renameat(2)` system call (`os.rename(..., src_dir_fd=cond_fd, dst_dir_fd=cond_fd)`).

### 2.2 Detailed Step-by-Step Execution Sequence

```
[Canonical Output Root: outputs/]
                 │
                 ▼ (Open dir_fd with O_DIRECTORY | O_NOFOLLOW -> root_fd)
[Condition Subdirectory: outputs/{condition_id}/]
                 │
                 ▼ (Open dir_fd relative to root_fd with O_DIRECTORY | O_NOFOLLOW -> cond_fd)
[Check Destination Symlink Status via os.stat(follow_symlinks=False, dir_fd=cond_fd)]
                 │ (If symlink -> RAISE OutputSecurityError immediately!)
                 ▼
[Create Unique Temp File: .tmp_{rec_id}_{pid}_{token}.rttm]
                 │ (Open via cond_fd with O_CREAT | O_EXCL | O_NOFOLLOW, mode 0644)
                 ▼ (Write full RTTM content -> flush -> fsync)
[Verify Temp File Descriptor: S_ISREG == True, st_nlink == 1, size == len]
                 │
                 ▼
[Atomic Publication via os.rename(src, dst, src_dir_fd=cond_fd, dst_dir_fd=cond_fd)]
                 │
                 ▼ (finally block: if not published, os.unlink(tmp, dir_fd=cond_fd))
[Post-Publication Verification: confirm target is regular file and not symlink]
```

---

## 3. Threat Model and Security Guarantees

### 3.1 Hard-Link Safety
- **Scenario**: An existing RTTM file `rec_01.rttm` is hard-linked to `/home/user/important_data.txt` or an authoritative benchmark metric file.
- **Behavior**: When `revalidate_and_write_rttm` executes:
  1. It writes new content to `.tmp_rec_01_....rttm` (a brand-new inode).
  2. It invokes `renameat()`.
  3. The Linux filesystem atomically updates the directory entry `rec_01.rttm` to point to the new inode.
  4. The link count of the old inode decrements from 2 to 1 (`st_nlink - 1`).
  5. The old inode and its data blocks are **never modified, truncated, or touched**.
  6. `/home/user/important_data.txt` retains 100% of its original bytes, unchanged.

### 3.2 Symlink Defense
- If a symlink exists at the target path, `os.stat(follow_symlinks=False)` detects it and raises `OutputSecurityError`.
- Even in a race where a symlink is placed immediately before `os.rename`, POSIX `renameat` replaces the symlink directory entry itself without following it to the target file.
- The post-publication check verifies that the published file is a regular file and not a symlink.

### 3.3 Guaranteed Temporary File Cleanup
A `try ... finally` construct monitors the `published` flag:
- If an exception occurs during writing, descriptor verification, or publication, `published` remains `False`.
- The `finally` block unlinks `.tmp_...` via `os.unlink(tmp_filename, dir_fd=cond_fd)`.
- No lingering temporary files are left in condition directories on success or failure.

---

## 4. Test Verification Results

All 10 tests in `tests/test_rttm_atomic_security.py` passed:
1. `test_1_existing_regular_destination_safely_replaced`: PASSED
2. `test_2_existing_hardlinked_destination_replaced_without_modifying_sentinel`: PASSED
3. `test_3_concurrent_hardlink_creation_cannot_cause_truncation`: PASSED
4. `test_4_existing_destination_symlink_not_followed`: PASSED
5. `test_5_parent_directory_substitution_does_not_redirect_write`: PASSED
6. `test_6_temporary_files_removed_after_successful_write`: PASSED
7. `test_7_temporary_files_removed_after_failed_write`: PASSED
8. `test_8_final_rttm_appears_only_at_canonical_destination`: PASSED
9. `test_9_no_trunc_used_on_existing_destination_inode`: PASSED
10. `test_10_external_sentinel_remains_byte_identical_after_all_attacks`: PASSED

Finding B is completely closed.
