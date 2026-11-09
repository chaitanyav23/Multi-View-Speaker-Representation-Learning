# Stage 5B.4-A.1.3.2 RTTM Security & Descriptor Traversal Report

## 1. Context and Problem Statement

Stage 5B.4 enforces strict containment: in production mode, all RTTM prediction files must be written exclusively to canonical subdirectories under `research_stages/stage5b/stage5b_4_fresh_diarization/outputs/{condition_id}/`.

In Stage 5B.4-A.1.3.1, `revalidate_and_write_rttm()` checked file paths and opened files with `O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW`.

An independent code audit identified two remaining vulnerabilities (**Finding 2**):
1. **Parent Directory Replacement Race**: Resolving path strings between `outputs/` and `outputs/{condition_id}/` could be exploited by replacing the parent directory with a symlink between path validation and file opening.
2. **Hard-Link Truncation Attack**: Opening an existing destination with `O_TRUNC` before inspecting its inode metadata would immediately truncate the target file. If an attacker had created a hard link from an existing RTTM target to an external sensitive file (`st_nlink > 1`), opening with `O_TRUNC` would destroy the contents of the external target before any permission or link check could execute.

---

## 2. Low-Level Descriptor Traversal Architecture

To eliminate path-string resolution races, Stage 5B.4-A.1.3.2 replaces string-based directory traversal with POSIX directory file descriptors (`dir_fd`):

### 2.1 Atomic Directory Resolution
1. **Canonical Root Anchoring**:
   The canonical output root is opened with directory-specific, non-following flags:
   ```python
   root_fd = os.open(str(canonical_root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
   ```
2. **Relative Condition Traversal**:
   The condition directory is opened relative to `root_fd` using `dir_fd`:
   ```python
   cond_fd = os.open(condition_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
   ```
   Because filesystem operations operate on the inode referenced by the open descriptor, renaming or replacing parent directory path strings has zero effect on the open file descriptors.

---

## 3. Hard-Link Truncation Defense

To defend against hard-link truncation, `revalidate_and_write_rttm` abandons unconditional `O_TRUNC` opening and adopts a two-phase check-before-truncate sequence:

```
[Target Destination Path]
         │
         ▼
[Phase 1: Exclusive Creation Attempt: O_CREAT | O_EXCL | O_NOFOLLOW]
         │
         ├── Success ──────────────────────────► Proceed to Write
         ▼ File Exists (FileExistsError)
[Phase 2: Non-Truncating Open: O_WRONLY | O_NOFOLLOW (NO O_TRUNC)]
         │
         ▼
[Descriptor Audit via os.fstat(fd)]
         │
         ├─ st.st_nlink > 1 ───────────────────► RAISE OutputSecurityError!
         │                                       (File untouched, zero bytes lost)
         ├─ not S_ISREG(st.st_mode) ───────────► RAISE OutputSecurityError!
         │
         ▼ Verified st_nlink == 1 and S_ISREG
[Safe Truncation: os.ftruncate(fd, 0)]
         │
         ▼
[Write RTTM Content -> flush -> os.fsync(fd)]
```

### 3.2 Defensive Guarantee
If an existing target file has `st_nlink > 1` (e.g., hard-linked to `/etc/passwd` or an external benchmark artifact), the open call succeeds in read-write mode **without modifying the file size**. The descriptor check immediately observes `st_nlink > 1` and raises `OutputSecurityError`. Not a single byte of the external hard-linked file is overwritten or truncated.

---

## 4. Verification & Audit Results

The descriptor and hard-link security enhancements were tested across 10 dedicated security tests in `tests/test_rttm_descriptor_security.py`:
- `test_1_normal_rttm_creation_succeeds`: Verified ordinary atomic file creation.
- `test_2_existing_regular_destination_safely_overwritten`: Verified safe overwrite when `st_nlink == 1`.
- `test_3_final_destination_symlink_rejected`: Destination symlink rejected by `O_NOFOLLOW`.
- `test_4_parent_directory_symlink_rejected`: Parent condition symlink rejected by `O_DIRECTORY | O_NOFOLLOW`.
- `test_5_parent_directory_race_defeated`: Defeated directory replacement races via `dir_fd`.
- `test_6_hardlink_truncation_attack_defeated`: Hardlinked files with `st_nlink > 1` are rejected without truncation.
- `test_7_condition_directory_traversal_rejected`: Traversal in `condition_id` (`../`) rejected.
- `test_8_recording_id_traversal_rejected`: Traversal in `recording_id` rejected.
- `test_9_output_root_substitution_rejected`: Unapproved output roots rejected.
- `test_10_external_sentinel_remains_byte_identical_after_all_attacks`: External sentinel file preserved byte-for-byte across all attack vectors.

All 10 tests passed in 0.05 seconds. Finding 2 is completely closed.
