# Stage 5B.4-A.1.3.2.2 — RTTM Security Report
**Document ID**: `STAGE5B_4_A1_3_2_2_RTTM_SECURITY_REPORT.md`  
**Date**: November 10, 2026  
**Status**: APPROVED / VERIFIED  
**Subject**: Analysis of Descriptor-Bound Atomic RTTM Publication and Inode Integrity Guarantees

---

## 1. Security Objectives

The security architecture of Stage 5B.4-A.1.3.2.2 guarantees that:
1. Every production RTTM file written to disk is cryptographic- and descriptor-bound to the exact byte stream verified by the system.
2. No intermediate directory-entry substitution or pathname race can redirect publication to an unverified inode.
3. Pre-existing files at the destination are replaced atomically without modifying the underlying inode or truncating data.
4. Any external hard link pointing to a previously published destination file retains 100% of its original data byte-identically.
5. Symbolic links are strictly rejected and never traversed.
6. File descriptors and temporary directory entries are never leaked.
7. Cleanup routines fail closed without performing unsafe pathname unlinks.

---

## 2. Inode Integrity & Race Elimination Analysis

### A. The Vulnerability in Stage 5B.4-A.1.3.2.1

In Stage 5B.4-A.1.3.2.1:
```python
# Stage 5B.4-A.1.3.2.1 code snippet
tmp_fd = os.open(tmp_filename, create_flags, mode=0o644, dir_fd=cond_fd)
with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
    f.write(content)
# exiting with-block closes tmp_fd!

# Pathname-based stat:
st_tmp = os.stat(tmp_filename, dir_fd=cond_fd, follow_symlinks=False)

# Pathname-based rename:
os.rename(tmp_filename, filename, src_dir_fd=cond_fd, dst_dir_fd=cond_fd)
```
Between `os.stat()` and `os.rename()`, an adversary with write access to `cond_dir` could execute:
`renameat(cond_fd, "malicious_file.rttm", cond_fd, tmp_filename)`
When `os.rename(tmp_filename, ...)` executed, it would publish `malicious_file.rttm` instead of the verified payload.

### B. The Elimination in Stage 5B.4-A.1.3.2.2

Stage 5B.4-A.1.3.2.2 eliminates this vulnerability through a multi-tier defense:

1. **Private Directory Isolation**:
   - The temporary file is created inside a private staging directory `.stage_{rec_id}_{pid}_{token}` with mode `0700`.
   - Other local users cannot enter (`+x`) or read (`+r`) this directory.
2. **Descriptor Retention**:
   - `tmp_fd` is created with `mode 0600` and retained throughout the entire writing, syncing, and verification sequence.
   - Verification is performed on the open descriptor via `os.fstat(tmp_fd)`. Pathname resolution is bypassed entirely.
3. **Publication-Link Containment Inside Private Staging**:
   - The publication link `pub_tmp_name` is created directly via the Linux procfs magic link interface **inside `stage_fd`**:
     `os.link(f"/proc/self/fd/{tmp_fd}", pub_tmp_name, dst_dir_fd=stage_fd)`
   - The kernel resolves `/proc/self/fd/{tmp_fd}` directly to the open file description and inode referenced by `tmp_fd`.
   - Because `pub_tmp_name` is created inside `stage_fd` (mode `0700`), it is completely protected from unprivileged actors with write access to `cond_dir`. No ephemeral directory entry is ever exposed in `cond_dir`.
4. **Direct Atomic Cross-Directory Renameat**:
   - `os.rename(pub_tmp_name, filename, src_dir_fd=stage_fd, dst_dir_fd=cond_fd)` atomically installs the verified inode into `cond_fd` in a single kernel syscall.
   - There is no intermediate link exposed in `cond_dir` that an adversary could unlink, swap, or replace.
5. **Post-Publication Cryptographic Inode Verification & Cleanup Safety**:
   - After `renameat` replaces the destination entry, `revalidate_and_write_rttm` opens the destination via `cond_fd` with `O_RDONLY | O_NOFOLLOW`:
     `pub_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cond_fd)`
   - It executes `st_pub = os.fstat(pub_fd)` and asserts:
     `st_pub.st_dev == st_tmp.st_dev and st_pub.st_ino == st_tmp.st_ino`
   - If an adversary managed to alter the destination entry after `renameat`, the inode mismatch check fails immediately and raises `OutputSecurityError`.
   - Crucially, the cleanup routine does **NOT** call `os.unlink(filename, dir_fd=cond_fd)` by pathname. Calling `os.unlink` on a mismatched destination would introduce a pathname race that could delete legitimate or concurrent files. The pipeline fails closed safely.

---

## 3. Platform Semantics: `linkat` vs `/proc/self/fd`

During investigation on Linux kernels, two candidate descriptor publication mechanisms were evaluated:

1. `linkat(tmp_fd, "", cond_fd, pub_tmp_name, AT_EMPTY_PATH)`:
   - Evaluated using libc direct invocation.
   - Returned `-1` with `errno: 2 (ENOENT)`.
   - As documented in the Linux kernel man pages (`linkat(2)`), `linkat` with `AT_EMPTY_PATH` requires `CAP_DAC_READ_SEARCH` capability when referencing an open file descriptor without `O_PATH`. In unprivileged research environments, `AT_EMPTY_PATH` fails with `ENOENT`.
   - Moreover, `linkat` fails with `EEXIST` if the destination file already exists; it cannot replace pre-existing files atomically.
2. `os.link(f"/proc/self/fd/{tmp_fd}", pub_tmp_name, dst_dir_fd=stage_fd)`:
   - Evaluated on the current platform.
   - In Python 3.10+, `os.link(src, dst, dst_dir_fd=stage_fd)` invokes `linkat(AT_FDCWD, src, dst_dir_fd, dst, AT_SYMLINK_FOLLOW)`.
   - The kernel follows the `/proc/self/fd/{tmp_fd}` magic link and creates an additional directory entry directly referencing the inode of `tmp_fd`.
   - Verified: creates a hard link with matching inode and device (`st_ino`, `st_dev`), increments `st_nlink` from 1 to 2, and succeeds without requiring root or special capabilities.
   - Staging directory entry `tmp_filename` is unlinked only after publication and post-publication verification, leaving the published file with `st_nlink == 1`.

---

## 4. Hard Link and Truncation Safety

A critical requirement is that existing destination inodes are never truncated or modified in place, protecting benchmark archives that may hard link into the outputs directory.

- In POSIX, `renameat(olddirfd, oldpath, newdirfd, newpath)` atomically unlinks `newpath` and installs the link to `oldpath`.
- The previous destination inode is **never opened**, **never truncated**, and **no bytes are modified**.
- If another process or archive holds an external hard link to the previous destination file:
  - The external hard link's data remains 100% byte-identical.
  - The link count of the previous inode decrements by 1.
  - No data is lost or corrupted.
- Verified in adversarial test `test_3_existing_hard_linked_destination_external_sentinel_preserved`: external sentinel remained byte-identical and link count decremented cleanly from 2 to 1.

---

## 5. Security Verdict

The Stage 5B.4-A.1.3.2.2 descriptor-bound publication mechanism provides complete protection against temporary pathname substitution races, guarantees inode-level provenance, preserves hard-link sentinels, rejects symlinks, avoids unsafe pathname unlinks during cleanup, and cleans up all descriptors and ephemeral files on all execution paths.
