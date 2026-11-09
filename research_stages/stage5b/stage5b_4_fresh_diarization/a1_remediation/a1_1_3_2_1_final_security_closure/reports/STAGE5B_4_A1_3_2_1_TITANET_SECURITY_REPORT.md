# Stage 5B.4-A.1.3.2.1 TitaNet Sealed memfd Security Report

## 1. Context and Problem Statement

In Stage 5B.4-A.1.3.2, `load_titanet_model_from_verified_bytes()` staged verified in-memory bytes into an ephemeral filesystem directory (`tempfile.mkdtemp`), wrote the bytes, closed the file descriptor, verified the file through a temporary descriptor, closed it, and passed the string path to NeMo's `restore_from()`.

An independent audit identified the remaining risk (**Finding A**):
1. **Filesystem Exposure**: Between closing the temporary verification descriptor and NeMo calling `tarfile.open()` on the path, the staging file existed in the filesystem. Although isolated in a `0o700` directory with mode `0o600`, passing a filesystem pathname did not theoretically exclude path replacement before NeMo opened it.
2. **Cleanup Overwrite Inaccuracy**: The cleanup block overwrote at most 4096 bytes with zeroes (`min(4096, len(ckpt_bytes))`), whereas documentation implied full overwrite. Furthermore, unlinking a regular file in Linux does not guarantee physical SSD/HDD sector erasure.
3. **Loader Override in Production**: `ControlledModelFactory.build_production_model()` passed `cls.nemo_loader_override` to the loader function without explicitly prohibiting test overrides in production mode.

---

## 2. Technical Investigation of Linux `memfd_create` with File Seals

To eliminate filesystem exposure entirely, we investigated Linux `memfd_create(2)` and file seals (`fcntl(2)` with `F_ADD_SEALS`):

### 2.1 Kernel Mechanics of `memfd_create`
- Introduced in Linux 3.17, `memfd_create` creates an anonymous file that lives exclusively in volatile memory (RAM/swap) backed by the tmpfs/shmem subsystem.
- The file has no name in any filesystem directory tree; it is identified purely by its file descriptor (`int fd`).
- In Linux, the file can be accessed by path via the process-local virtual directory `/proc/self/fd/{fd}`.
- Unlike regular files, `/proc/self/fd/{fd}` cannot be replaced, unlinked, or symlinked away by another process because it is a direct representation of the running process's file descriptor table.

### 2.2 Kernel File Seals
File seals allow an application to place permanent constraints on a `memfd` file descriptor:
- `F_SEAL_WRITE`: Prevents any further writes to the memory file. Any subsequent `write(2)` or `mmap(..., PROT_WRITE)` is rejected by the Linux kernel with `EPERM` (Operation not permitted).
- `F_SEAL_GROW`: Prevents the file size from increasing (e.g., via `ftruncate(2)` or write past end).
- `F_SEAL_SHRINK`: Prevents the file size from decreasing (e.g., via `ftruncate(2)` to zero).
- `F_SEAL_SEAL`: Prevents any further seals from being added or modified.

Once these seals are applied, the memory contents become completely immutable at the kernel level. Even the creating process, child processes, or a process running as root cannot modify or truncate the sealed memfd bytes!

### 2.3 Compatibility with NeMo `SaveRestoreConnector`
We audited NeMo's checkpoint restoration code (`SaveRestoreConnector._unpack_nemo_file` and `_tar_open`):
```python
@staticmethod
@contextmanager
def _tar_open(path2file: str) -> Generator[tarfile.TarFile, None, None]:
    if not os.path.exists(path2file):
        raise FileNotFoundError(f"{path2file} does not exist")
    ...
    tar = tarfile.open(path2file, tar_header)
```
When `path2file` is set to `/proc/self/fd/{memfd}`:
1. `os.path.exists('/proc/self/fd/{memfd}')` returns `True`.
2. `os.path.isfile('/proc/self/fd/{memfd}')` returns `True`.
3. Python `tarfile.open()` opens `/proc/self/fd/{memfd}` and extracts model configuration and checkpoint weights seamlessly.
4. Any attempt to write to `/proc/self/fd/{memfd}` from another thread or process is rejected with `PermissionError: [Errno 1] Operation not permitted`.

---

## 3. Architecture & Implementation

### 3.1 Ingestion Flow in `scripts/model_factory.py`

```
[Verified In-Memory Checkpoint Bytes]
                 │
                 ▼
[memfd = os.memfd_create("nemo_titanet_...", MFD_CLOEXEC | MFD_ALLOW_SEALING)]
                 │
                 ▼ (Write exact verified bytes)
[Apply Kernel Seals: F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE]
                 │
                 ▼ (Query fcntl.F_GET_SEALS: verify all 4 seals active)
[Pre-Handoff Verification: read back bytes through memfd descriptor -> SHA-256 check]
                 │
                 ▼ (Keep memfd descriptor open!)
[NeMo restore_from(restore_path=f"/proc/self/fd/{memfd}")]
                 │
                 ▼ (finally block)
[os.close(memfd): Kernel reclaims anonymous inode and frees RAM pages]
```

### 3.2 Elimination of Arbitrary Model Injection
In `ControlledModelFactory.build_production_model()`:
```python
if cls.nemo_loader_override is not None:
    raise ModelFactoryError(
        "Custom loader overrides are strictly prohibited in production mode; "
        "loader overrides are restricted exclusively to synthetic testing."
    )
```
Any attempt to bypass production checkpoint ingestion by supplying a test loader override is rejected immediately.

### 3.3 Accurate Cleanup Documentation
- **No false secure erasure claims**: In volatile RAM, closing the last descriptor to a sealed anonymous memory object (`os.close(memfd)`) causes the Linux kernel to immediately drop the inode reference count to zero and deallocate the physical pages back to the kernel free-page pool.
- The documentation explicitly acknowledges that sealed memory objects cannot be overwritten in user-space before closing (because `F_SEAL_WRITE` prevents user-space writing), and release is handled securely by operating system kernel memory reclamation.

---

## 4. Test Verification Results

All 10 tests in `tests/test_titanet_memfd_security.py` passed:
1. `test_1_exact_digest_consumed_by_loader`: PASSED
2. `test_2_attempted_mutation_after_sealing_rejected`: PASSED
3. `test_3_attempted_truncation_and_growth_after_sealing_rejected`: PASSED
4. `test_4_proc_path_write_rejected`: PASSED
5. `test_5_original_checkpoint_replacement_cannot_alter_staged_bytes`: PASSED
6. `test_6_loader_exceptions_guarantee_descriptor_cleanup`: PASSED
7. `test_7_production_model_injection_bypass_rejected`: PASSED
8. `test_8_unsupported_platform_fails_closed`: PASSED
9. `test_9_corrupted_digest_rejected_before_sealing`: PASSED
10. `test_10_descriptor_closed_after_successful_load`: PASSED

Finding A is completely closed.
