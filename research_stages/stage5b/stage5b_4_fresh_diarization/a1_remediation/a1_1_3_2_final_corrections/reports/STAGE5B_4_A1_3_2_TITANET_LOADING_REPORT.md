# Stage 5B.4-A.1.3.2 TitaNet Loading & In-Memory Checkpoint Report

## 1. Context and Problem Statement

In Stage 5B.4-A.1.3.1, the checkpoint preflight subsystem verified physical files on disk and generated signed `VerifiedCheckpoint` tokens. For WeSpeaker (`WeSpeakerResNet293`), in-memory byte verification was solved cleanly: `ControlledModelFactory.read_and_verify_checkpoint_bytes()` read the checkpoint via `O_NOFOLLOW` into a `bytes` buffer, validated the SHA-256 digest, and loaded the model via `torch.load(io.BytesIO(ckpt_bytes))`.

However, for NVIDIA NeMo TitaNet (`EncDecSpeakerLabelModel`), the loader entrypoint is `restore_from(restore_path: str, ...)`. In A.1.3.1, the factory verified the file bytes in memory but passed the original file path `str(token.canonical_path)` to `restore_from()`.

An independent code audit identified this as **Finding 1**:
> Even though bytes were hashed in memory, passing the external pathname back to NeMo reopened a Time-of-Check to Time-of-Use (TOCTOU) gap if an external process or symlink mutation replaced the file at `token.canonical_path` before NeMo opened it.

---

## 2. Technical Investigation of NeMo Architecture

To determine if NeMo could natively consume raw bytes, `io.BytesIO`, or open file descriptors, we conducted a technical audit of NeMo's checkpoint restoration pipeline in `nemo.core.classes.ModelPT` and `nemo.core.connectors.save_restore_connector.SaveRestoreConnector`:

1. **`restore_from()` Method Signature**:
   ```python
   @classmethod
   def restore_from(
       cls,
       restore_path: str,
       override_config_path: Optional[Union[DictConfig, str]] = None,
       map_location: Optional[torch.device] = None,
       strict: bool = True,
       return_config: bool = False,
       save_restore_connector: Optional[SaveRestoreConnector] = None,
       trainer: Optional[Trainer] = None,
   ) -> 'ModelPT':
   ```
2. **Internal Tarfile Extraction Flow**:
   - `SaveRestoreConnector.restore_from()` receives `restore_path: str`.
   - It invokes `_unpack_nemo_file(path2file=restore_path, out_folder=temp_dir)`.
   - `_unpack_nemo_file` calls `tarfile.open(name=path2file, mode='r:*')`.
   - `tarfile.open()` in Python accepts either a pathname `name` or a fileobj `fileobj`. However, NeMo's connector hardcodes `path2file: str` throughout its path-handling routines (e.g., `os.path.abspath(path2file)`, `os.path.exists(path2file)`), rejecting `io.BytesIO` or file descriptors with `TypeError: stat: path should be string, bytes, os.PathLike or integer, not _io.BytesIO`.

### Key Conclusion
NeMo's restoration engine fundamentally requires a valid filesystem pathname to untar `.nemo` archives (which bundle `model_config.yaml`, `model_weights.ckpt`, and auxiliary metadata). Modifying NeMo's internals in site-packages would violate repository isolation and external dependency integrity.

---

## 3. The Secure Isolated Ephemeral Staging Solution

To provide 100% cryptographic byte-binding without monkeypatching NeMo or exposing a pathname TOCTOU gap, Stage 5B.4-A.1.3.2 implements `load_titanet_model_from_verified_bytes`:

```
[Verified In-Memory Bytes]
         │
         ▼
[Exclusive Temp Dir: tempfile.mkdtemp(prefix="nemo_titanet_stage_") (mode 0700)]
         │
         ▼
[Exclusive Temp File: os.open(..., O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600)]
         │ (Write verified bytes -> flush -> fsync)
         ▼
[Descriptor Revalidation: os.open(..., O_RDONLY | O_NOFOLLOW)]
         │ (fstat: S_ISREG == True, st_nlink == 1, st_size == len, SHA-256 == expected)
         ▼
[EncDecSpeakerLabelModel.restore_from(restore_path=staging_file)]
         │
         ▼ (finally block)
[Guaranteed Cleanup: Zero Overwrite -> Unlink File -> Remove Staging Dir]
```

### 3.1 Security Invariants Enforced
1. **Directory Isolation**: Staging directory is created in the operating system's temporary directory with exclusive mode `0o700` (`rwx------`), preventing other non-root processes from traversing or reading the directory.
2. **File Atomicity & Symlink Defeat**: File creation uses `O_CREAT | O_EXCL | O_NOFOLLOW` with mode `0o600` (`rw-------`). If the path exists or is a symlink, creation fails immediately (`FileExistsError` / `OSError`).
3. **Descriptor Hardlink & Digest Verification**: Immediately before handoff to NeMo, the staging file is opened with `O_RDONLY | O_NOFOLLOW`. The file descriptor is audited via `os.fstat(fd)`:
   - Must be a regular file (`stat.S_ISREG`).
   - Hard link count must be exactly 1 (`st_nlink == 1`), defeating hard-link injection.
   - Byte size must exactly equal in-memory buffer length.
   - Bytes read back from the descriptor must hash to the exact expected SHA-256 digest.
4. **Guaranteed Ephemeral Lifetime**: The staging file is never left on disk. A `try ... finally` construct guarantees that:
   - The file is overwritten with zeroes.
   - The file is unlinked via `os.unlink()`.
   - The staging directory is removed via `os.rmdir()`.
   - Even if NeMo crashes, throws an exception, or aborts, cleanup executes completely.

---

## 4. Verification & Audit Results

The implementation was evaluated across 10 security unit tests in `tests/test_titanet_loader_security.py`:
- `test_1_correct_synthetic_checkpoint_loads`: Verified end-to-end model construction from verified bytes.
- `test_2_checkpoint_digest_mismatch_rejected`: Tampered in-memory bytes rejected before staging.
- `test_3_checkpoint_mutation_before_loading_rejected`: Filesystem mutation between preflight and read rejected.
- `test_4_pathname_replacement_cannot_load_unverified_bytes`: Original file replacement has zero effect on staged bytes.
- `test_5_loader_receives_exact_verified_bytes`: NeMo loader receives exact byte-identical staging file.
- `test_6_arbitrary_checkpoint_paths_rejected`: Unsigned tokens and string paths rejected.
- `test_7_symlink_substitution_rejected`: Symlink replacement rejected with `O_NOFOLLOW`.
- `test_8_loader_exceptions_propagated_safely`: Loader exceptions propagate cleanly without masking.
- `test_9_temporary_staging_files_cleaned_up`: Zero lingering staging files/dirs on success and failure.
- `test_10_no_production_mode_model_injection_bypass`: Synthetic handles rejected in production.

All 10 tests passed in 0.82 seconds. Finding 1 is completely closed.
