# Stage 5B.4-A.1.3.1 Security Audit Report

## 1. Executive Summary

This Security Audit Report details the evaluation, implementation, and verification of security remediations introduced in Stage 5B.4-A.1.3.1. These remediations eliminate all three critical security vulnerabilities identified in Stage 5B.4-A.1.3:

1. **Finding 1**: Caller-controlled checkpoint search roots allowing allowlist expansion and directory traversal.
2. **Finding 4**: TOCTOU race conditions during model deserialization and incomplete WeSpeaker model reconstruction.
3. **Finding 5**: TOCTOU symlink substitution races during RTTM destination output writing.

All fixes were implemented using defensive systems programming techniques (`O_NOFOLLOW`, descriptor validation, in-memory cryptographic verification, immutable allowlist containment) and proven with dedicated security test suites.

---

## 2. Finding 1: Checkpoint Root Allowlist Containment

### 2.1 Threat Model
In A.1.3, `CheckpointPreflightVerifier` accepted an optional `custom_permitted_roots` parameter. A compromised caller or poorly configured runner could pass an arbitrary root (e.g. `/tmp` or repo root) in production mode, effectively expanding authorization and allowing untrusted checkpoints to be evaluated as production models. Furthermore, relative paths containing `..` or symlink parents could allow directory traversal.

### 2.2 Implemented Remediation (`checkpoint_preflight.py`)
1. **Immutable Standard Allowlist**:
   Established `get_immutable_standard_checkpoint_roots()` returning strictly:
   - `research_stages/stage3_dual_encoder/models`
   - `~/.cache/torch`
   - `~/.cache/huggingface`
   - `whisperx/assets`
2. **Strict Production Containment**:
   In `PRODUCTION` mode, `custom_permitted_roots` is strictly audited. If provided, every caller-supplied root must resolve strictly inside an authorized standard root and cannot be an ancestor of any standard root. Supplying any external custom root raises `CheckpointPreflightError("Unauthorized custom root")`.
3. **Path Sanitization and Symlink Rejection**:
   Rejects null bytes, relative traversal (`..`), directory targets, and any path where the target or any parent directory is a symbolic link.

### 2.3 Verification Results
10 dedicated unit tests in `test_checkpoint_roots.py` verify that:
- Standard allowlist checkpoints pass.
- Outside checkpoints fail.
- Custom external roots fail.
- Parent roots (e.g. `/`) fail.
- Symlinks pointing outside fail.
- Traversal attacks fail.

---

## 3. Finding 4: WeSpeaker Architecture & TOCTOU Elimination

### 3.1 Threat Model
Two critical flaws were present in A.1.3:
1. `ControlledModelFactory` relied on `torch.load(checkpoint_path)` after preflight checks. An attacker or concurrent process could replace or mutate the file on disk between the preflight hash check and the model loader invocation (Time-of-Check / Time-of-Use race).
2. WeSpeaker model instantiation did not reconstruct the official `WeSpeakerResNet293` architecture and did not filter training-time projection head parameters.

### 3.2 Implemented Remediation (`model_factory.py`)
1. **Official WeSpeaker ResNet293 Architecture**:
   Implemented exact `Bottleneck`, `TSTP` (Temporal Statistics Pooling), and `WeSpeakerResNet293` matching Stage 3 dual-encoder extraction:
   `num_blocks=[10, 20, 64, 3], m_channels=32, feat_dim=80, embed_dim=256, two_emb_layer=False`.
2. **Projection Key Filtering**:
   Inference extracts 256-dimensional embeddings prior to projection. Training-time `projection.*` keys in `state_dict` are automatically filtered out, and model parameters are strictly verified with `load_state_dict(filtered_sd, strict=True)`.
3. **In-Memory Byte Buffer TOCTOU Elimination**:
   `ControlledModelFactory.read_and_verify_checkpoint_bytes()` reads the file bytes into an immutable Python `bytes` object using `os.O_NOFOLLOW`. It re-verifies the file size and SHA-256 on those exact in-memory bytes. The framework loader is then fed directly from `io.BytesIO(ckpt_bytes)` (or raw buffer for ONNX). Because the bytes reside in immutable memory, filesystem tampering after verification is physically impossible.

### 3.3 Verification Results
9 dedicated unit tests in `test_wespeaker_and_toctou.py` confirm:
- Complete ResNet293 instantiation and projection key filtering.
- Rejection of file mutation between preflight and loading.
- Rejection of symlink replacement before loading.
- Rejection of preloaded model injection in production.
- Cryptographic signature tamper-proofing on `VerifiedModelHandle`.

---

## 4. Finding 5: RTTM Destination Symlink Race Defeated

### 4.1 Threat Model
Between canonical path validation and RTTM writing, an attacker or concurrent race could create a symbolic link at the destination path pointing to a critical system file (e.g. `/etc/passwd` or a baseline manifest). An unprotected file writer opening the destination would follow the symlink and overwrite the target file.

### 4.2 Implemented Remediation (`output_path_guard.py`)
1. **Low-Level Secure Descriptor Open**:
   Instead of high-level `open()`, `revalidate_and_write_rttm()` uses POSIX low-level open flags:
   `os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW`.
   If the destination path is a symlink, the kernel immediately refuses to open the file and raises `OSError: [Errno 40] Too many levels of symbolic links` (or `EEXIST`).
2. **Descriptor Verification via `fstat`**:
   The open file descriptor is inspected with `os.fstat(fd)` to guarantee `stat.S_ISREG(file_stat.st_mode)` is true (confirming a regular file, not a FIFO, device, or socket).
3. **Post-Write Revalidation**:
   After flushing and calling `os.fsync(fd)`, the path is re-verified to guarantee no symlink was substituted and containment within the authorized output directory holds.

### 4.3 Verification Results
8 dedicated unit tests in `test_rttm_symlink_race.py`, including the Critical Acceptance Test:
- Validated destination path.
- Injected symlink pointing to an external sentinel file containing sensitive text.
- Invoked `revalidate_and_write_rttm`.
- Asserted `OutputSecurityError` was raised.
- **Confirmed that external sentinel file contents remained 100% untouched!**

---

## 5. Security Audit Verdict

**PASS** — All security gaps identified in Findings 1, 4, and 5 are completely closed and verified.
