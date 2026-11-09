# Stage 5B.4-A.1.3.2 Integration Patch Guide

## 1. Overview

This document specifies the exact code differences, interface modifications, and security enhancements between Stage 5B.4-A.1.3.1 and Stage 5B.4-A.1.3.2. If an engineer or pipeline runner needs to backport or integrate these fixes into another branch or staging environment, this guide details each changed function and file.

---

## 2. Modified Components & Function Signatures

### 2.1 `scripts/model_factory.py`

#### A. Added Secure Staging Function
```python
def load_titanet_model_from_verified_bytes(
    verified_checkpoint: VerifiedCheckpoint,
    ckpt_bytes: bytes,
    device: str = "cpu",
    loader_fn: Optional[Callable[[str, str], Any]] = None,
) -> Any:
```
- **Replaces**: Direct passing of `token.canonical_path` to `EncDecSpeakerLabelModel.restore_from()`.
- **Implementation Mechanism**:
  1. Validates `VerifiedCheckpoint` signature and memory buffer hash.
  2. Creates temp directory with `tempfile.mkdtemp(prefix="nemo_titanet_stage_")` and sets `os.chmod(stage_dir, 0o700)`.
  3. Creates staging file with flags `os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW` and mode `0o600`.
  4. Writes bytes and flushes with `os.fsync(fd)`.
  5. Re-opens via descriptor with `O_RDONLY | O_NOFOLLOW` and verifies:
     - `stat.S_ISREG(st.st_mode)`
     - `st.st_nlink == 1`
     - `st.st_size == len(ckpt_bytes)`
     - SHA-256 digest of read-back bytes matches `verified_checkpoint.computed_sha256`.
  6. Calls `loader_fn(str(staging_file), device)` or `EncDecSpeakerLabelModel.restore_from(restore_path=str(staging_file), map_location=device)`.
  7. In a `finally` block:
     - Overwrites staging file bytes with zeroes.
     - Unlinks staging file.
     - Removes staging directory.

#### B. Updated `ControlledModelFactory.build_production_model()`
```python
# Before (A.1.3.1):
model = EncDecSpeakerLabelModel.restore_from(
    restore_path=str(verified_checkpoint.canonical_path),
    map_location="cpu",
)

# After (A.1.3.2):
model = load_titanet_model_from_verified_bytes(
    verified_checkpoint=verified_checkpoint,
    ckpt_bytes=ckpt_bytes,
    device="cpu",
    loader_fn=cls.nemo_loader_override,
)
```

---

### 2.2 `scripts/output_path_guard.py`

#### A. Updated `validate_final_rttm_path()`
- Implemented descriptor traversal using `os.open(..., os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)`.
- Traverses from `root_fd` to `cond_fd`, preventing symlink insertion into condition directory hierarchies.

#### B. Updated `revalidate_and_write_rttm()`
- **Replaces**: Unconditional `O_TRUNC` opening.
- **New Check-Before-Truncate Flow**:
  ```python
  # Attempt exclusive creation first
  flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
  if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW

  try:
      fd = os.open(str(final_path), flags, 0o644)
  except FileExistsError:
      # If file exists, open WITHOUT O_TRUNC
      open_flags = os.O_WRONLY
      if hasattr(os, "O_NOFOLLOW"):
          open_flags |= os.O_NOFOLLOW
      fd = os.open(str(final_path), open_flags)
      st = os.fstat(fd)
      if not stat.S_ISREG(st.st_mode):
          os.close(fd)
          raise OutputSecurityError("Not a regular file")
      if st.st_nlink > 1:
          os.close(fd)
          raise OutputSecurityError(f"Multiple hard links detected ({st.st_nlink})")
      # Safe to truncate now
      os.ftruncate(fd, 0)
  ```

---

### 2.3 `scripts/overlap_provenance.py`

#### A. Elimination of Default Fallback Substitution
```python
# Before (A.1.3.1):
gen_code = entry.get(
    "generation_code_version",
    TRUSTED_PYANNET_POSTERIOR_SPEC["generation_code_version"],
)

# After (A.1.3.2):
for req_field in MANDATORY_PROVENANCE_FIELDS:
    if req_field not in entry:
        errors.append(f"Mandatory provenance field missing: '{req_field}'")
        continue
    val = entry[req_field]
    if val is None or str(val).strip() == "":
        errors.append(f"Mandatory provenance field '{req_field}' is null or empty.")
        continue
```

#### B. Mandatory Fields List
```python
MANDATORY_PROVENANCE_FIELDS = [
    "schema_version",
    "model_architecture",
    "model_checkpoint_sha256",
    "generation_config_id",
    "posterior_semantics",
    "timestamp_convention",
    "frame_step_sec",
    "first_frame_center_sec",
    "source_sample_rate",
    "generation_code_version",
]
```

---

### 2.4 `manifests/overlap_posterior_provenance_manifest.json`
- Explicit `generation_code_version: "stage2_baseline_v1"` inserted into all 16 entries.
- Unrecorded mono streams explicitly marked `"source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST"`.

---

## 3. Migration Instructions

To migrate a runtime environment from Stage 5B.4-A.1.3.1 to Stage 5B.4-A.1.3.2:
1. Replace `scripts/model_factory.py`, `scripts/output_path_guard.py`, and `scripts/overlap_provenance.py` with their A.1.3.2 counterparts.
2. Replace `manifests/overlap_posterior_provenance_manifest.json` with the A.1.3.2 manifest.
3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
4. Verify all 50 tests pass.
