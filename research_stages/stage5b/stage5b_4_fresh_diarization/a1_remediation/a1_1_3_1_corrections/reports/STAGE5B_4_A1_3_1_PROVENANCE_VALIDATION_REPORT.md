# Stage 5B.4-A.1.3.1 Provenance Validation Report

## 1. Executive Summary

This Provenance Validation Report documents the remediation of Findings 2 and 3 in Stage 5B.4-A.1.3.1.

Previous releases suffered from two critical provenance vulnerabilities:
1. Manifest string claims were accepted without performing physical byte hashing on source audio files and the PyanNet model checkpoint.
2. Posterior generation configurations and semantics were recorded as passive JSON metadata without validation against an authoritative, independent specification.

Stage 5B.4-A.1.3.1 resolves both issues by establishing physical disk byte hashing and formal validation against `TRUSTED_PYANNET_POSTERIOR_SPEC`.

---

## 2. Finding 2: Physical Byte Hashing of Source Audio & Checkpoint

### 2.1 Technical Deficiency in A.1.3
In A.1.3, `verify_provenance()` compared manifest string digests against each other without independently hashing the source audio files or the PyanNet model checkpoint on disk. An attacker or corrupted environment could swap disk files while keeping manifest strings intact, bypassing validation.

### 2.2 Implemented Remediation (`overlap_provenance.py`)
1. **PyanNet Checkpoint Physical Byte Hashing**:
   - The verifier locates `whisperx/assets/pytorch_model.bin` on disk.
   - Computes physical byte SHA-256 (`compute_file_sha256()`).
   - Asserts exact equality with `0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea`.
   - Fails closed if the checkpoint is missing, a symlink, corrupted, or mismatched.
2. **Source Audio Physical Byte Hashing**:
   - Resolves the source audio WAV file path securely from `source_audio_rel`.
   - Rejects relative path traversal (`..`) and symlinks.
   - Directly hashes physical bytes on disk.
   - Compares computed hash to the authoritative recording manifest.
3. **Historical Mono Limitation Preserved**:
   - Historical downmixed mono WAV files were not hashed in Stage 1 `recording_manifest.json`.
   - Manifest entries for mono audio explicitly state `"source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST"`.
   - Physical byte verification cannot match an unrecorded digest, so the verifier records `source_audio_sha256` in `unverified_fields`.
   - Consequently, `provenance_verified = False` and `production_eligible = False`.
   - Condition M4 strictly fails closed on mono audio.

---

## 3. Finding 3: Validation Against `TRUSTED_PYANNET_POSTERIOR_SPEC`

### 3.1 Trusted Configuration Specification
To eliminate configuration drift and ambiguous semantics, `overlap_provenance.py` defines the canonical, version-controlled specification:

```python
TRUSTED_PYANNET_POSTERIOR_SPEC = {
    "schema_version": "1.0",
    "model_architecture": "pyannote/segmentation",
    "model_checkpoint_sha256": "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea",
    "generation_config_id": "pyannote_3_1_continuous_top2",
    "posterior_semantics": "continuous_top2_second_highest",
    "timestamp_convention": "sliding_window_middle_centering",
    "frame_step_sec": 0.016875,          # 270 samples at 16 kHz
    "first_frame_center_sec": 0.03096875, # sliding window centering
    "sample_rate": 16000,
    "generation_code_version": "stage2_baseline_v1",
}
```

### 3.2 Strict Validation Logic
During `verify_provenance()`, every manifest artifact is checked field-by-field against `TRUSTED_PYANNET_POSTERIOR_SPEC`:
- `schema_version`: Must equal `"1.0"`.
- `model_architecture`: Must equal `"pyannote/segmentation"`.
- `generation_config_id`: Must equal `"pyannote_3_1_continuous_top2"`.
- `posterior_semantics`: Must equal `"continuous_top2_second_highest"`.
- `timestamp_convention`: Must equal `"sliding_window_middle_centering"`.
- `frame_step_sec`: Must match `0.016875` within $10^{-6}$.
- `first_frame_center_sec`: Must match `0.03096875` within $10^{-6}$.
- `source_sample_rate`: Must equal `16000`.
- `generation_code_version`: Must equal `"stage2_baseline_v1"`.

Any deviation or unrecognized configuration is recorded in `provenance_errors` and strictly fails closed.

---

## 4. Multi-State Validation Architecture

The three validation states remain strictly decoupled:

```mermaid
flowchart TD
    A["Raw Posterior Array (.npz)"] --> B["Structural Acoustic Validation"]
    B -->|Checks Stride, Centering, Monotonicity, Range| C{"Structurally Valid?"}
    C -->|No| F["overall_state.structurally_valid = False<br>production_eligible = False"]
    C -->|Yes| D["Cryptographic & Spec Verification"]
    D -->|Hashes Source Audio & PyanNet Checkpoint<br>Validates TRUSTED_PYANNET_POSTERIOR_SPEC| E{"Provenance Verified?"}
    E -->|No / Mono Unrecorded| G["overall_state.provenance_verified = False<br>production_eligible = False"]
    E -->|Yes (Headset Stream)| H["overall_state.provenance_verified = True<br>production_eligible = True"]
```

---

## 5. Verification Matrix

| Test Suite | Tests | Findings Covered | Verdict |
|------------|-------|------------------|---------|
| `test_source_and_checkpoint_hashes.py` | 11 | Finding 2 (Disk byte hashing, missing files, symlink escapes, placeholders) | **11/11 PASS** |
| `test_provenance_schema_and_config.py` | 11 | Finding 3 (Schema version, architecture, frame stride, centering, semantics) | **11/11 PASS** |
| `test_m0_m4_eligibility.py` | 3 | Finding 6 (Decoupled M0 vs M4 production eligibility) | **3/3 PASS** |

---

## 6. Conclusion

All provenance verification requirements are fully satisfied. File and configuration integrity are mathematically verified from physical bytes on disk.
