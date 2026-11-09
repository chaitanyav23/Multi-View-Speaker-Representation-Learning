# Stage 5B.4-A.1.3.2 Provenance & Mandatory Field Enforcement Report

## 1. Context and Problem Statement

Stage 5B.4 introduces cryptographic provenance verification to ensure that overlap posterior arrays used in Condition M4 (`E01`, `E11`, `E21`) originate from authenticated model executions over genuine audio recordings.

In Stage 5B.4-A.1.3.1, `overlap_provenance.py` verified manifest records against `TRUSTED_PYANNET_POSTERIOR_SPEC`. However, for `generation_code_version`, it used:
```python
entry.get("generation_code_version", TRUSTED_PYANNET_POSTERIOR_SPEC["generation_code_version"])
```
An independent code audit identified this as **Finding 3**:
> Using fallback default substitutions violates the principle of explicit provenance. If an artifact's manifest omits `generation_code_version`, the verifier silently supplies the trusted default, allowing incomplete records to pass verification.

---

## 2. Mandatory Fields Specification & Enforcement

Stage 5B.4-A.1.3.2 completely eliminates default substitutions and establishes 10 non-negotiable mandatory provenance fields:

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

### 2.1 Enforcement Logic in `scripts/overlap_provenance.py`
For every posterior artifact queried, the verifier iterates through `MANDATORY_PROVENANCE_FIELDS`:
1. **Presence Check**: Field must exist as an explicit key in the manifest entry.
2. **Non-Null & Non-Empty Check**: Value must not be `None`, and strings must not be empty or whitespace (`str(val).strip() != ""`).
3. **Spec Alignment Check**:
   - `model_architecture` must equal `"pyannote/segmentation"`.
   - `model_checkpoint_sha256` must equal the trusted PyanNet checkpoint hash.
   - `generation_config_id` must equal `"pyannote_3_1_continuous_top2"`.
   - `posterior_semantics` must equal `"continuous_top2_second_highest"`.
   - `timestamp_convention` must equal `"sliding_window_middle_centering"`.
   - `frame_step_sec` must equal `0.016875` (within float tolerance).
   - `first_frame_center_sec` must equal `0.03096875` (within float tolerance).
   - `source_sample_rate` must equal `16000`.
   - `generation_code_version` must explicitly equal `"stage2_baseline_v1"`.

If any field is missing, null, empty, or mismatched, `provenance_verified` is immediately set to `False`, and specific error descriptions are appended to `provenance_errors`.

---

## 3. Updated Canonical Provenance Manifest

The authoritative manifest:
`research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_2_final_corrections/manifests/overlap_posterior_provenance_manifest.json`
was regenerated to include explicit entries for all 16 posterior artifacts (R01–R08 for both `stream_headset` and `stream_mono`).

Each record contains:
```json
{
  "schema_version": "1.0",
  "posterior_file_rel": "posteriors/rec_01_stream_headset_overlap.npz",
  "posterior_file_sha256": "4b6be29e...",
  "posterior_file_size_bytes": 35568,
  "model_architecture": "pyannote/segmentation",
  "model_checkpoint_sha256": "a3b2c1...",
  "generation_config_id": "pyannote_3_1_continuous_top2",
  "posterior_semantics": "continuous_top2_second_highest",
  "timestamp_convention": "sliding_window_middle_centering",
  "frame_step_sec": 0.016875,
  "first_frame_center_sec": 0.03096875,
  "source_sample_rate": 16000,
  "generation_code_version": "stage2_baseline_v1",
  "source_audio_rel": "audio/rec_01_headset.wav",
  "source_audio_sha256": "8f3a1b...",
  "source_duration_sec": 300.0
}
```

For historical mono streams where source audio was not tracked in Stage 2 manifests, the field is explicitly set to `"source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST"`, causing provenance verification for mono M4 conditions to fail closed cleanly as documented.

---

## 4. Verification & Audit Results

The mandatory provenance enforcement was verified across 16 unit tests in `tests/test_mandatory_provenance_fields.py`:
- `test_1_complete_valid_manifest_passes`: Full valid manifest passes.
- `test_2_missing_generation_code_version_fails`: Missing `generation_code_version` fails closed.
- `test_3_null_generation_code_version_fails`: Null `generation_code_version` fails closed.
- `test_4_empty_generation_code_version_fails`: Empty string `generation_code_version` fails closed.
- `test_5_incorrect_generation_code_version_fails`: Incorrect version string fails closed.
- `test_6_missing_other_mandatory_fields_fail`: Parameterized tests across all 8 remaining spec fields confirm each individually fails closed when missing.
- `test_6b_missing_schema_version_fails`: Missing `schema_version` fails closed.
- `test_7_m4_production_eligibility_false_on_incomplete_provenance`: Production M4 execution is blocked on incomplete provenance.
- `test_8_m0_execution_independent_of_overlap_provenance`: Condition M0 executes unimpeded regardless of overlap provenance.

All 16 tests passed in 1.54 seconds. Finding 3 is completely closed.
