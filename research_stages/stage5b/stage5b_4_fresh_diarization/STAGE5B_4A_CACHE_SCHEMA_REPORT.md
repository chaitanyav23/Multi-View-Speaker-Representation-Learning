# Stage 5B.4-A — Embedding Cache Schema & Isolation Report

**Execution Status**: Specification & Validation Complete  
**Date**: 2026-11-06  
**Auditor**: Senior ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4A_CACHE_SCHEMA_REPORT.md`

---

## 1. Executive Summary

In Stage 3, speaker embeddings were cached using simple directory keys (`{cond}/{rec_id}/titanet_large__{cond}__scale{scale_k}.npz`), with subsegments hardcoded to the Stage 2 Path A DFN VAD (`C00`). When evaluating alternative VAD systems such as `C50` (Raw Mono) or `C62` (Consensus), the speech boundaries differ, resulting in different numbers and timestamps of subsegments.

Reusing the Stage 3 cache for Stage 5B.4 would result in:
1. Contamination or overwriting of frozen Stage 3 artifacts.
2. Silent cache hits on stale `C00` embeddings when evaluating `C50` or `C62`.
3. Dimension mismatch crashes when subsegment counts vary across segmentations.

Stage 5B.4-A implements a **cryptographically validated, VAD-aware isolated cache** exclusively under `research_stages/stage5b/stage5b_4_fresh_diarization/cache/`.

---

## 2. Canonical Cache Identity Specification

Every cache entry is governed by a complete, immutable `CacheIdentity` dataclass:

```python
@dataclass(frozen=True)
class CacheIdentity:
    schema_version: str            # "5b4_v1"
    recording_id: str              # e.g., "R01_sara_sharif"
    vad_config_id: str             # e.g., "C00", "C50", "C62"
    vad_intervals_sha256: str      # 64-character hex hash of speech intervals
    encoder_name: str              # e.g., "titanet_large", "wespeaker_resnet293_lm"
    encoder_checkpoint_sha256: str # SHA-256 hash of model weights
    audio_source_id: str           # "dfn" | "16k_mono"
    audio_preprocessing_id: str    # "raw_16k"
    scale_index: int               # 0 .. 4
    window_sec: float              # e.g., 0.50, 1.00, 1.25, 1.50, 1.75
    shift_sec: float               # e.g., 0.25, 0.50
    sample_rate: int               # 16000
    frontend_config_version: str   # "nemo_msdd_v1" | "wespeaker_kaldi_fbank_v1"
```

---

## 3. Canonical VAD Interval Hash

The `vad_intervals_sha256` represents a deterministic, canonical serialization of ordered speech intervals:
1. Intervals are sorted by start time and end time.
2. Boundaries are formatted to an explicit rounding precision of 4 decimal places: `{start:.4f},{end:.4f}`.
3. The canonical string is constructed as:  
   `"{rec_id}|{vad_id}|v1|" + ";".join(elements)`
4. The 256-bit SHA-256 digest is computed over the UTF-8 bytes.

This ensures:
- Identical intervals in different order produce the exact same hash.
- Any shift or alteration in boundary $> 10^{-4}$s produces a completely different hash.
- Different VAD IDs on identical boundaries produce different hashes.

---

## 4. On-Disk Layout & Filename Convention

All cache files are stored within the isolated Stage 5B.4 workspace:

```
research_stages/stage5b/stage5b_4_fresh_diarization/cache/
└── {vad_config_id}/
    └── {recording_id}/
        └── {encoder_name}__{audio_source_id}__scale{scale_index}__{vad_intervals_sha256[:16]}.npz
```

### Example On-Disk Filename:
`titanet_large__dfn__scale4__3a9f0e12d45c8b71.npz`

---

## 5. Metadata Schema & NPZ Payload

Each `.npz` file contains the following compressed arrays:
- `embeddings`: Float32 array of shape `(N_subsegments, embedding_dim)` (192 for TitaNet, 256 for WeSpeaker).
- `l2_norms`: Float32 array of shape `(N_subsegments,)` containing pre-normalization vector norms.
- `subsegment_starts`: Float64 array of shape `(N_subsegments,)` recording subsegment start times.
- `subsegment_ends`: Float64 array of shape `(N_subsegments,)` recording subsegment end times.
- `metadata_json`: Serialized JSON string containing the complete `CacheIdentity` dictionary.

---

## 6. Atomic Write Behavior

To prevent partial or corrupted cache files resulting from interrupted jobs:
1. Data is written to a named temporary file in the target directory using `tempfile.mkstemp(prefix="cache_write_", suffix=".npz")`.
2. The compressed `.npz` file is fully flushed and closed.
3. The temporary file is atomically renamed to the final destination via `temp_path.replace(final_path)`.
4. If an exception occurs during serialization, the temporary file is unlinked immediately.

---

## 7. Strict Cache Validation on Read

Before any cached embedding is returned by `load_embeddings`, the cache manager executes 10 sequential checks:
1. File existence check at destination path.
2. NPZ container integrity check.
3. Metadata presence check (`metadata_json` key).
4. Schema version equality (`schema_version == "5b4_v1"`).
5. Exact field-by-field equality of all 13 `CacheIdentity` fields.
6. Matrix shape verification (2D array, row count matching subsegment count).
7. Feature dimension verification (expected 192 for TitaNet, 256 for WeSpeaker).
8. Numeric validity check: rejects NaN and Infinite values.
9. Non-zero vector check: rejects degenerate zero-norm vectors.
10. Subsegment timestamp array alignment against expected subsegment definitions.

If any check fails, a `ValueError` or `FileNotFoundError` is raised. The cache manager **never** silently falls back to stale data.

---

## 8. Proof of Prohibited Stage 3 Path Isolation

In `embedding_cache.py`:
```python
PROHIBITED_STAGE3_CACHE = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache"

@staticmethod
def _assert_safe_cache_root(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_prohibited = PROHIBITED_STAGE3_CACHE.resolve()
    if resolved_prohibited in resolved_path.parents or resolved_path == resolved_prohibited:
        raise PermissionError(
            f"FATAL: Attempted to use prohibited Stage 3 cache directory: {resolved_path}"
        )
```

This safety check is enforced in `__init__`, `get_entry_dir`, and `save_embeddings`. Any attempt to point the cache root or file operations toward Stage 3 immediately triggers a `PermissionError`.

Unit test `test_prohibited_stage3_cache_rejection` verifies this behavior and passed successfully.
