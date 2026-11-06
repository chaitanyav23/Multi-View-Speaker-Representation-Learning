# Stage 5B.4-A.1.1 Integration and Path Audit Report
## Forensic Analysis of Path Resolution, Component Interfaces, and Data Flow

**Author**: Lead Research Engineer / Code Auditor  
**Date**: November 6, 2026  
**Scope**: Stage 5B.4-A.1.1 Architectural Audit  

---

## 1. Path Resolution Architecture Audit

### 1.1 Forensic Analysis of Prior Flaw
In Stage 5B.4-A.1, multiple scripts resolved the repository root by taking five parent directories from `__file__`:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
```
Evaluating this from `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/scripts/` produced:
1. `scripts` -> `a1_remediation`
2. `a1_remediation` -> `stage5b_4_fresh_diarization`
3. `stage5b_4_fresh_diarization` -> `stage5b`
4. `stage5b` -> `research_stages`
5. `research_stages` -> `research_stages` (depth 5 is `research_stages`, NOT repo root!)

Consequently, expressions such as `REPO_ROOT / "research_stages" / "stage2_baseline"` expanded to:
`.../Multi-View-Speaker-Representation-Learning/research_stages/research_stages/stage2_baseline`
This caused silent path resolution errors when attempting to locate baseline posteriors.

### 1.2 Remediated Centralized Path Resolver
In `scripts/path_utils.py`, path resolution is decoupled from directory nesting depth by traversing upward until the repository anchor (`.git` directory or marker) is discovered:
```python
def get_repo_root(start_path: Optional[Path] = None) -> Path:
    current = Path(start_path or __file__).resolve()
    if current.is_file():
        current = current.parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent.resolve()
    # Fallback to unique project directory anchors
    for parent in [current] + list(current.parents):
        if (parent / "research_stages").is_dir() and (parent / "phase20_standalone").is_dir():
            return parent.resolve()
    raise RuntimeError(f"Could not determine repository root from {start_path or __file__}")
```

All subsystem directories are derived directly from this single authoritative root:
- `get_research_stages_dir()` -> `{REPO_ROOT}/research_stages`
- `get_stage5b_dir()` -> `{REPO_ROOT}/research_stages/stage5b`
- `get_stage5b_4_dir()` -> `{REPO_ROOT}/research_stages/stage5b/stage5b_4_fresh_diarization`
- `get_stage5b_4_cache_dir()` -> `{REPO_ROOT}/research_stages/stage5b/stage5b_4_fresh_diarization/cache`
- `get_stage2_posteriors_dir()` -> `{REPO_ROOT}/research_stages/stage2_baseline/posteriors`
- `get_phase20_scripts_dir()` -> `{REPO_ROOT}/phase20_standalone/scripts`

This guarantees exact, non-duplicating path resolution from any working directory.

---

## 2. Component Interface Integrity Audit

### 2.1 Overlap Posterior Loader Interface
- **Module**: `scripts/overlap_posterior_loader.py`
- **Class**: `OverlapPosteriorLoader`
- **Source of Truth**: `research_stages/stage2_baseline/posteriors/{rec_id}_16k_mono_posteriors.npz`
- **NPZ Schema Verification**:
  - `frame_times`: 1D array of float timestamps (sampled at 10ms frame resolution)
  - `p_overlap`: 1D array of continuous overlap probabilities $\in [0.0, 1.0]$
- **Interface Behavior**:
  - `load_overlap_posteriors(rec_id, expected_duration=None) -> Tuple[np.ndarray, np.ndarray]`
  - Validates array presence, shape parity, finite values, and probability boundaries.
  - Decoupling guarantee: M0 conditions never invoke this loader. M4 conditions fail immediately if posteriors are unavailable.

### 2.2 Production cpWER Adapter Interface
- **Module**: `scripts/cpwer_adapter.py`
- **Audited Core**: `phase20_standalone/scripts/cpwer.py:cpwer_stats`
- **Schema Compliance**:
  - Requires hypothesis segments to be wrapped in canonical envelope:
    `{"segments": [{"speaker": str, "text": str, "start": float, "end": float}], "words": [...]}`
  - Reference transcript input supports both `{"segments": [...]}` and bare segment lists.
  - Downstream output returns full Hungarian assignment diagnostics:
    `{"cpwer": float, "insertions": int, "deletions": int, "substitutions": int, "reference_words": int, "hypothesis_words": int}`

### 2.3 Checkpoint Preflight Verifier Interface
- **Module**: `scripts/checkpoint_preflight.py`
- **Audited Core**: `a1_remediation/scripts/checkpoint_verifier.py:verify_checkpoint`
- **Preflight Requirement**:
  - Hexadecimal format check: length strictly 64 chars, valid hex symbols.
  - Byte-by-byte SHA-256 digest calculation in 64KB blocks.
  - Zero weight deserialization or PyTorch / ONNX graph construction.

### 2.4 Output Path Guard Interface
- **Module**: `scripts/output_path_guard.py`
- **Enforcement Function**: `validate_output_path(output_path, allowed_roots=None, allow_creation=True)`
- **Containment Policy**:
  - Rejects traversal sequences (`..`).
  - Rejects null bytes.
  - Rejects symlink escapes.
  - Rejects paths targeting any prohibited historical stage (`stage1_data_prep`, `stage2_baseline`, `stage3_dual_encoder`, `stage5a`, `stage5b_1`, `stage5b_2`, `stage5b_3`).

---

## 3. Data Flow and Isolation Verification

```mermaid
flowchart TD
    subgraph Storage ["On-Disk Data Sources"]
        P20["phase20_standalone/scripts/cpwer.py"]
        P_OVL["stage2_baseline/posteriors/*.npz"]
        CKPT["model_checkpoints/*.nemo, *.onnx"]
    end

    subgraph Preflight ["Orchestrator Preflight"]
        V_CKPT["checkpoint_preflight.py<br/>(Byte-level SHA-256)"]
        PATH_CHK["path_utils.py<br/>(Git Anchor Resolution)"]
    end

    subgraph Cache ["Isolated Embedding Cache"]
        LOAD_STATUS["cache_manager_remediated.py<br/>load_or_status()"]
        C_HIT["HIT: Return cached"]
        C_MISS["MISS: Extract & Cache"]
        C_INVAL["INVALIDATION: Re-extract"]
        C_SEC["SECURITY_ERR: Abort"]
    end

    subgraph Pipeline ["Diarization Pipeline"]
        VAD["VAD Binarization<br/>(C00, C50, C62)"]
        FUSION["Condition J Fusion<br/>(0.5 A_zt + 0.5 A_zr)"]
        CLUST["Fresh Spectral Clustering<br/>(Single-speaker base)"]
        OVL_LOAD["overlap_posterior_loader.py<br/>(M4 only)"]
        OVL_APPLY["apply_overlap_attribution()<br/>(M0 / M4)"]
    end

    subgraph Output_Guard ["Output & Evaluation Guard"]
        OUT_GUARD["output_path_guard.py<br/>(Containment Barrier)"]
        RTTM_OUT["output_dir/*.rttm"]
        EVAL_PY["pyannote.metrics<br/>(DER 0.0s, DER 0.25s, JER)"]
        EVAL_CP["cpwer_adapter.py<br/>(Hungarian cpWER)"]
    end

    CKPT --> V_CKPT
    PATH_CHK --> LOAD_STATUS
    LOAD_STATUS --> C_HIT
    LOAD_STATUS --> C_MISS
    LOAD_STATUS --> C_INVAL
    LOAD_STATUS --> C_SEC
    VAD --> FUSION
    FUSION --> CLUST
    P_OVL --> OVL_LOAD
    CLUST --> OVL_APPLY
    OVL_LOAD --> OVL_APPLY
    OVL_APPLY --> OUT_GUARD
    OUT_GUARD --> RTTM_OUT
    RTTM_OUT --> EVAL_PY
    RTTM_OUT --> EVAL_CP
    P20 --> EVAL_CP
```

---

## 4. Audit Conclusion

The integration paths, module boundaries, and error handling regimes have been systematically inspected and audited. The implementation is cleanly isolated, defensively guarded, and free of path duplication bugs or unhandled exceptions.
