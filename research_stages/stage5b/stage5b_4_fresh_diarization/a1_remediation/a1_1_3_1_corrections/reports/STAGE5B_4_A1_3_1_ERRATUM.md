# Stage 5B.4-A.1.3.1 Documentation Erratum: M0 vs M4 Eligibility & Provenance States

## 1. Executive Summary

During the independent source-code audit of Stage 5B.4-A.1.3, Finding 6 identified an inconsistency between the prose documentation in `STAGE5B_4_A1_3_PROVENANCE_VALIDATION_REPORT.md` and the actual experimental design of the fresh diarization study.

Specifically, earlier documentation stated that Condition M0 (oracle and VAD conditions without overlap detection, namely E00, E10, and E20) was "production eligible" despite overlap posterior provenance remaining unverified for mono audio. In contrast, other sections claimed that unverified overlap provenance prevented all production execution.

This Erratum formally resolves this ambiguity by establishing distinct, decoupled semantic definitions for **Execution Eligibility** versus **Overlap Posterior Provenance Verification**.

---

## 2. Decoupled State Semantics

In Stage 5B.4, experimental conditions are partitioned into two architectural categories based on their consumption of Stage 2 overlap posteriors:

### 2.1 Condition M0 (No Overlap Detection / Baseline Partition)
- **Experimental Conditions**: E00 (Oracle VAD + M0), E10 (Baseline VAD + M0), E20 (Optimized VAD + M0).
- **Acoustic Data Flow**: M0 generates subsegments exclusively from VAD speech intervals (or oracle intervals), extracts multiscale speaker embeddings from TitaNet-Large and WeSpeaker ResNet293, and clusters them using NME-SC.
- **Overlap Posterior Consumption**: **ZERO**. Condition M0 does not load, inspect, or apply overlap posteriors (`p_overlap` is strictly `None`).
- **Eligibility State**:
  - `m0_execution_eligible = True` (authorized for production execution).
  - `overlap_provenance_verified = False` (posteriors are not verified).
  - `overlap_production_eligible = False` (posteriors are not eligible).
  - `m4_execution_eligible = False` (M4 is not authorized).

### 2.2 Condition M4 (Conservative Abstention Overlap Diarization)
- **Experimental Conditions**: E01 (Oracle VAD + M4), E11 (Baseline VAD + M4), E21 (Optimized VAD + M4).
- **Acoustic Data Flow**: M4 loads Stage 2 frame-level overlap posteriors, validates timestamp monotonicity and 16.875ms stride, downsamples/aligns overlap probabilities with base subsegments, and assigns second-speaker labels to high-confidence overlap frames.
- **Overlap Posterior Consumption**: **MANDATORY**. M4 cannot execute without loading and trusting Stage 2 overlap posteriors.
- **Eligibility State**:
  - `m0_execution_eligible = False`.
  - `overlap_provenance_verified = True` (requires cryptographic byte hashing of posterior, source audio, and model checkpoint).
  - `overlap_production_eligible = True` (requires structural validity AND independent provenance verification).
  - `m4_execution_eligible = True` (authorized only when `overlap_production_eligible = True`).

---

## 3. The Historical Mono Source Audio Provenance Limitation

Stage 1 (`research_stages/stage1_data_prep/manifests/recording_manifest.json`) cataloged multi-channel headset WAV files and calculated their authoritative SHA-256 digests. However, the downmixed historical `16k_mono.wav` files were generated without recording independent SHA-256 checksums in the frozen Stage 1 manifest.

Consequently:
1. `overlap_posterior_provenance_manifest.json` accurately marks mono audio records as:
   `"source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST"`
2. Because physical byte hashing cannot verify against an unrecorded manifest digest, the verifier records `"source_audio_sha256"` in `unverified_fields`.
3. As a result, historical mono overlap posteriors have:
   `provenance_verified = False` and `production_eligible = False`.
4. In production mode, **Condition M4 strictly fails closed** when evaluating mono audio, raising `OverlapProvenanceError`.
5. Conversely, **Condition M0 is 100% unaffected** by this limitation because M0 does not consume overlap posteriors.

---

## 4. Formal Schema Specification

The `PipelineExecutionStats` container and all orchestration checkpoints in `a1_1_3_1_corrections` explicitly represent these decoupled states:

```python
@dataclass
class PipelineExecutionStats:
    condition_id: str
    recording_id: str
    # ...
    structurally_valid: bool
    m0_execution_eligible: bool          # True for E00, E10, E20
    m4_execution_eligible: bool          # True ONLY if overlap_production_eligible is True
    overlap_provenance_verified: bool    # True ONLY if physical bytes match manifest
    overlap_production_eligible: bool    # True ONLY if structurally_valid AND overlap_provenance_verified
    # Compatibility aliases
    provenance_verified: bool            # Synchronized with overlap_provenance_verified
    production_eligible: bool            # Synchronized with overlap_production_eligible
```

This formalization eliminates all previous ambiguity and ensures strict fail-closed security for production deployment.
