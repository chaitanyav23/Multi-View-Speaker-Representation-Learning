# Stage 5B.2 Corrective Audit: Execution Log

## 1. Environment & Hardware Specifications
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi`
- **Execution Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10.20, Torch 2.6.0, Pyannote Audio 3.3.2)
- **Audit Execution Date**: 2026-11-04

---

## 2. Chronological Log of Corrective Actions

1. **Pre-Audit Integrity Verification**:
   - Re-verified all 298 frozen research artifacts from Stages 1–5A.1.2 against `frozen_artifact_hashes_after.csv`. Result: **100% bitwise match (0 missing, 0 mismatches)**.
   - Verified all 316 original Stage 5B.2 artifacts against `STAGE5B_2_MANIFEST.json`. Result: **100% bitwise match (0 missing, 0 mismatches)**.
   - Generated `STAGE5B_2_CORRECTIVE_HASH_AUDIT.csv` recording 614 verified artifact hashes.

2. **Audit A — R04 Theoretical Overlap Miss Floor Forensic Analysis**:
   - Identified the source of the 95s vs 98s discrepancy: $T_{\text{overlap}} = \int \mathbb{1}[|Y(t)| \ge 2] dt = 95.0\text{ s}$, whereas $T_{\text{excess}} = \int \max(0, |Y(t)| - 1) dt = 98.0\text{ s}$ due to 3.0s of 3-speaker simultaneous speech.
   - Identified the root cause of the 32.00% reported miss: `phase20_standalone/scripts/common.py` line 348 lacked track IDs (`ann[Segment(s, e)] = spk`), causing Line 30 to overwrite Line 29 in `R04_piers_robinson.rttm` and deleting 11.0s of Speaker_1 speech (reducing reference time from 284.0s to 273.0s).
   - Proved the mathematical single-label miss floor: $T_{\text{excess}} / T_{\text{total}} = 98.0 / 284.0 = \mathbf{34.51\%}$.
   - Created `R04_OVERLAP_FLOOR_RECONCILIATION.md`.

3. **Audit B — Overlap Posterior Binarization & Timestamp Alignment**:
   - Discovered that cached `frame_times` represented sliding window centers ($t_0 = 0.03096875\text{ s}$), truncating ~31ms at segment boundaries in legacy binarizer.
   - Implemented `binarize_overlap_corrected()` with exact physical window support: receptive field 0.0619375s, step 0.016875s (270 samples at 16kHz).
   - Created `OVERLAP_TIMESTAMP_ALIGNMENT_AUDIT.md`.

4. **Audit C — Repair of M2 Dual-Encoder Prototype Affinity**:
   - Discovered that legacy M2 never evaluated cosine similarity, instead counting frames in a temporal window and falling back to list indexing.
   - Implemented genuine mathematical prototype affinity using frozen Stage 3 scale 0 embeddings (TitaNet 192d, WeSpeaker 256d, Concat 448d, and Normalized Cosine Average).
   - Created `M2_PROTOTYPE_AFFINITY_CORRECTION.md`.

5. **Audit D — Correction of M1 Temporal Context Attribution**:
   - Removed arbitrary `global_alts[0]` list-order fallback.
   - Enforced 3.0s local context window, minimum evidence threshold (0.20s), and strict abstention.
   - Created `M1_TEMPORAL_ATTRIBUTION_CORRECTION.md`.

6. **Audit E — Correction of M3 Local Turn Alternation & M4 Conservative Abstention**:
   - Removed global full-file slicing (`frame_spk[:mid_idx]`).
   - Implemented local turn extraction bounded by $\Delta t_{\text{max}} = 5.0\text{ s}$ and minimum turn duration 0.20s.
   - Created `M3_M4_LOCAL_TURN_ATTRIBUTION_CORRECTION.md`.

7. **Audit F — Redefinition of Oracle Diagnostics**:
   - Removed artificial `oracle_<gt_spk>` label prefix that previously corrupted DER.
   - Implemented Hungarian bipartite matching mapping GT speaker identities into hypothesis cluster space.
   - Created `ORACLE_DIAGNOSTIC_PROTOCOL.md`.

8. **Experimental Re-Execution (`run_stage5b_2_corrected.py`)**:
   - Evaluated 22 configurations across all 8 recordings (176 total runs).
   - Generated track-aware RTTMs, computed collar=0.0s and collar=0.25s DER, JER, frame-level overlap detection metrics, and cpWER/saWER.
   - Emitted 4 CSV deliverables under `metrics/`.
   - Executed 10,000-resample cluster bootstrap and Wilcoxon signed-rank tests.
   - Generated 7 analytical diagnostic plots under `STAGE5B_2_CORRECTED_PLOTS/`.
   - Total runtime: 222.34 seconds. Exit code: 0.
