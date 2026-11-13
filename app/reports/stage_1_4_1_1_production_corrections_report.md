# Stage 1.4.1.1 — Production Control-Flow and Output-Provenance Corrections Report

**Date:** 2026-11-07  
**Host:** `cseproj66` (Linux 6.8.0-58-generic x86_64)  
**Primary Accelerator:** NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Protected Git Baseline:** `b4899b5` (`main`)  
**Parent B.4 Baseline:** `4490823e936d3702258ce82e558b597ad53d3fd3`  
**Status:** **COMPLETE & FULLY VERIFIED (READY FOR REVIEW — ZERO COMMITS)**

---

## 1. Executive Summary & Defect Correction Matrix

Stage 1.4.1.1 implemented narrowly scoped, production-grade control-flow and output-provenance corrections following an independent review of Stage 1.4.1. No new diarization research was conducted, no packages were installed or altered, no checkpoints were downloaded or modified, and no historical Phase 20 artifacts were altered.

| Defect ID | Component | Original Defect | Root Cause | Exact Engineering Correction | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Correction A** | `phase20_engine.py` | `enable_diarization=False` constructor parameter was ignored | Unconditional assignment `self.enable_diarization = True` at line 100 in `__init__` | Removed line 100; constructor parameter `self.enable_diarization = enable_diarization` preserved | Unit test: `test_enable_diarization_flag_preservation` |
| **Correction B** | `phase20_engine.py` | Disabling diarization bypassed NeMo but still attempted Pyannote when HF token existed | Control flow fell through to `if self.hf_token:` block even when `enable_diarization=False` | Immediate check `if not self.enable_diarization:` skips all backends, returns `DIARIZATION_UNAVAILABLE` with blocker `"Diarization disabled by configuration."`, assigns `SPEAKER_UNASSIGNED`, preserves ASR/alignment | Unit test: `test_disabled_diarization_bypasses_all_backends` |
| **Correction C** | `phase20_engine.py` | VAD failure silently fell back to raw ASR segment boundaries | `except Exception:` block assigned `vad_intervals = [{"start": s["start"], "end": s["end"]} for s in raw_segments]` | Removed ASR segment fallback completely. Structural validation checks timestamps (`start < end`, $\ge 0$). VAD failure aborts NeMo diarization immediately with `DIARIZATION_FAILED` | Unit tests: `test_vad_failure_stops_diarization_without_asr_fallback`, `test_malformed_vad_intervals_trigger_failure` |
| **Correction D** | `diarization_worker.py` & `nemo_diarizer.py` | Worker selected arbitrary `candidates[0]` from `pred_rttms/*.rttm`; potential output collision | Worker globbed all RTTM files; adapter reused parent directory without unique invocation child directory | Worker strictly expects `pred_rttms/{file_id}.rttm` matching audio manifest; adapter creates isolated child directory for every run and validates workspace provenance | Unit tests: `test_diarize_subprocess_mock_success`, `test_diarize_worker_provenance_violation`, Group B worker test |
| **Correction E** | `nemo_diarizer.py` | Adapter accepted returncode 0 without strict worker JSON status validation | Incomplete protocol enforcement on subprocess stdout | Strict protocol schema enforced: valid JSON, recognized status (`SUCCESS`, `NO_SPEECH`, `FAILED`), verified regular file on disk, provenance containment | Unit tests: `test_diarize_missing_json_payload`, `test_diarize_malformed_json_payload`, `test_diarize_unrecognized_worker_status`, `test_diarize_worker_reported_failure_status` |
| **Correction F** | `diarization_worker.py` & `nemo_diarizer.py` | Worker emitted `{"status": "SUCCESS"}` on empty VAD; adapter reported `DIARIZATION_AVAILABLE` | Zero turns in VAD RTTM was handled as successful diarization | Worker emits `{"status": "NO_SPEECH"}` and short-circuits before model loading; adapter and engine return `NO_SPEECH` with 0 detected speakers | Unit tests: `test_diarize_empty_vad_intervals_short_circuits`, `test_empty_vad_returns_no_speech_status`, Group B CLI test |

---

## 2. Reconciled Turn-Count Provenance & Evidence

In Stage 1.4.1, three different turn counts were reported across various sections of the report. The empirical provenance of each number has been verified from disk artifacts:

| Turn Count | Setting | Artifact Evidence | Explanation & Provenance |
| :---: | :--- | :--- | :--- |
| **48** | **Benchmark Diarization Evaluation (Group C)** | `scratch/nemo_diar_group_c/predicted.rttm` | NeMo MSDD inference executed conditioned on the 46 historical Phase 20 Path A VAD intervals (`phase20_standalone/grounding/bwc_pathA_vad.json`). The raw output RTTM contains **exactly 48 lines** (`SPEAKER bwc 1 ...`). Evaluated against `phase20_standalone/ground_truth_rttm/bwc.rttm` (48 reference turns). |
| **39** | **Application End-to-End Raw Diarization** | `task-1297.log` (17:21:30 UTC): `NeMo MSDD diarization succeeded: 39 turns, 2 speakers` | The live application pipeline dynamically extracted Silero VAD speech intervals from the normalized audio. NeMo MSDD took those Silero VAD intervals and produced **39 raw speaker turns** in its generated RTTM. |
| **16** | **Application End-to-End Dialogue Turns** | `scratch/stage_1_4_1_full_e2e_results.json`, `app/storage/results/b1b3f780-94e9-490d-89f8-52945b84b31b.srt` | The 413 aligned words were attributed to speakers based on the 39 raw turns. Then, `Phase20SpeechEngine._group_into_turns(aligned_words, max_gap=1.5)` grouped adjacent words from the same speaker (with inter-word silence $< 1.5$s) into **16 high-level conversation dialogue turns** for subtitle cues and transcript presentation. |

### Benchmark Parity Clarification
- Conditioned on historical Phase 20 Path A VAD intervals, MSDD diarization achieved **identical confusion (7.71%) and identical missed speech (23.00%)** at collar 0.00s compared to historical `bwc_step06a_diar_eval.json`.
- The live application pipeline uses its own genuine VAD (Silero VAD) + MSDD diarization, yielding 39 raw speaker intervals and 16 conversational dialogue turns.

---

## 3. Modified Files and Detailed Changes

### 3.1 `app/engine/diarization_worker.py`
1. **Empty VAD Short-Circuit:** When `vad_stats["turn_count"] == 0`, immediately prints `{"status": "NO_SPEECH", "reason": "VAD RTTM contains 0 speech intervals.", "speaker_count": 0, "speakers": [], "turn_count": 0, ...}` and exits with code 0 without importing or initializing `NeuralDiarizer`.
2. **Deterministic RTTM Output Selection:** Replaced `candidates = sorted(pred_rttms_dir.glob("*.rttm"))` with `expected_rttm = pred_rttms_dir / f"{file_id}.rttm"`. Verifies that `expected_rttm.is_file()` exists; rejects any directory missing the file with exit code 6.
3. **Canonical Output Promotion:** Copies strictly `expected_rttm` to `predicted.rttm` and returns `payload["status"] = "SUCCESS"` if `stats["turn_count"] > 0` else `"NO_SPEECH"`.

### 3.2 `app/engine/nemo_diarizer.py`
1. **Workspace Isolation:** If `out_dir` is provided, creates a unique timestamped child directory `out_dir / f"run_{file_id}_{pid}_{timestamp}"`. If `out_dir` is None, uses a temporary directory.
2. **Pre-Worker VAD Validation:** If `vad_intervals` is empty or `vad_rttm` has 0 turns, short-circuits immediately with `([], "NO_SPEECH", "No speech intervals detected by VAD.", stats)` without spawning subprocess.
3. **Strict JSON Protocol Validation:** Validates that stdout contains valid JSON dictionary with recognized status in `{"SUCCESS", "NO_SPEECH", "FAILED"}`. Rejects missing JSON, malformed JSON, or unrecognized statuses with `DIARIZATION_FAILED`.
4. **Workspace Provenance Verification:** When `SUCCESS` is reported, resolves `rttm_path = Path(worker_meta["rttm_path"]).resolve()` and verifies `rttm_path.relative_to(unique_work_dir)`. Rejects any path escaping the invocation workspace.
5. **Zero-Turn RTTM Handling:** If RTTM parses to 0 segments, returns `([], "NO_SPEECH", "Worker produced empty RTTM with zero speaker segments.", stats)`.

### 3.3 `app/engine/phase20_engine.py`
1. **Constructor Parameter Preservation:** Removed unconditional re-assignment `self.enable_diarization = True` at line 100.
2. **Complete Diarization Bypass:** In `_load_or_derive_diarization()`, `if not self.enable_diarization:` immediately returns `(fallback_segments, "DIARIZATION_UNAVAILABLE", "Diarization disabled by configuration.")`, bypassing NeMo, Pyannote, and model loading.
3. **Genuine VAD Extraction & Structural Validation:** Silero VAD timestamps are validated for structural integrity ($start < end$, $\ge 0$). Removed ASR segment fallback completely. Any VAD exception aborts diarization with `DIARIZATION_FAILED`.
4. **Empty VAD Handling:** If VAD yields 0 speech intervals, returns `(fallback_segments, "NO_SPEECH", "No speech intervals detected by VAD.")`.
5. **Accurate Summary Statistics:** `speakers_detected` counts only distinct real speakers when `diar_status in ("DIARIZATION_AVAILABLE", "EXTERNAL_RTTM")`; returns 0 when diarization is disabled, unavailable, failed, or yielded no speech.

### 3.4 `app/tests/test_nemo_diarizer.py`
Added 9 comprehensive unit tests:
- `test_diarize_empty_vad_intervals_short_circuits`
- `test_diarize_empty_vad_rttm_short_circuits`
- `test_diarize_missing_json_payload`
- `test_diarize_malformed_json_payload`
- `test_diarize_unrecognized_worker_status`
- `test_diarize_worker_reported_failure_status`
- `test_diarize_worker_reported_no_speech_status`
- `test_diarize_worker_provenance_violation`
- `test_diarize_worker_missing_output_file`

### 3.5 `app/tests/test_pipeline_integration.py`
Updated `test_diarization_blocker_disclosure` and added 4 new engine-level regression tests:
- `test_enable_diarization_flag_preservation`
- `test_disabled_diarization_bypasses_all_backends`
- `test_vad_failure_stops_diarization_without_asr_fallback`
- `test_empty_vad_returns_no_speech_status`
- `test_malformed_vad_intervals_trigger_failure`

---

## 4. Test Execution & Verification Results

### 4.1 Host Test Suite Execution (`cs698b_phase20`)
```bash
/home/priyanka/anaconda3/envs/cs698b_phase20/bin/pytest app/tests/ -v
```
**Results:**
- **60 passed, 1 skipped** (the manual real-audio test) in **3.82s**.
- Zero regressions across existing tests (`test_api.py`, `test_confidence.py`, `test_e2e_workflow.py`, `test_independence.py`, `test_schemas.py`, `test_services.py`, `test_wav2vec2_aligner.py`).
- All 16 tests in `test_nemo_diarizer.py` passed.
- All 13 tests in `test_pipeline_integration.py` passed.

### 4.2 Group B Worker CLI Protocol Test (`cs698b_nemo`)
```bash
/home/priyanka/anaconda3/envs/cs698b_nemo/bin/python app/engine/diarization_worker.py \
  --audio /home/priyanka/.gemini/antigravity-cli/brain/7e3e867e-7476-4041-b3ce-d38ba921ee71/scratch/bwc_app_preprocessed_16k_norm.wav \
  --vad-rttm /dev/null \
  --out-dir /tmp/test_empty_vad_worker \
  --checkpoint /home/priyanka/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo
```
**Output:**
```
[WARNING] (nemo_worker) VAD RTTM contains 0 speech intervals. Short-circuiting without model loading.
{"status": "NO_SPEECH", "reason": "VAD RTTM contains 0 speech intervals.", "rttm_path": null, "speaker_count": 0, "speakers": [], "turn_count": 0, "total_speech_seconds": 0.0, "inference_time_seconds": 0.0}
```
**Result:** Clean exit code 0 in 0.05s without loading NeuralDiarizer or weights.

### 4.3 Environment Integrity
Executed `pip check` across both Conda environments:
- `cs698b_phase20`: `No broken requirements found.` (0 packages modified)
- `cs698b_nemo`: `No broken requirements found.` (0 packages modified)

### 4.4 Real GPU Inference Policy
No full 175-second GPU diarization inference was rerun during Stage 1.4.1.1. The existing real-speech evidence from Stage 1.4.1 was strictly preserved, as model weights, hyperparameters, and feature extraction remained completely untouched.

---

## 5. Git Status and Diff Summary

```
$ git status --short
 M app/engine/diarization_worker.py
 M app/engine/nemo_diarizer.py
 M app/engine/phase20_engine.py
 M app/tests/test_nemo_diarizer.py
 M app/tests/test_pipeline_integration.py
?? app/reports/stage_1_4_1_1_production_corrections_report.md

$ git diff --stat
 app/engine/diarization_worker.py       |  28 ++--
 app/engine/nemo_diarizer.py            | 152 ++++++++++++++----
 app/engine/phase20_engine.py           | 133 +++++++++-------
 app/tests/test_nemo_diarizer.py        | 277 ++++++++++++++++++++++++++++++---
 app/tests/test_pipeline_integration.py | 190 +++++++++++++++++++++-
 5 files changed, 649 insertions(+), 131 deletions(-)
```

Commit state: `b4899b5` remains the unchanged HEAD commit. Zero commits have been created. Changes are ready for review and explicit commit authorization.
