#!/usr/bin/env python3
"""
Stage 5B.1: Corrective Evaluation Script.
Executes the corrected VAD evaluation, projected DER audit, and cpWER computation
in the isolated audit_corrections/ workspace.

Key corrections implemented:
1. Binary VAD Reference Duration: Uses exact union of reference speech intervals (T_union)
   so overlapping speakers are not double-counted in VAD miss and FA denominators.
2. Downstream DER: Preserved with explicit labeling as "VAD-mask-projected DER under
   frozen baseline speaker assignments", and records NN-propagated duration & percentage.
3. cpWER: Corrects the JSON payload schema (wrapped with {"segments": segs_attr})
   to compute authentic cpWER without fallback.
4. Timestamp & Consensus Verification: Explicitly confirms bitwise timestamp equality.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common
from run_stage2_baseline import compute_der_collar025_noovl

AUDIT_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_1_vad_optimization" / "audit_corrections"
METRICS_DIR = AUDIT_DIR / "metrics"
PRED_DIR = AUDIT_DIR / "predictions"
CONFIGS_DIR = AUDIT_DIR / "configs"

for d in [METRICS_DIR, PRED_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_posteriors(rec_id: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads frame_times and p_speech for dfn, mono, and normalized streams."""
    base_dir = REPO_ROOT / "research_stages" / "stage2_baseline" / "posteriors"
    streams = {}
    for stream_name, suffix in [("dfn", "_dfn_posteriors.npz"), ("mono", "_16k_mono_posteriors.npz"), ("normalized", "_normalized_posteriors.npz")]:
        npz_p = base_dir / f"{rec_id}{suffix}"
        if npz_p.exists():
            data = np.load(npz_p)
            streams[stream_name] = (data["frame_times"], data["p_speech"])
    return streams


def binarize_posteriors(
    times: np.ndarray,
    probs: np.ndarray,
    onset: float = 0.481,
    offset: float = 0.481,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """Binarizes continuous probabilities with hysteresis thresholds, minimum turn duration, and gap filling."""
    is_speech = False
    segments: List[Dict[str, float]] = []
    seg_start = 0.0

    for t, p in zip(times, probs):
        if not is_speech and p >= onset:
            is_speech = True
            seg_start = float(t)
        elif is_speech and p < offset:
            is_speech = False
            dur = float(t) - seg_start
            if dur >= min_duration_on:
                segments.append({"start": round(seg_start, 3), "end": round(float(t), 3)})

    if is_speech:
        dur = float(times[-1]) - seg_start
        if dur >= min_duration_on:
            segments.append({"start": round(seg_start, 3), "end": round(float(times[-1]), 3)})

    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    filtered = [s for s in merged if (s["end"] - s["start"]) >= min_duration_on]
    return filtered


def evaluate_vad_corrected(
    gt_rttm_segs: List[Dict[str, Any]],
    vad_segs: List[Dict[str, Any]],
    duration: float,
    resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Computes rigorous frame-level VAD metrics with both:
    1. Union reference speech duration (the correct binary VAD denominator).
    2. Sum of speaker segment durations (for comparison and reconciliation).
    """
    res = p20_common.vad_metrics(gt_rttm_segs, vad_segs, duration, resolution=resolution)

    # 1. Union reference speech
    ref_mask = p20_common.activity_mask(gt_rttm_segs, duration, resolution)
    union_gt_frames = sum(ref_mask)
    union_gt_speech = union_gt_frames * resolution
    total_silence_sec = (len(ref_mask) - union_gt_frames) * resolution

    # 2. Sum of speaker segments (legacy uncorrected)
    sum_gt_speech = sum(s["end"] - s["start"] for s in gt_rttm_segs)
    total_pred_speech = sum(s["end"] - s["start"] for s in vad_segs)

    fp_sec = res["fp_frames"] * resolution
    fn_sec = res["fn_frames"] * resolution

    # Correct Union-normalized metrics
    miss_union_rate = fn_sec / union_gt_speech if union_gt_speech > 0 else 0.0
    fa_union_rate = fp_sec / union_gt_speech if union_gt_speech > 0 else 0.0
    fa_silence_rate = res["false_alarm"]

    # Legacy sum-normalized metrics (for exact reconciliation)
    miss_sum_rate = fn_sec / sum_gt_speech if sum_gt_speech > 0 else 0.0
    fa_sum_rate = fp_sec / sum_gt_speech if sum_gt_speech > 0 else 0.0

    return {
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "f1": float(res["f1"]),
        "union_gt_speech_sec": float(round(union_gt_speech, 3)),
        "sum_gt_speech_sec": float(round(sum_gt_speech, 3)),
        "pred_speech_sec": float(round(total_pred_speech, 3)),
        "miss_sec": float(round(fn_sec, 3)),
        "fa_sec": float(round(fp_sec, 3)),
        "miss_union_rate": float(miss_union_rate),
        "fa_union_rate": float(fa_union_rate),
        "miss_sum_rate_legacy": float(miss_sum_rate),
        "fa_sum_rate_legacy": float(fa_sum_rate),
        "fa_silence_rate": float(fa_silence_rate),
        "speech_coverage": float(total_pred_speech / duration) if duration > 0 else 0.0,
        "segment_count": len(vad_segs),
    }


def project_vad_to_diarization_audited(
    vad_segs: List[Dict[str, Any]],
    base_rttm_path: Path,
    out_rttm_path: Path,
    dfn_stem: str,
    duration_sec: float,
    resolution: float = 0.01,
) -> Tuple[Path, float, float]:
    """
    Constructs an updated hypothesis RTTM by conditioning the frozen M1 cluster assignments
    on the new VAD speech intervals. Returns output path and nearest-neighbor propagation metrics.
    """
    base_segs = p20_common.rttm_to_segments(base_rttm_path)
    n_frames = int(round(duration_sec / resolution)) + 1

    frame_spk = [None] * n_frames
    base_active_mask = np.zeros(n_frames, dtype=bool)
    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        base_active_mask[st_i:en_i] = True
        for i in range(st_i, en_i):
            frame_spk[i] = s["speaker"]

    # Vectorized nearest neighbor fill
    known_indices = np.array([i for i, spk in enumerate(frame_spk) if spk is not None], dtype=np.int64)
    if len(known_indices) > 0:
        missing_indices = np.array([i for i, spk in enumerate(frame_spk) if spk is None], dtype=np.int64)
        if len(missing_indices) > 0:
            idx = np.searchsorted(known_indices, missing_indices)
            idx = np.clip(idx, 0, len(known_indices) - 1)
            left_idx = np.clip(idx - 1, 0, len(known_indices) - 1)
            dist_right = np.abs(known_indices[idx] - missing_indices)
            dist_left = np.abs(known_indices[left_idx] - missing_indices)
            best_idx = np.where(dist_left < dist_right, left_idx, idx)
            for m_i, b_i in zip(missing_indices, best_idx):
                frame_spk[m_i] = frame_spk[known_indices[b_i]]
    else:
        frame_spk = ["speaker_0"] * n_frames

    # Apply new VAD mask
    active_mask = np.zeros(n_frames, dtype=bool)
    for vs in vad_segs:
        st_i = max(0, int(round(vs["start"] / resolution)))
        en_i = min(n_frames, int(round(vs["end"] / resolution)))
        active_mask[st_i:en_i] = True

    # Quantify nearest-neighbor propagation
    nn_propagated_frames = active_mask & (~base_active_mask)
    total_active_frames = np.sum(active_mask)
    nn_propagated_sec = float(np.sum(nn_propagated_frames) * resolution)
    total_active_sec = float(total_active_frames * resolution)
    nn_propagated_pct = float((nn_propagated_sec / total_active_sec * 100.0) if total_active_sec > 0 else 0.0)

    # Build RTTM segments
    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
    rttm_lines = []
    in_turn = False
    cur_spk = None
    turn_st = 0.0

    for i in range(n_frames):
        t = i * resolution
        is_active = active_mask[i]
        spk = frame_spk[i]

        if is_active:
            if not in_turn:
                in_turn = True
                cur_spk = spk
                turn_st = t
            elif spk != cur_spk:
                turn_dur = max(0.01, t - turn_st)
                rttm_lines.append(f"SPEAKER {dfn_stem} 1 {turn_st:.3f} {turn_dur:.3f} <NA> <NA> {cur_spk} <NA> <NA>")
                cur_spk = spk
                turn_st = t
        else:
            if in_turn:
                in_turn = False
                turn_dur = max(0.01, t - turn_st)
                rttm_lines.append(f"SPEAKER {dfn_stem} 1 {turn_st:.3f} {turn_dur:.3f} <NA> <NA> {cur_spk} <NA> <NA>")
                cur_spk = None

    if in_turn:
        turn_dur = max(0.01, (n_frames * resolution) - turn_st)
        rttm_lines.append(f"SPEAKER {dfn_stem} 1 {turn_st:.3f} {turn_dur:.3f} <NA> <NA> {cur_spk} <NA> <NA>")

    out_rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    return out_rttm_path, nn_propagated_sec, nn_propagated_pct


def run_corrections():
    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    configurations = [
        # Exp 0: Baseline reproduction
        ("C00_baseline", "Baseline Pyannote PathA (DFN, T=0.481)", "dfn", 0.481, 0.481, 0.10, 0.15),
        # Exp 1: Short-turn preservation sweep
        ("C10_min_on_005", "Short-Turn min_dur=0.05s", "dfn", 0.481, 0.481, 0.05, 0.15),
        ("C11_min_on_020", "Short-Turn min_dur=0.20s", "dfn", 0.481, 0.481, 0.20, 0.15),
        ("C12_min_on_030", "Short-Turn min_dur=0.30s", "dfn", 0.481, 0.481, 0.30, 0.15),
        ("C13_min_on_050", "Short-Turn min_dur=0.50s (Legacy)", "dfn", 0.481, 0.481, 0.50, 0.15),
        # Exp 2: Threshold optimization sweep
        ("C20_thresh_030", "Single Threshold T=0.30", "dfn", 0.300, 0.300, 0.10, 0.15),
        ("C21_thresh_040", "Single Threshold T=0.40", "dfn", 0.400, 0.400, 0.10, 0.15),
        ("C22_thresh_055", "Single Threshold T=0.55", "dfn", 0.550, 0.550, 0.10, 0.15),
        ("C23_thresh_060", "Single Threshold T=0.60", "dfn", 0.600, 0.600, 0.10, 0.15),
        ("C24_thresh_070", "Single Threshold T=0.70", "dfn", 0.700, 0.700, 0.10, 0.15),
        ("C25_thresh_080", "Single Threshold T=0.80", "dfn", 0.800, 0.800, 0.10, 0.15),
        # Exp 3: Hysteresis dual-threshold sweep
        ("C30_hyst_55_40", "Hysteresis onset=0.55, offset=0.40", "dfn", 0.550, 0.400, 0.10, 0.15),
        ("C31_hyst_60_35", "Hysteresis onset=0.60, offset=0.35", "dfn", 0.600, 0.350, 0.10, 0.15),
        ("C32_hyst_65_35", "Hysteresis onset=0.65, offset=0.35", "dfn", 0.650, 0.350, 0.10, 0.15),
        ("C33_hyst_70_40", "Hysteresis onset=0.70, offset=0.40", "dfn", 0.700, 0.400, 0.10, 0.15),
        ("C34_hyst_70_30", "Hysteresis onset=0.70, offset=0.30", "dfn", 0.700, 0.300, 0.10, 0.15),
        # Exp 4: Gap-filling & minimum silence sweep
        ("C40_gap_005", "Gap Filling merge_gap=0.05s", "dfn", 0.481, 0.481, 0.10, 0.05),
        ("C41_gap_030", "Gap Filling merge_gap=0.30s", "dfn", 0.481, 0.481, 0.10, 0.30),
        ("C42_gap_050", "Gap Filling merge_gap=0.50s", "dfn", 0.481, 0.481, 0.10, 0.50),
        # Exp 5: Input audio stream comparison
        ("C50_stream_mono", "Stream: Raw 16k Mono (T=0.481)", "mono", 0.481, 0.481, 0.10, 0.15),
        ("C51_stream_norm", "Stream: Normalized PathB (T=0.481)", "normalized", 0.481, 0.481, 0.10, 0.15),
        ("C52_consensus", "Stream: Linear Consensus (DFN + Mono)", "consensus", 0.481, 0.481, 0.10, 0.15),
        # Exp 6: Combined configurations
        ("C60_comb_hyst_gap", "Combined: Hysteresis (0.60/0.35) + Gap 0.30s", "dfn", 0.600, 0.350, 0.10, 0.30),
        ("C61_comb_consensus_hyst", "Combined: Consensus + Hysteresis (0.60/0.35)", "consensus", 0.600, 0.350, 0.10, 0.15),
        ("C62_comb_consensus_hyst_gap", "Combined: Consensus + Hysteresis (0.60/0.35) + Gap 0.30s", "consensus", 0.600, 0.350, 0.10, 0.30),
        ("C63_comb_anti_noise", "Combined Noise Suppressor: Hysteresis (0.70/0.40) + Turn 0.15s", "dfn", 0.700, 0.400, 0.15, 0.15),
    ]

    all_vad_rows = []
    all_diar_rows = []

    recording_data = {}
    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_rttm_path = REPO_ROOT / entry["rttm_uniform"]
        gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
        base_rttm_path = REPO_ROOT / f"research_stages/stage5a_1/predictions/rttms/condition_J_norm_equal_weight_fusion/{dfn_stem}.rttm"
        aligned_words_p = REPO_ROOT / f"research_stages/stage2_baseline/whisperx/{rec_id}_aligned.json"

        streams = load_posteriors(rec_id)

        # Audit consensus alignment
        if "dfn" in streams and "mono" in streams:
            t_dfn, p_dfn = streams["dfn"]
            t_mono, p_mono = streams["mono"]
            assert len(t_dfn) == len(t_mono), f"Length mismatch in {rec_id}"
            assert np.array_equal(t_dfn, t_mono), f"Timestamp mismatch in {rec_id}"
            p_joint = 0.5 * p_dfn + 0.5 * p_mono
            streams["consensus"] = (t_dfn, p_joint)

        recording_data[rec_id] = {
            "entry": entry,
            "duration": dur,
            "dfn_stem": dfn_stem,
            "gt_rttm_path": gt_rttm_path,
            "gt_rttm_segs": gt_rttm_segs,
            "base_rttm_path": base_rttm_path,
            "aligned_words_p": aligned_words_p,
            "streams": streams,
        }

    for c_id, c_name, stream_key, onset, offset, min_on, min_off in configurations:
        print(f"Executing corrected evaluation for: {c_id} ...")

        for entry in manifest:
            rec_id = entry["rec_id"]
            rdata = recording_data[rec_id]
            dur = rdata["duration"]
            dfn_stem = rdata["dfn_stem"]
            gt_rttm_path = rdata["gt_rttm_path"]
            gt_rttm_segs = rdata["gt_rttm_segs"]
            base_rttm_path = rdata["base_rttm_path"]

            times, probs = rdata["streams"][stream_key]
            v_segs = binarize_posteriors(times, probs, onset, offset, min_on, min_off)
            vad_res = evaluate_vad_corrected(gt_rttm_segs, v_segs, dur)

            vad_row = {
                "rec_id": rec_id,
                "config_id": c_id,
                "config_name": c_name,
                "stream": stream_key,
                "onset": onset,
                "offset": offset,
                "min_duration_on": min_on,
                "min_duration_off": min_off,
                "duration_sec": dur,
                **vad_res,
            }
            all_vad_rows.append(vad_row)

            # Projected Diarization Evaluation
            out_rttm = PRED_DIR / c_id / f"{dfn_stem}.rttm"
            _, nn_sec, nn_pct = project_vad_to_diarization_audited(v_segs, base_rttm_path, out_rttm, dfn_stem, dur)

            diar_eval = p20_common.diarization_metrics(gt_rttm_path, out_rttm)
            der_025 = compute_der_collar025_noovl(gt_rttm_path, out_rttm)

            diar_segs = p20_common.rttm_to_segments(out_rttm)
            pred_spks = len(set(s["speaker"] for s in diar_segs))
            ref_spks = len(set(s["speaker"] for s in gt_rttm_segs))

            # Correct cpWER computation with properly wrapped dictionary payload
            cpwer = None
            if rdata["aligned_words_p"].exists():
                try:
                    aligned_words = p20_common.read_json(rdata["aligned_words_p"])
                    gt_json_path = REPO_ROOT / rdata["entry"]["json_uniform"]
                    words_attr = p20_common.attributed_words(aligned_words, diar_segs)
                    segs_attr = p20_common.group_words(words_attr)
                    tmp_attr_path = PRED_DIR / c_id / f"{rec_id}_attr.json"
                    
                    # Exact schema fix: wrap segments list inside dictionary
                    wrapped_payload = {"segments": segs_attr}
                    p20_common.write_json(tmp_attr_path, wrapped_payload)
                    
                    f_metrics = p20_common.fusion_metrics(gt_json_path, tmp_attr_path, dur)
                    if "cpwer" in f_metrics and f_metrics["cpwer"] is not None:
                        cpwer = float(f_metrics["cpwer"])
                    else:
                        cpwer = None
                except Exception as e:
                    print(f"cpWER computation failed for {rec_id} {c_id}: {e}")
                    cpwer = None

            diar_row = {
                "rec_id": rec_id,
                "config_id": c_id,
                "config_name": c_name,
                "projected_der_c0": float(diar_eval["der"]),
                "projected_der_c025": float(der_025),
                "jer": float(diar_eval.get("jer") or 0.0),
                "diar_false_alarm": float(diar_eval["false_alarm"]),
                "diar_miss": float(diar_eval["miss"]),
                "diar_confusion": float(diar_eval["confusion"]),
                "ref_k": ref_spks,
                "pred_k": pred_spks,
                "single_spk_collapse": pred_spks == 1,
                "nn_propagated_sec": round(nn_sec, 3),
                "nn_propagated_pct": round(nn_pct, 2),
                "cpwer": cpwer,
            }
            all_diar_rows.append(diar_row)

    # Save corrected per-recording CSVs
    vad_csv_p = METRICS_DIR / "STAGE5B_1_CORRECTED_VAD_METRICS_PER_RECORDING.csv"
    with vad_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_vad_rows[0].keys()))
        w.writeheader()
        w.writerows(all_vad_rows)

    diar_csv_p = METRICS_DIR / "STAGE5B_1_CORRECTED_DOWNSTREAM_METRICS_PER_RECORDING.csv"
    with diar_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_diar_rows[0].keys()))
        w.writeheader()
        w.writerows(all_diar_rows)

    # Aggregate corrected summary
    summary_rows = []
    for c_id, c_name, stream_key, onset, offset, min_on, min_off in configurations:
        v_sub = [r for r in all_vad_rows if r["config_id"] == c_id]
        d_sub = [r for r in all_diar_rows if r["config_id"] == c_id]

        cpwers = [r["cpwer"] for r in d_sub if r["cpwer"] is not None]

        summary_rows.append({
            "config_id": c_id,
            "config_name": c_name,
            "stream": stream_key,
            "onset": onset,
            "offset": offset,
            "min_on": min_on,
            "min_off": min_off,
            "vad_macro_precision": float(np.mean([r["precision"] for r in v_sub])),
            "vad_macro_recall": float(np.mean([r["recall"] for r in v_sub])),
            "vad_macro_f1": float(np.mean([r["f1"] for r in v_sub])),
            "vad_macro_miss_union": float(np.mean([r["miss_union_rate"] for r in v_sub])),
            "vad_macro_fa_union": float(np.mean([r["fa_union_rate"] for r in v_sub])),
            "vad_macro_miss_sum_legacy": float(np.mean([r["miss_sum_rate_legacy"] for r in v_sub])),
            "vad_macro_fa_sum_legacy": float(np.mean([r["fa_sum_rate_legacy"] for r in v_sub])),
            "vad_macro_fa_silence": float(np.mean([r["fa_silence_rate"] for r in v_sub])),
            "projected_macro_der_c0": float(np.mean([r["projected_der_c0"] for r in d_sub])),
            "projected_macro_der_c025": float(np.mean([r["projected_der_c025"] for r in d_sub])),
            "diar_macro_fa": float(np.mean([r["diar_false_alarm"] for r in d_sub])),
            "diar_macro_miss": float(np.mean([r["diar_miss"] for r in d_sub])),
            "diar_macro_conf": float(np.mean([r["diar_confusion"] for r in d_sub])),
            "diar_macro_jer": float(np.mean([r["jer"] for r in d_sub])),
            "macro_nn_propagated_pct": float(np.mean([r["nn_propagated_pct"] for r in d_sub])),
            "macro_cpwer": float(np.mean(cpwers)) if cpwers else None,
            "single_spk_collapse_count": sum(1 for r in d_sub if r["single_spk_collapse"]),
        })

    summary_csv_p = METRICS_DIR / "STAGE5B_1_CORRECTED_SUMMARY_COMPARISON.csv"
    with summary_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\nCorrected evaluation completed successfully across all 26 configurations!")


if __name__ == "__main__":
    run_corrections()
