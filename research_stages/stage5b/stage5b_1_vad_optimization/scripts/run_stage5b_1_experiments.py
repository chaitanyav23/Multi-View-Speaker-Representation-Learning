#!/usr/bin/env python3
"""
Stage 5B.1: Voice Activity Detection (VAD) Optimization & Diarization Impact.
Controlled experimental evaluation across all 8 benchmark recordings:
1. Exp 0: Baseline Reproduction (Path A, onset=0.481, offset=0.481, min_on=0.10, min_off=0.15)
2. Exp 1: Short-Turn Preservation Sweep (min_on in [0.05, 0.10, 0.20, 0.30, 0.50]s)
3. Exp 2: Threshold Optimization Sweep (T in [0.30, 0.40, 0.481, 0.55, 0.60, 0.70, 0.80])
4. Exp 3: Hysteresis Dual-Threshold Sweep (onset/offset pairs)
5. Exp 4: Gap-Filling & Minimum Silence Sweep (min_off in [0.05, 0.15, 0.30, 0.50]s)
6. Exp 5: Input Audio Stream & Consensus (dfn vs mono vs pathB vs consensus)
7. Exp 6: Combined Configurations & Ablations
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage3_dual_encoder"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common
from run_stage2_baseline import compute_der_collar025_noovl

STAGE5B_1_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_1_vad_optimization"
CONFIGS_DIR = STAGE5B_1_DIR / "configs"
METRICS_DIR = STAGE5B_1_DIR / "metrics"
PLOTS_DIR = STAGE5B_1_DIR / "plots"
PRED_DIR = STAGE5B_1_DIR / "predictions"

for d in [CONFIGS_DIR, METRICS_DIR, PLOTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_posteriors(rec_id: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads frame_times and p_speech for dfn, 16k_mono, and normalized streams."""
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

    # Gap-filling: merge segments separated by less than min_duration_off
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    # Post-filtering by minimum duration after merging
    filtered = [s for s in merged if (s["end"] - s["start"]) >= min_duration_on]
    return filtered


def evaluate_vad(
    gt_rttm_segs: List[Dict[str, Any]],
    vad_segs: List[Dict[str, Any]],
    duration: float,
    resolution: float = 0.01,
) -> Dict[str, Any]:
    """Computes comprehensive frame-level VAD metrics."""
    res = p20_common.vad_metrics(gt_rttm_segs, vad_segs, duration, resolution=resolution)
    total_gt_speech = sum(s["end"] - s["start"] for s in gt_rttm_segs)
    total_pred_speech = sum(s["end"] - s["start"] for s in vad_segs)

    fp_sec = res["fp_frames"] * resolution
    fn_sec = res["fn_frames"] * resolution

    # Dual False Alarm and Miss reporting:
    # 1. Standard VAD (silence-normalized FA)
    # 2. NIST/DER-aligned (reference-speech-normalized FA)
    fa_silence = res["false_alarm"]
    fa_ref = fp_sec / total_gt_speech if total_gt_speech > 0 else 0.0
    miss_ref = fn_sec / total_gt_speech if total_gt_speech > 0 else 0.0

    return {
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "f1": float(res["f1"]),
        "miss_ref": float(miss_ref),
        "fa_silence": float(fa_silence),
        "fa_ref": float(fa_ref),
        "total_gt_speech": float(total_gt_speech),
        "total_pred_speech": float(total_pred_speech),
        "speech_coverage": float(total_pred_speech / duration) if duration > 0 else 0.0,
        "segment_count": len(vad_segs),
        "mean_segment_dur": float(np.mean([s["end"] - s["start"] for s in vad_segs])) if vad_segs else 0.0,
    }


def project_vad_to_diarization(
    vad_segs: List[Dict[str, Any]],
    base_rttm_path: Path,
    out_rttm_path: Path,
    dfn_stem: str,
    duration_sec: float,
    resolution: float = 0.01,
) -> Path:
    """
    Constructs an updated hypothesis RTTM by conditioning the frozen M1 cluster assignments
    on the new VAD speech intervals. This precisely evaluates downstream DER impact.
    """
    base_segs = p20_common.rttm_to_segments(base_rttm_path)
    n_frames = int(round(duration_sec / resolution)) + 1

    # Map baseline speaker labels to time frames
    frame_spk = [None] * n_frames
    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            frame_spk[i] = s["speaker"]

    # Vectorized nearest neighbor fill using searchsorted
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

    # Now apply the new VAD mask
    active_mask = np.zeros(n_frames, dtype=bool)
    for vs in vad_segs:
        st_i = max(0, int(round(vs["start"] / resolution)))
        en_i = min(n_frames, int(round(vs["end"] / resolution)))
        active_mask[st_i:en_i] = True

    # Build continuous RTTM segments
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
                # Speaker change
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

    if in_turn and cur_spk is not None:
        turn_dur = max(0.01, duration_sec - turn_st)
        rttm_lines.append(f"SPEAKER {dfn_stem} 1 {turn_st:.3f} {turn_dur:.3f} <NA> <NA> {cur_spk} <NA> <NA>")

    out_rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    return out_rttm_path


def run_all_experiments():
    print("[Stage 5B.1] Initializing VAD Optimization & Diarization Impact Experiments...")

    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Define experimental conditions
    # Each condition specifies: (exp_id, exp_name, stream, onset, offset, min_on, min_off)
    configurations = [
        # Exp 0: Baseline
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

    all_vad_rows: List[Dict[str, Any]] = []
    all_diar_rows: List[Dict[str, Any]] = []

    # Pre-load posteriors and ground truths
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

        # Precompute consensus stream
        if "dfn" in streams and "mono" in streams:
            t_dfn, p_dfn = streams["dfn"]
            t_mono, p_mono = streams["mono"]
            min_len = min(len(p_dfn), len(p_mono))
            p_joint = 0.5 * p_dfn[:min_len] + 0.5 * p_mono[:min_len]
            streams["consensus"] = (t_dfn[:min_len], p_joint)

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

    # Execute all configurations
    for c_id, c_name, stream_key, onset, offset, min_on, min_off in configurations:
        print(f"Executing {c_id:25s} [{c_name}]...")
        for rec_id, rdata in recording_data.items():
            dur = rdata["duration"]
            gt_rttm_segs = rdata["gt_rttm_segs"]
            gt_rttm_path = rdata["gt_rttm_path"]
            base_rttm_path = rdata["base_rttm_path"]
            dfn_stem = rdata["dfn_stem"]

            times, probs = rdata["streams"][stream_key]
            v_segs = binarize_posteriors(
                times=times,
                probs=probs,
                onset=onset,
                offset=offset,
                min_duration_on=min_on,
                min_duration_off=min_off,
            )

            vad_metrics = evaluate_vad(gt_rttm_segs, v_segs, dur, resolution=0.01)
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
                **vad_metrics,
            }
            all_vad_rows.append(vad_row)

            # Downstream Diarization Evaluation
            out_rttm = PRED_DIR / c_id / f"{dfn_stem}.rttm"
            project_vad_to_diarization(v_segs, base_rttm_path, out_rttm, dfn_stem, dur)

            diar_eval = p20_common.diarization_metrics(gt_rttm_path, out_rttm)
            der_025 = compute_der_collar025_noovl(gt_rttm_path, out_rttm)

            diar_segs = p20_common.rttm_to_segments(out_rttm)
            pred_spks = len(set(s["speaker"] for s in diar_segs))
            ref_spks = len(set(s["speaker"] for s in gt_rttm_segs))

            # Conversational metrics (cpWER) where WhisperX words exist
            cpwer = None
            if rdata["aligned_words_p"].exists():
                try:
                    aligned_words = p20_common.read_json(rdata["aligned_words_p"])
                    gt_json_path = REPO_ROOT / rdata["entry"]["json_uniform"]
                    words_attr = p20_common.attributed_words(aligned_words, diar_segs)
                    segs_attr = p20_common.group_words(words_attr)
                    tmp_attr_path = PRED_DIR / c_id / f"{rec_id}_attr.json"
                    p20_common.write_json(tmp_attr_path, segs_attr)
                    f_metrics = p20_common.fusion_metrics(gt_json_path, tmp_attr_path, dur)
                    cpwer = float(f_metrics.get("cpwer", 0.0))
                except Exception:
                    cpwer = None

            diar_row = {
                "rec_id": rec_id,
                "config_id": c_id,
                "config_name": c_name,
                "der_c0": float(diar_eval["der"]),
                "der_c025": float(der_025),
                "jer": float(diar_eval.get("jer") or 0.0),
                "false_alarm": float(diar_eval["false_alarm"]),
                "miss": float(diar_eval["miss"]),
                "confusion": float(diar_eval["confusion"]),
                "ref_k": ref_spks,
                "pred_k": pred_spks,
                "single_spk_collapse": pred_spks == 1,
                "cpwer": cpwer,
            }
            all_diar_rows.append(diar_row)

    # Save CSVs
    vad_csv_p = STAGE5B_1_DIR / "STAGE5B_1_VAD_METRICS_PER_RECORDING.csv"
    with vad_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_vad_rows[0].keys()))
        w.writeheader()
        w.writerows(all_vad_rows)

    diar_csv_p = STAGE5B_1_DIR / "STAGE5B_1_DOWNSTREAM_METRICS_PER_RECORDING.csv"
    with diar_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_diar_rows[0].keys()))
        w.writeheader()
        w.writerows(all_diar_rows)

    # Aggregate Macro Summaries across Configurations
    summary_rows = []
    for c_id, c_name, stream_key, onset, offset, min_on, min_off in configurations:
        v_sub = [r for r in all_vad_rows if r["config_id"] == c_id]
        d_sub = [r for r in all_diar_rows if r["config_id"] == c_id]

        total_dur = sum(r["duration_sec"] for r in v_sub)
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
            "vad_macro_miss_ref": float(np.mean([r["miss_ref"] for r in v_sub])),
            "vad_macro_fa_ref": float(np.mean([r["fa_ref"] for r in v_sub])),
            "vad_macro_fa_silence": float(np.mean([r["fa_silence"] for r in v_sub])),
            "diar_macro_der_c0": float(np.mean([r["der_c0"] for r in d_sub])),
            "diar_macro_der_c025": float(np.mean([r["der_c025"] for r in d_sub])),
            "diar_macro_fa": float(np.mean([r["false_alarm"] for r in d_sub])),
            "diar_macro_miss": float(np.mean([r["miss"] for r in d_sub])),
            "diar_macro_conf": float(np.mean([r["confusion"] for r in d_sub])),
            "diar_macro_jer": float(np.mean([r["jer"] for r in d_sub])),
            "macro_cpwer": float(np.mean(cpwers)) if cpwers else None,
            "single_spk_collapse_count": sum(1 for r in d_sub if r["single_spk_collapse"]),
        })

    summary_csv_p = METRICS_DIR / "stage5b_1_summary_comparison.csv"
    with summary_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    sens_csv_p = STAGE5B_1_DIR / "STAGE5B_1_PARAMETER_SENSITIVITY.csv"
    with sens_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Print Summary Table
    print("\n" + "=" * 115)
    print("STAGE 5B.1 VAD OPTIMIZATION & DIARIZATION IMPACT: SUMMARY")
    print("=" * 115)
    print(f"{'Config ID':26s} | {'VAD F1':7s} | {'VAD Recall':10s} | {'VAD Prec':8s} | {'DER (c=0)':9s} | {'DER (c=0.25)':12s} | {'Diar FA':8s} | {'Diar Miss':9s} | {'Diar Conf':9s}")
    print("-" * 115)
    for s in summary_rows:
        print(f"{s['config_id']:26s} | {s['vad_macro_f1']*100:6.2f}% | {s['vad_macro_recall']*100:9.2f}% | {s['vad_macro_precision']*100:7.2f}% | {s['diar_macro_der_c0']*100:8.2f}% | {s['diar_macro_der_c025']*100:11.2f}% | {s['diar_macro_fa']*100:7.2f}% | {s['diar_macro_miss']*100:8.2f}% | {s['diar_macro_conf']*100:8.2f}%")
    print("=" * 115)

    # Generate Analytical Plots
    generate_plots(summary_rows, all_vad_rows, all_diar_rows)


def generate_plots(summary_rows: List[Dict[str, Any]], all_vad: List[Dict[str, Any]], all_diar: List[Dict[str, Any]]):
    """Generates publication-quality analytical plots for Stage 5B.1."""
    print("[Plots] Generating Stage 5B.1 analytical figures...")

    # Plot 1: Precision-Recall Curve across Threshold Sweeps
    thresh_configs = [s for s in summary_rows if s["config_id"].startswith("C2") or s["config_id"] == "C00_baseline"]
    thresh_configs.sort(key=lambda x: x["onset"])

    plt.figure(figsize=(8, 6))
    recalls = [s["vad_macro_recall"] * 100 for s in thresh_configs]
    precisions = [s["vad_macro_precision"] * 100 for s in thresh_configs]
    ders = [s["diar_macro_der_c0"] * 100 for s in thresh_configs]
    labels = [f"T={s['onset']:.2f}" for s in thresh_configs]

    plt.plot(recalls, precisions, "o-", color="navy", lw=2, label="Threshold Sweep (T=0.30 to 0.80)")
    for r, p, lbl in zip(recalls, precisions, labels):
        plt.annotate(lbl, (r, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.xlabel("Macro Recall (%)", fontsize=11)
    plt.ylabel("Macro Precision (%)", fontsize=11)
    plt.title("Stage 5B.1: VAD Precision-Recall Trade-off across Speech Probability Thresholds", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vad_precision_recall_tradeoff.png", dpi=200)
    plt.close()

    # Plot 2: Macro DER vs VAD Configuration
    plt.figure(figsize=(12, 6))
    c_names = [s["config_id"] for s in summary_rows]
    der_c0 = [s["diar_macro_der_c0"] * 100 for s in summary_rows]
    der_c025 = [s["diar_macro_der_c025"] * 100 for s in summary_rows]

    x = np.arange(len(c_names))
    width = 0.35

    plt.bar(x - width/2, der_c0, width, label="DER (Collar=0.0)", color="crimson", alpha=0.85)
    plt.bar(x + width/2, der_c025, width, label="DER (Collar=0.25, noovl)", color="royalblue", alpha=0.85)
    plt.axhline(der_c0[0], color="black", linestyle="--", alpha=0.7, label=f"Baseline (c=0: {der_c0[0]:.2f}%)")

    plt.ylabel("Macro Diarization Error Rate (%)", fontsize=11)
    plt.title("Macro DER across All Tested VAD Configurations", fontsize=12)
    plt.xticks(x, c_names, rotation=45, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_der_vs_vad_config.png", dpi=200)
    plt.close()

    # Plot 3: False Alarm vs Missed Detection Error Stack
    plt.figure(figsize=(12, 6))
    fa_vals = [s["diar_macro_fa"] * 100 for s in summary_rows]
    miss_vals = [s["diar_macro_miss"] * 100 for s in summary_rows]
    conf_vals = [s["diar_macro_conf"] * 100 for s in summary_rows]

    plt.bar(c_names, fa_vals, label="False Alarm Rate (%)", color="#e74c3c", alpha=0.85)
    plt.bar(c_names, miss_vals, bottom=fa_vals, label="Missed Speech Rate (%)", color="#3498db", alpha=0.85)
    bottom_conf = np.array(fa_vals) + np.array(miss_vals)
    plt.bar(c_names, conf_vals, bottom=bottom_conf, label="Speaker Confusion (%)", color="#2ecc71", alpha=0.85)

    plt.ylabel("Error Component Contribution to DER (%)", fontsize=11)
    plt.title("Decomposition of Diarization Error (FA, Miss, Confusion) across VAD Configurations", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "der_error_decomposition.png", dpi=200)
    plt.close()

    # Plot 4: Per-Recording False Alarm Comparison (R05 and R06 focus)
    plt.figure(figsize=(10, 6))
    recs = ["R01_sara_sharif", "R02_dennis_skinner", "R03_disabled_parking", "R04_piers_robinson",
            "R05_southwest_pilot", "R06_rich_karen", "R07_bwc_cheap_shot", "R08_uk_parliament"]
    base_fa = [r["false_alarm"] * 100 for r in all_diar if r["config_id"] == "C00_baseline"]
    # Best anti-noise configuration (C63 or C31)
    opt_fa = [r["false_alarm"] * 100 for r in all_diar if r["config_id"] == "C63_comb_anti_noise"]

    xr = np.arange(len(recs))
    plt.bar(xr - 0.18, base_fa, 0.35, label="Baseline VAD FA (%)", color="#e74c3c")
    plt.bar(xr + 0.18, opt_fa, 0.35, label="Noise-Suppressed VAD FA (%)", color="#27ae60")
    plt.ylabel("False Alarm Error Rate (%)", fontsize=11)
    plt.title("Per-Recording VAD False Alarm Rate: Baseline vs Anti-Noise VAD", fontsize=12)
    plt.xticks(xr, [r[:12] for r in recs], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_recording_false_alarm_reduction.png", dpi=200)
    plt.close()

    print("[Plots] Finished saving all figures to plots/ directory.")


if __name__ == "__main__":
    run_all_experiments()
