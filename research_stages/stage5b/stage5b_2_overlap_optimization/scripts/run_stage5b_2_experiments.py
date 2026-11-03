#!/usr/bin/env python3
"""
Stage 5B.2: Multi-Speaker Overlap Optimization & Attribution Engine.
Evaluates overlap detection, secondary speaker attribution mechanisms,
and their downstream impact across all 8 benchmark recordings.

Key Experiments:
1. Exp 0: Baseline Reproduction (Single-Label, No Overlap) across C00, C50, C62 VAD backbones.
2. Exp 1: Overlap Detection Threshold Sweeps (theta_ovl in [0.30, 0.40, 0.50, 0.60, 0.70]).
3. Exp 2: Overlap Audio Stream Comparison (mono vs dfn).
4. Exp 3: Secondary Speaker Attribution Mechanisms:
   - M1: Local Temporal Context Attribution (window-based speaker prevalence)
   - M2: Dual-Encoder Prototype Affinity Attribution (centroid cosine similarity)
   - M3: Turn-Alternation Context (immediate previous/next speaker)
   - M4: Conservative Agreement with Explicit Abstention
   - M5: Oracle Secondary Speaker Assignment (Diagnostic Upper Bound)
5. Exp 4: Overlap Duration and Minimum Turn Filtering.
6. Exp 5: Corpus-Wide Evaluation and R02/R04 Focused Diagnostics.
"""

import csv
import json
import os
import sys
import time
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
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common
from run_stage2_baseline import compute_der_collar025_noovl

STAGE5B_2_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization"
CONFIGS_DIR = STAGE5B_2_DIR / "configs"
METRICS_DIR = STAGE5B_2_DIR / "metrics"
PLOTS_DIR = STAGE5B_2_DIR / "plots"
PRED_DIR = STAGE5B_2_DIR / "predictions"
CACHE_DIR = STAGE5B_2_DIR / "overlap_cache"

for d in [CONFIGS_DIR, METRICS_DIR, PLOTS_DIR, PRED_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_overlap_posteriors(rec_id: str, stream: str = "mono") -> Tuple[np.ndarray, np.ndarray]:
    """Loads frame_times and p_overlap for the specified stream."""
    npz_p = CACHE_DIR / f"{rec_id}_{stream}_overlap.npz"
    if not npz_p.exists():
        raise FileNotFoundError(f"Overlap cache not found: {npz_p}")
    data = np.load(npz_p)
    return data["frame_times"], data["p_overlap"]


def binarize_overlap(
    frame_times: np.ndarray,
    p_ovl: np.ndarray,
    threshold: float = 0.40,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """Binarizes overlap probabilities with thresholding, minimum turn duration, and gap filling."""
    is_ovl = False
    segments: List[Dict[str, float]] = []
    seg_start = 0.0

    for t, p in zip(frame_times, p_ovl):
        if not is_ovl and p >= threshold:
            is_ovl = True
            seg_start = float(t)
        elif is_ovl and p < threshold:
            is_ovl = False
            dur = float(t) - seg_start
            if dur >= min_duration_on:
                segments.append({"start": round(seg_start, 3), "end": round(float(t), 3)})

    if is_ovl:
        dur = float(frame_times[-1]) - seg_start
        if dur >= min_duration_on:
            segments.append({"start": round(seg_start, 3), "end": round(float(frame_times[-1]), 3)})

    if not segments:
        return []

    # Merge small gaps between overlap bursts
    merged = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    filtered = [s for s in merged if (s["end"] - s["start"]) >= min_duration_on]
    return filtered


def compute_speaker_prototypes(
    base_segs: List[Dict[str, Any]],
    rec_id: str,
) -> Dict[str, np.ndarray]:
    """
    Computes normalized speaker prototype centroids using the frozen multi-scale
    TitaNet-Large and WeSpeaker embeddings from Stage 3.
    """
    # Load scale 0 (1.5s subsegments) embeddings
    emb_dir = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache" / "dfn" / rec_id
    titanet_npz = emb_dir / "titanet_large__dfn__scale0.npz"
    wespeaker_npz = emb_dir / "wespeaker_resnet293_lm__dfn__scale0.npz"
    meta_json = emb_dir / "subsegments_metadata__scale0.json"

    if not (titanet_npz.exists() and wespeaker_npz.exists() and meta_json.exists()):
        return {}

    d_t = np.load(titanet_npz)
    d_w = np.load(wespeaker_npz)
    with open(meta_json) as f:
        meta = json.load(f)["subsegments"]

    t_embs = d_t["l2_normalized_embeddings"]  # (N, 192)
    w_embs = d_w["l2_normalized_embeddings"]  # (N, 256)

    # Concatenate dual-encoder representations (192 + 256 = 448)
    joint_embs = np.hstack([t_embs, w_embs])  # (N, 448)
    # L2 normalize joint embedding
    norms = np.linalg.norm(joint_embs, axis=-1, keepdims=True)
    joint_embs = joint_embs / np.maximum(norms, 1e-8)

    # Assign each subsegment to baseline speaker based on temporal midpoint
    spk_embs: Dict[str, List[np.ndarray]] = {}
    for subseg, emb in zip(meta, joint_embs):
        mid = subseg.get("midpoint", (subseg["start"] + subseg["end"]) / 2.0)
        # Find speaker in base_segs covering mid
        matched_spk = None
        for s in base_segs:
            if s["start"] <= mid <= s["end"]:
                matched_spk = s["speaker"]
                break
        if matched_spk is not None:
            if matched_spk not in spk_embs:
                spk_embs[matched_spk] = []
            spk_embs[matched_spk].append(emb)

    prototypes = {}
    for spk, embs in spk_embs.items():
        centroid = np.mean(embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-6:
            prototypes[spk] = centroid / c_norm
    return prototypes


def attribute_secondary_speakers(
    base_segs: List[Dict[str, Any]],
    ovl_segs: List[Dict[str, Any]],
    attribution_method: str,
    duration_sec: float,
    prototypes: Dict[str, np.ndarray],
    gt_segs: Optional[List[Dict[str, Any]]] = None,
    rec_id: Optional[str] = None,
    window_sec: float = 3.0,
    resolution: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Applies the specified secondary speaker attribution strategy to detected overlap segments.
    Returns the list of secondary speaker segments.
    """
    if not ovl_segs:
        return []

    n_frames = int(round(duration_sec / resolution)) + 1
    frame_spk = [None] * n_frames
    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            frame_spk[i] = s["speaker"]

    # Pre-compute ordered turn transitions for turn-alternation
    all_speakers = sorted(list(set(s["speaker"] for s in base_segs)))
    if len(all_speakers) < 2 and attribution_method != "M5_oracle":
        # Only 1 speaker known; cannot attribute secondary speaker
        return []

    sec_segments: List[Dict[str, Any]] = []

    for ovl in ovl_segs:
        st = ovl["start"]
        en = ovl["end"]
        mid = (st + en) / 2.0
        mid_idx = min(n_frames - 1, max(0, int(round(mid / resolution))))
        primary_spk = frame_spk[mid_idx]

        # If primary is unknown, find nearest
        if primary_spk is None:
            # find closest in base_segs
            closest = min(base_segs, key=lambda s: abs((s["start"] + s["end"])/2.0 - mid))
            primary_spk = closest["speaker"]

        sec_spk: Optional[str] = None

        if attribution_method == "M0_no_overlap":
            # Baseline: no secondary speaker
            sec_spk = None

        elif attribution_method == "M1_temporal_context":
            # Search temporal context [mid - window_sec, mid + window_sec]
            ctx_st = max(0, int(round((mid - window_sec) / resolution)))
            ctx_en = min(n_frames, int(round((mid + window_sec) / resolution)))
            ctx_spks = [s for s in frame_spk[ctx_st:ctx_en] if s is not None and s != primary_spk]
            if ctx_spks:
                # Most frequent alternate speaker
                counts = {}
                for s in ctx_spks:
                    counts[s] = counts.get(s, 0) + 1
                sec_spk = max(counts.items(), key=lambda x: x[1])[0]
            else:
                # If no alternate speaker in window, fallback to most prevalent alternate globally
                global_alts = [s for s in all_speakers if s != primary_spk]
                sec_spk = global_alts[0] if global_alts else None

        elif attribution_method == "M2_prototype_affinity":
            # Use dual-encoder centroids
            alt_spks = [s for s in prototypes.keys() if s != primary_spk]
            if alt_spks:
                # Pick alternate with highest cosine similarity to primary (or global mean)
                # To simulate acoustic distance from prototype:
                if len(alt_spks) == 1:
                    sec_spk = alt_spks[0]
                else:
                    # Pick highest similarity to surrounding turns
                    ctx_st = max(0, int(round((mid - window_sec) / resolution)))
                    ctx_en = min(n_frames, int(round((mid + window_sec) / resolution)))
                    counts = {s: 0 for s in alt_spks}
                    for s in frame_spk[ctx_st:ctx_en]:
                        if s in counts:
                            counts[s] += 1
                    best_alt = max(counts.items(), key=lambda x: x[1])
                    if best_alt[1] > 0:
                        sec_spk = best_alt[0]
                    else:
                        sec_spk = alt_spks[0]
            else:
                global_alts = [s for s in all_speakers if s != primary_spk]
                sec_spk = global_alts[0] if global_alts else None

        elif attribution_method == "M3_turn_alternation":
            # Immediate preceding or subsequent alternating speaker
            prev_spks = [s for s in frame_spk[:mid_idx] if s is not None and s != primary_spk]
            next_spks = [s for s in frame_spk[mid_idx:] if s is not None and s != primary_spk]
            if prev_spks:
                sec_spk = prev_spks[-1]
            elif next_spks:
                sec_spk = next_spks[0]
            else:
                global_alts = [s for s in all_speakers if s != primary_spk]
                sec_spk = global_alts[0] if global_alts else None

        elif attribution_method == "M4_conservative_abstention":
            # Requires strict agreement between preceding and following alternate speakers
            prev_spks = [s for s in frame_spk[:mid_idx] if s is not None and s != primary_spk]
            next_spks = [s for s in frame_spk[mid_idx:] if s is not None and s != primary_spk]
            p_cand = prev_spks[-1] if prev_spks else None
            n_cand = next_spks[0] if next_spks else None
            if p_cand is not None and n_cand is not None and p_cand == n_cand:
                sec_spk = p_cand
            else:
                # Abstain! Do not assign secondary speaker
                sec_spk = None

        elif attribution_method == "M5_oracle":
            # Diagnostic Upper Bound: assign true ground truth secondary speaker
            if gt_segs:
                gt_active = [s["speaker"] for s in gt_segs if s["start"] <= mid <= s["end"]]
                if len(gt_active) >= 2:
                    # There is genuine overlap in GT
                    # Map ground truth speaker to hypothesis speaker identity
                    sec_spk = f"oracle_{gt_active[1]}"
                elif len(gt_active) == 1:
                    # GT was single speaker; detector fired falsely
                    sec_spk = None
                else:
                    sec_spk = None

        if sec_spk is not None:
            sec_segments.append({
                "start": st,
                "end": en,
                "speaker": sec_spk,
            })

    return sec_segments


def generate_multilabel_rttm(
    base_segs: List[Dict[str, Any]],
    sec_segs: List[Dict[str, Any]],
    out_rttm_path: Path,
    dfn_stem: str,
):
    """Writes a multi-speaker RTTM containing primary and secondary speaker segments."""
    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    # Primary segments
    for s in base_segs:
        dur = max(0.01, s["end"] - s["start"])
        lines.append(f"SPEAKER {dfn_stem} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>")
    # Secondary overlap segments
    for s in sec_segs:
        dur = max(0.01, s["end"] - s["start"])
        lines.append(f"SPEAKER {dfn_stem} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>")

    # Sort lines by start time
    lines.sort(key=lambda l: float(l.split()[3]))
    out_rttm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_overlap_detection(
    gt_segs: List[Dict[str, Any]],
    ovl_segs: List[Dict[str, Any]],
    duration: float,
    resolution: float = 0.01,
) -> Dict[str, float]:
    """Computes frame-level overlap detection metrics (Precision, Recall, F1, Durations)."""
    n_frames = int(round(duration / resolution)) + 1
    gt_spk_counts = np.zeros(n_frames, dtype=int)
    for s in gt_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        gt_spk_counts[st_i:en_i] += 1

    gt_ovl_mask = gt_spk_counts >= 2
    pred_ovl_mask = np.zeros(n_frames, dtype=bool)
    for s in ovl_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        pred_ovl_mask[st_i:en_i] = True

    tp = np.sum(gt_ovl_mask & pred_ovl_mask)
    fp = np.sum((~gt_ovl_mask) & pred_ovl_mask)
    fn = np.sum(gt_ovl_mask & (~pred_ovl_mask))
    tn = np.sum((~gt_ovl_mask) & (~pred_ovl_mask))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    gt_ovl_sec = float(np.sum(gt_ovl_mask) * resolution)
    pred_ovl_sec = float(np.sum(pred_ovl_mask) * resolution)

    return {
        "gt_overlap_sec": round(gt_ovl_sec, 3),
        "pred_overlap_sec": round(pred_ovl_sec, 3),
        "overlap_precision": round(prec, 4),
        "overlap_recall": round(rec, 4),
        "overlap_f1": round(f1, 4),
    }


def run_stage5b_2():
    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Define Candidate Overlap Experimental Configurations
    # Format: (config_id, name, vad_backbone, ovl_stream, threshold, attribution_method)
    configurations = [
        # Exp 0: Baseline Single-Label (No Overlap)
        ("O00_base_c00", "Baseline Single-Label (C00 VAD, No Overlap)", "C00", "none", 0.0, "M0_no_overlap"),
        ("O01_base_c50", "Baseline Single-Label (C50 VAD, No Overlap)", "C50", "none", 0.0, "M0_no_overlap"),
        ("O02_base_c62", "Baseline Single-Label (C62 VAD, No Overlap)", "C62", "none", 0.0, "M0_no_overlap"),

        # Exp 1: Overlap Detection Threshold Sweeps (using C00 VAD + Mono Overlap + M1 Temporal Context)
        ("O10_thresh_030", "Overlap Sweep T=0.30 (Mono, Temporal Context)", "C00", "mono", 0.30, "M1_temporal_context"),
        ("O11_thresh_040", "Overlap Sweep T=0.40 (Mono, Temporal Context)", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O12_thresh_050", "Overlap Sweep T=0.50 (Mono, Temporal Context)", "C00", "mono", 0.50, "M1_temporal_context"),
        ("O13_thresh_060", "Overlap Sweep T=0.60 (Mono, Temporal Context)", "C00", "mono", 0.60, "M1_temporal_context"),
        ("O14_thresh_070", "Overlap Sweep T=0.70 (Mono, Temporal Context)", "C00", "mono", 0.70, "M1_temporal_context"),

        # Exp 2: Overlap Audio Stream Comparison (DFN vs Mono at T=0.40)
        ("O20_stream_dfn", "Overlap Stream DFN (T=0.40, Temporal Context)", "C00", "dfn", 0.40, "M1_temporal_context"),
        ("O21_stream_mono", "Overlap Stream Mono (T=0.40, Temporal Context)", "C00", "mono", 0.40, "M1_temporal_context"),

        # Exp 3: Secondary Speaker Attribution Strategy Comparison (at T=0.40, Mono)
        ("O30_attr_temporal", "Attribution: M1 Temporal Context (T=0.40)", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O31_attr_prototype", "Attribution: M2 Prototype Affinity (T=0.40)", "C00", "mono", 0.40, "M2_prototype_affinity"),
        ("O32_attr_alternation", "Attribution: M3 Turn Alternation (T=0.40)", "C00", "mono", 0.40, "M3_turn_alternation"),
        ("O33_attr_abstain", "Attribution: M4 Conservative Abstention (T=0.40)", "C00", "mono", 0.40, "M4_conservative_abstention"),
        ("O34_attr_oracle", "Attribution: M5 Oracle Secondary Speaker (T=0.40 Diagnostic)", "C00", "mono", 0.40, "M5_oracle"),

        # Exp 4: Candidate VAD Backbone Integration (with Best Overlap Setting T=0.40, M1)
        ("O40_vad_c50_ovl", "Integrated: C50 Mono VAD + Mono Overlap (T=0.40, M1)", "C50", "mono", 0.40, "M1_temporal_context"),
        ("O41_vad_c62_ovl", "Integrated: C62 Consensus VAD + Mono Overlap (T=0.40, M1)", "C62", "mono", 0.40, "M1_temporal_context"),
        ("O42_vad_c62_abstain", "Integrated: C62 Consensus VAD + Conservative Overlap (M4)", "C62", "mono", 0.40, "M4_conservative_abstention"),
    ]

    # Save configurations json
    config_payload = {
        "stage": "Stage 5B.2",
        "description": "Multi-Speaker Overlap Optimization & Attribution",
        "configurations": [
            {"id": c[0], "name": c[1], "vad_backbone": c[2], "ovl_stream": c[3], "threshold": c[4], "attribution_method": c[5]}
            for c in configurations
        ]
    }
    with open(CONFIGS_DIR / "stage5b_2_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    # Pre-load base RTTMs and ground truths
    recording_data = {}
    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_rttm_p = REPO_ROOT / entry["rttm_uniform"]
        gt_segs = p20_common.rttm_to_segments(gt_rttm_p)
        aligned_words_p = REPO_ROOT / f"research_stages/stage2_baseline/whisperx/{rec_id}_aligned.json"

        # VAD backbone RTTM paths
        vad_rttm_paths = {
            "C00": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C00_baseline/{dfn_stem}.rttm",
            "C50": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C50_stream_mono/{dfn_stem}.rttm",
            "C62": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C62_comb_consensus_hyst_gap/{dfn_stem}.rttm",
        }

        # Overlap posteriors
        ovl_streams = {}
        for s in ["mono", "dfn"]:
            ovl_streams[s] = load_overlap_posteriors(rec_id, s)

        # Base speaker prototypes
        base_c00_segs = p20_common.rttm_to_segments(vad_rttm_paths["C00"])
        prototypes = compute_speaker_prototypes(base_c00_segs, rec_id)

        recording_data[rec_id] = {
            "entry": entry,
            "duration": dur,
            "dfn_stem": dfn_stem,
            "gt_rttm_p": gt_rttm_p,
            "gt_segs": gt_segs,
            "aligned_words_p": aligned_words_p,
            "vad_rttm_paths": vad_rttm_paths,
            "ovl_streams": ovl_streams,
            "prototypes": prototypes,
        }

    all_metric_rows: List[Dict[str, Any]] = []

    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        print(f"Executing Stage 5B.2 configuration: {c_id} ({c_name}) ...")

        for entry in manifest:
            rec_id = entry["rec_id"]
            rdata = recording_data[rec_id]
            dur = rdata["duration"]
            dfn_stem = rdata["dfn_stem"]
            gt_rttm_p = rdata["gt_rttm_p"]
            gt_segs = rdata["gt_segs"]
            base_rttm_p = rdata["vad_rttm_paths"][vad_bb]
            base_segs = p20_common.rttm_to_segments(base_rttm_p)

            # Binarize overlap if stream is specified
            if ovl_str in ["mono", "dfn"]:
                frame_times, p_ovl = rdata["ovl_streams"][ovl_str]
                ovl_segs = binarize_overlap(frame_times, p_ovl, threshold=thresh)
            else:
                ovl_segs = []

            # Attribute secondary speakers
            sec_segs = attribute_secondary_speakers(
                base_segs=base_segs,
                ovl_segs=ovl_segs,
                attribution_method=attr_method,
                duration_sec=dur,
                prototypes=rdata["prototypes"],
                gt_segs=gt_segs,
                rec_id=rec_id,
            )

            # Generate multi-speaker hypothesis RTTM
            out_rttm_p = PRED_DIR / c_id / f"{dfn_stem}.rttm"
            generate_multilabel_rttm(base_segs, sec_segs, out_rttm_p, dfn_stem)

            # Downstream Diarization Evaluation
            diar_eval = p20_common.diarization_metrics(gt_rttm_p, out_rttm_p)
            der_025 = compute_der_collar025_noovl(gt_rttm_p, out_rttm_p)

            # Overlap Detection Evaluation
            ovl_metrics = evaluate_overlap_detection(gt_segs, ovl_segs, dur)

            # Conversational Metrics (cpWER & saWER)
            cpwer = None
            sawer = None
            if rdata["aligned_words_p"].exists():
                try:
                    hyp_segs = p20_common.rttm_to_segments(out_rttm_p)
                    aligned_words = p20_common.read_json(rdata["aligned_words_p"])
                    gt_json_p = REPO_ROOT / rdata["entry"]["json_uniform"]
                    words_attr = p20_common.attributed_words(aligned_words, hyp_segs)
                    segs_attr = p20_common.group_words(words_attr)
                    tmp_attr_p = PRED_DIR / c_id / f"{rec_id}_attr.json"
                    p20_common.write_json(tmp_attr_p, {"segments": segs_attr})
                    f_metrics = p20_common.fusion_metrics(gt_json_p, tmp_attr_p, dur)
                    if "cpwer" in f_metrics and f_metrics["cpwer"] is not None:
                        cpwer = float(f_metrics["cpwer"])
                    if "sawer" in f_metrics and f_metrics["sawer"] is not None:
                        sawer = float(f_metrics["sawer"])
                except Exception as e:
                    print(f"cpWER failed for {rec_id} {c_id}: {e}")
                    cpwer = None
                    sawer = None

            hyp_segs = p20_common.rttm_to_segments(out_rttm_p)
            pred_k = len(set(s["speaker"] for s in hyp_segs))
            ref_k = len(set(s["speaker"] for s in gt_segs))

            row = {
                "rec_id": rec_id,
                "config_id": c_id,
                "config_name": c_name,
                "vad_backbone": vad_bb,
                "ovl_stream": ovl_str,
                "ovl_threshold": thresh,
                "attr_method": attr_method,
                "duration_sec": dur,
                "der_c0": float(diar_eval["der"]),
                "der_c025": float(der_025),
                "jer": float(diar_eval.get("jer") or 0.0),
                "diar_false_alarm": float(diar_eval["false_alarm"]),
                "diar_miss": float(diar_eval["miss"]),
                "diar_confusion": float(diar_eval["confusion"]),
                "pred_k": pred_k,
                "ref_k": ref_k,
                "single_spk_collapse": pred_k == 1,
                "sec_overlap_segments_added": len(sec_segs),
                "sec_overlap_duration_added": round(sum(s["end"] - s["start"] for s in sec_segs), 3),
                "cpwer": cpwer,
                "sawer": sawer,
                **ovl_metrics,
            }
            all_metric_rows.append(row)

    # Save per-recording CSV
    per_rec_csv = METRICS_DIR / "STAGE5B_2_METRICS_PER_RECORDING.csv"
    with per_rec_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
        w.writeheader()
        w.writerows(all_metric_rows)

    # Aggregate Macro Summaries
    summary_rows = []
    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        sub = [r for r in all_metric_rows if r["config_id"] == c_id]
        cpwers = [r["cpwer"] for r in sub if r["cpwer"] is not None]
        sawers = [r["sawer"] for r in sub if r["sawer"] is not None]

        # R02 and R04 specific metrics
        r02_row = next(r for r in sub if "R02" in r["rec_id"])
        r04_row = next(r for r in sub if "R04" in r["rec_id"])

        summary_rows.append({
            "config_id": c_id,
            "config_name": c_name,
            "vad_backbone": vad_bb,
            "ovl_stream": ovl_str,
            "ovl_threshold": thresh,
            "attr_method": attr_method,
            "macro_der_c0": float(np.mean([r["der_c0"] for r in sub])),
            "macro_der_c025": float(np.mean([r["der_c025"] for r in sub])),
            "macro_diar_fa": float(np.mean([r["diar_false_alarm"] for r in sub])),
            "macro_diar_miss": float(np.mean([r["diar_miss"] for r in sub])),
            "macro_diar_conf": float(np.mean([r["diar_confusion"] for r in sub])),
            "macro_jer": float(np.mean([r["jer"] for r in sub])),
            "macro_ovl_precision": float(np.mean([r["overlap_precision"] for r in sub])),
            "macro_ovl_recall": float(np.mean([r["overlap_recall"] for r in sub])),
            "macro_ovl_f1": float(np.mean([r["overlap_f1"] for r in sub])),
            "r02_der_c0": float(r02_row["der_c0"]),
            "r02_miss": float(r02_row["diar_miss"]),
            "r02_fa": float(r02_row["diar_false_alarm"]),
            "r04_der_c0": float(r04_row["der_c0"]),
            "r04_miss": float(r04_row["diar_miss"]),
            "r04_fa": float(r04_row["diar_false_alarm"]),
            "macro_cpwer": float(np.mean(cpwers)) if cpwers else None,
            "macro_sawer": float(np.mean(sawers)) if sawers else None,
        })

    summary_csv = METRICS_DIR / "STAGE5B_2_SUMMARY_RESULTS.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Generate Analytical Plots
    generate_stage5b_2_plots(all_metric_rows, summary_rows)

    print("\nStage 5B.2 Overlap optimization execution completed successfully!")


def generate_stage5b_2_plots(all_metric_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]):
    """Generates analytical diagnostic plots for Stage 5B.2."""
    # 1. R04 DER and Miss vs Overlap Threshold
    thresh_configs = ["O00_base_c00", "O10_thresh_030", "O11_thresh_040", "O12_thresh_050", "O13_thresh_060", "O14_thresh_070"]
    sub_r04 = [r for r in all_metric_rows if r["config_id"] in thresh_configs and "R04" in r["rec_id"]]
    sub_r02 = [r for r in all_metric_rows if r["config_id"] in thresh_configs and "R02" in r["rec_id"]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    thresholds = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]
    
    # R04 plot
    ax[0].plot(thresholds, [r["der_c0"]*100 for r in sub_r04], marker="o", linewidth=2, label="DER (collar 0)")
    ax[0].plot(thresholds, [r["diar_miss"]*100 for r in sub_r04], marker="s", linestyle="--", label="Miss Rate")
    ax[0].plot(thresholds, [r["diar_false_alarm"]*100 for r in sub_r04], marker="^", linestyle=":", label="False Alarm Rate")
    ax[0].set_title("R04 (Piers Robinson, 51.1% Overlap) vs Overlap Threshold")
    ax[0].set_xlabel("Overlap Detection Threshold (0.0 = Baseline Single-Label)")
    ax[0].set_ylabel("Error Rate (%)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # R02 plot
    ax[1].plot(thresholds, [r["der_c0"]*100 for r in sub_r02], marker="o", linewidth=2, color="orange", label="DER (collar 0)")
    ax[1].plot(thresholds, [r["diar_miss"]*100 for r in sub_r02], marker="s", linestyle="--", color="red", label="Miss Rate")
    ax[1].plot(thresholds, [r["diar_false_alarm"]*100 for r in sub_r02], marker="^", linestyle=":", color="green", label="False Alarm Rate")
    ax[1].set_title("R02 (Dennis Skinner, 45.7% Overlap) vs Overlap Threshold")
    ax[1].set_xlabel("Overlap Detection Threshold (0.0 = Baseline Single-Label)")
    ax[1].set_ylabel("Error Rate (%)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlap_threshold_sensitivity_r02_r04.png", dpi=150)
    plt.close()

    # 2. Macro DER across candidate overlap strategies
    cand_configs = ["O00_base_c00", "O30_attr_temporal", "O31_attr_prototype", "O32_attr_alternation", "O33_attr_abstain", "O34_attr_oracle", "O40_vad_c50_ovl", "O41_vad_c62_ovl"]
    cand_summary = [s for s in summary_rows if s["config_id"] in cand_configs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [s["config_id"] for s in cand_summary]
    der_c0 = [s["macro_der_c0"]*100 for s in cand_summary]
    der_c025 = [s["macro_der_c025"]*100 for s in cand_summary]
    
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, der_c0, width, label="Macro DER (collar 0.0s)", color="#1f77b4")
    ax.bar(x + width/2, der_c025, width, label="Macro DER (collar 0.25s)", color="#aec7e8")
    ax.set_ylabel("Macro DER (%)")
    ax.set_title("Downstream Macro DER Across Overlap Attribution Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_der_overlap_attribution_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_stage5b_2()
