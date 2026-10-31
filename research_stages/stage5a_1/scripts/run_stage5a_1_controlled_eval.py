#!/usr/bin/env python3
"""
Stage 5A.1 Controlled Evaluation Script:
Implements fair affinity-normalization controls (H: normalized TitaNet, I: normalized ResNet,
J: normalized equal fusion, K: normalized reliability gate) alongside historical conditions (A, B, C),
evaluates downstream diarization, WhisperX word attribution, LORO verification, error complementarity,
and paired clustered bootstrap uncertainty.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage1_data_prep"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage3_dual_encoder"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common  # noqa: E402
from run_stage2_baseline import compute_der_collar025_noovl, compute_macro_sacc  # noqa: E402
from run_stage4_complementarity import SCALES, compute_eer_and_mindcf, load_cache  # noqa: E402
from fusion_models import ReliabilityGate, offdiag_zscore_affinity  # noqa: E402

from nemo.collections.asr.parts.utils.offline_clustering import (  # noqa: E402
    SpeakerClustering,
    getMultiScaleCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.speaker_utils import (  # noqa: E402
    generate_cluster_labels,
    labels_to_rttmfile,
)

STAGE5A_1_DIR = REPO_ROOT / "research_stages" / "stage5a_1"
STAGE2_DIR = REPO_ROOT / "research_stages" / "stage2_baseline"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def assign_subsegments_to_gt_turns(
    meta: List[Dict[str, Any]], gt_segments: List[Dict[str, Any]], min_purity: float = 0.75
) -> List[Optional[int]]:
    clean_gt = [g for g in gt_segments if g.get("usable_for_clean_speaker_eval", False)]
    assigned: List[Optional[int]] = []
    for m in meta:
        lbl = str(m["gt_label"])
        if m["gt_is_overlap"] or lbl.startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS")):
            assigned.append(None)
            continue
        st = float(m["start"])
        en = float(m["end"])
        dur = max(en - st, 1e-6)
        best_seg_idx = None
        best_ov = 0.0
        for g in clean_gt:
            gst = float(g["start"])
            gen = float(g["end_clipped_to_audio"])
            ov = min(en, gen) - max(st, gst)
            if ov > best_ov:
                best_ov = ov
                best_seg_idx = int(g["seg_idx"])
        if best_seg_idx is not None and (best_ov / dur) >= min_purity:
            assigned.append(best_seg_idx)
        else:
            assigned.append(None)
    return assigned


def build_hybrid_turn_records(
    manifest: List[Dict[str, Any]], scale_k: int = 4
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    audit_summary: Dict[str, Any] = {"total_gt_turns": 0, "usable_clean_turns": 0}

    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_payload = p20_common.read_json(REPO_ROOT / entry["json_uniform"])
        gt_segments = gt_payload["segments"]

        t_d, _, sub_meta_tita = load_cache("dfn", rec_id, scale_k)
        _, r_d, _ = load_cache("16k_mono", rec_id, scale_k)

        t_norm = t_d["l2_normalized_embeddings"]
        r_norm = r_d["l2_normalized_embeddings"]

        clean_gt = [g for g in gt_segments if g.get("usable_for_clean_speaker_eval", False)]
        audit_summary["total_gt_turns"] += len(gt_segments)
        audit_summary["usable_clean_turns"] += len(clean_gt)

        for g in clean_gt:
            gst = float(g["start"])
            gen = float(g["end_clipped_to_audio"])
            idxs = np.where(
                np.array([
                    (float(m["start"]) >= gst - 0.05)
                    and (float(m["end"]) <= gen + 0.05)
                    and not m["gt_is_overlap"]
                    and (m["gt_label"] == g["speaker"])
                    for m in sub_meta_tita
                ])
            )[0]

            if idxs.size == 0:
                continue

            t_vec = np.mean(t_norm[idxs], axis=0)
            r_vec = np.mean(r_norm[idxs], axis=0)

            t_vec = (t_vec / max(float(np.linalg.norm(t_vec)), 1e-12)).astype(np.float32)
            r_vec = (r_vec / max(float(np.linalg.norm(r_vec)), 1e-12)).astype(np.float32)

            turns.append({
                "rec_id": rec_id,
                "seg_idx": g["seg_idx"],
                "speaker": g["speaker"],
                "global_speaker_id": g["global_speaker_id"],
                "start": gst,
                "end": gen,
                "duration": round(gen - gst, 3),
                "num_subsegments_pooled": int(idxs.size),
                "resnet293_l2_norm": float(np.mean([sub_meta_tita[i]["resnet293_l2_norm"] for i in idxs])),
                "dfn_noise_reduction_db": float(np.mean([sub_meta_tita[i]["dfn_noise_reduction_db"] for i in idxs])),
                "vad_conf_mean": float(np.mean([sub_meta_tita[i]["vad_conf_mean"] for i in idxs])),
                "tita_emb": t_vec,
                "res_emb": r_vec,
            })
    return turns, audit_summary


def eval_at_frozen_threshold(scores: np.ndarray, targets: np.ndarray, thr: float) -> Dict[str, Any]:
    pos_mask = targets == 1
    neg_mask = targets == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return {
            "frozen_train_thr": float(thr),
            "far_at_frozen_thr": None,
            "frr_at_frozen_thr": None,
            "hter_at_frozen_thr": None,
            "roc_auc": None,
        }

    preds = scores >= thr
    far = float(np.sum(preds[neg_mask])) / n_neg
    frr = float(np.sum(~preds[pos_mask])) / n_pos
    hter = 0.5 * (far + frr)
    auc = float(roc_auc_score(targets, scores))

    return {
        "frozen_train_thr": float(thr),
        "far_at_frozen_thr": round(far, 4),
        "frr_at_frozen_thr": round(frr, 4),
        "hter_at_frozen_thr": round(hter, 4),
        "roc_auc": round(auc, 4),
    }


def compute_act_dcf(far: float, frr: float, p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    cost = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    return float(cost / c_def)


def main():
    print("=" * 80)
    print("Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit")
    print("=" * 80)

    # 1. Hardware & Environment Check
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"Git Commit Hash: {commit}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config_p = REPO_ROOT / "research_stages" / "stage5a" / "configs" / "stage5a_config.json"
    cfg = json.loads(config_p.read_text(encoding="utf-8"))
    manifest = json.loads((REPO_ROOT / cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    recordings = cfg["recordings"]

    # Save Stage 5A.1 specific configuration
    stage5a_1_cfg = {
        "stage": "5a_1",
        "name": "Affinity Normalization, Fusion Fairness & Implementation Audit",
        "base_commit": commit,
        "seed": 42,
        "recordings": recordings,
        "conditions": {
            "A": "condition_A_raw_titanet_dfn",
            "B": "condition_B_raw_resnet293_mono",
            "C": "condition_C_norm_equal_weight_stage5a",
            "H": "condition_H_norm_titanet_dfn_alone",
            "I": "condition_I_norm_resnet293_mono_alone",
            "J": "condition_J_norm_equal_weight_fusion",
            "K": "condition_K_norm_reliability_gate",
            "K_sig": "condition_K_sigmoid_gate",
        },
        "multiscale_weights": [1.0, 2.0, 3.0, 3.0, 3.0],
        "clustering_parameters": cfg["clustering_parameters"],
        "bootstrap": {"n_resamples": 2000, "confidence_level": 0.95, "seed": 42},
    }
    (STAGE5A_1_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (STAGE5A_1_DIR / "configs" / "stage5a_1_config.json").write_text(json.dumps(stage5a_1_cfg, indent=2), encoding="utf-8")

    # 2. Extract Turn-Disjoint LOTO Dataset
    print("\n[Stage 5A.1] Extracting Turn-Disjoint LOTO clean subsegments for hybrid gate training...")
    loto_subseg_items: List[Dict[str, Any]] = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
        t_d, _, meta_dfn = load_cache("dfn", rec_id, 4)
        _, r_d, meta_mono = load_cache("16k_mono", rec_id, 4)

        t_norm = t_d["l2_normalized_embeddings"]
        r_norm = r_d["l2_normalized_embeddings"]

        assigned_turns = assign_subsegments_to_gt_turns(meta_dfn, gt_segs, min_purity=0.75)
        clean_idxs = [i for i, seg_idx in enumerate(assigned_turns) if seg_idx is not None]

        spk_to_turn_idxs: Dict[str, Dict[int, List[int]]] = {}
        for i in clean_idxs:
            spk = str(meta_dfn[i]["gt_label"])
            t_id = assigned_turns[i]
            assert t_id is not None
            spk_to_turn_idxs.setdefault(spk, {}).setdefault(t_id, []).append(i)

        multi_turn_spks = sorted([s for s, t_dict in spk_to_turn_idxs.items() if len(t_dict) >= 2])
        if len(multi_turn_spks) < 2:
            continue

        t_cent_full = {}
        r_cent_full = {}
        for s in multi_turn_spks:
            all_s_ix = [idx for ix_list in spk_to_turn_idxs[s].values() for idx in ix_list]
            t_mu = np.mean(t_norm[all_s_ix], axis=0)
            r_mu = np.mean(r_norm[all_s_ix], axis=0)
            t_cent_full[s] = t_mu / max(float(np.linalg.norm(t_mu)), 1e-12)
            r_cent_full[s] = r_mu / max(float(np.linalg.norm(r_mu)), 1e-12)

        for s in multi_turn_spks:
            for t_id, turn_ixs in spk_to_turn_idxs[s].items():
                pos_ixs = [idx for t2, ix_list in spk_to_turn_idxs[s].items() if t2 != t_id for idx in ix_list]
                assert len(pos_ixs) > 0
                t_pos_mu = np.mean(t_norm[pos_ixs], axis=0)
                r_pos_mu = np.mean(r_norm[pos_ixs], axis=0)
                t_pos_proto = t_pos_mu / max(float(np.linalg.norm(t_pos_mu)), 1e-12)
                r_pos_proto = r_pos_mu / max(float(np.linalg.norm(r_pos_mu)), 1e-12)

                other_spks = [o for o in multi_turn_spks if o != s]

                for i in turn_ixs:
                    t_vec = t_norm[i]
                    r_vec = r_norm[i]

                    t_pos_cos = float(np.dot(t_vec, t_pos_proto))
                    r_pos_cos = float(np.dot(r_vec, r_pos_proto))

                    t_neg_cos = max(float(np.dot(t_vec, t_cent_full[o])) for o in other_spks)
                    r_neg_cos = max(float(np.dot(r_vec, r_cent_full[o])) for o in other_spks)

                    tm = t_pos_cos - t_neg_cos
                    rm = r_pos_cos - r_neg_cos
                    delta_m = rm - tm

                    loto_subseg_items.append({
                        "rec_id": rec_id,
                        "subseg_idx": i,
                        "turn_id": t_id,
                        "speaker": s,
                        "delta_margin": float(delta_m),
                        "res_margin": float(rm),
                        "tita_margin": float(tm),
                        "resnet293_l2_norm": float(meta_mono[i]["resnet293_l2_norm"]),
                        "dfn_noise_reduction_db": float(meta_dfn[i]["dfn_noise_reduction_db"]),
                        "vad_conf_mean": float(meta_mono[i]["vad_conf_mean"]),
                        "res_wins": int(rm > tm),
                    })

    print(f"[Stage 5A.1] Extracted {len(loto_subseg_items)} clean turn-disjoint LOTO subsegments across eligible recordings.")

    # 3. Fit Gate Models under 8-Fold LORO
    gate_features = ["resnet293_l2_norm", "dfn_noise_reduction_db", "vad_conf_mean"]
    gate_models_linear: Dict[str, ReliabilityGate] = {}

    for held_rec in recordings:
        tr_items = [x for x in loto_subseg_items if x["rec_id"] != held_rec]
        gate = ReliabilityGate(features=gate_features, regularization_lambda=0.05, clip_min=0.15, clip_max=0.85, gain_factor=2.0)
        gate.fit(tr_items)
        gate_models_linear[held_rec] = gate

    # 4. Downstream Diarization & WhisperX Attribution for All Conditions
    print("\n[Stage 5A.1] Executing Downstream Diarization & WhisperX Attribution for all conditions...")
    ms_weights_tensor = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32, device="cuda")
    clustering = SpeakerClustering(cuda=True)

    conditions_list = [
        ("A", "condition_A_raw_titanet_dfn"),
        ("B", "condition_B_raw_resnet293_mono"),
        ("C", "condition_C_norm_equal_weight_stage5a"),
        ("H", "condition_H_norm_titanet_dfn_alone"),
        ("I", "condition_I_norm_resnet293_mono_alone"),
        ("J", "condition_J_norm_equal_weight_fusion"),
        ("K", "condition_K_norm_reliability_gate"),
        ("K_sig", "condition_K_sigmoid_gate"),
    ]

    per_rec_diar_rows: List[Dict[str, Any]] = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_json_uni = REPO_ROOT / entry["json_uniform"]
        gt_rttm_uni = REPO_ROOT / entry["rttm_uniform"]
        aligned_words_p = STAGE2_DIR / "whisperx" / f"{rec_id}_aligned.json"

        gt_payload = p20_common.read_json(gt_json_uni)
        gt_segments = gt_payload["segments"]
        gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_uni)
        ref_spks_pos = len(set(s["speaker"] for s in gt_rttm_segs))
        aligned_words = p20_common.read_json(aligned_words_p)

        t_scales, r_scales, ts_scales = [], [], []
        meta_s4_dfn = []
        meta_s4_mono = []

        for scale_k, _, _, _ in SCALES:
            t_d, _, m_d = load_cache("dfn", rec_id, scale_k)
            _, r_d, m_m = load_cache("16k_mono", rec_id, scale_k)
            t_scales.append(torch.from_numpy(t_d["l2_normalized_embeddings"]).to("cuda"))
            r_scales.append(torch.from_numpy(r_d["l2_normalized_embeddings"]).to("cuda"))
            ts_scales.append(torch.from_numpy(np.stack([t_d["starts"], t_d["ends"]], axis=-1).astype(np.float32)).to("cuda"))
            if scale_k == 4:
                meta_s4_dfn = m_d
                meta_s4_mono = m_m

        base_ts = ts_scales[-1]

        # Compute Raw Multiscale Affinities
        A_tita = getMultiScaleCosAffinityMatrix(ms_weights_tensor, t_scales, ts_scales, torch.device("cuda"))
        A_res = getMultiScaleCosAffinityMatrix(ms_weights_tensor, r_scales, ts_scales, torch.device("cuda"))

        # Compute Normalized Multiscale Affinities (Unsupervised per-recording transductive)
        A_zt = offdiag_zscore_affinity(A_tita)
        A_zr = offdiag_zscore_affinity(A_res)

        # Equal-weight fusion
        A_eq = 0.5 * A_zt + 0.5 * A_zr

        # Linear Reliability Gate Matrix
        meta_gate_items = [
            {
                "resnet293_l2_norm": float(meta_s4_mono[i]["resnet293_l2_norm"]),
                "dfn_noise_reduction_db": float(meta_s4_dfn[i]["dfn_noise_reduction_db"]),
                "vad_conf_mean": float(meta_s4_mono[i]["vad_conf_mean"]),
            }
            for i in range(len(meta_s4_dfn))
        ]
        g_model = gate_models_linear[rec_id]
        G_mat_linear = g_model.build_affinity_gate_matrix(meta_gate_items, device=torch.device("cuda"))
        A_gated_linear = (1.0 - G_mat_linear) * A_zt + G_mat_linear * A_zr

        # Sigmoid Gate Matrix (Sensitivity analysis)
        pred_deltas = g_model.predict_delta_margin(meta_gate_items)
        g_res_sig = np.clip(1.0 / (1.0 + np.exp(-2.0 * pred_deltas)), 0.15, 0.85).astype(np.float32)
        G_mat_sig = torch.from_numpy(0.5 * (g_res_sig[:, None] + g_res_sig[None, :])).to("cuda").float()
        A_gated_sig = (1.0 - G_mat_sig) * A_zt + G_mat_sig * A_zr

        affinity_by_condition: Dict[str, torch.Tensor] = {
            "condition_A_raw_titanet_dfn": A_tita,
            "condition_B_raw_resnet293_mono": A_res,
            "condition_C_norm_equal_weight_stage5a": A_eq,
            "condition_H_norm_titanet_dfn_alone": A_zt,
            "condition_I_norm_resnet293_mono_alone": A_zr,
            "condition_J_norm_equal_weight_fusion": A_eq,
            "condition_K_norm_reliability_gate": A_gated_linear,
            "condition_K_sigmoid_gate": A_gated_sig,
        }

        for cond_id, m_name in conditions_list:
            A_mat = affinity_by_condition[m_name]
            torch.manual_seed(42)
            np.random.seed(42)

            c_labels = clustering.forward_unit_infer(
                mat=A_mat.to(torch.device("cuda")).float(),
                oracle_num_speakers=-1,
                max_num_speakers=10,
                max_rp_threshold=0.25,
                sparse_search_volume=30,
                est_num_of_spk_enhanced=torch.tensor(-1),
                fixed_thres=-1.0,
                kmeans_random_trials=1,
            )
            labels, _ = generate_cluster_labels(base_ts.cpu().numpy(), c_labels.cpu().numpy())

            rttm_dir = STAGE5A_1_DIR / "predictions" / "rttms" / m_name
            rttm_dir.mkdir(parents=True, exist_ok=True)
            labels_to_rttmfile(labels, dfn_stem, str(rttm_dir))
            pred_rttm_path = rttm_dir / f"{dfn_stem}.rttm"

            diar_segments = p20_common.rttm_to_segments(pred_rttm_path)
            diar_eval = p20_common.diarization_metrics(gt_rttm_uni, pred_rttm_path)
            der_025 = compute_der_collar025_noovl(gt_rttm_uni, pred_rttm_path)
            words_attr = p20_common.attributed_words(aligned_words, diar_segments)
            segs_attr = p20_common.group_words(words_attr)

            attr_dir = STAGE5A_1_DIR / "predictions" / "attributions" / m_name
            attr_dir.mkdir(parents=True, exist_ok=True)
            attr_path = attr_dir / f"{rec_id}_{m_name}_attributed.json"
            p20_common.write_json(
                attr_path,
                {"file": f"{rec_id}.wav", "method": m_name, "words": words_attr, "segments": segs_attr},
            )

            f_metrics = p20_common.fusion_metrics(gt_json_uni, attr_path, dur)
            macro_info = compute_macro_sacc(gt_segments, words_attr, f_metrics.get("speaker_mapping", {}))

            pred_spks = len(set(s["speaker"] for s in diar_segments))
            single_spk_collapse = bool(pred_spks == 1 and ref_spks_pos > 1)

            per_rec_diar_rows.append({
                "rec_id": rec_id,
                "condition_id": cond_id,
                "method": m_name,
                "duration_sec": dur,
                "ref_speakers_pos": ref_spks_pos,
                "pred_speakers": pred_spks,
                "speaker_count_error_abs": abs(pred_spks - ref_spks_pos),
                "speaker_count_signed_bias": pred_spks - ref_spks_pos,
                "single_speaker_collapse": single_spk_collapse,
                "der_collar0_ovl": round(float(diar_eval["der"]), 4),
                "der_collar025_noovl": round(float(der_025), 4),
                "jer_collar0": round(float(diar_eval["jer"]), 4) if diar_eval["jer"] is not None else None,
                "false_alarm": round(float(diar_eval["false_alarm"]), 4),
                "miss": round(float(diar_eval["miss"]), 4),
                "confusion": round(float(diar_eval["confusion"]), 4),
                "cpwer": round(float(f_metrics["cpwer"]), 4),
                "sawer": round(float(f_metrics["sawer"]), 4),
                "micro_sacc": round(float(f_metrics["speaker_accuracy"]), 4),
                "micro_wae": round(float(f_metrics["word_attribution_error"]), 4),
                "timeline_accuracy": round(float(f_metrics["timeline_accuracy"]), 4),
                "macro_sacc": round(float(macro_info["macro_sacc"]), 4),
            })

    # Save per-recording diarization CSV
    write_csv(STAGE5A_1_DIR / "downstream_metrics" / "stage5a_1_per_recording_diarization.csv", per_rec_diar_rows)

    # Aggregate Diarization Summary
    summary_diar_rows: List[Dict[str, Any]] = []
    for cond_id, m_name in conditions_list:
        m_rows = [r for r in per_rec_diar_rows if r["method"] == m_name]
        tot_d = sum(r["duration_sec"] for r in m_rows)
        summary_diar_rows.append({
            "condition_id": cond_id,
            "method": m_name,
            "macro_der_collar0_ovl": round(float(np.mean([r["der_collar0_ovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar0_ovl": round(float(sum(r["der_collar0_ovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_der_collar025_noovl": round(float(np.mean([r["der_collar025_noovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar025_noovl": round(float(sum(r["der_collar025_noovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_jer": round(float(np.mean([r["jer_collar0"] for r in m_rows])), 4),
            "macro_false_alarm": round(float(np.mean([r["false_alarm"] for r in m_rows])), 4),
            "macro_miss": round(float(np.mean([r["miss"] for r in m_rows])), 4),
            "macro_confusion": round(float(np.mean([r["confusion"] for r in m_rows])), 4),
            "spk_count_mae": round(float(np.mean([r["speaker_count_error_abs"] for r in m_rows])), 4),
            "spk_count_signed_bias": round(float(np.mean([r["speaker_count_signed_bias"] for r in m_rows])), 4),
            "single_spk_collapse_count": int(sum(1 for r in m_rows if r["single_speaker_collapse"])),
            "macro_cpwer": round(float(np.mean([r["cpwer"] for r in m_rows])), 4),
            "macro_sawer": round(float(np.mean([r["sawer"] for r in m_rows])), 4),
            "macro_micro_sacc": round(float(np.mean([r["micro_sacc"] for r in m_rows])), 4),
            "macro_macro_sacc": round(float(np.mean([r["macro_sacc"] for r in m_rows])), 4),
        })
    write_csv(STAGE5A_1_DIR / "downstream_metrics" / "stage5a_1_summary_diarization.csv", summary_diar_rows)

    # 5. Verification Evaluation Audit under 8-Fold LORO
    print("\n[Stage 5A.1] Auditing Turn-Level Verification under 8-fold LORO with Fair Controls...")
    usable_turns, audit_summary = build_hybrid_turn_records(manifest, scale_k=4)
    print(f"[Stage 5A.1] Clean turns extracted: {len(usable_turns)}")

    within_rec_pairs: List[Dict[str, Any]] = []
    recs_present = sorted(set(t["rec_id"] for t in usable_turns))

    for rec_id in recs_present:
        rec_t = [t for t in usable_turns if t["rec_id"] == rec_id]
        for i in range(len(rec_t)):
            for j in range(i + 1, len(rec_t)):
                t1, t2 = rec_t[i], rec_t[j]
                same = int(t1["global_speaker_id"] == t2["global_speaker_id"])
                within_rec_pairs.append({
                    "rec_id": rec_id,
                    "t1_idx": i,
                    "t2_idx": j,
                    "same": same,
                    "tita_cos": float(np.dot(t1["tita_emb"], t2["tita_emb"])),
                    "res_cos": float(np.dot(t1["res_emb"], t2["res_emb"])),
                    "t1": t1,
                    "t2": t2,
                })
    print(f"[Stage 5A.1] Revalidated {len(within_rec_pairs)} within-recording clean turn pairs.")

    verif_conditions = [
        ("A", "condition_A_raw_titanet_dfn"),
        ("B", "condition_B_raw_resnet293_mono"),
        ("C", "condition_C_norm_equal_weight_stage5a"),
        ("H", "condition_H_norm_titanet_dfn_alone"),
        ("I", "condition_I_norm_resnet293_mono_alone"),
        ("J", "condition_J_norm_equal_weight_fusion"),
        ("K", "condition_K_norm_reliability_gate"),
        ("K_sig", "condition_K_sigmoid_gate"),
    ]

    verif_methods = [c[1] for c in verif_conditions]
    loro_verif_per_rec_rows: List[Dict[str, Any]] = []

    oof_scores: Dict[str, List[float]] = {m: [] for m in verif_methods}
    oof_preds: Dict[str, List[int]] = {m: [] for m in verif_methods}
    oof_targets: Dict[str, List[int]] = {m: [] for m in verif_methods}

    # Per-recording ROC-AUC accumulation across eligible recordings
    rec_auc_by_method: Dict[str, Dict[str, float]] = {m: {} for m in verif_methods}

    # Decision error predictions per method for error complementarity
    turn_pair_predictions: Dict[str, List[int]] = {m: [] for m in verif_methods}
    turn_pair_ground_truth: List[int] = []
    turn_pair_recordings: List[str] = []

    for held_rec in recordings:
        tr_pairs = [p for p in within_rec_pairs if p["rec_id"] != held_rec]
        te_pairs = [p for p in within_rec_pairs if p["rec_id"] == held_rec]

        tr_same = np.array([p["same"] for p in tr_pairs], dtype=np.int32)
        te_same = np.array([p["same"] for p in te_pairs], dtype=np.int32)

        tr_tita = np.array([p["tita_cos"] for p in tr_pairs], dtype=np.float64)
        tr_res = np.array([p["res_cos"] for p in tr_pairs], dtype=np.float64)
        te_tita = np.array([p["tita_cos"] for p in te_pairs], dtype=np.float64)
        te_res = np.array([p["res_cos"] for p in te_pairs], dtype=np.float64)

        # LORO training-fold z-scoring parameters (strictly out-of-fold)
        mu_t, sd_t = float(np.mean(tr_tita)), max(float(np.std(tr_tita)), 1e-12)
        mu_r, sd_r = float(np.mean(tr_res)), max(float(np.std(tr_res)), 1e-12)

        tr_zt = (tr_tita - mu_t) / sd_t
        tr_zr = (tr_res - mu_r) / sd_r
        te_zt = (te_tita - mu_t) / sd_t
        te_zr = (te_res - mu_r) / sd_r

        tr_eq = 0.5 * tr_zt + 0.5 * tr_zr
        te_eq = 0.5 * te_zt + 0.5 * te_zr

        g_model = gate_models_linear[held_rec]
        tr_g1 = g_model.predict_subsegment_weights([p["t1"] for p in tr_pairs])
        tr_g2 = g_model.predict_subsegment_weights([p["t2"] for p in tr_pairs])
        tr_gpair = 0.5 * (tr_g1 + tr_g2)
        tr_gated = (1.0 - tr_gpair) * tr_zt + tr_gpair * tr_zr

        if len(te_pairs) > 0:
            te_g1 = g_model.predict_subsegment_weights([p["t1"] for p in te_pairs])
            te_g2 = g_model.predict_subsegment_weights([p["t2"] for p in te_pairs])
            te_gpair = 0.5 * (te_g1 + te_g2)
            te_gated = (1.0 - te_gpair) * te_zt + te_gpair * te_zr

            # Sigmoid gate weights
            tr_d1 = g_model.predict_delta_margin([p["t1"] for p in tr_pairs])
            tr_d2 = g_model.predict_delta_margin([p["t2"] for p in tr_pairs])
            tr_gs1 = np.clip(1.0 / (1.0 + np.exp(-2.0 * tr_d1)), 0.15, 0.85)
            tr_gs2 = np.clip(1.0 / (1.0 + np.exp(-2.0 * tr_d2)), 0.15, 0.85)
            tr_gpair_sig = 0.5 * (tr_gs1 + tr_gs2)
            tr_gated_sig = (1.0 - tr_gpair_sig) * tr_zt + tr_gpair_sig * tr_zr

            te_d1 = g_model.predict_delta_margin([p["t1"] for p in te_pairs])
            te_d2 = g_model.predict_delta_margin([p["t2"] for p in te_pairs])
            te_gs1 = np.clip(1.0 / (1.0 + np.exp(-2.0 * te_d1)), 0.15, 0.85)
            te_gs2 = np.clip(1.0 / (1.0 + np.exp(-2.0 * te_d2)), 0.15, 0.85)
            te_gpair_sig = 0.5 * (te_gs1 + te_gs2)
            te_gated_sig = (1.0 - te_gpair_sig) * te_zt + te_gpair_sig * te_zr
        else:
            te_gated = np.array([])
            te_gated_sig = np.array([])
            tr_gated_sig = tr_gated

        scores_tr: Dict[str, np.ndarray] = {
            "condition_A_raw_titanet_dfn": tr_tita,
            "condition_B_raw_resnet293_mono": tr_res,
            "condition_C_norm_equal_weight_stage5a": tr_eq,
            "condition_H_norm_titanet_dfn_alone": tr_zt,
            "condition_I_norm_resnet293_mono_alone": tr_zr,
            "condition_J_norm_equal_weight_fusion": tr_eq,
            "condition_K_norm_reliability_gate": tr_gated,
            "condition_K_sigmoid_gate": tr_gated_sig,
        }

        scores_te: Dict[str, np.ndarray] = {
            "condition_A_raw_titanet_dfn": te_tita,
            "condition_B_raw_resnet293_mono": te_res,
            "condition_C_norm_equal_weight_stage5a": te_eq,
            "condition_H_norm_titanet_dfn_alone": te_zt,
            "condition_I_norm_resnet293_mono_alone": te_zr,
            "condition_J_norm_equal_weight_fusion": te_eq,
            "condition_K_norm_reliability_gate": te_gated,
            "condition_K_sigmoid_gate": te_gated_sig,
        }

        n_pos = int(np.sum(te_same == 1))
        n_neg = int(np.sum(te_same == 0))
        eligible = bool(n_pos > 0 and n_neg > 0)

        for m_name in verif_methods:
            s_tr = scores_tr[m_name]
            s_te = scores_te[m_name]

            tr_eer_res = compute_eer_and_mindcf(s_tr, tr_same)
            thr_tr = float(tr_eer_res["threshold"]) if tr_eer_res["threshold"] is not None else 0.0

            te_eval = eval_at_frozen_threshold(s_te, te_same, thr=thr_tr)

            if eligible:
                rec_auc_by_method[m_name][held_rec] = float(te_eval["roc_auc"])

            if len(te_pairs) > 0:
                preds = (s_te >= thr_tr).astype(np.int32)
                oof_scores[m_name].extend([float(v) for v in s_te])
                oof_preds[m_name].extend([int(v) for v in preds])
                turn_pair_predictions[m_name].extend([int(v) for v in preds])

            loro_verif_per_rec_rows.append({
                "held_out_rec_id": held_rec,
                "method": m_name,
                "test_pos_pairs": n_pos,
                "test_neg_pairs": n_neg,
                "eligible_for_eer": eligible,
                "frozen_train_thr": round(thr_tr, 4),
                "far_at_frozen_thr": te_eval["far_at_frozen_thr"],
                "frr_at_frozen_thr": te_eval["frr_at_frozen_thr"],
                "hter_at_frozen_thr": te_eval["hter_at_frozen_thr"],
                "held_out_roc_auc": te_eval["roc_auc"],
            })

        if len(te_pairs) > 0:
            oof_targets["all"].extend([int(v) for v in te_same]) if "all" in oof_targets else oof_targets.setdefault("all", [int(v) for v in te_same])
            turn_pair_ground_truth.extend([int(v) for v in te_same])
            turn_pair_recordings.extend([held_rec] * len(te_same))

    write_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_verification_per_recording.csv", loro_verif_per_rec_rows)

    # Macro vs Pooled Verification Summary
    verif_summary_rows: List[Dict[str, Any]] = []
    targets_arr = np.array(oof_targets["all"], dtype=np.int32)

    for cond_id, m_name in verif_conditions:
        scores_arr = np.array(oof_scores[m_name], dtype=np.float64)
        preds_arr = np.array(oof_preds[m_name], dtype=np.int32)

        n_pos_tot = int(np.sum(targets_arr == 1))
        n_neg_tot = int(np.sum(targets_arr == 0))
        pooled_far = float(np.sum(preds_arr[targets_arr == 0])) / n_neg_tot
        pooled_frr = float(np.sum(~preds_arr[targets_arr == 1].astype(bool))) / n_pos_tot
        pooled_hter = 0.5 * (pooled_far + pooled_frr)
        pooled_auc = float(roc_auc_score(targets_arr, scores_arr))
        pooled_act_dcf_01 = compute_act_dcf(pooled_far, pooled_frr, p_target=0.01)
        pooled_act_dcf_10 = compute_act_dcf(pooled_far, pooled_frr, p_target=0.10)

        # Macro ROC-AUC over the 6 eligible recordings
        rec_aucs = list(rec_auc_by_method[m_name].values())
        macro_auc = float(np.mean(rec_aucs)) if rec_aucs else None

        # Macro HTER over the 6 eligible recordings
        m_rec_rows = [r for r in loro_verif_per_rec_rows if r["method"] == m_name and r["eligible_for_eer"]]
        macro_hter = float(np.mean([r["hter_at_frozen_thr"] for r in m_rec_rows])) if m_rec_rows else None

        verif_summary_rows.append({
            "condition_id": cond_id,
            "method": m_name,
            "eligible_recordings_count": len(rec_aucs),
            "macro_6rec_roc_auc": round(macro_auc, 4) if macro_auc else None,
            "macro_6rec_frozen_hter": round(macro_hter, 4) if macro_hter else None,
            "oof_pooled_frozen_hter": round(pooled_hter, 4),
            "oof_pooled_frozen_far": round(pooled_far, 4),
            "oof_pooled_frozen_frr": round(pooled_frr, 4),
            "oof_pooled_roc_auc": round(pooled_auc, 4),
            "oof_pooled_act_dcf_p001": round(pooled_act_dcf_01, 4),
            "oof_pooled_act_dcf_p01": round(pooled_act_dcf_10, 4),
            "cross_fold_comparability_caveat": "Macro ROC-AUC is primary unbiased indicator; pooled OOF ROC-AUC is pair-weighted and subject to score offset variation across folds.",
        })
    write_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_verification_summary.csv", verif_summary_rows)

    # 6. Error Complementarity Analysis (Raw A vs B vs Normalized H vs I)
    print("\n[Stage 5A.1] Computing Error Complementarity Analysis...")
    gt_arr = np.array(turn_pair_ground_truth, dtype=np.int32)

    def analyze_pair_complementarity(m1: str, m2: str, label: str) -> Dict[str, Any]:
        p1 = np.array(turn_pair_predictions[m1], dtype=np.int32)
        p2 = np.array(turn_pair_predictions[m2], dtype=np.int32)
        err1 = (p1 != gt_arr)
        err2 = (p2 != gt_arr)

        m1_only = int(np.sum(err1 & ~err2))
        m2_only = int(np.sum(~err1 & err2))
        shared = int(np.sum(err1 & err2))
        neither = int(np.sum(~err1 & ~err2))
        union_err = m1_only + m2_only + shared
        total_pairs = len(gt_arr)

        jaccard = float(shared / union_err) if union_err > 0 else 0.0
        comp_fraction = float((m1_only + m2_only) / union_err) if union_err > 0 else 0.0

        return {
            "comparison": label,
            "method_1": m1,
            "method_2": m2,
            "total_pairs": total_pairs,
            "method_1_only_errors": m1_only,
            "method_2_only_errors": m2_only,
            "shared_errors": shared,
            "both_correct": neither,
            "total_unique_errors": union_err,
            "error_jaccard_similarity": round(jaccard, 4),
            "complementary_error_fraction": round(comp_fraction, 4),
        }

    comp_rows = [
        analyze_pair_complementarity("condition_A_raw_titanet_dfn", "condition_B_raw_resnet293_mono", "Raw Baselines (A vs B)"),
        analyze_pair_complementarity("condition_H_norm_titanet_dfn_alone", "condition_I_norm_resnet293_mono_alone", "Normalized Controls (H vs I)"),
    ]
    write_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_error_complementarity.csv", comp_rows)

    # 7. Paired Clustered Bootstrap Analysis (B=2000, seed=42)
    print("\n[Stage 5A.1] Computing Paired Clustered Bootstrap Analysis (B=2000, seed=42)...")
    np.random.seed(42)
    n_boot = 2000
    rec_ids = sorted(recordings)
    n_recs = len(rec_ids)

    # Comparisons to evaluate
    eval_comparisons = [
        # Question A: Representation Fusion Effect
        ("J_vs_H", "condition_J_norm_equal_weight_fusion", "condition_H_norm_titanet_dfn_alone", "Normalized Equal Fusion vs Normalized TitaNet"),
        ("J_vs_I", "condition_J_norm_equal_weight_fusion", "condition_I_norm_resnet293_mono_alone", "Normalized Equal Fusion vs Normalized ResNet"),
        # Question B: Reliability Gate Effect
        ("K_vs_J", "condition_K_norm_reliability_gate", "condition_J_norm_equal_weight_fusion", "Normalized Reliability Gate vs Normalized Equal Fusion"),
        ("K_vs_H", "condition_K_norm_reliability_gate", "condition_H_norm_titanet_dfn_alone", "Normalized Reliability Gate vs Normalized TitaNet"),
        ("K_vs_I", "condition_K_norm_reliability_gate", "condition_I_norm_resnet293_mono_alone", "Normalized Reliability Gate vs Normalized ResNet"),
        # Normalization Effect Controls
        ("H_vs_A", "condition_H_norm_titanet_dfn_alone", "condition_A_raw_titanet_dfn", "Normalized TitaNet vs Raw TitaNet"),
        ("I_vs_B", "condition_I_norm_resnet293_mono_alone", "condition_B_raw_resnet293_mono", "Normalized ResNet vs Raw ResNet"),
        # Gate Sensitivity
        ("K_sig_vs_K", "condition_K_sigmoid_gate", "condition_K_norm_reliability_gate", "Sigmoid Gate vs Linear Gate"),
        # Historical Comparisons
        ("C_vs_A", "condition_C_norm_equal_weight_stage5a", "condition_A_raw_titanet_dfn", "Equal Fusion vs Raw TitaNet (Stage 5A)"),
        ("C_vs_B", "condition_C_norm_equal_weight_stage5a", "condition_B_raw_resnet293_mono", "Equal Fusion vs Raw ResNet (Stage 5A)"),
        ("K_vs_A", "condition_K_norm_reliability_gate", "condition_A_raw_titanet_dfn", "Reliability Gate vs Raw TitaNet (Stage 5A)"),
        ("K_vs_B", "condition_K_norm_reliability_gate", "condition_B_raw_resnet293_mono", "Reliability Gate vs Raw ResNet (Stage 5A)"),
    ]

    target_metrics = [
        "der_collar0_ovl",
        "der_collar025_noovl",
        "jer_collar0",
        "cpwer",
        "sawer",
        "micro_sacc",
        "macro_sacc",
        "speaker_count_error_abs",
    ]

    # Pre-structure recording metric lookups: rec_metrics[method][metric][rec_id]
    rec_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in per_rec_diar_rows:
        m = r["method"]
        rec = r["rec_id"]
        rec_metrics.setdefault(m, {})
        for k in target_metrics:
            rec_metrics[m].setdefault(k, {})[rec] = float(r[k])

    paired_boot_rows: List[Dict[str, Any]] = []

    for comp_id, m1, m2, desc in eval_comparisons:
        for metric in target_metrics:
            v1_arr = np.array([rec_metrics[m1][metric][r] for r in rec_ids])
            v2_arr = np.array([rec_metrics[m2][metric][r] for r in rec_ids])

            # Point difference: Method 1 minus Method 2
            point_diff = float(np.mean(v1_arr) - np.mean(v2_arr))

            boot_diffs = []
            for _ in range(n_boot):
                sample_idx = np.random.choice(n_recs, size=n_recs, replace=True)
                diff = float(np.mean(v1_arr[sample_idx]) - np.mean(v2_arr[sample_idx]))
                boot_diffs.append(diff)

            ci_low = float(np.percentile(boot_diffs, 2.5))
            ci_high = float(np.percentile(boot_diffs, 97.5))
            ci_width = ci_high - ci_low
            inc_zero = bool(ci_low <= 0.0 <= ci_high)

            # Wins/losses
            # For error metrics (DER, JER, cpWER, saWER, count_err), smaller is better
            # For accuracy metrics (sacc), larger is better
            is_error_metric = metric not in ["micro_sacc", "macro_sacc"]
            if is_error_metric:
                wins_m1 = int(np.sum(v1_arr < v2_arr))
                wins_m2 = int(np.sum(v2_arr < v1_arr))
            else:
                wins_m1 = int(np.sum(v1_arr > v2_arr))
                wins_m2 = int(np.sum(v2_arr > v1_arr))
            ties = int(np.sum(v1_arr == v2_arr))

            paired_boot_rows.append({
                "comparison_id": comp_id,
                "description": desc,
                "method_1": m1,
                "method_2": m2,
                "metric": metric,
                "point_diff_m1_minus_m2": round(point_diff, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "ci95_width": round(ci_width, 4),
                "includes_zero": inc_zero,
                "m1_wins": wins_m1,
                "m2_wins": wins_m2,
                "ties": ties,
            })

    write_csv(STAGE5A_1_DIR / "bootstrap" / "stage5a_1_paired_bootstrap_differences.csv", paired_boot_rows)

    # Master results JSON
    master_results = {
        "stage": "Stage 5A.1 \u2014 Affinity Normalization, Fusion Fairness & Implementation Audit",
        "git_commit": commit,
        "summary_diarization": summary_diar_rows,
        "verification_summary": verif_summary_rows,
        "error_complementarity": comp_rows,
        "paired_bootstrap_differences": paired_boot_rows,
    }
    with open(STAGE5A_1_DIR / "downstream_metrics" / "stage5a_1_master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print(f"\n[Stage 5A.1] Master results saved to {STAGE5A_1_DIR / 'downstream_metrics' / 'stage5a_1_master_results.json'}")
    print("[Stage 5A.1] Main controlled evaluation completed successfully!")


if __name__ == "__main__":
    main()
