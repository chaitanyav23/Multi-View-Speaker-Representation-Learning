#!/usr/bin/env python3
"""
Stage 5A Main Execution & Evaluation Script:
Controlled Validation of Hybrid Speaker-Affinity Fusion (TitaNet-DFN + ResNet293-mono)
and Controlled Feature Ablations under Strict 8-Fold LORO, Turn-Disjoint LOTO,
and Clustered Bootstrap Uncertainty (B=2000, seed=42).
"""

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["XDG_CACHE_HOME"] = "/home/priyanka/.cache"
os.environ["HF_HOME"] = "/home/priyanka/.cache/huggingface"
os.environ["TORCH_HOME"] = "/home/priyanka/.cache/torch"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import scipy.stats
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
STAGE1_DIR = REPO_ROOT / "research_stages" / "stage1_data_prep"
STAGE2_DIR = REPO_ROOT / "research_stages" / "stage2_baseline"
STAGE3_DIR = REPO_ROOT / "research_stages" / "stage3_dual_encoder"
STAGE4_DIR = REPO_ROOT / "research_stages" / "stage4_complementarity"
STAGE4_1_DIR = REPO_ROOT / "research_stages" / "stage4_1"
STAGE5A_DIR = REPO_ROOT / "research_stages" / "stage5a"

sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(STAGE2_DIR))
sys.path.insert(0, str(STAGE4_DIR))
sys.path.insert(0, str(STAGE5A_DIR / "scripts"))

import common as p20_common  # noqa: E402
from run_stage2_baseline import compute_der_collar025_noovl, compute_macro_sacc  # noqa: E402
from run_stage4_complementarity import SCALES, compute_eer_and_mindcf, load_cache  # noqa: E402
from fusion_models import ReliabilityGate, offdiag_zscore_affinity  # noqa: E402
from validate_inputs_and_caches import validate_all  # noqa: E402

from nemo.collections.asr.parts.utils.offline_clustering import (  # noqa: E402
    SpeakerClustering,
    getMultiScaleCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.speaker_utils import (  # noqa: E402
    generate_cluster_labels,
    labels_to_rttmfile,
)


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds clean turn representations for the hybrid setup:
    - TitaNet-Large from DFN
    - WeSpeaker ResNet293-LM from 16k_mono
    """
    usable_turns: List[Dict[str, Any]] = []
    all_clean_gt_turns: List[Dict[str, Any]] = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
        t_d, _, sub_meta_dfn = load_cache("dfn", rec_id, scale_k)
        _, r_d, sub_meta_mono = load_cache("16k_mono", rec_id, scale_k)

        t_norm = t_d["l2_normalized_embeddings"]
        r_norm = r_d["l2_normalized_embeddings"]
        sub_st = np.array([m["start"] for m in sub_meta_dfn], dtype=np.float32)
        sub_en = np.array([m["end"] for m in sub_meta_dfn], dtype=np.float32)

        for g in gt_segs:
            if not g.get("usable_for_clean_speaker_eval", False):
                continue
            gst = float(g["start"])
            gen = float(g["end_clipped_to_audio"])
            dur = gen - gst
            ov = np.minimum(sub_en, gen) - np.maximum(sub_st, gst)
            sub_dur = np.maximum(sub_en - sub_st, 1e-6)
            idxs = np.where((ov / sub_dur) >= 0.75)[0]

            rec_info = {
                "rec_id": rec_id,
                "seg_idx": int(g["seg_idx"]),
                "speaker": str(g["speaker"]),
                "global_speaker_id": str(g["global_speaker_id"]),
                "start": gst,
                "end": gen,
                "duration": round(dur, 3),
                "num_subsegments_pooled": int(idxs.size),
                "included_in_turn_pairs": bool(dur >= 0.5 and idxs.size > 0),
            }
            all_clean_gt_turns.append(rec_info)

            if dur >= 0.5 and idxs.size > 0:
                t_vec = np.mean(t_norm[idxs], axis=0)
                t_vec = (t_vec / max(float(np.linalg.norm(t_vec)), 1e-12)).astype(np.float32)
                r_vec = np.mean(r_norm[idxs], axis=0)
                r_vec = (r_vec / max(float(np.linalg.norm(r_vec)), 1e-12)).astype(np.float32)

                usable_turns.append({
                    **rec_info,
                    "snr_local_db": float(np.mean([sub_meta_mono[i]["snr_local_db"] for i in idxs])),
                    "rms_dbfs": float(np.mean([sub_meta_mono[i]["rms_dbfs"] for i in idxs])),
                    "dfn_noise_reduction_db": float(np.mean([sub_meta_dfn[i]["dfn_noise_reduction_db"] for i in idxs])),
                    "spectral_flatness": float(np.mean([sub_meta_mono[i]["spectral_flatness"] for i in idxs])),
                    "vad_conf_mean": float(np.mean([sub_meta_mono[i]["vad_conf_mean"] for i in idxs])),
                    "vad_conf_std": float(np.mean([sub_meta_mono[i]["vad_conf_std"] for i in idxs])),
                    "titanet_l2_norm": float(np.mean([sub_meta_dfn[i]["titanet_l2_norm"] for i in idxs])),
                    "resnet293_l2_norm": float(np.mean([sub_meta_mono[i]["resnet293_l2_norm"] for i in idxs])),
                    "titanet_cross_scale_cos": float(np.mean([sub_meta_dfn[i]["titanet_cross_scale_cos"] for i in idxs])),
                    "resnet293_cross_scale_cos": float(np.mean([sub_meta_mono[i]["resnet293_cross_scale_cos"] for i in idxs])),
                    "tita_emb": t_vec,
                    "res_emb": r_vec,
                })
    return usable_turns, all_clean_gt_turns


def eval_at_frozen_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, Any]:
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return {
            "far_at_frozen_thr": None,
            "frr_at_frozen_thr": None,
            "hter_at_frozen_thr": None,
            "act_dcf_p001": None,
            "act_dcf_p01": None,
            "roc_auc": None,
            "exploratory_in_sample_eer": None,
            "exploratory_in_sample_mindcf": None,
        }
    preds = (scores >= thr).astype(np.int32)
    far = float(np.mean(preds[neg_mask] == 1))
    frr = float(np.mean(preds[pos_mask] == 0))
    hter = 0.5 * (far + frr)

    act_dcf_001 = (0.01 * frr + 0.99 * far) / 0.01
    act_dcf_010 = (0.10 * frr + 0.90 * far) / 0.10

    auc = float(roc_auc_score(labels, scores))
    in_sample = compute_eer_and_mindcf(scores, labels)

    return {
        "far_at_frozen_thr": round(far, 4),
        "frr_at_frozen_thr": round(frr, 4),
        "hter_at_frozen_thr": round(hter, 4),
        "act_dcf_p001": round(act_dcf_001, 4),
        "act_dcf_p01": round(act_dcf_010, 4),
        "roc_auc": round(auc, 4),
        "exploratory_in_sample_eer": in_sample["eer"],
        "exploratory_in_sample_mindcf": in_sample["mindcf_p001"],
    }


def main() -> None:
    print("=" * 80)
    print("Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion")
    print("=" * 80)

    # 1. Environment & Pre-Execution Logging
    torch.manual_seed(42)
    np.random.seed(42)

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], text=True).strip()

    print(f"Git Commit Hash: {git_hash}")
    print(f"Git Working-Tree Status:\n{git_status if git_status else 'Clean'}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Run Pre-Execution Validation
    validate_all()

    with open(STAGE5A_DIR / "configs" / "stage5a_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    manifest = json.loads((REPO_ROOT / cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    recordings = cfg["recordings"]

    # 3. Build Turn-Disjoint LOTO Subsegment Dataset for Hybrid Setup
    # TitaNet on DFN, ResNet293 on 16k_mono
    print("\n[Stage 5A] Extracting Turn-Disjoint LOTO clean subsegments for hybrid gate training...")
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
            continue  # R02 (only 1 clean spk) and R04 (each spk 1 turn) cannot form multi-speaker LOTO prototypes

        # Compute full centroids of all other speakers for the margin negative term
        t_cent_full = {}
        r_cent_full = {}
        for s in multi_turn_spks:
            all_s_ix = [idx for ix_list in spk_to_turn_idxs[s].values() for idx in ix_list]
            tc = np.mean(t_norm[all_s_ix], axis=0)
            t_cent_full[s] = tc / max(float(np.linalg.norm(tc)), 1e-12)
            rc = np.mean(r_norm[all_s_ix], axis=0)
            r_cent_full[s] = rc / max(float(np.linalg.norm(rc)), 1e-12)

        for s in multi_turn_spks:
            others = [o for o in multi_turn_spks if o != s]
            for t_id, cur_idxs in spk_to_turn_idxs[s].items():
                other_turn_idxs = [
                    idx for tid_other, ix_list in spk_to_turn_idxs[s].items() if tid_other != t_id for idx in ix_list
                ]
                if not other_turn_idxs:
                    continue
                tc_loto = np.mean(t_norm[other_turn_idxs], axis=0)
                tc_loto = tc_loto / max(float(np.linalg.norm(tc_loto)), 1e-12)
                rc_loto = np.mean(r_norm[other_turn_idxs], axis=0)
                rc_loto = rc_loto / max(float(np.linalg.norm(rc_loto)), 1e-12)

                for i in cur_idxs:
                    m_d = meta_dfn[i]
                    m_m = meta_mono[i]
                    tm = float(np.dot(t_norm[i], tc_loto)) - max(float(np.dot(t_norm[i], t_cent_full[o])) for o in others)
                    rm = float(np.dot(r_norm[i], rc_loto)) - max(float(np.dot(r_norm[i], r_cent_full[o])) for o in others)
                    loto_subseg_items.append({
                        "rec_id": rec_id,
                        "subseg_idx": i,
                        "turn_id": t_id,
                        "speaker": s,
                        "resnet293_l2_norm": float(m_m["resnet293_l2_norm"]),
                        "dfn_noise_reduction_db": float(m_d["dfn_noise_reduction_db"]),
                        "vad_conf_mean": float(m_m["vad_conf_mean"]),
                        "tita_margin": tm,
                        "res_margin": rm,
                        "delta_margin": rm - tm,
                        "res_wins": int(rm > tm),
                    })

    print(f"[Stage 5A] Extracted {len(loto_subseg_items)} clean turn-disjoint LOTO subsegments across eligible recordings.")

    # 4. Fit Gates for Condition D and Ablations E, F, G under 8-fold LORO
    print("\n[Stage 5A] Fitting Condition D (Full 3-Feature) and Ablation Gates (E, F, G) under strict LORO...")
    gate_conditions = [
        ("condition_D_three_feature_reliability_gate", ["resnet293_l2_norm", "dfn_noise_reduction_db", "vad_conf_mean"]),
        ("condition_E_gate_ablation_no_resnet_l2", ["dfn_noise_reduction_db", "vad_conf_mean"]),
        ("condition_F_gate_ablation_no_dfn_nr", ["resnet293_l2_norm", "vad_conf_mean"]),
        ("condition_G_gate_ablation_no_vad_conf", ["resnet293_l2_norm", "dfn_noise_reduction_db"]),
    ]

    gate_models: Dict[str, Dict[str, ReliabilityGate]] = {g_name: {} for g_name, _ in gate_conditions}
    gate_param_records: Dict[str, Dict[str, Any]] = {g_name: {} for g_name, _ in gate_conditions}
    gate_eval_rows: List[Dict[str, Any]] = []

    for held_rec in recordings:
        tr_items = [x for x in loto_subseg_items if x["rec_id"] != held_rec]
        te_items = [x for x in loto_subseg_items if x["rec_id"] == held_rec]

        for g_name, feats in gate_conditions:
            gate = ReliabilityGate(features=feats, regularization_lambda=0.05, clip_min=0.15, clip_max=0.85, gain_factor=2.0)
            gate.fit(tr_items)
            gate_models[g_name][held_rec] = gate
            gate_param_records[g_name][held_rec] = gate.get_params_dict()

            tr_pred = gate.predict_delta_margin(tr_items)
            y_tr = np.array([x["delta_margin"] for x in tr_items])
            tr_rho = float(scipy.stats.spearmanr(tr_pred, y_tr)[0])

            if len(te_items) >= 5:
                te_pred = gate.predict_delta_margin(te_items)
                y_te = np.array([x["delta_margin"] for x in te_items])
                te_rho = float(scipy.stats.spearmanr(te_pred, y_te)[0])
                te_r2 = 1.0 - float(np.sum((te_pred - y_te) ** 2)) / max(float(np.sum((y_te - np.mean(y_te)) ** 2)), 1e-12)
                te_win_auc = float(roc_auc_score([x["res_wins"] for x in te_items], te_pred))
            else:
                te_rho = None
                te_r2 = None
                te_win_auc = None

            gate_eval_rows.append({
                "condition": g_name,
                "features": "+".join(feats),
                "held_out_rec_id": held_rec,
                "train_sample_count": len(tr_items),
                "test_sample_count": len(te_items),
                "train_spearman_rho": round(tr_rho, 4),
                "held_out_spearman_rho": round(te_rho, 4) if te_rho is not None else None,
                "held_out_r2": round(te_r2, 4) if te_r2 is not None else None,
                "held_out_res_wins_auc": round(te_win_auc, 4) if te_win_auc is not None else None,
            })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_gate_loto_evaluation.csv", gate_eval_rows)
    with open(STAGE5A_DIR / "metrics" / "stage5a_gate_parameters.json", "w", encoding="utf-8") as f:
        json.dump(gate_param_records, f, indent=2)

    # 5. Downstream Diarization & Speaker Attribution
    print("\n[Stage 5A] Executing Downstream Diarization & WhisperX Attribution for all conditions...")
    ms_weights_tensor = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32, device="cuda")
    clustering = SpeakerClustering(cuda=True)

    methods = [
        ("condition_A_titanet_dfn_baseline", "A"),
        ("condition_B_resnet293_mono_baseline", "B"),
        ("condition_C_equal_weight_affinity", "C"),
        ("condition_D_three_feature_reliability_gate", "D"),
        ("condition_E_gate_ablation_no_resnet_l2", "E"),
        ("condition_F_gate_ablation_no_dfn_nr", "F"),
        ("condition_G_gate_ablation_no_vad_conf", "G"),
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

        # Load multiscale scales: TitaNet from DFN, ResNet293 from 16k_mono
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

        A_tita = getMultiScaleCosAffinityMatrix(ms_weights_tensor, t_scales, ts_scales, torch.device("cuda"))
        A_res = getMultiScaleCosAffinityMatrix(ms_weights_tensor, r_scales, ts_scales, torch.device("cuda"))
        A_zt = offdiag_zscore_affinity(A_tita)
        A_zr = offdiag_zscore_affinity(A_res)
        A_eq = 0.5 * A_zt + 0.5 * A_zr

        # Assemble meta items for gate inference (using ResNet norm from mono, DFN noise reduction from dfn, VAD conf from mono)
        meta_gate_items = [
            {
                "resnet293_l2_norm": float(meta_s4_mono[i]["resnet293_l2_norm"]),
                "dfn_noise_reduction_db": float(meta_s4_dfn[i]["dfn_noise_reduction_db"]),
                "vad_conf_mean": float(meta_s4_mono[i]["vad_conf_mean"]),
            }
            for i in range(len(meta_s4_dfn))
        ]

        affinity_by_condition: Dict[str, torch.Tensor] = {
            "condition_A_titanet_dfn_baseline": A_tita,
            "condition_B_resnet293_mono_baseline": A_res,
            "condition_C_equal_weight_affinity": A_eq,
        }

        for g_name, _ in gate_conditions:
            g_model = gate_models[g_name][rec_id]
            G_mat = g_model.build_affinity_gate_matrix(meta_gate_items, device=torch.device("cuda"))
            affinity_by_condition[g_name] = (1.0 - G_mat) * A_zt + G_mat * A_zr

        for m_name, cond_id in methods:
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

            rttm_dir = STAGE5A_DIR / "predictions" / "rttms" / m_name
            rttm_dir.mkdir(parents=True, exist_ok=True)
            labels_to_rttmfile(labels, dfn_stem, str(rttm_dir))
            pred_rttm_path = rttm_dir / f"{dfn_stem}.rttm"

            diar_segments = p20_common.rttm_to_segments(pred_rttm_path)
            diar_eval = p20_common.diarization_metrics(gt_rttm_uni, pred_rttm_path)
            der_025 = compute_der_collar025_noovl(gt_rttm_uni, pred_rttm_path)
            words_attr = p20_common.attributed_words(aligned_words, diar_segments)
            segs_attr = p20_common.group_words(words_attr)

            attr_dir = STAGE5A_DIR / "predictions" / "attributions" / m_name
            attr_dir.mkdir(parents=True, exist_ok=True)
            attr_path = attr_dir / f"{rec_id}_{m_name}_attributed.json"
            p20_common.write_json(
                attr_path,
                {"file": f"{rec_id}.wav", "method": m_name, "words": words_attr, "segments": segs_attr},
            )

            f_metrics = p20_common.fusion_metrics(gt_json_uni, attr_path, dur)
            macro_info = compute_macro_sacc(gt_segments, words_attr, f_metrics.get("speaker_mapping", {}))
            pred_spks = len(set(s["speaker"] for s in diar_segments))

            per_rec_diar_rows.append({
                "rec_id": rec_id,
                "method_id": cond_id,
                "method": m_name,
                "duration_sec": round(dur, 3),
                "ref_speakers_pos": ref_spks_pos,
                "pred_speakers": pred_spks,
                "speaker_count_error_abs": abs(pred_spks - ref_spks_pos),
                "speaker_count_signed_bias": pred_spks - ref_spks_pos,
                "single_speaker_collapse": pred_spks == 1,
                "der_collar0_ovl": round(float(diar_eval["der"]), 4),
                "der_collar025_noovl": round(der_025, 4),
                "jer_collar0": round(float(diar_eval["jer"]), 4) if diar_eval["jer"] is not None else None,
                "false_alarm": round(float(diar_eval["false_alarm"]), 4),
                "miss": round(float(diar_eval["miss"]), 4),
                "confusion": round(float(diar_eval["confusion"]), 4),
                "cpwer": round(float(f_metrics["cpwer"]), 4),
                "sawer": round(float(f_metrics["sawer"]), 4),
                "micro_sacc": round(float(f_metrics["speaker_accuracy"]), 4),
                "micro_wae": round(float(f_metrics["word_attribution_error"]), 4),
                "timeline_accuracy": round(float(f_metrics["timeline_accuracy"]), 4),
                "macro_sacc": macro_info["macro_sacc"],
            })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_per_recording_diarization.csv", per_rec_diar_rows)

    # 6. Aggregate Diarization Summary
    summary_diar_rows: List[Dict[str, Any]] = []
    for m_name, cond_id in methods:
        m_rows = [r for r in per_rec_diar_rows if r["method"] == m_name]
        tot_d = sum(r["duration_sec"] for r in m_rows)
        summary_diar_rows.append({
            "method_id": cond_id,
            "method": m_name,
            "macro_der_collar0_ovl": round(float(np.mean([r["der_collar0_ovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar0_ovl": round(float(sum(r["der_collar0_ovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_der_collar025_noovl": round(float(np.mean([r["der_collar025_noovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar025_noovl": round(float(sum(r["der_collar025_noovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_jer": round(float(np.mean([r["jer_collar0"] for r in m_rows])), 4),
            "spk_count_mae": round(float(np.mean([r["speaker_count_error_abs"] for r in m_rows])), 4),
            "spk_count_signed_bias": round(float(np.mean([r["speaker_count_signed_bias"] for r in m_rows])), 4),
            "single_spk_collapse_count": int(sum(1 for r in m_rows if r["single_speaker_collapse"])),
            "macro_cpwer": round(float(np.mean([r["cpwer"] for r in m_rows])), 4),
            "macro_sawer": round(float(np.mean([r["sawer"] for r in m_rows])), 4),
            "macro_micro_sacc": round(float(np.mean([r["micro_sacc"] for r in m_rows])), 4),
            "macro_macro_sacc": round(float(np.mean([r["macro_sacc"] for r in m_rows])), 4),
        })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_summary_diarization.csv", summary_diar_rows)

    # 7. Turn-Level Pairwise Verification under 8-Fold LORO
    print("\n[Stage 5A] Evaluating Turn-Level Pairwise Verification under 8-fold LORO...")
    usable_turns, _ = build_hybrid_turn_records(manifest, scale_k=4)

    # Form strictly within-recording turn pairs
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

    print(f"[Stage 5A] Assembled {len(within_rec_pairs)} within-recording clean turn pairs.")

    loro_verif_per_rec_rows: List[Dict[str, Any]] = []
    verif_methods = [m[0] for m in methods]

    # Precompute fold score calibrations and thresholds
    fold_train_stats: Dict[str, Dict[str, float]] = {}

    oof_scores_by_method: Dict[str, List[float]] = {m: [] for m in verif_methods}
    oof_preds_by_method: Dict[str, List[int]] = {m: [] for m in verif_methods}
    oof_targets_by_method: Dict[str, List[int]] = {m: [] for m in verif_methods}

    for held_rec in recordings:
        tr_pairs = [p for p in within_rec_pairs if p["rec_id"] != held_rec]
        te_pairs = [p for p in within_rec_pairs if p["rec_id"] == held_rec]

        tr_same = np.array([p["same"] for p in tr_pairs], dtype=np.int32)
        te_same = np.array([p["same"] for p in te_pairs], dtype=np.int32)

        # Base scores
        tr_tita = np.array([p["tita_cos"] for p in tr_pairs], dtype=np.float64)
        tr_res = np.array([p["res_cos"] for p in tr_pairs], dtype=np.float64)
        te_tita = np.array([p["tita_cos"] for p in te_pairs], dtype=np.float64)
        te_res = np.array([p["res_cos"] for p in te_pairs], dtype=np.float64)

        mu_t, sd_t = float(np.mean(tr_tita)), max(float(np.std(tr_tita)), 1e-12)
        mu_r, sd_r = float(np.mean(tr_res)), max(float(np.std(tr_res)), 1e-12)

        tr_zt = (tr_tita - mu_t) / sd_t
        tr_zr = (tr_res - mu_r) / sd_r
        te_zt = (te_tita - mu_t) / sd_t
        te_zr = (te_res - mu_r) / sd_r

        tr_eq = 0.5 * tr_zt + 0.5 * tr_zr
        te_eq = 0.5 * te_zt + 0.5 * te_zr

        scores_by_method_tr: Dict[str, np.ndarray] = {
            "condition_A_titanet_dfn_baseline": tr_tita,
            "condition_B_resnet293_mono_baseline": tr_res,
            "condition_C_equal_weight_affinity": tr_eq,
        }
        scores_by_method_te: Dict[str, np.ndarray] = {
            "condition_A_titanet_dfn_baseline": te_tita,
            "condition_B_resnet293_mono_baseline": te_res,
            "condition_C_equal_weight_affinity": te_eq,
        }

        # Gate scores
        for g_name, _ in gate_conditions:
            g_model = gate_models[g_name][held_rec]
            # Predict pair weights
            tr_g1 = g_model.predict_subsegment_weights([p["t1"] for p in tr_pairs])
            tr_g2 = g_model.predict_subsegment_weights([p["t2"] for p in tr_pairs])
            tr_gpair = 0.5 * (tr_g1 + tr_g2)
            scores_by_method_tr[g_name] = (1.0 - tr_gpair) * tr_zt + tr_gpair * tr_zr

            if len(te_pairs) > 0:
                te_g1 = g_model.predict_subsegment_weights([p["t1"] for p in te_pairs])
                te_g2 = g_model.predict_subsegment_weights([p["t2"] for p in te_pairs])
                te_gpair = 0.5 * (te_g1 + te_g2)
                scores_by_method_te[g_name] = (1.0 - te_gpair) * te_zt + te_gpair * te_zr
            else:
                scores_by_method_te[g_name] = np.array([])

        for m_name in verif_methods:
            s_tr = scores_by_method_tr[m_name]
            s_te = scores_by_method_te[m_name]

            # Fit threshold strictly on training pairs
            tr_eer_res = compute_eer_and_mindcf(s_tr, tr_same)
            thr_tr = float(tr_eer_res["threshold"]) if tr_eer_res["threshold"] is not None else 0.0

            te_eval = eval_at_frozen_threshold(s_te, te_same, thr=thr_tr)
            n_pos = int(np.sum(te_same == 1))
            n_neg = int(np.sum(te_same == 0))

            if len(te_pairs) > 0:
                preds = (s_te >= thr_tr).astype(np.int32)
                oof_scores_by_method[m_name].extend([float(v) for v in s_te])
                oof_preds_by_method[m_name].extend([int(v) for v in preds])
                oof_targets_by_method[m_name].extend([int(v) for v in te_same])

            loro_verif_per_rec_rows.append({
                "held_out_rec_id": held_rec,
                "method": m_name,
                "test_pos_pairs": n_pos,
                "test_neg_pairs": n_neg,
                "eligible_for_eer": bool(n_pos > 0 and n_neg > 0),
                "frozen_train_thr": round(thr_tr, 4),
                "far_at_frozen_thr": te_eval["far_at_frozen_thr"],
                "frr_at_frozen_thr": te_eval["frr_at_frozen_thr"],
                "hter_at_frozen_thr": te_eval["hter_at_frozen_thr"],
                "act_dcf_p001": te_eval["act_dcf_p001"],
                "act_dcf_p01": te_eval["act_dcf_p01"],
                "roc_auc": te_eval["roc_auc"],
                "exploratory_in_sample_eer": te_eval["exploratory_in_sample_eer"],
                "exploratory_in_sample_mindcf": te_eval["exploratory_in_sample_mindcf"],
            })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_loro_verification_per_recording.csv", loro_verif_per_rec_rows)

    # LORO Verification Summary
    loro_verif_summary_rows: List[Dict[str, Any]] = []

    for m_name in verif_methods:
        rec_m = [r for r in loro_verif_per_rec_rows if r["method"] == m_name and r["eligible_for_eer"]]
        macro_hter = float(np.mean([r["hter_at_frozen_thr"] for r in rec_m]))
        macro_auc = float(np.mean([r["roc_auc"] for r in rec_m]))
        macro_eer = float(np.mean([r["exploratory_in_sample_eer"] for r in rec_m]))

        # Pooled OOF metrics across all 9817 within-recording pairs
        all_scores = np.array(oof_scores_by_method[m_name], dtype=np.float64)
        all_preds = np.array(oof_preds_by_method[m_name], dtype=np.int32)
        all_targs = np.array(oof_targets_by_method[m_name], dtype=np.int32)

        pos_m = all_targs == 1
        neg_m = all_targs == 0
        oof_far = float(np.mean(all_preds[neg_m] == 1))
        oof_frr = float(np.mean(all_preds[pos_m] == 0))
        oof_hter = 0.5 * (oof_far + oof_frr)
        oof_act_dcf_001 = (0.01 * oof_frr + 0.99 * oof_far) / 0.01
        oof_act_dcf_010 = (0.10 * oof_frr + 0.90 * oof_far) / 0.10
        oof_auc = float(roc_auc_score(all_targs, all_scores))
        leaked_in_sample = compute_eer_and_mindcf(all_scores, all_targs)

        loro_verif_summary_rows.append({
            "method": m_name,
            "total_within_rec_pairs": len(all_targs),
            "eligible_recordings_count": len(rec_m),
            "oof_pooled_far_frozen": round(oof_far, 4),
            "oof_pooled_frr_frozen": round(oof_frr, 4),
            "oof_pooled_hter_frozen": round(oof_hter, 4),
            "oof_pooled_act_dcf_p001": round(oof_act_dcf_001, 4),
            "oof_pooled_act_dcf_p01": round(oof_act_dcf_010, 4),
            "oof_pooled_roc_auc": round(oof_auc, 4),
            "macro_6rec_hter_frozen": round(macro_hter, 4),
            "macro_6rec_roc_auc": round(macro_auc, 4),
            "macro_6rec_exploratory_eer": round(macro_eer, 4),
            "exploratory_full_dataset_eer_leaked": leaked_in_sample["eer"],
            "exploratory_full_dataset_mindcf_leaked": leaked_in_sample["mindcf_p001"],
        })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_loro_verification_summary.csv", loro_verif_summary_rows)

    # 8. Paired Clustered Bootstrap Uncertainty (B=2000, seed=42)
    print("\n[Stage 5A] Computing Paired Clustered Bootstrap Uncertainty (B=2000, seed=42)...")
    rng = np.random.default_rng(42)
    B = 2000

    diar_by_method_rec: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in per_rec_diar_rows:
        diar_by_method_rec.setdefault(r["method"], {})[r["rec_id"]] = r

    rec_ids_ordered = [e["rec_id"] for e in manifest]
    n_rec = len(rec_ids_ordered)
    boot_indices = rng.integers(0, n_rec, size=(B, n_rec))

    clustered_bootstrap_metric_rows: List[Dict[str, Any]] = []
    paired_diff_rows: List[Dict[str, Any]] = []

    metrics_to_eval = [
        "der_collar0_ovl",
        "der_collar025_noovl",
        "jer_collar0",
        "cpwer",
        "sawer",
        "micro_sacc",
        "macro_sacc",
        "speaker_count_error_abs",
    ]

    for m_name, cond_id in methods:
        for met in metrics_to_eval:
            vals = np.array([diar_by_method_rec[m_name][rid][met] for rid in rec_ids_ordered], dtype=np.float64)
            pt = float(np.mean(vals))
            b_means = np.mean(vals[boot_indices], axis=1)
            ci_lo = float(np.percentile(b_means, 2.5))
            ci_hi = float(np.percentile(b_means, 97.5))
            clustered_bootstrap_metric_rows.append({
                "method_id": cond_id,
                "method": m_name,
                "metric": f"macro_{met}",
                "point_estimate": round(pt, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "ci95_width": round(ci_hi - ci_lo, 4),
            })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_clustered_bootstrap_metrics.csv", clustered_bootstrap_metric_rows)

    # Paired comparisons
    comparisons = [
        ("condition_B_resnet293_mono_baseline", "condition_A_titanet_dfn_baseline", "B_vs_A"),
        ("condition_C_equal_weight_affinity", "condition_A_titanet_dfn_baseline", "C_vs_A"),
        ("condition_C_equal_weight_affinity", "condition_B_resnet293_mono_baseline", "C_vs_B"),
        ("condition_D_three_feature_reliability_gate", "condition_A_titanet_dfn_baseline", "D_vs_A"),
        ("condition_D_three_feature_reliability_gate", "condition_B_resnet293_mono_baseline", "D_vs_B"),
        ("condition_D_three_feature_reliability_gate", "condition_C_equal_weight_affinity", "D_vs_C"),
        ("condition_E_gate_ablation_no_resnet_l2", "condition_D_three_feature_reliability_gate", "E_vs_D"),
        ("condition_F_gate_ablation_no_dfn_nr", "condition_D_three_feature_reliability_gate", "F_vs_D"),
        ("condition_G_gate_ablation_no_vad_conf", "condition_D_three_feature_reliability_gate", "G_vs_D"),
    ]

    for m_a, m_b, label in comparisons:
        for met in metrics_to_eval:
            v_a = np.array([diar_by_method_rec[m_a][rid][met] for rid in rec_ids_ordered], dtype=np.float64)
            v_b = np.array([diar_by_method_rec[m_b][rid][met] for rid in rec_ids_ordered], dtype=np.float64)
            diff = v_a - v_b
            pt_diff = float(np.mean(diff))
            b_diffs = np.mean(diff[boot_indices], axis=1)
            ci_lo = float(np.percentile(b_diffs, 2.5))
            ci_hi = float(np.percentile(b_diffs, 97.5))
            inc_zero = bool(ci_lo <= 0.0 <= ci_hi)

            # Wins/ties/losses: for error metrics (der, cpwer, etc.), lower is better; for acc (sacc), higher is better
            if "sacc" in met:
                wins_a = int(np.sum(v_a > v_b))
                wins_b = int(np.sum(v_b > v_a))
            else:
                wins_a = int(np.sum(v_a < v_b))
                wins_b = int(np.sum(v_b < v_a))
            ties = int(np.sum(v_a == v_b))

            paired_diff_rows.append({
                "comparison_label": label,
                "method_a": m_a,
                "method_b": m_b,
                "metric": f"delta_macro_{met} (A - B)",
                "point_diff_a_minus_b": round(pt_diff, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "ci95_width": round(ci_hi - ci_lo, 4),
                "includes_zero": inc_zero,
                "rec_wins_a_better": wins_a,
                "rec_wins_b_better": wins_b,
                "rec_ties": ties,
            })

    write_csv(STAGE5A_DIR / "metrics" / "stage5a_paired_bootstrap_differences.csv", paired_diff_rows)

    # 9. Master JSON Compilation
    master_results = {
        "stage": "Stage 5A — Controlled Validation of Hybrid Speaker-Affinity Fusion",
        "git_commit": git_hash,
        "pre_execution_audit": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "seed": 42,
        },
        "summary_diarization": summary_diar_rows,
        "loro_verification_summary": loro_verif_summary_rows,
        "ablation_evaluation": gate_eval_rows,
    }

    with open(STAGE5A_DIR / "metrics" / "stage5a_master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\n[Stage 5A] Execution completed successfully! All metrics and artifacts saved under research_stages/stage5a/.")


if __name__ == "__main__":
    main()
