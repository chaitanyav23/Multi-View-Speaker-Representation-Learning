#!/usr/bin/env python3
"""
Audit baseline VAD across all 8 recordings.
Computes frame-level Precision, Recall, F1, Miss, and False Alarm rates
under both reference-speech normalization and reference-silence normalization.
"""

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
import common as p20_common

manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

print(f"{'Recording':15s} | {'Path':6s} | {'Precision':9s} | {'Recall':9s} | {'F1':7s} | {'Miss(ref)':9s} | {'FA(sil)':9s} | {'FA(ref)':9s} | {'GT Dur':8s} | {'Pred Dur':8s}")
print("-" * 105)

all_rows = []

for entry in manifest:
    rec_id = entry["rec_id"]
    dur = float(entry["duration_sec"])
    gt_rttm_path = REPO_ROOT / entry["rttm_uniform"]
    gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
    total_gt_speech = sum(s["end"] - s["start"] for s in gt_rttm_segs)

    for path_name in ["pathA", "pathB", "mono"]:
        vad_file = REPO_ROOT / f"research_stages/stage2_baseline/vad/{rec_id}_{path_name}_vad.json"
        if not vad_file.exists():
            continue
        vad_segs = p20_common.read_json(vad_file)["segments"]
        total_pred_speech = sum(s["end"] - s["start"] for s in vad_segs)

        eval_res = p20_common.vad_metrics(gt_rttm_segs, vad_segs, dur, resolution=0.01)
        fp_sec = eval_res["fp_frames"] * 0.01
        fn_sec = eval_res["fn_frames"] * 0.01
        fa_ref = fp_sec / total_gt_speech if total_gt_speech > 0 else 0.0
        miss_ref = fn_sec / total_gt_speech if total_gt_speech > 0 else 0.0

        p = eval_res["precision"] * 100
        r = eval_res["recall"] * 100
        f = eval_res["f1"] * 100
        m = eval_res["miss"] * 100
        fa_sil = eval_res["false_alarm"] * 100
        fa_r = fa_ref * 100

        all_rows.append({
            "rec_id": rec_id,
            "path": path_name,
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1": eval_res["f1"],
            "miss_ref": miss_ref,
            "fa_silence": eval_res["false_alarm"],
            "fa_ref": fa_ref,
            "gt_dur": total_gt_speech,
            "pred_dur": total_pred_speech,
            "seg_count": len(vad_segs)
        })

        print(f"{rec_id[:15]:15s} | {path_name:6s} | {p:8.2f}% | {r:8.2f}% | {f:6.2f}% | {m:8.2f}% | {fa_sil:8.2f}% | {fa_r:8.2f}% | {total_gt_speech:7.1f}s | {total_pred_speech:7.1f}s")

print("-" * 105)
for path_name in ["pathA", "pathB", "mono"]:
    subset = [r for r in all_rows if r["path"] == path_name]
    m_p = np.mean([r["precision"] for r in subset]) * 100
    m_r = np.mean([r["recall"] for r in subset]) * 100
    m_f = np.mean([r["f1"] for r in subset]) * 100
    m_m = np.mean([r["miss_ref"] for r in subset]) * 100
    m_fa_sil = np.mean([r["fa_silence"] for r in subset]) * 100
    m_fa_ref = np.mean([r["fa_ref"] for r in subset]) * 100
    print(f"MACRO {path_name:6s} | --     | {m_p:8.2f}% | {m_r:8.2f}% | {m_f:6.2f}% | {m_m:8.2f}% | {m_fa_sil:8.2f}% | {m_fa_ref:8.2f}% | --      | --")
