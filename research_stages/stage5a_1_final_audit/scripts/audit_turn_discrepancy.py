#!/usr/bin/env python3
"""
Audit Turn Discrepancy:
Investigates the exact difference between:
- Stage 4.1 / Stage 5A: 254 turns, 9,817 pairs
- Stage 5A.1: 250 turns, 9,489 pairs
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))

import common as p20_common
from run_stage4_complementarity import load_cache

manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

# 1. Reconstruct Stage 5A method
def get_stage5a_turns():
    turns = []
    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
        t_d, _, sub_meta_dfn = load_cache("dfn", rec_id, 4)
        _, r_d, sub_meta_mono = load_cache("16k_mono", rec_id, 4)

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

            if dur >= 0.5 and idxs.size > 0:
                turns.append({
                    "rec_id": rec_id,
                    "seg_idx": int(g["seg_idx"]),
                    "speaker": str(g["speaker"]),
                    "global_speaker_id": str(g["global_speaker_id"]),
                    "start": gst,
                    "end": gen,
                    "duration": round(dur, 3),
                    "num_subsegments_pooled": int(idxs.size),
                    "pooled_idxs": [int(x) for x in idxs],
                })
    return turns

# 2. Reconstruct Stage 5A.1 method
def get_stage5a_1_turns():
    turns = []
    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_payload = p20_common.read_json(REPO_ROOT / entry["json_uniform"])
        gt_segments = gt_payload["segments"]
        t_d, _, sub_meta_tita = load_cache("dfn", rec_id, 4)

        clean_gt = [g for g in gt_segments if g.get("usable_for_clean_speaker_eval", False)]

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

            turns.append({
                "rec_id": rec_id,
                "seg_idx": int(g["seg_idx"]),
                "speaker": str(g["speaker"]),
                "global_speaker_id": str(g["global_speaker_id"]),
                "start": gst,
                "end": gen,
                "duration": round(gen - gst, 3),
                "num_subsegments_pooled": int(idxs.size),
                "pooled_idxs": [int(x) for x in idxs],
            })
    return turns

turns_5a = get_stage5a_turns()
turns_5a_1 = get_stage5a_1_turns()

print(f"Total turns: Stage 5A = {len(turns_5a)}, Stage 5A.1 = {len(turns_5a_1)}")

# Find differing turns
keys_5a = {(t["rec_id"], t["seg_idx"]): t for t in turns_5a}
keys_5a_1 = {(t["rec_id"], t["seg_idx"]): t for t in turns_5a_1}

diff_in_5a_only = [t for k, t in keys_5a.items() if k not in keys_5a_1]
diff_in_5a_1_only = [t for k, t in keys_5a_1.items() if k not in keys_5a]

print("\nTurns in Stage 5A but NOT in Stage 5A.1:")
for t in diff_in_5a_only:
    print(f"  Recording: {t['rec_id']}, seg_idx: {t['seg_idx']}, speaker: {t['speaker']}, start: {t['start']}, end: {t['end']}, dur: {t['duration']}, pooled: {t['num_subsegments_pooled']}")

print("\nTurns in Stage 5A.1 but NOT in Stage 5A:")
for t in diff_in_5a_1_only:
    print(f"  Recording: {t['rec_id']}, seg_idx: {t['seg_idx']}, speaker: {t['speaker']}")

# Build pair counts for both
def build_pairs(turns_list):
    pairs = []
    recs = sorted(set(t["rec_id"] for t in turns_list))
    rec_breakdown = {}
    for r in recs:
        rt = [t for t in turns_list if t["rec_id"] == r]
        n = len(rt)
        pos = 0
        neg = 0
        for i in range(n):
            for j in range(i + 1, n):
                same = int(rt[i]["global_speaker_id"] == rt[j]["global_speaker_id"])
                if same:
                    pos += 1
                else:
                    neg += 1
                pairs.append((r, i, j, same))
        rec_breakdown[r] = {
            "clean_turns": n,
            "pos_pairs": pos,
            "neg_pairs": neg,
            "total_pairs": len(pairs),
            "n_choose_2": n * (n - 1) // 2,
            "eligible_for_roc_auc": bool(pos > 0 and neg > 0)
        }
    return pairs, rec_breakdown

pairs_5a, recs_5a = build_pairs(turns_5a)
pairs_5a_1, recs_5a_1 = build_pairs(turns_5a_1)

print(f"\nPairs: Stage 5A = {len(pairs_5a)}, Stage 5A.1 = {len(pairs_5a_1)}")

print("\nRecording-level breakdown for Stage 5A (254 turns):")
for r, v in recs_5a.items():
    print(f"  {r}: turns={v['clean_turns']}, pos={v['pos_pairs']}, neg={v['neg_pairs']}, pairs={v['n_choose_2']}, eligible={v['eligible_for_roc_auc']}")

print("\nRecording-level breakdown for Stage 5A.1 (250 turns):")
for r, v in recs_5a_1.items():
    print(f"  {r}: turns={v['clean_turns']}, pos={v['pos_pairs']}, neg={v['neg_pairs']}, pairs={v['n_choose_2']}, eligible={v['eligible_for_roc_auc']}")

# Save JSON report
out_data = {
    "stage5a_turns_count": len(turns_5a),
    "stage5a_1_turns_count": len(turns_5a_1),
    "diff_in_5a_only": diff_in_5a_only,
    "diff_in_5a_1_only": diff_in_5a_1_only,
    "stage5a_recording_breakdown": recs_5a,
    "stage5a_1_recording_breakdown": recs_5a_1,
}

out_p = REPO_ROOT / "research_stages" / "stage5a_1_final_audit" / "reconciliation" / "turn_pair_reconciliation.json"
out_p.write_text(json.dumps(out_data, indent=2))
print(f"\nSaved reconciliation to {out_p}")
