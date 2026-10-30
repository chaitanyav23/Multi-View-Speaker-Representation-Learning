import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf

BASE_DIR = Path("experiments/phase12_karen_evaluation")
AUDIO_PATH = BASE_DIR / "audio_preprocessed/karen_normalized.wav"
SYSTEM_VAD_RTTM = BASE_DIR / "grounding/karen_system_vad.rttm"
ALIGNED_OUTPUT = BASE_DIR / "outputs/asr/karen_aligned.json"

DIAR_OUT_DIR = BASE_DIR / "outputs/diarization"
DIAR_RTTM = DIAR_OUT_DIR / "karen_diarization.rttm"
FUSION_OUTPUT = BASE_DIR / "outputs/fusion/karen_fused.json"
TIMELINE_JSON = BASE_DIR / "outputs/conversational_timeline/karen_conversation.json"
TIMELINE_TXT = BASE_DIR / "outputs/conversational_timeline/karen_conversation.txt"
DIAG_SPK_COUNT = BASE_DIR / "diagnostics/speaker_count_analysis"

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def step7_msdd():
    print("\n--- STEP 7: MSDD DIARIZATION (AUTO SPEAKER COUNT) ---")
    manifest_path = DIAR_OUT_DIR / "karen_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(AUDIO_PATH.absolute()),
            "offset": 0, "duration": None, "label": "infer", "text": "-",
            "num_speakers": None, "rttm_filepath": str(SYSTEM_VAD_RTTM.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(DIAR_OUT_DIR),
            "oracle_vad": True, "collar": 0.25, "ignore_overlap": False,
            "vad": {"model_path": None, "external_vad_manifest": None, "parameters": {}},
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                    "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                    "multiscale_weights": [1, 2, 3, 3, 3], "save_embeddings": True,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_backim_for_multiscale": True, "sigmoid_threshold": [0.7],
                    "use_adaptive_thres": False, "overlap_infer_spk_limit": 5,
                    "diar_window_length": 50, "infer_batch_size": 16,
                    "seq_eval_mode": False, "split_infer": True, "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {"max_num_speakers": 10, "oracle_num_speakers": False, "max_rp_threshold": 0.25, "sparse_search_volume": 30},
            },
        },
        "batch_size": 16, "num_workers": 4, "sample_rate": 16000, "verbose": True, "device": DEVICE,
    })

    msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
    try:
        msdd_model.diarize()
    except Exception as e:
        print(f"MSDD Failed (potentially zero segments or silence): {e}")

    # The output is at DIAR_OUT_DIR / pred_rttms / karen_normalized.rttm
    pred_rttm = DIAR_OUT_DIR / "pred_rttms" / "karen_normalized.rttm"
    if pred_rttm.exists():
        os.system(f"cp {pred_rttm} {DIAR_RTTM}")
    else:
        print("Warning: pred_rttm not found. Creating empty.")
        with open(DIAR_RTTM, "w") as f: pass

    segments = []
    if DIAR_RTTM.exists():
        with open(DIAR_RTTM, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[7]
                    segments.append({"start": start, "end": start+dur, "speaker": spk})
    return segments

def step8_speaker_count_analysis(segments):
    print("\n--- STEP 8: SPEAKER COUNT ANALYSIS ---")
    DIAG_SPK_COUNT.mkdir(parents=True, exist_ok=True)
    spk_durs = {}
    spk_turns = {}
    total_dur = 0
    for s in segments:
        dur = s["end"] - s["start"]
        spk = s["speaker"]
        spk_durs[spk] = spk_durs.get(spk, 0) + dur
        spk_turns[spk] = spk_turns.get(spk, 0) + 1
        total_dur += dur
        
    num_pred = len(spk_durs)
    report_path = DIAG_SPK_COUNT / "speaker_count_analysis.md"
    
    with open(report_path, "w") as f:
        f.write("# SPEAKER COUNT ANALYSIS\n\n")
        f.write(f"- **Ground Truth Speakers:** 6\n") # Given by prompt
        f.write(f"- **Predicted Speakers:** {num_pred}\n")
        f.write(f"- **Difference:** {num_pred - 6}\n\n")
        
        f.write("### Speaker Metrics\n")
        for spk, dur in spk_durs.items():
            pct = (dur / total_dur * 100) if total_dur > 0 else 0
            f.write(f"- **{spk}**: {dur:.2f}s ({pct:.1f}%), {spk_turns[spk]} turns\n")

    print(f"Predicted Speakers: {num_pred}")
    print(f"Analysis saved to {report_path}")

def step9_fusion(aligned_res, msdd_segments):
    print("\n--- STEP 9: MAXIMUM OVERLAP FUSION ---")
    for seg in aligned_res.get("segments", []):
        for word_info in seg.get("words", []):
            if "start" not in word_info or "end" not in word_info: continue
            w_start, w_end = word_info["start"], word_info["end"]
            
            overlaps = []
            for msdd in msdd_segments:
                if msdd["start"] <= w_end and msdd["end"] >= w_start:
                    o_start = max(w_start, msdd["start"])
                    o_end = min(w_end, msdd["end"])
                    o_dur = o_end - o_start
                    if o_dur > 0: overlaps.append({"speaker": msdd["speaker"], "overlap_dur": o_dur})
            
            if overlaps:
                overlaps.sort(key=lambda x: x["overlap_dur"], reverse=True)
                word_info["speaker"] = overlaps[0]["speaker"]
            else:
                word_info["speaker"] = "UNKNOWN"
        
        # Segment speaker is majority of word speakers
        spks = [w["speaker"] for w in seg.get("words", []) if w.get("speaker", "UNKNOWN") != "UNKNOWN"]
        if spks:
            from collections import Counter
            seg["speaker"] = Counter(spks).most_common(1)[0][0]
        else:
            seg["speaker"] = "UNKNOWN"
            
    with open(FUSION_OUTPUT, "w") as f:
        json.dump(aligned_res, f, indent=2)
    return aligned_res

def step10_conversational_timeline(fused_res):
    print("\n--- STEP 10: CONVERSATIONAL TIMELINE ---")
    with open(TIMELINE_JSON, "w") as f:
        json.dump(fused_res, f, indent=2)
        
    with open(TIMELINE_TXT, "w") as f:
        for s in fused_res.get("segments", []):
            f.write(f"{s.get('speaker', 'UNKNOWN')} [{s['start']:06.2f} - {s['end']:06.2f}]\n")
            f.write(f"{s.get('text', '')}\n\n")

if __name__ == "__main__":
    msdd_segments = step7_msdd()
    step8_speaker_count_analysis(msdd_segments)
    
    with open(ALIGNED_OUTPUT, "r") as f:
        aligned_res = json.load(f)
        
    fused_res = step9_fusion(aligned_res, msdd_segments)
    step10_conversational_timeline(fused_res)
