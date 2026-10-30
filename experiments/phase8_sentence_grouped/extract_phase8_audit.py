"""
experiments/phase8_sentence_grouped/extract_phase8_audit.py
==========================================================
Extracts audio clips for ASR deletions and diarization errors (±2.0s context).
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

import numpy as np
import torchaudio

from evaluate_diarization_improved import parse_ground_truth

PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase8_sentence_grouped"
GT_DIR = PROJECT_ROOT / "ground_truth"
AUDIO_DIR = PHASE_DIR / "audio"
OUTPUT_DIR = PHASE_DIR / "outputs"
AUDIT_DIR = PHASE_DIR / "error_audit"

def extract_clip(audio_tensor, sr, start_s, end_s, output_path, buffer=2.0):
    t_start = max(0, start_s - buffer)
    t_end = min(audio_tensor.shape[1] / sr, end_s + buffer)
    start_frame = int(t_start * sr)
    end_frame = int(t_end * sr)
    clip = audio_tensor[:, start_frame:end_frame]
    torchaudio.save(str(output_path), clip, sr)

def audit_asr_deletions(stem, gt_segments, pred_segments, audio_tensor, sr):
    audit_dir = AUDIT_DIR / "asr_deletions"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for gt_seg in gt_segments:
        found = False
        for p_seg in pred_segments:
            if abs(gt_seg["start"] - p_seg["start"]) < 0.5 and p_seg.get("text", "").strip():
                found = True
                break
        if not found:
            out_path = audit_dir / f"{stem}_deletion_{gt_seg['start']:.2f}_{gt_seg['end']:.2f}.wav"
            extract_clip(audio_tensor, sr, gt_seg["start"], gt_seg["end"], out_path)

def audit_diarization_errors(stem, gt_segments, pred_segments, audio_tensor, sr, step=0.1):
    max_t = max([s["end"] for s in gt_segments + pred_segments])
    n_frames = int(max_t / step) + 1
    gt_map = {s["speaker"]: i for i, s in enumerate(gt_segments)}
    pred_map = {s.get("speaker", "UNKNOWN"): i for i, s in enumerate(pred_segments)}
    gt_frames = np.full(n_frames, -1, dtype=int)
    pred_frames = np.full(n_frames, -1, dtype=int)
    for s in gt_segments: gt_frames[int(s["start"]/step):int(s["end"]/step)] = gt_map.get(s["speaker"], 0)
    for s in pred_segments: pred_frames[int(s["start"]/step):int(s["end"]/step)] = pred_map.get(s.get("speaker", "UNKNOWN"), 0)
    for err in ["diar_miss", "diar_false_alarm", "diar_confusion"]: (AUDIT_DIR / err).mkdir(parents=True, exist_ok=True)
    for i in range(1, n_frames):
        t = i * step
        gt_spk, pred_spk = gt_frames[i], pred_frames[i]
        err_type = None
        if gt_spk != -1 and pred_spk == -1: err_type = "diar_miss"
        elif gt_spk == -1 and pred_spk != -1: err_type = "diar_false_alarm"
        elif gt_spk != -1 and pred_spk != -1 and gt_spk != pred_spk: err_type = "diar_confusion"
        if err_type and i % 50 == 0:
            out_path = AUDIT_DIR / err_type / f"{stem}_{err_type}_{t:.2f}_{t+step:.2f}.wav"
            extract_clip(audio_tensor, sr, t, t+step, out_path)

def process_audit():
    for pred_path in OUTPUT_DIR.glob("*_final.json"):
        stem = pred_path.name.replace("_final.json", "")
        gt_path, audio_path = GT_DIR / f"{stem}.txt", AUDIO_DIR / f"{stem}.wav"
        if not gt_path.exists() or not audio_path.exists(): continue
        gt_segments = parse_ground_truth(gt_path)
        with open(pred_path, "r") as f: pred_data = json.load(f)
        pred_segments = pred_data.get("segments", [])
        waveform, sr = torchaudio.load(str(audio_path))
        audit_asr_deletions(stem, gt_segments, pred_segments, waveform, sr)
        audit_diarization_errors(stem, gt_segments, pred_segments, waveform, sr)

def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    process_audit()

if __name__ == "__main__":
    main()
