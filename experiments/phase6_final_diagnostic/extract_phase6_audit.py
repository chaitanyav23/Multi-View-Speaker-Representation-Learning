"""
experiments/phase6_final_diagnostic/extract_phase6_audit.py
=========================================================
Surgically extracts audio clips for ASR Deletions and Diarization Errors.
Includes ±2.0s context buffer.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
import torchaudio
from evaluate_asr_improved import normalize_text_improved, levenshtein_distance
from evaluate_diarization_improved import parse_ground_truth

# Paths
PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase6_final_diagnostic"
GT_DIR = PROJECT_ROOT / "ground_truth"
AUDIT_BASE_DIR = PHASE_DIR / "error_audit"
AUDIO_PROC_DIR = PHASE_DIR / "audio_preprocessed"

def extract_clip(audio_tensor, sr, start_s, end_s, output_path, buffer=2.0):
    """Extract audio clip with buffer."""
    t_start = max(0, start_s - buffer)
    t_end = min(audio_tensor.shape[1] / sr, end_s + buffer)
    
    start_frame = int(t_start * sr)
    end_frame = int(t_end * sr)
    
    clip = audio_tensor[:, start_frame:end_frame]
    torchaudio.save(str(output_path), clip, sr)

def audit_asr_deletions(enhancer, stem, gt_segments, pred_segments, audio_tensor, sr):
    """
    Find words deleted by ASR. 
    Note: For surgical accuracy, this would need word-level alignment with GT.
    As a heuristic, we identify segments with high deletion counts.
    """
    audit_dir = AUDIT_BASE_DIR / enhancer / "asr_deletions"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple heuristic: If a GT segment has no corresponding pred segment text, it's a deletion.
    for i, gt_seg in enumerate(gt_segments):
        found = False
        for p_seg in pred_segments:
            if abs(gt_seg["start"] - p_seg["start"]) < 0.5:
                if len(p_seg["text"].strip()) > 0:
                    found = True
                    break
        if not found:
            out_path = audit_dir / f"deletion_{gt_seg['start']:.2f}_{gt_seg['end']:.2f}.wav"
            extract_clip(audio_tensor, sr, gt_seg["start"], gt_seg["end"], out_path)

def audit_diarization_errors(enhancer, stem, gt_segments, pred_segments, audio_tensor, sr, step=0.1):
    """Identify Miss, FA, and Confusion zones and extract clips."""
    max_t = max([s["end"] for s in gt_segments + pred_segments])
    n_frames = int(max_t / step) + 1
    
    gt_map = {s["speaker"]: i for i, s in enumerate(gt_segments)}
    pred_map = {s["speaker"]: i for i, s in enumerate(pred_segments)}
    
    gt_frames = np.full(n_frames, -1, dtype=int)
    pred_frames = np.full(n_frames, -1, dtype=int)
    
    for s in gt_segments: gt_frames[int(s["start"]/step):int(s["end"]/step)] = gt_map.get(s["speaker"], 0)
    for s in pred_segments: pred_frames[int(s["start"]/step):int(s["end"]/step)] = pred_map.get(s["speaker"], 0)

    error_types = ["diar_miss", "diar_false_alarm", "diar_confusion"]
    for err in error_types: (AUDIT_BASE_DIR / enhancer / err).mkdir(parents=True, exist_ok=True)

    # Simplified error zone detection
    for i in range(1, n_frames):
        t = i * step
        gt_spk = gt_frames[i]
        pred_spk = pred_frames[i]
        
        err_type = None
        if gt_spk != -1 and pred_spk == -1: err_type = "diar_miss"
        elif gt_spk == -1 and pred_spk != -1: err_type = "diar_false_alarm"
        elif gt_spk != -1 and pred_spk != -1 and gt_spk != pred_spk: err_type = "diar_confusion"
        
        if err_type:
            # Only extract every 5 seconds of error to avoid flooding
            if i % 50 == 0:
                out_path = AUDIT_BASE_DIR / enhancer / err_type / f"{err_type}_{t:.2f}_{t+step:.2f}.wav"
                extract_clip(audio_tensor, sr, t, t+step, out_path)

def process_audit(enhancer):
    print(f"Auditing: {enhancer}")
    output_dir = PHASE_DIR / "outputs" / enhancer
    for pred_path in output_dir.glob("*_final.json"):
        stem = pred_path.name.replace("_final.json", "")
        gt_path = GT_DIR / f"{stem}.txt"
        audio_path = AUDIO_PROC_DIR / f"{stem}_{'norm' if enhancer == 'baseline' else enhancer}.wav"
        
        if not gt_path.exists() or not audio_path.exists(): continue
        
        # Load
        gt_segments = parse_ground_truth(gt_path)
        with open(pred_path, "r") as f: pred_data = json.load(f)
        pred_segments = pred_data["segments"]
        waveform, sr = torchaudio.load(str(audio_path))
        
        audit_asr_deletions(enhancer, stem, gt_segments, pred_segments, waveform, sr)
        audit_diarization_errors(enhancer, stem, gt_segments, pred_segments, waveform, sr)

def main():
    enhancers = ["baseline", "deepfilternet", "speechbrain"]
    for enh in enhancers:
        process_audit(enh)

if __name__ == "__main__":
    main()
