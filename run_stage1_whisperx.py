"""
run_stage1_whisperx.py (V5 - BWC Optimized)
==========================================
Specialized for Body-Worn Camera (BWC) audio.
Prioritizes high recall for distant speakers and noise-robustness.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scripts.utils.audio_utils import load_audio, SILERO_SR, get_duration_sec, save_audio_clip, extract_segment
from scripts.utils.device import get_device
from scripts.utils.logger import get_logger

PROJECT_ROOT = Path.cwd()
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
VAD_OUTPUT_DIR = PROJECT_ROOT / "vad_outputs_whisperx"
SPEECH_SEGMENTS_DIR = PROJECT_ROOT / "speech_segments_whisperx"
LOGS_DIR = PROJECT_ROOT / "logs"

VAD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger(name="bwc_vad", level="INFO", log_dir=LOGS_DIR, log_filename="bwc_vad_pipeline.log")

def apply_min_cut(scores, start_idx, end_idx, max_len):
    """Recursively split a segment at the quietest point."""
    duration = end_idx - start_idx
    if duration <= max_len:
        return [(start_idx, end_idx)]

    # Search window for surgical cut
    search_start = start_idx + max_len // 2
    search_end = start_idx + max_len
    search_end = min(search_end, end_idx)
    
    if search_start >= search_end:
        return [(start_idx, end_idx)]

    min_rel_idx = np.argmin(scores[search_start:search_end])
    cut_idx = search_start + min_rel_idx

    return apply_min_cut(scores, start_idx, cut_idx, max_len) + apply_min_cut(scores, cut_idx, end_idx, max_len)

def run_bwc_optimized_vad(wav_path, model, utils, device):
    t0 = time.perf_counter()
    waveform, sr = load_audio(wav_path, target_sr=SILERO_SR, mono=True)
    total_duration = get_duration_sec(waveform, sr)
    
    # BWC-SPECIALIZED PARAMETERS
    threshold_onset = 0.30  # More sensitive for far-away subjects
    threshold_offset = 0.25 # Latch onto speech to bridge noise
    min_silence_ms = 300     # Bridge gaps in noisy environments
    speech_pad_ms = 400     # Generous padding to avoid clipping word tails
    
    window_size = 512
    timestep_sec = window_size / sr
    max_len_frames = int(25.0 / timestep_sec)
    padding_frames = int(speech_pad_ms / 1000.0 / timestep_sec)
    
    get_speech_timestamps, _, _, VADIterator, _ = utils
    vad_iterator = VADIterator(
        model, 
        threshold=threshold_onset, 
        sampling_rate=sr, 
        min_silence_duration_ms=min_silence_ms
    )
    
    scores = []
    raw_segments = []
    waveform_1d = waveform.squeeze(0)
    
    model.reset_states()
    for i in range(0, waveform_1d.shape[0], window_size):
        chunk = waveform_1d[i : i + window_size]
        if chunk.shape[0] < window_size:
            chunk = torch.nn.functional.pad(chunk, (0, window_size - chunk.shape[0]))
        
        chunk = chunk.to(device)
        with torch.no_grad():
            prob = model(chunk, sr).item()
            scores.append(prob)
            
            speech_dict = vad_iterator(chunk, return_seconds=False)
            if speech_dict:
                if "start" in speech_dict:
                    raw_segments.append({"start": speech_dict["start"] // window_size})
                if "end" in speech_dict and raw_segments:
                    raw_segments[-1]["end"] = speech_dict["end"] // window_size

    if raw_segments and "end" not in raw_segments[-1]:
        raw_segments[-1]["end"] = len(scores)

    scores_arr = np.array(scores)
    
    # 1. Pad natural boundaries
    padded_segments = []
    for seg in raw_segments:
        s = max(0, seg["start"] - padding_frames)
        e = min(len(scores_arr), seg["end"] + padding_frames)
        padded_segments.append((s, e))

    # 2. Apply surgical cuts for ASR-readiness
    final_indices = []
    for s_idx, e_idx in padded_segments:
        final_indices.extend(apply_min_cut(scores_arr, s_idx, e_idx, max_len_frames))
    
    # 3. Final Formatting
    segments = []
    for i, (s_idx, e_idx) in enumerate(final_indices):
        start_s = round(s_idx * timestep_sec, 3)
        end_s = round(e_idx * timestep_sec, 3)
        
        if (end_s - start_s) >= 0.2:
            segments.append({
                "start": start_s,
                "end": end_s,
                "duration": round(end_s - start_s, 3)
            })

    # 4. Export
    output_json = VAD_OUTPUT_DIR / f"{wav_path.stem}_vad.json"
    with open(output_json, "w") as f:
        json.dump({
            "file_name": wav_path.name, 
            "pipeline": "bwc_optimized_v5",
            "duration_sec": total_duration, 
            "segments": segments
        }, f, indent=2)
        
    # 5. Clips
    clip_dir = SPEECH_SEGMENTS_DIR / wav_path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(segments):
        clip = extract_segment(waveform, sr, seg["start"], seg["end"])
        save_audio_clip(clip, sr, clip_dir / f"{wav_path.stem}_seg{i:03d}_{seg['start']:.3f}_{seg['end']:.3f}.wav")

    log.info(f"BWC-Optimized {wav_path.name}: {len(segments)} segments in {time.perf_counter()-t0:.2f}s")

def main():
    device = get_device()
    repo = PROJECT_ROOT / "external" / "silero-vad"
    model, utils = torch.hub.load(repo_or_dir=str(repo), model="silero_vad", source="local", onnx=False)
    model = model.to(device)
    
    audio_files = list(RAW_AUDIO_DIR.glob("*.wav"))
    
    for audio_path in tqdm(audio_files):
        run_bwc_optimized_vad(audio_path, model, utils, device)

if __name__ == "__main__":
    main()
