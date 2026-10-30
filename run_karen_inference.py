import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
import whisperx
from pyannote.audio import Model, Pipeline as PyannotePipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import datetime
from collections import Counter

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

INPUT_AUDIO = PROJECT_ROOT / "raw_audio" / "karen.mp3"
OUTPUT_DIR = PROJECT_ROOT / "karen_inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    # Get total seconds and microseconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def apply_normalization(audio, target_db=-20.0):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        curr_db = 20 * np.log10(rms)
        gain = 10**((target_db - curr_db) / 20)
        return audio * gain, gain
    return audio, 1.0

def merge_segments(segments, max_dur_sec=28.0):
    if not segments: return []
    merged = []
    curr_start, curr_end = segments[0]
    for i in range(1, len(segments)):
        next_start, next_end = segments[i]
        if (next_end - curr_start) <= max_dur_sec:
            curr_end = next_end
        else:
            merged.append({"start": round(curr_start, 3), "end": round(curr_end, 3)})
            curr_start, curr_end = next_start, next_end
    merged.append({"start": round(curr_start, 3), "end": round(curr_end, 3)})
    return merged

def write_rttm(segments, out_path, file_id="karen"):
    with open(out_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            f.write(f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

def get_msdd_segments_from_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({"start": start, "end": start+duration, "speaker": speaker})
    return segments

# Phase 11A Fusion Logic
def assign_speakers_to_words(aligned_res, msdd_segments, tolerance=0.5):
    for seg in aligned_res["segments"]:
        for word_info in seg.get("words", []):
            if "start" not in word_info or "end" not in word_info:
                continue
            w_start, w_end = word_info["start"], word_info["end"]
            
            overlaps = []
            for msdd in msdd_segments:
                if msdd["start"] <= w_end and msdd["end"] >= w_start:
                    overlap_start = max(w_start, msdd["start"])
                    overlap_end = min(w_end, msdd["end"])
                    overlap_dur = overlap_end - overlap_start
                    if overlap_dur > 0:
                        overlaps.append({
                            "speaker": msdd["speaker"],
                            "overlap_dur": overlap_dur
                        })
            
            assigned_spk = None
            if len(overlaps) >= 1:
                overlaps.sort(key=lambda x: x["overlap_dur"], reverse=True)
                assigned_spk = overlaps[0]["speaker"]
            else:
                nearest_spk = None
                min_dist = float('inf')
                for msdd in msdd_segments:
                    dist = min(abs(msdd["start"] - w_end), abs(msdd["end"] - w_start))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spk = msdd["speaker"]
                
                if min_dist <= tolerance and nearest_spk:
                    assigned_spk = nearest_spk
                else:
                    assigned_spk = "UNK"
            word_info["speaker"] = assigned_spk
            
    for seg in aligned_res["segments"]:
        speakers = []
        for w in seg.get("words", []):
            if "speaker" in w and w["speaker"] != "UNK":
                speakers.append(w["speaker"])
        if speakers:
            seg["speaker"] = Counter(speakers).most_common(1)[0][0]
        else:
            seg["speaker"] = "UNK"
    return aligned_res

def main():
    print(f"--- Inference on {INPUT_AUDIO.name} (Phase 11A Architecture) ---")
    
    # 1. LOAD & NORMALIZE
    print("[1/6] Loading & Normalizing Audio...")
    audio = whisperx.load_audio(str(INPUT_AUDIO))
    audio, gain = apply_normalization(audio)
    sr = 16000
    
    # Save normalized version for NeMo (MSDD needs 16k wav)
    norm_wav_path = OUTPUT_DIR / "karen_normalized.wav"
    import soundfile as sf
    sf.write(norm_wav_path, audio, sr)

    # 2. VAD GROUNDING (Pyannote 3.1)
    print("[2/6] Running Pyannote VAD Grounding...")
    vad_model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
    vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
    vad_pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})
    vad_pipeline.to(torch.device(DEVICE))
    
    vad_res = vad_pipeline(str(norm_wav_path))
    raw_segs = [(s.start, s.end) for s in vad_res.itersegments()]
    grounding_segments = merge_segments(raw_segs)
    
    # Create Oracle RTTM from VAD for NeMo MSDD
    oracle_rttm_path = OUTPUT_DIR / "karen_vad_oracle.rttm"
    write_rttm(grounding_segments, oracle_rttm_path, file_id="karen")

    # 3. NeMo MSDD DIARIZATION (Phase 11A Core)
    print("[3/6] Running NeMo MSDD (Automatic Speaker Count)...")
    manifest_path = OUTPUT_DIR / "karen_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(norm_wav_path.absolute()),
            "offset": 0, "duration": None, "label": "infer", "text": "-",
            "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(OUTPUT_DIR / "msdd_outputs"),
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
                "parameters": {
                    "max_num_speakers": 20, 
                    "oracle_num_speakers": False, 
                    "max_rp_threshold": 0.25, 
                    "sparse_search_volume": 30
                },
            },
        },
        "batch_size": 16, "num_workers": 1, "sample_rate": 16000, "verbose": True, "device": DEVICE,
    })

    msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
    msdd_model.diarize()
    
    rttm_dir = OUTPUT_DIR / "msdd_outputs" / "pred_rttms"
    rttm_files = list(rttm_dir.glob("*.rttm"))
    if not rttm_files:
        raise FileNotFoundError(f"No RTTM file found in {rttm_dir}")
    msdd_segments = get_msdd_segments_from_rttm(rttm_files[0])

    # 4. WhisperX ASR
    print("[4/6] Running WhisperX ASR (Manual Slicing)...")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    transcript_segments = []
    for seg in tqdm(grounding_segments, desc="Transcribing"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(audio_slice, language="en", task="transcribe", condition_on_previous_text=False)
        text = "".join([s.text for s in res]).strip()
        if text:
            transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

    # 5. Alignment & Fusion (11A Logic)
    print("[5/6] Performing Word Alignment & 11A Fusion...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    aligned_res = whisperx.align(transcript_segments, model_a, metadata, audio, DEVICE)
    final_result = assign_speakers_to_words(aligned_res, msdd_segments)

    # 6. Formatting & Output
    print("[6/6] Saving Results...")
    # Clean up speaker labels (convert 'speaker_0' to 'Speaker_1' etc)
    spk_map = {}
    next_idx = 1
    
    # Sort results by time
    final_result["segments"].sort(key=lambda x: x['start'])
    
    out_txt = OUTPUT_DIR / "karen_timeline.txt"
    with open(out_txt, "w") as f:
        for i, s in enumerate(final_result["segments"]):
            raw_spk = s.get("speaker", "UNK")
            if raw_spk not in spk_map and raw_spk != "UNK":
                spk_map[raw_spk] = f"Speaker_{next_idx}"
                next_idx += 1
            
            friendly_spk = spk_map.get(raw_spk, "Speaker_Unknown")
            start_ts = format_timestamp(s['start'])
            end_ts = format_timestamp(s['end'])
            
            f.write(f"{friendly_spk} [{start_ts} - {end_ts}]\n")
            f.write(f"{s['text']}\n\n")

    # Save JSON for programmatic use
    with open(OUTPUT_DIR / "karen_timeline.json", "w") as f:
        json.dump(final_result, f, indent=2, default=str)

    print(f"\n[SUCCESS] Karen Inference Complete!")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
