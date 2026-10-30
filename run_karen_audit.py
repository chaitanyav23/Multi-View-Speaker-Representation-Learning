import os
import json
import torch
import yaml
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
import whisperx
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import soundfile as sf
import subprocess

# Paths
KAREN_DIR = Path("experiments/karen_11A_audit")
AUDIO_PATH = KAREN_DIR / "audio" / "karen_normalized.wav"
GROUNDING_PATH = KAREN_DIR / "grounding" / "karen_grounding.json"
ORACLE_RTTM_PATH = KAREN_DIR / "grounding" / "karen_oracle.rttm"
OUTPUT_DIR = KAREN_DIR / "outputs"
DIAG_DIR = KAREN_DIR / "diagnostics"
SEGMENTS_DIR = KAREN_DIR / "vad_segments"

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pyannote_vad_pipeline(hf_token):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=hf_token)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {"checkpoint": checkpoint, "revision": revision}
    return Pipeline.from_pretrained(config, token=hf_token)

def run_vad(audio_path, hf_token):
    print(">>> Running Pyannote VAD Grounding...")
    pipeline = load_pyannote_vad_pipeline(hf_token).to(torch.device(DEVICE))
    vad = pipeline(str(audio_path))
    segments = []
    for s in vad.get_timeline().support():
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3)})
    
    print(f"Found {len(segments)} raw segments.")
    # Merge gap logic (Phase 11 consistency)
    merge_gap = 0.15
    cleaned = []
    if segments:
        curr = segments[0]
        for next_seg in segments[1:]:
            if next_seg["start"] - curr["end"] < merge_gap:
                curr["end"] = next_seg["end"]
            else:
                cleaned.append(curr)
                curr = next_seg
        cleaned.append(curr)
    
    print(f"Merged into {len(cleaned)} segments.")
    with open(GROUNDING_PATH, "w") as f:
        json.dump({"segments": cleaned}, f, indent=2)
    
    with open(ORACLE_RTTM_PATH, "w") as f:
        for seg in cleaned:
            dur = seg["end"] - seg["start"]
            f.write(f"SPEAKER karen 1 {seg['start']:.3f} {dur:.3f} <NA> <NA> speaker <NA> <NA>\n")
    return cleaned

def run_asr_align(audio_path, vad_segments):
    print(">>> Running WhisperX ASR + Alignment...")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    audio = whisperx.load_audio(str(audio_path))
    sr = 16000
    transcript_segments = []
    for seg in tqdm(vad_segments, desc="ASR"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(audio_slice, language="en", task="transcribe", condition_on_previous_text=False)
        text = "".join([s.text for s in res]).strip()
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})
    
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    aligned_res = whisperx.align(transcript_segments, align_model, align_metadata, audio, DEVICE)
    return aligned_res

def run_msdd(audio_path, oracle_rttm_path):
    print(">>> Running NeMo Oracle MSDD...")
    manifest_path = KAREN_DIR / "karen_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0, "duration": None, "label": "infer", "text": "-",
            "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)
    
    msdd_out_dir = KAREN_DIR / "msdd_raw"
    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(msdd_out_dir),
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
    msdd_model.diarize()
    
    rttm_out_path = msdd_out_dir / "pred_rttms" / "karen_normalized.rttm"
    msdd_segments = []
    with open(rttm_out_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                msdd_segments.append({"start": start, "end": start+dur, "speaker": spk})
    return msdd_segments

def fuse_results(aligned_res, msdd_segments):
    print(">>> Fusing ASR and Diarization...")
    tolerance = 0.5
    for seg in aligned_res["segments"]:
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
                nearest_spk, min_dist = None, float('inf')
                for msdd in msdd_segments:
                    dist = min(abs(msdd["start"] - w_end), abs(msdd["end"] - w_start))
                    if dist < min_dist: min_dist, nearest_spk = dist, msdd["speaker"]
                word_info["speaker"] = nearest_spk if min_dist <= tolerance else "UNK"
        
        spks = [w["speaker"] for w in seg.get("words", []) if w.get("speaker") != "UNK"]
        if spks:
            from collections import Counter
            seg["speaker"] = Counter(spks).most_common(1)[0][0]
        else:
            seg["speaker"] = "UNK"
    return aligned_res

def main():
    vad_segments = run_vad(AUDIO_PATH, HF_TOKEN)
    msdd_segments = run_msdd(AUDIO_PATH, ORACLE_RTTM_PATH)
    aligned_res = run_asr_align(AUDIO_PATH, vad_segments)
    final_res = fuse_results(aligned_res, msdd_segments)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "karen_final.json", "w") as f:
        json.dump(final_res, f, indent=2)
    with open(OUTPUT_DIR / "karen_final.txt", "w") as f:
        for s in final_res["segments"]:
            f.write(f"[{s['start']:06.2f}s] {s['speaker']:10}: {s['text']}\n")
    
    # Save VAD segments for audit
    print(">>> Saving VAD segments...")
    audio, sr = sf.read(AUDIO_PATH)
    for i, seg in enumerate(vad_segments):
        start_samp = int(seg["start"] * sr)
        end_samp = int(seg["end"] * sr)
        clip = audio[start_samp:end_samp]
        sf.write(SEGMENTS_DIR / f"karen_seg{i:03d}_{seg['start']:.2f}_{seg['end']:.2f}.wav", clip, sr)

    print("\n>>> Pipeline Complete. Now run evaluation.")

if __name__ == "__main__":
    main()
