"""
experiments/phase6_final_diagnostic/run_phase6_main.py
=====================================================
Master orchestrator for Phase 6: Multi-Enhancer Pipeline.
Iterates through Baseline, DeepFilterNet, and SpeechBrain.
"""

import os
import json
import torch
import time
import argparse
import numpy as np
import pandas as pd
import whisperx
import yaml
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from dotenv import dotenv_values
import importlib.util
import subprocess

# Paths
PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase6_final_diagnostic"
AUDIO_OUT_DIR = PHASE_DIR / "audio_preprocessed"
GROUNDING_OUT_DIR = PHASE_DIR / "grounding"
OUTPUT_BASE_DIR = PHASE_DIR / "outputs"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

# Auth & Device
ENV_PATH = EXTERNAL_WX_DIR / ".env"
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_command(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command Failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

def normalize_audio(input_path, output_path):
    """Normalize loudness to -20 LUFS."""
    if output_path.exists(): return
    print(f"Normalizing: {input_path.name}")
    run_command([
        "ffmpeg", "-i", str(input_path), 
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11", 
        "-ar", "16000", "-ac", "1", 
        str(output_path), "-y"
    ])

def enhance_audio(input_path, output_path, method):
    """Enhance audio using DeepFilterNet or SpeechBrain."""
    if output_path.exists(): return
    print(f"Enhancing ({method}): {input_path.name}")
    tmp_dir = output_path.parent / f"_tmp_{method}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        if method == "deepfilternet":
            # Try GPU first
            res = subprocess.run([
                "python3", "experiments/phase2_enhancement/pipeline_enhancement_deepfilternet.py",
                "--file", str(input_path),
                "--output-dir", str(tmp_dir)
            ], capture_output=True, text=True)
            
            if res.returncode != 0:
                print(f"DeepFilterNet GPU failed, trying CPU fallback...")
                run_command([
                    "python3", "experiments/phase2_enhancement/pipeline_enhancement_deepfilternet.py",
                    "--file", str(input_path),
                    "--output-dir", str(tmp_dir),
                    "--device", "cpu"
                ])
        elif method == "speechbrain":
            run_command([
                "python3", "experiments/phase2_enhancement/pipeline_enhancement_speechbrain.py",
                "--file", str(input_path),
                "--output-dir", str(tmp_dir)
            ])
    except Exception as e:
        print(f"Enhancement Failed for {input_path.name}: {e}")
        raise
    if method == "deepfilternet":
        candidates = [
            tmp_dir / input_path.name,
            tmp_dir / f"{input_path.stem}_DeepFilterNet3.wav",
        ]
    else:
        candidates = [tmp_dir / input_path.name]
    for candidate in candidates:
        if candidate.exists():
            os.replace(candidate, output_path)
            return
    raise FileNotFoundError(f"Enhanced audio not found for {input_path.name} in {tmp_dir}")

def load_pyannote_vad_pipeline(hf_token):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=hf_token)
    if config_path is None:
        raise ValueError(f"Could not download pipeline config for {model_id}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {
            "checkpoint": checkpoint,
            "revision": revision,
        }
    return Pipeline.from_pretrained(config, token=hf_token)

def run_vad_pyannote31(audio_path, output_json, hf_token, merge_gap, min_duration, overwrite=False):
    """Run Pyannote 3.1 VAD and clean segments."""
    if output_json.exists() and not overwrite:
        return
    print(f"Running VAD: {audio_path.name}")
    pipeline = load_pyannote_vad_pipeline(hf_token).to(torch.device(DEVICE))
    vad = pipeline(str(audio_path))
    
    segments = []
    for s in vad.get_timeline().support():
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3)})
    
    # Cleaning
    cleaned = []
    if segments:
        curr = segments[0]
        for next_seg in segments[1:]:
            if next_seg["start"] - curr["end"] < merge_gap:
                curr["end"] = next_seg["end"]
            else:
                if (curr["end"] - curr["start"]) >= min_duration:
                    cleaned.append(curr)
                curr = next_seg
        if (curr["end"] - curr["start"]) >= min_duration:
            cleaned.append(curr)

    with open(output_json, "w") as f:
        json.dump({"file": audio_path.name, "segments": cleaned}, f, indent=2)
    return cleaned

def process_pipeline(enhancer, audio_files, asr_model, diarizer, args):
    print(f"\n>>> Starting Pipeline: {enhancer if enhancer else 'Baseline'}")
    enhancer_label = enhancer if enhancer else "baseline"
    output_dir = OUTPUT_BASE_DIR / enhancer_label
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_files:
        stem = audio_path.stem
        # 1. Preprocess
        norm_path = AUDIO_OUT_DIR / f"{stem}_norm.wav"
        normalize_audio(audio_path, norm_path)
        
        proc_path = norm_path
        if enhancer:
            # Skip enhancement for podcast.wav as per user request
            if "podcast" in stem.lower():
                print(f"Skipping enhancement for {stem} (enhancer: {enhancer})")
                continue
                
            proc_path = AUDIO_OUT_DIR / f"{stem}_{enhancer}.wav"
            enhance_audio(norm_path, proc_path, enhancer)

        # 2. VAD
        vad_json = GROUNDING_OUT_DIR / f"{stem}_{enhancer_label}_vad.json"
        vad_segments = run_vad_pyannote31(
            proc_path,
            vad_json,
            HF_TOKEN,
            merge_gap=args.vad_merge_gap,
            min_duration=args.vad_min_duration,
            overwrite=args.overwrite_vad,
        )
        if not vad_segments:
            with open(vad_json, "r") as f:
                vad_segments = json.load(f)["segments"]

        # 3. Parallel ASR & Diarization (Fixed-2)
        print(f"Processing {stem} ({enhancer_label})...")
        audio = whisperx.load_audio(str(proc_path))
        sr = 16000

        # Diarization (Muting Trick)
        mask = np.zeros_like(audio)
        for seg in vad_segments:
            mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
        muted_audio = audio * mask
        audio_tensor = torch.from_numpy(muted_audio[None, :]).to(DEVICE)
        
        diar_res = diarizer(
            {"waveform": audio_tensor, "sample_rate": sr},
            min_speakers=2, max_speakers=2
        )
        
        diar_list = []
        for turn, _, speaker in diar_res.speaker_diarization.itertracks(yield_label=True):
            diar_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        diar_df = pd.DataFrame(diar_list)

        # ASR (Manual Slices)
        transcript_segments = []
        for seg in tqdm(vad_segments, desc="ASR"):
            audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
            res, _ = asr_model.model.transcribe(
                audio_slice, language="en", task="transcribe", 
                condition_on_previous_text=False
            )
            text = "".join([s.text for s in res]).strip()
            transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

        # Alignment
        model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        aligned_res = whisperx.align(transcript_segments, model_a, metadata, audio, DEVICE)
        final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

        # Save
        out_json = output_dir / f"{stem}_final.json"
        out_txt = output_dir / f"{stem}_final.txt"
        with open(out_json, "w") as f:
            json.dump(final_result, f, indent=2, default=str)
        with open(out_txt, "w") as f:
            for s in final_result["segments"]:
                f.write(f"[{s['start']:06.2f}s] {s.get('speaker', 'UNK'):10}: {s['text']}\n")

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Multi-Enhancer Pipeline")
    parser.add_argument("--vad-min-duration", type=float, default=0.1)
    parser.add_argument("--vad-merge-gap", type=float, default=0.15)
    parser.add_argument("--overwrite-vad", action="store_true", help="Recompute VAD segments even if cached")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN required.")
        return

    # Load shared models
    print("Loading Models...")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(torch.device(DEVICE))

    audio_files = sorted(list(RAW_AUDIO_DIR.glob("*.wav")))
    enhancers = [None, "deepfilternet", "speechbrain"]

    for enhancer in enhancers:
        process_pipeline(enhancer, audio_files, asr_model, diarizer, args)

if __name__ == "__main__":
    main()
