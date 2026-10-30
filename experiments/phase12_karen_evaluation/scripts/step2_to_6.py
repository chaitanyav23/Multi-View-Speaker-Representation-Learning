import os
import json
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
from pyannote.audio import Pipeline
import whisperx

BASE_DIR = Path("experiments/phase12_karen_evaluation")
AUDIO_PATH = BASE_DIR / "audio_preprocessed/karen_normalized.wav"
MUTED_AUDIO_PATH = BASE_DIR / "audio_preprocessed/karen_muted.wav"
GROUNDING_JSON = BASE_DIR / "grounding/karen_vad.json"
SYSTEM_VAD_RTTM = BASE_DIR / "grounding/karen_system_vad.rttm"
VAD_SEGMENTS_DIR = BASE_DIR / "vad_segments"
VAD_INDEX = VAD_SEGMENTS_DIR / "index.csv"
ASR_OUTPUT = BASE_DIR / "outputs/asr/karen_asr.json"
ALIGNED_OUTPUT = BASE_DIR / "outputs/asr/karen_aligned.json"

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
import yaml

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

def step2_3_master_grounding():
    print("\n--- STEP 2: MASTER GROUNDING & STEP 3: RTTM ---")
    pipeline = load_pyannote_vad_pipeline(HF_TOKEN)
    pipeline.to(torch.device(DEVICE))
    
    vad = pipeline(str(AUDIO_PATH))
    segments = []
    for s in vad.get_timeline().support():
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3)})
    
    # Merge gap logic (merge gaps < 0.15s)
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
    
    with open(GROUNDING_JSON, "w") as f:
        json.dump({"segments": cleaned}, f, indent=2)
        
    with open(SYSTEM_VAD_RTTM, "w") as f:
        for seg in cleaned:
            dur = seg["end"] - seg["start"]
            f.write(f"SPEAKER karen_normalized 1 {seg['start']:.3f} {dur:.3f} <NA> <NA> speaker <NA> <NA>\n")
            
    # Export EVERY speech segment
    audio_data, sr = sf.read(AUDIO_PATH)
    index_data = []
    for i, seg in enumerate(cleaned):
        start_samp = int(seg["start"] * sr)
        end_samp = int(seg["end"] * sr)
        clip = audio_data[start_samp:end_samp]
        clip_name = f"segment_{i+1:04d}.wav"
        sf.write(VAD_SEGMENTS_DIR / clip_name, clip, sr)
        index_data.append({
            "segment_id": clip_name,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["end"] - seg["start"]
        })
    pd.DataFrame(index_data).to_csv(VAD_INDEX, index=False)
    print(f"Generated {len(cleaned)} VAD segments.")
    return cleaned

def step4_muted_audio(segments):
    print("\n--- STEP 4: MUTED AUDIO ---")
    audio, sr = sf.read(AUDIO_PATH)
    mask = np.zeros_like(audio)
    for seg in segments:
        mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
    muted_audio = audio * mask
    sf.write(MUTED_AUDIO_PATH, muted_audio, sr)
    print("Muted audio created.")

def step5_asr(segments):
    print("\n--- STEP 5: ASR (WhisperX Large-v3) ---")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    audio = whisperx.load_audio(str(AUDIO_PATH))
    sr = 16000
    transcript_segments = []
    
    for seg in tqdm(segments, desc="ASR on slices"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        if len(audio_slice) == 0: continue
        res, _ = asr_model.model.transcribe(audio_slice, language="en", task="transcribe", condition_on_previous_text=False)
        text = "".join([s.text for s in res]).strip()
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})
        
    with open(ASR_OUTPUT, "w") as f:
        json.dump({"segments": transcript_segments}, f, indent=2)
    return transcript_segments, audio

def step6_alignment(transcript_segments, audio):
    print("\n--- STEP 6: ALIGNMENT (Wav2Vec2) ---")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    aligned_res = whisperx.align(transcript_segments, align_model, align_metadata, audio, DEVICE)
    with open(ALIGNED_OUTPUT, "w") as f:
        json.dump(aligned_res, f, indent=2)
    return aligned_res

if __name__ == "__main__":
    segments = step2_3_master_grounding()
    step4_muted_audio(segments)
    ts, audio = step5_asr(segments)
    step6_alignment(ts, audio)
