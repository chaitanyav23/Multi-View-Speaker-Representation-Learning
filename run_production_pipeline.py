"""
run_production_pipeline.py
==========================
Final Hardened Stabilized Pipeline for BWC.
Workflow: Normalization -> Pyannote VAD -> Muted Diarization -> WhisperX ASR -> Word Alignment.
"""

import json
import torch
import numpy as np
import pandas as pd
import whisperx
import argparse
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import VoiceActivityDetection
from dotenv import dotenv_values
import os

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

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

def main():
    parser = argparse.ArgumentParser(description="BWC Production Pipeline")
    parser.add_argument("--input", required=True, help="Path to raw audio file")
    parser.add_argument("--output-dir", default="production_outputs", help="Output directory")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    audio_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    # 1. LOAD & PREPROCESS (Normalization)
    print(f"\n[1/5] Loading & Normalizing: {audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))
    audio, gain = apply_normalization(audio)
    sr = 16000

    # 2. VAD GROUNDING (Pyannote 3.1)
    print("[2/5] Running VAD Grounding (Pyannote 3.1)")
    # Using the segmentation model as VAD for maximum coverage
    vad_model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
    vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
    vad_pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})
    vad_pipeline.to(torch.device(DEVICE))
    
    vad_res = vad_pipeline(str(audio_path)) # VAD runs on raw file for speed, we apply gain logic to chunks later
    raw_segs = [(s.start, s.end) for s in vad_res.itersegments()]
    grounding_segments = merge_segments(raw_segs)

    # 3. STABILIZED DIARIZATION (The Muting Trick)
    print("[3/5] Running Stabilized Diarization")
    mask = np.zeros_like(audio)
    for seg in grounding_segments:
        mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
    muted_audio = audio * mask # Use normalized + muted audio
    
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(torch.device(DEVICE))
    audio_tensor = torch.from_numpy(muted_audio[None, :]).to(DEVICE)
    diar_res = diarizer({"waveform": audio_tensor, "sample_rate": sr}, 
                        min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    
    diar_list = [{"start": t.start, "end": t.end, "speaker": s} 
                 for t, _, s in diar_res.speaker_diarization.itertracks(yield_label=True)]
    diar_df = pd.DataFrame(diar_list)

    # 4. HARDENED ASR (Manual Slicing)
    print("[4/5] Running Hardened ASR (WhisperX large-v3)")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    transcript_segments = []
    for seg in tqdm(grounding_segments, desc="Transcribing"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(audio_slice, language="en", task="transcribe", condition_on_previous_text=False)
        text = "".join([s.text for s in res]).strip()
        if text:
            transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

    # 5. ALIGNMENT & ATTRIBUTION
    print("[5/5] Finalizing Conversational Timeline")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    aligned_res = whisperx.align(transcript_segments, model_a, metadata, audio, DEVICE)
    final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

    # SAVE OUTPUTS
    out_json = out_dir / f"{stem}_timeline.json"
    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
    
    out_txt = out_dir / f"{stem}_timeline.txt"
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            spk = s.get("speaker", "UNKNOWN")
            f.write(f"[{s['start']:06.2f}s] {spk:10}: {s['text']}\n")

    print(f"\n[SUCCESS] Final Conversational Timeline saved to: {out_dir}")

if __name__ == "__main__":
    main()
