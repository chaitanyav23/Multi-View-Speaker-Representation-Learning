import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
import whisperx
from pyannote.audio import Pipeline
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf

# Path setups
BASE_DIR = Path("experiments/phase10_dual_branch_fusion")
AUDIO_DIR = BASE_DIR / "audio_variants"
GROUNDING_DIR = BASE_DIR / "grounding_master"
OUT_10A = BASE_DIR / "outputs_phase10A"
OUT_10B = BASE_DIR / "outputs_phase10B"
OUT_10C = BASE_DIR / "outputs_phase10C"

NORM_WAV = AUDIO_DIR / "bwc_normalized.wav"
DFN_WAV = AUDIO_DIR / "bwc_deepfilternet.wav"
MASTER_VAD_JSON = GROUNDING_DIR / "bwc_vad.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

from custom_fusion import assign_speakers_to_words

def write_rttm(segments, out_path, file_id="bwc"):
    with open(out_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            f.write(f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

def get_msdd_segments(diar_json_dir, file_stem):
    diar_json_path = diar_json_dir / f"{file_stem}_diarization.json"
    if diar_json_path.exists():
        with open(diar_json_path, 'r') as f:
            lines = f.readlines()
        
        segments = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
                segments.append({"start": start, "end": end, "speaker": speaker})
        return segments
    return []
    
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

def run_msdd_fixed2(audio_path: Path, oracle_rttm_path: Path, output_dir: Path):
    rttm_out_path = output_dir / audio_path.stem / "pred_rttms" / f"{audio_path.stem}.rttm"
    if rttm_out_path.exists():
        print(f"Loading MSDD from cache: {rttm_out_path}")
        return get_msdd_segments_from_rttm(rttm_out_path)

    manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": 2,
            "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(output_dir / audio_path.stem),
            "oracle_vad": True,
            "collar": 0.25,
            "ignore_overlap": False,
            "vad": {
                "model_path": None,
                "external_vad_manifest": None,
                "parameters": {},
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                    "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                    "multiscale_weights": [1, 2, 3, 3, 3],
                    "save_embeddings": True,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_backim_for_multiscale": True,
                    "sigmoid_threshold": [0.7],
                    "use_adaptive_thres": False,
                    "overlap_infer_spk_limit": 3,
                    "diar_window_length": 50,
                    "infer_batch_size": 16,
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {
                    "max_num_speakers": 2,
                    "oracle_num_speakers": True,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                },
            },
        },
        "batch_size": 16,
        "num_workers": 4,
        "sample_rate": 16000,
        "verbose": True,
        "device": DEVICE,
    })

    print(f"Running MSDD: {audio_path.name}")
    msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
    msdd_model.diarize()
    
    return get_msdd_segments_from_rttm(rttm_out_path)

def run_asr_and_align(audio_path, vad_segments, asr_model, align_model, align_metadata):
    audio = whisperx.load_audio(str(audio_path))
    sr = 16000
    transcript_segments = []
    
    print(f"Running WhisperX ASR on {len(vad_segments)} segments...")
    for seg in tqdm(vad_segments, desc="ASR"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(
            audio_slice, language="en", task="transcribe",
            condition_on_previous_text=False
        )
        text = "".join([s.text for s in res]).strip()
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})
        
    print("Running WhisperX Alignment...")
    aligned_res = whisperx.align(transcript_segments, align_model, align_metadata, audio, DEVICE)
    return aligned_res, audio

def save_result(output_dir, final_result, stem, word_metadata=None, diagnostics=None):
    out_json = output_dir / f"{stem}_final.json"
    out_txt = output_dir / f"{stem}_final.txt"
    
    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
        
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            f.write(f"[{s.get('start', 0.0):06.2f}s] {s.get('speaker', 'UNK'):10}: {s.get('text', '')}\n")
            
    if word_metadata:
        meta_json = output_dir / f"{stem}_word_metadata.json"
        with open(meta_json, "w") as f:
            json.dump(word_metadata, f, indent=2)
            
    if diagnostics:
        diag_json = output_dir / f"{stem}_diagnostics.json"
        with open(diag_json, "w") as f:
            json.dump(diagnostics, f, indent=2)

def run_pipeline_10A(asr_model, align_model, align_metadata, diarizer, vad_segments):
    print("\n=== Running Phase 10A (Reference Baseline) ===")
    OUT_10A.mkdir(parents=True, exist_ok=True)
    
    # Muting Trick Diarization
    audio = whisperx.load_audio(str(NORM_WAV))
    sr = 16000
    mask = np.zeros_like(audio)
    for seg in vad_segments:
        mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
    muted_audio = audio * mask
    audio_tensor = torch.from_numpy(muted_audio[None, :]).to(DEVICE)
    
    print("Running Pyannote 3.1 Diarization...")
    diar_res = diarizer({"waveform": audio_tensor, "sample_rate": sr}, min_speakers=2, max_speakers=2)
    diar_list = []
    for turn, _, speaker in diar_res.speaker_diarization.itertracks(yield_label=True):
        diar_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    diar_df = pd.DataFrame(diar_list)
    
    aligned_res, _ = run_asr_and_align(NORM_WAV, vad_segments, asr_model, align_model, align_metadata)
    
    print("Running WhisperX Speaker Attribution...")
    final_result = whisperx.assign_word_speakers(diar_df, aligned_res)
    
    save_result(OUT_10A, final_result, "bwc")

def run_pipeline_10B(asr_model, align_model, align_metadata, oracle_rttm_path, vad_segments):
    print("\n=== Running Phase 10B (Dual Branch Fusion) ===")
    OUT_10B.mkdir(parents=True, exist_ok=True)
    
    msdd_segments = run_msdd_fixed2(NORM_WAV, oracle_rttm_path, OUT_10B)
    
    # ASR on DFN Audio
    aligned_res, _ = run_asr_and_align(DFN_WAV, vad_segments, asr_model, align_model, align_metadata)
    
    # Custom Fusion
    print("Running Custom Fusion Attribution...")
    final_result, word_meta, diag = assign_speakers_to_words(aligned_res, msdd_segments)
    
    save_result(OUT_10B, final_result, "bwc", word_meta, diag)

def run_pipeline_10C(asr_model, align_model, align_metadata, oracle_rttm_path, vad_segments):
    print("\n=== Running Phase 10C (MSDD Control) ===")
    OUT_10C.mkdir(parents=True, exist_ok=True)
    
    msdd_segments = run_msdd_fixed2(NORM_WAV, oracle_rttm_path, OUT_10C)
    
    # ASR on NORM Audio
    aligned_res, _ = run_asr_and_align(NORM_WAV, vad_segments, asr_model, align_model, align_metadata)
    
    # Custom Fusion
    print("Running Custom Fusion Attribution...")
    final_result, word_meta, diag = assign_speakers_to_words(aligned_res, msdd_segments)
    
    save_result(OUT_10C, final_result, "bwc", word_meta, diag)

def main():
    with open(MASTER_VAD_JSON, "r") as f:
        vad_segments = json.load(f)["segments"]
        
    oracle_rttm = GROUNDING_DIR / "bwc_oracle.rttm"
    write_rttm(vad_segments, oracle_rttm)
    
    print("Loading Models...")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(torch.device(DEVICE))
    
    run_pipeline_10A(asr_model, align_model, align_metadata, diarizer, vad_segments)
    run_pipeline_10B(asr_model, align_model, align_metadata, oracle_rttm, vad_segments)
    run_pipeline_10C(asr_model, align_model, align_metadata, oracle_rttm, vad_segments)
    
    print("\nPhase 10 Pipelines complete!")

if __name__ == "__main__":
    main()
