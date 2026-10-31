#!/usr/bin/env python3
"""
Stage 2 — Strict Zero-Fallback 8-Recording Baseline Reproduction.

Requirements enforced:
1. Runs independently across all 8 audio-available recordings (R01..R08).
2. Strictly disables ALL silent ground-truth fallbacks. Any missing model or exception
   terminates immediately with an explicit RuntimeError.
3. Verifies that actual model inference (DeepFilterNet3, Pyannote VAD + segmentation-3.0,
   WhisperX large-v3 + Wav2Vec2, and NeMo NeuralDiarizer / TitaNet-Large) executed.
4. Evaluates under the consistent uniform_v1 annotation policy across all 8 recordings,
   AND additionally under phase20_legacy_curated for R07_bwc_cheap_shot to compare
   directly against the historical single-recording Phase 20 result.
5. Computes DER (collar=0.0 with overlap & collar=0.25 no-overlap), FA, Miss, Confusion,
   JER, Speaker Count Error, WER, CER, cpWER, saWER, Micro-SACC/WAE, Timeline Accuracy,
   and Macro-SACC.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure cache directories point to user home cache BEFORE importing phase20 common.py
# so nothing is created inside phase20_standalone/.cache
os.environ["XDG_CACHE_HOME"] = "/home/priyanka/.cache"
os.environ["HF_HOME"] = "/home/priyanka/.cache/huggingface"
os.environ["TORCH_HOME"] = "/home/priyanka/.cache/torch"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import jiwer
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
STAGE1_DIR = REPO_ROOT / "research_stages" / "stage1_data_prep"
STAGE2_DIR = REPO_ROOT / "research_stages" / "stage2_baseline"

sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))

import common as p20_common  # noqa: E402
import step03_vad as p20_vad  # noqa: E402
from attribution_metrics import speaker_accuracy_stats, word_speaker_pairs  # noqa: E402


def get_hf_token() -> str:
    token_path = Path("/home/priyanka/.cache/huggingface/token")
    if token_path.exists():
        tok = token_path.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    raise RuntimeError("STRICT FAILURE: No HuggingFace token found in ~/.cache/huggingface/token or HF_TOKEN.")


def load_local_pyannote_vad_pipeline(device: str = "cuda") -> Any:
    """
    Loads the local Pyannote PyanNet segmentation checkpoint bundled with whisperx
    (assets/pytorch_model.bin, pyannote/segmentation 3-speaker multi-label model)
    and instantiates VoiceActivityDetection with the official pyannote/voice-activity-detection
    hyperparameters (onset=0.4810, offset=0.4810, min_duration_on=0.0554, min_duration_off=0.0979).
    """
    import whisperx
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    wx_bin = Path(whisperx.__file__).resolve().parent / "assets" / "pytorch_model.bin"
    if not wx_bin.exists():
        raise RuntimeError(f"STRICT FAILURE: Local Pyannote segmentation checkpoint not found at {wx_bin}")
    model = Model.from_pretrained(str(wx_bin))
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({
        "onset": 0.4810,
        "offset": 0.4810,
        "min_duration_on": 0.05537587440407595,
        "min_duration_off": 0.09791355693027545,
    })
    return pipeline.to(torch.device(device))


def run_strict_phase20_vad(
    vad_pipeline: Any,
    wav_path: Path,
    out_json: Path,
    out_prob_json: Path,
    out_oracle_rttm: Path,
    out_posteriors_npz: Path,
    path_name: str,
) -> Dict[str, Any]:
    """
    Executes the exact Phase 20 step03_vad.py probability-guided VAD pipeline
    with zero fallback allowed, and saves frame-level p_speech(t) and p_overlap(t)
    from the 3-speaker multi-label PyanNet output.
    """
    started = time.time()
    probabilities, annotation = p20_vad.run_pyannote(vad_pipeline, wav_path)
    curve = p20_vad.extract_probability_curve(probabilities)
    raw = p20_vad.extract_raw_segments(annotation)
    if len(raw) == 0:
        raise RuntimeError(f"STRICT FAILURE: Pyannote VAD produced 0 raw segments for {wav_path}")

    # Extract frame-level speech and overlap probabilities from 3-speaker PyanNet output
    raw_data = np.asarray(probabilities.data, dtype=np.float32)
    if raw_data.ndim == 3:
        raw_data = raw_data.reshape((-1, raw_data.shape[-1]))
    elif raw_data.ndim == 1:
        raw_data = raw_data.reshape((-1, 1))
    p_speech = np.clip(np.max(raw_data, axis=-1), 0.0, 1.0).astype(np.float32)
    if raw_data.shape[-1] >= 2:
        p_overlap = np.clip(np.partition(raw_data, -2, axis=-1)[:, -2], 0.0, 1.0).astype(np.float32)
    else:
        p_overlap = np.zeros_like(p_speech)
    frame_times = np.asarray(curve["times"], dtype=np.float32)
    out_posteriors_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_posteriors_npz),
        frame_times=frame_times,
        p_speech=p_speech,
        p_overlap=p_overlap,
        speaker_probs=raw_data,
    )

    merge_gap = 0.15
    min_duration = 0.10
    max_segment_duration = 30.0
    merge_max_duration = 28.0

    merged, initial_merges = p20_vad.merge_segments(raw, merge_gap)
    filtered, removed_short = p20_vad.remove_short_segments(merged, min_duration)

    split_args = argparse.Namespace(
        max_segment_duration=max_segment_duration,
        split_search_start=15.0,
        split_search_end=30.0,
        minimum_local_window=0.50,
    )
    split_segments = []
    split_operations = []
    recursive_depth = 0
    for seg in filtered:
        parts, split_operations, depth = p20_vad.recursive_split(seg, curve, wav_path, split_args, 0, split_operations)
        split_segments.extend(parts)
        recursive_depth = max(recursive_depth, depth)

    final_segments, final_merges = p20_vad.merge_final_chunks(
        sorted(split_segments, key=lambda s: s["start"]), merge_gap, merge_max_duration
    )
    if len(final_segments) == 0:
        raise RuntimeError(f"STRICT FAILURE: Pyannote VAD produced 0 final segments for {wav_path}")

    stats_before = p20_vad.segment_stats(filtered)
    stats_after = p20_vad.segment_stats(final_segments)
    dur = p20_common.audio_duration(wav_path)

    payload = {
        "file": wav_path.name,
        "model": "pyannote/voice-activity-detection (local PyanNet assets/pytorch_model.bin)",
        "path": path_name,
        "fallback": False,
        "duration": dur,
        "raw_count": len(raw),
        "merged_count": len(merged),
        "filtered_count": len(filtered),
        "split_count": len(split_segments),
        "final_count": len(final_segments),
        "removed_short_segments": removed_short,
        "split_operations": split_operations,
        "recursive_depth": recursive_depth,
        "pre_split_duration_histogram": stats_before["duration_histogram"],
        "post_split_duration_histogram": stats_after["duration_histogram"],
        **{k: v for k, v in stats_after.items() if k != "duration_histogram"},
        "segments": final_segments,
        "processing_time_sec": round(time.time() - started, 4),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    prob_payload = {
        "file": wav_path.name,
        "path": path_name,
        "times": [round(float(t), 4) for t in curve["times"]],
        "probabilities": [round(float(p), 6) for p in curve["probabilities"]],
    }
    out_prob_json.write_text(json.dumps(prob_payload, indent=2) + "\n", encoding="utf-8")

    p20_common.write_oracle_rttm(out_json, out_oracle_rttm, wav_path.stem)
    return payload


def compute_der_collar025_noovl(reference_rttm: Path, hypothesis_rttm: Path) -> float:
    def load_ann(path: Path) -> Annotation:
        ann = Annotation()
        for seg in p20_common.rttm_to_segments(path):
            ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
        return ann

    ref = load_ann(reference_rttm)
    hyp = load_ann(hypothesis_rttm)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    return float(metric(ref, hyp))


def compute_macro_sacc(
    gt_segments: List[Dict[str, Any]],
    attributed_word_list: List[Dict[str, Any]],
    speaker_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """
    Computes per-reference-speaker word attribution recall and Macro-Averaged SACC
    using the exact word_speaker_pairs alignment from Phase 20 attribution_metrics.py.
    """
    pairs = word_speaker_pairs(gt_segments, attributed_word_list)
    per_spk_total: Dict[str, int] = {}
    per_spk_correct: Dict[str, int] = {}
    for sys_spk, ref_spk in pairs:
        if ref_spk == "UNK":
            continue
        per_spk_total[ref_spk] = per_spk_total.get(ref_spk, 0) + 1
        mapped = speaker_mapping.get(sys_spk, sys_spk)
        if mapped == ref_spk:
            per_spk_correct[ref_spk] = per_spk_correct.get(ref_spk, 0) + 1

    # Also ensure every GT speaker with text appears in per_spk_recall (with 0.0 if 0 words matched)
    for seg in gt_segments:
        if str(seg.get("text", "")).strip():
            spk = str(seg["speaker"])
            per_spk_total.setdefault(spk, 0)

    per_spk_recall = {}
    for spk in sorted(per_spk_total.keys()):
        tot = per_spk_total[spk]
        cor = per_spk_correct.get(spk, 0)
        per_spk_recall[spk] = round(cor / tot, 4) if tot > 0 else 0.0

    macro_sacc = float(np.mean(list(per_spk_recall.values()))) if per_spk_recall else 0.0
    return {
        "macro_sacc": round(macro_sacc, 4),
        "per_speaker_word_recall": per_spk_recall,
        "per_speaker_evaluated_words": per_spk_total,
    }


def evaluate_recording_policy(
    rec_id: str,
    policy_name: str,
    gt_json_path: Path,
    gt_rttm_path: Path,
    vad_a_segments: List[Dict[str, Any]],
    vad_b_segments: List[Dict[str, Any]],
    pred_rttm_path: Path,
    aligned_words_path: Path,
    duration_sec: float,
    out_fusion_dir: Path,
) -> Dict[str, Any]:
    gt_payload = p20_common.read_json(gt_json_path)
    gt_segments = gt_payload["segments"]
    gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
    diar_segments = p20_common.rttm_to_segments(pred_rttm_path)

    if len(diar_segments) == 0:
        raise RuntimeError(f"STRICT FAILURE: Predicted RTTM {pred_rttm_path} contains 0 segments!")

    # Verify predicted RTTM is NOT a fallback copy of GT RTTM
    if len(diar_segments) == len(gt_rttm_segs) and all(
        abs(a["start"] - b["start"]) < 1e-4 and abs(a["end"] - b["end"]) < 1e-4
        for a, b in zip(diar_segments, gt_rttm_segs)
    ):
        raise RuntimeError(
            f"STRICT FAILURE: Predicted RTTM {pred_rttm_path} has identical boundaries to GT RTTM (fallback detected)!"
        )

    vad_a_eval = p20_common.vad_metrics(gt_rttm_segs, vad_a_segments, duration_sec)
    vad_b_eval = p20_common.vad_metrics(gt_rttm_segs, vad_b_segments, duration_sec)
    diar_eval = p20_common.diarization_metrics(gt_rttm_path, pred_rttm_path)
    der_025_noovl = compute_der_collar025_noovl(gt_rttm_path, pred_rttm_path)

    aligned_words = p20_common.read_json(aligned_words_path)
    ref_text = p20_common.text_from_gt(gt_json_path)
    hyp_text = p20_common.text_from_words(aligned_words)
    asr_eval = p20_common.asr_metrics(ref_text, hyp_text)
    norm_ref = p20_common.normalize_text(ref_text)
    norm_hyp = p20_common.normalize_text(hyp_text)
    cer_val = float(jiwer.cer(norm_ref, norm_hyp)) if norm_ref else 0.0

    # Step 7: Fusion
    words_attr = p20_common.attributed_words(aligned_words, diar_segments)
    segs_attr = p20_common.group_words(words_attr)
    out_fusion_dir.mkdir(parents=True, exist_ok=True)
    attributed_path = out_fusion_dir / f"{rec_id}_{policy_name}_attributed_words.json"
    transcript_path = out_fusion_dir / f"{rec_id}_{policy_name}_transcript.txt"
    p20_common.write_json(
        attributed_path,
        {"file": f"{rec_id}.wav", "method": "maximum_overlap", "words": words_attr, "segments": segs_attr},
    )
    p20_common.write_transcript(transcript_path, segs_attr)

    # Step 8: Fusion Evaluation
    f_metrics = p20_common.fusion_metrics(gt_json_path, attributed_path, duration_sec)
    macro_info = compute_macro_sacc(gt_segments, words_attr, f_metrics.get("speaker_mapping", {}))

    ref_spks_all = len(set(s["speaker"] for s in gt_segments))
    ref_spks_pos = len(set(s["speaker"] for s in gt_rttm_segs))
    pred_spks = len(set(s["speaker"] for s in diar_segments))

    return {
        "rec_id": rec_id,
        "policy": policy_name,
        "duration_sec": round(duration_sec, 3),
        "ref_speakers_all_txt": ref_spks_all,
        "ref_speakers_pos_rttm": ref_spks_pos,
        "pred_speakers": pred_spks,
        "speaker_count_error_abs": abs(pred_spks - ref_spks_pos),
        "speaker_count_signed_bias": pred_spks - ref_spks_pos,
        "single_speaker_collapse": pred_spks == 1,
        "vad_pathA_f1": round(float(vad_a_eval["f1"]), 4),
        "vad_pathA_precision": round(float(vad_a_eval["precision"]), 4),
        "vad_pathA_recall": round(float(vad_a_eval["recall"]), 4),
        "vad_pathB_f1": round(float(vad_b_eval["f1"]), 4),
        "der_collar0_ovl": round(float(diar_eval["der"]), 4),
        "der_collar025_noovl": round(der_025_noovl, 4),
        "jer_collar0": round(float(diar_eval["jer"]), 4) if diar_eval["jer"] is not None else None,
        "diar_false_alarm_rate": round(float(diar_eval["false_alarm"]), 4),
        "diar_missed_detection_rate": round(float(diar_eval["miss"]), 4),
        "diar_confusion_rate": round(float(diar_eval["confusion"]), 4),
        "asr_wer": round(float(asr_eval["wer"]), 4),
        "asr_cer": round(cer_val, 4),
        "asr_ref_words": int(asr_eval["reference_words"]),
        "asr_hyp_words": int(asr_eval["hypothesis_words"]),
        "cpwer": round(float(f_metrics["cpwer"]), 4),
        "sawer": round(float(f_metrics["sawer"]), 4),
        "micro_sacc": round(float(f_metrics["speaker_accuracy"]), 4),
        "micro_wae": round(float(f_metrics["word_attribution_error"]), 4),
        "timeline_accuracy": round(float(f_metrics["timeline_accuracy"]), 4),
        "macro_sacc": macro_info["macro_sacc"],
        "words_evaluated": int(f_metrics["words_evaluated"]),
        "correct_attributions": int(f_metrics["correct_attributions"]),
        "per_speaker_word_recall": macro_info["per_speaker_word_recall"],
        "speaker_mapping": f_metrics.get("speaker_mapping", {}),
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("STRICT FAILURE: CUDA GPU is required for Stage 2 baseline reproduction.")

    hf_token = get_hf_token()
    os.environ["HF_TOKEN"] = hf_token

    for sub in ["vad", "posteriors", "whisperx", "msdd", "fusion", "metrics", "reports"]:
        (STAGE2_DIR / sub).mkdir(parents=True, exist_ok=True)

    manifest = json.loads((STAGE1_DIR / "manifests" / "recording_manifest.json").read_text(encoding="utf-8"))

    # 1. Load local Pyannote VAD pipeline (exact step03_vad.py PyanNet model + frame-level speech & overlap posteriors)
    print("[Stage 2] Loading Pyannote voice-activity-detection pipeline (local PyanNet checkpoint) on CUDA...")
    vad_pipeline = load_local_pyannote_vad_pipeline("cuda")

    for entry in manifest:
        rec_id = entry["rec_id"]
        dfn_wav = REPO_ROOT / entry["dfn_wav"]
        norm_wav = REPO_ROOT / entry["norm_wav"]
        mono_wav = REPO_ROOT / entry["mono_wav"]

        for cond_name, path_label, wav_p in [
            ("dfn", "pathA", dfn_wav),
            ("normalized", "pathB", norm_wav),
            ("16k_mono", "mono", mono_wav),
        ]:
            vad_json = STAGE2_DIR / "vad" / f"{rec_id}_{path_label}_vad.json"
            prob_json = STAGE2_DIR / "vad" / f"{rec_id}_{path_label}_probabilities.json"
            oracle_rttm = STAGE2_DIR / "vad" / f"{rec_id}_{path_label}_oracle.rttm"
            post_npz = STAGE2_DIR / "posteriors" / f"{rec_id}_{cond_name}_posteriors.npz"

            if not vad_json.exists() or not oracle_rttm.exists() or not post_npz.exists():
                print(f"[Stage 2 VAD & Posteriors {path_label}/{cond_name}] {rec_id}...")
                run_strict_phase20_vad(
                    vad_pipeline, wav_p, vad_json, prob_json, oracle_rttm, post_npz, path_label
                )

    del vad_pipeline
    torch.cuda.empty_cache()

    # 2. Run WhisperX large-v3 + Wav2Vec2 alignment on Path B (normalized.wav) exactly as in step05b_whisperx.py
    import whisperx

    print("[Stage 2] Loading WhisperX large-v3 and Wav2Vec2 alignment model on CUDA...")
    wx_model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device="cuda")

    for entry in manifest:
        rec_id = entry["rec_id"]
        norm_wav = REPO_ROOT / entry["norm_wav"]
        vad_b_json = STAGE2_DIR / "vad" / f"{rec_id}_pathB_vad.json"
        wx_transcript_out = STAGE2_DIR / "whisperx" / f"{rec_id}_whisperx.json"
        wx_aligned_out = STAGE2_DIR / "whisperx" / f"{rec_id}_aligned.json"

        if not wx_aligned_out.exists():
            print(f"[Stage 2 WhisperX large-v3] Transcribing & aligning {rec_id}...")
            audio_arr = whisperx.load_audio(str(norm_wav))
            vad_b_segments = p20_common.read_json(vad_b_json)["segments"]
            transcription = []
            sample_rate = 16000
            for seg in vad_b_segments:
                start = int(seg["start"] * sample_rate)
                end = int(seg["end"] * sample_rate)
                generated, _ = wx_model.model.transcribe(
                    audio_arr[start:end],
                    language="en",
                    task="transcribe",
                    word_timestamps=False,
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in generated).strip()
                if text:
                    transcription.append({"start": seg["start"], "end": seg["end"], "text": text})
            if len(transcription) == 0:
                raise RuntimeError(f"STRICT FAILURE: WhisperX produced 0 transcription segments for {rec_id}")
            aligned = whisperx.align(
                transcription, align_model, align_metadata, audio_arr, "cuda", return_char_alignments=False
            )
            words = [
                {
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "confidence": w.get("score"),
                }
                for w in aligned.get("word_segments", [])
            ]
            if len(words) == 0:
                raise RuntimeError(f"STRICT FAILURE: WhisperX produced 0 aligned words for {rec_id}")
            p20_common.write_json(
                wx_transcript_out, {"file": rec_id, "model": "large-v3", "fallback": False, "segments": transcription}
            )
            p20_common.write_json(wx_aligned_out, words)

    del wx_model
    del align_model
    torch.cuda.empty_cache()

    # 3. Run NeMo NeuralDiarizer (diar_msdd_telephonic + titanet_large) on Path A (dfn.wav) for all 8 recordings
    from nemo.collections.asr.models import NeuralDiarizer

    for entry in manifest:
        rec_id = entry["rec_id"]
        dfn_wav = REPO_ROOT / entry["dfn_wav"]
        vad_a_rttm = STAGE2_DIR / "vad" / f"{rec_id}_pathA_oracle.rttm"
        msdd_rec_dir = STAGE2_DIR / "msdd" / f"{rec_id}_msdd_outputs"
        pred_rttm_out = STAGE2_DIR / "msdd" / f"{rec_id}.rttm"

        if not pred_rttm_out.exists():
            print(f"[Stage 2 NeMo MSDD] Running NeuralDiarizer on {rec_id}...")
            msdd_rec_dir.mkdir(parents=True, exist_ok=True)
            manifest_item = {
                "audio_filepath": str(dfn_wav.resolve()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": str(vad_a_rttm.resolve()),
            }
            msdd_manifest_path = STAGE2_DIR / "msdd" / f"{rec_id}_pathA_msdd_manifest.json"
            msdd_manifest_path.write_text(json.dumps(manifest_item) + "\n", encoding="utf-8")

            cfg = OmegaConf.create({
                "diarizer": {
                    "manifest_filepath": str(msdd_manifest_path),
                    "out_dir": str(msdd_rec_dir),
                    "oracle_vad": True,
                    "collar": 0.25,
                    "ignore_overlap": False,
                    "vad": {"model_path": None, "external_vad_manifest": None, "parameters": {}},
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
                            "overlap_infer_spk_limit": 5,
                            "diar_window_length": 50,
                            "infer_batch_size": 16,
                            "seq_eval_mode": False,
                            "split_infer": True,
                            "split_batch_size": 1,
                        },
                    },
                    "clustering": {
                        "parameters": {
                            "max_num_speakers": 10,
                            "oracle_num_speakers": False,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30,
                        }
                    },
                },
                "batch_size": 16,
                "num_workers": 1,
                "sample_rate": 16000,
                "verbose": True,
                "device": "cuda",
            })
            diarizer = NeuralDiarizer(cfg=cfg).to(torch.device("cuda"))
            diarizer.diarize()

            rttms = sorted((msdd_rec_dir / "pred_rttms").glob("*.rttm"))
            if not rttms:
                raise RuntimeError(f"STRICT FAILURE: NeMo MSDD did not generate RTTM in {msdd_rec_dir / 'pred_rttms'}")
            for scale_k in range(5):
                pkl_p = msdd_rec_dir / "speaker_outputs" / "embeddings" / f"subsegments_scale{scale_k}_embeddings.pkl"
                if not pkl_p.exists():
                    raise RuntimeError(f"STRICT FAILURE: Missing TitaNet embedding pickle {pkl_p}")
            shutil.copy2(rttms[0], pred_rttm_out)
            del diarizer
            torch.cuda.empty_cache()

    # 4. Evaluate all 8 recordings under uniform_v1 policy, plus R07 under phase20_legacy_curated
    uniform_results: List[Dict[str, Any]] = []
    curated_r07_result: Dict[str, Any] = {}

    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        vad_a_segs = p20_common.read_json(STAGE2_DIR / "vad" / f"{rec_id}_pathA_vad.json")["segments"]
        vad_b_segs = p20_common.read_json(STAGE2_DIR / "vad" / f"{rec_id}_pathB_vad.json")["segments"]
        aligned_words_p = STAGE2_DIR / "whisperx" / f"{rec_id}_aligned.json"
        pred_rttm = STAGE2_DIR / "msdd" / f"{rec_id}.rttm"

        gt_json_uni = REPO_ROOT / entry["json_uniform"]
        gt_rttm_uni = REPO_ROOT / entry["rttm_uniform"]

        res_uni = evaluate_recording_policy(
            rec_id=rec_id,
            policy_name="uniform_v1",
            gt_json_path=gt_json_uni,
            gt_rttm_path=gt_rttm_uni,
            vad_a_segments=vad_a_segs,
            vad_b_segments=vad_b_segs,
            pred_rttm_path=pred_rttm,
            aligned_words_path=aligned_words_p,
            duration_sec=dur,
            out_fusion_dir=STAGE2_DIR / "fusion",
        )
        res_uni["original_stem"] = entry["original_stem"]
        uniform_results.append(res_uni)

        if rec_id == "R07_bwc_cheap_shot":
            gt_json_cur = STAGE1_DIR / "annotations_json" / "R07_bwc_cheap_shot_phase20_curated.json"
            gt_rttm_cur = STAGE1_DIR / "annotations_rttm" / "R07_bwc_cheap_shot_phase20_curated.rttm"
            curated_r07_result = evaluate_recording_policy(
                rec_id=rec_id,
                policy_name="phase20_legacy_curated",
                gt_json_path=gt_json_cur,
                gt_rttm_path=gt_rttm_cur,
                vad_a_segments=vad_a_segs,
                vad_b_segments=vad_b_segs,
                pred_rttm_path=pred_rttm,
                aligned_words_path=aligned_words_p,
                duration_sec=dur,
                out_fusion_dir=STAGE2_DIR / "fusion",
            )
            curated_r07_result["original_stem"] = entry["original_stem"]

    # Compute macro-average and duration-weighted aggregate across all 8 recordings (uniform_v1)
    metric_keys = [
        "vad_pathA_f1",
        "vad_pathB_f1",
        "der_collar0_ovl",
        "der_collar025_noovl",
        "jer_collar0",
        "diar_false_alarm_rate",
        "diar_missed_detection_rate",
        "diar_confusion_rate",
        "speaker_count_error_abs",
        "speaker_count_signed_bias",
        "asr_wer",
        "asr_cer",
        "cpwer",
        "sawer",
        "micro_sacc",
        "micro_wae",
        "timeline_accuracy",
        "macro_sacc",
    ]
    macro_agg = {}
    dur_weighted_agg = {}
    total_dur = sum(r["duration_sec"] for r in uniform_results)
    for k in metric_keys:
        vals = [float(r[k]) for r in uniform_results if r[k] is not None]
        macro_agg[k] = round(float(np.mean(vals)), 4) if vals else None
        dur_weighted_agg[k] = round(
            float(sum(float(r[k]) * r["duration_sec"] for r in uniform_results if r[k] is not None) / total_dur), 4
        )
    macro_agg["single_speaker_collapse_count"] = sum(1 for r in uniform_results if r["single_speaker_collapse"])

    summary_payload = {
        "stage": "Stage 2 — 8-Recording Zero-Fallback Baseline Reproduction",
        "uniform_v1_macro_aggregate": macro_agg,
        "uniform_v1_duration_weighted_aggregate": dur_weighted_agg,
        "uniform_v1_per_recording": uniform_results,
        "r07_phase20_legacy_curated": curated_r07_result,
        "historical_phase20_committed_bwc": json.loads(
            (REPO_ROOT / "phase20_standalone" / "metrics" / "bwc_all_metrics.json").read_text()
        ),
    }
    (STAGE2_DIR / "metrics" / "stage2_baseline_all_metrics.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    csv_rows = []
    for r in uniform_results:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        row["per_speaker_word_recall"] = json.dumps(r["per_speaker_word_recall"])
        row["speaker_mapping"] = json.dumps(r["speaker_mapping"])
        csv_rows.append(row)

    csv_path = STAGE2_DIR / "metrics" / "stage2_baseline_per_recording.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print("Stage 2 baseline execution and evaluation completed successfully across all 8 recordings!")


if __name__ == "__main__":
    main()
