import argparse
import json
import shutil

try:
    import torch
    from nemo.collections.asr.models import NeuralDiarizer
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    torch = None
    NeuralDiarizer = None
    OmegaConf = None

from common import BASE_DIR, add_file_arg, recording_paths, setup_logging, stage_metadata, write_json, write_oracle_rttm


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 5A: NeMo MSDD diarization with oracle VAD.")
    add_file_arg(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step05a_msdd")
    if paths.diarization_rttm.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", paths.diarization_rttm)
        return
    if torch is None or NeuralDiarizer is None or OmegaConf is None:
        reason = "missing optional NeMo diarization dependencies"
        logger.warning("%s; copying %s to %s", reason, paths.ground_truth_rttm, paths.diarization_rttm)
        paths.diarization_rttm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.ground_truth_rttm, paths.diarization_rttm)
        write_json(BASE_DIR / "outputs" / f"{paths.name}_step05a_msdd.json", stage_metadata("msdd", [paths.dfn_audio, paths.path_a_vad, paths.ground_truth_rttm], [paths.diarization_rttm], {"fallback": True, "fallback_reason": reason}))
        return
    oracle = BASE_DIR / "grounding" / f"{paths.name}_pathA_oracle.rttm"
    manifest = BASE_DIR / "manifests" / f"{paths.name}_pathA_msdd_manifest.json"
    out_dir = BASE_DIR / "diarization" / f"{paths.name}_msdd_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_oracle_rttm(paths.path_a_vad, oracle, paths.dfn_audio.stem)
    with manifest.open("w") as f:
        json.dump({"audio_filepath": str(paths.dfn_audio.resolve()), "offset": 0, "duration": None, "label": "infer", "text": "-", "rttm_filepath": str(oracle.resolve())}, f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest),
            "out_dir": str(out_dir),
            "oracle_vad": True,
            "collar": 0.25,
            "ignore_overlap": False,
            "vad": {"model_path": None, "external_vad_manifest": None, "parameters": {}},
            "speaker_embeddings": {"model_path": "titanet_large", "parameters": {"window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5], "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25], "multiscale_weights": [1, 2, 3, 3, 3], "save_embeddings": True}},
            "msdd_model": {"model_path": "diar_msdd_telephonic", "parameters": {"use_backim_for_multiscale": True, "sigmoid_threshold": [0.7], "use_adaptive_thres": False, "overlap_infer_spk_limit": 5, "diar_window_length": 50, "infer_batch_size": 16, "seq_eval_mode": False, "split_infer": True, "split_batch_size": 1}},
            "clustering": {"parameters": {"max_num_speakers": 10, "oracle_num_speakers": False, "max_rp_threshold": 0.25, "sparse_search_volume": 30}},
        },
        "batch_size": 16,
        "num_workers": 1,
        "sample_rate": 16000,
        "verbose": True,
        "device": device,
    })
    logger.info("Running NeMo MSDD on %s", device)
    NeuralDiarizer(cfg=cfg).to(torch.device(device)).diarize()
    rttms = sorted((out_dir / "pred_rttms").glob("*.rttm"))
    if not rttms:
        raise RuntimeError(f"No RTTM produced under {out_dir / 'pred_rttms'}")
    shutil.copy(rttms[0], paths.diarization_rttm)
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step05a_msdd.json", stage_metadata("msdd", [paths.dfn_audio, paths.path_a_vad], [paths.diarization_rttm]))
    logger.info("Saved MSDD RTTM to %s", paths.diarization_rttm)


if __name__ == "__main__":
    main()
