#!/usr/bin/env python3
"""
Stage 5B.4 Production Diarization Runner & CLI
Coordinates execution of the authorized fresh diarization pilot:
- E20: C62 Linear Consensus VAD x M0 (Single-label baseline)
- E21: C62 Linear Consensus VAD x M4-DFN (Conservative overlap attribution)
Wires CLI directly to FreshDiarizationOrchestratorRemediated using the frozen condition registry.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
import torch

# Ensure local script directory is first on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import (
    get_repo_root,
    get_stage5b_4_dir,
    get_stage5b_4_cache_dir,
    get_stage5b_4_outputs_dir,
    get_research_stages_dir,
)
from condition_registry import CONDITIONS, BENCHMARK_RECORDINGS, get_condition, validate_registry
from checkpoint_preflight import ExecutionMode, run_production_checkpoint_preflight
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from vad_adapter_remediated import VADAdapterRemediated
from clustering_backends import NeMoProductionClusteringBackend, SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import ProductionEvaluationAdapterRemediated, SyntheticEvaluationAdapter
from overlap_posterior_loader import OverlapPosteriorLoaderStrict
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated, PipelineExecutionStats


DEFAULT_TITANET_PATH = Path.home() / ".cache" / "torch" / "NeMo" / "NeMo_2.7.3" / "diar_msdd_telephonic" / "3c3697a0a46f945574fa407149975a13" / "diar_msdd_telephonic.nemo"
DEFAULT_TITANET_SHA256 = "450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6"

DEFAULT_WESPEAKER_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--Wespeaker--wespeaker-voxceleb-resnet293-LM" / "blobs" / "dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98"
DEFAULT_WESPEAKER_SHA256 = "dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5B.4 Production Diarization Runner")
    parser.add_argument("mode", choices=["validate", "test-synthetic", "run"], help="Execution mode.")
    parser.add_argument("--authorize-inference", action="store_true", default=False, help="Explicit authorization flag required for real inference in 'run' mode.")
    parser.add_argument("--recordings", nargs="+", default=["R01_sara_sharif"], help="Recording IDs to process.")
    parser.add_argument("--conditions", nargs="+", default=["E20", "E21"], help="Condition IDs to process.")
    parser.add_argument("--titanet-path", type=Path, default=DEFAULT_TITANET_PATH, help="Path to TitaNet / MSDD checkpoint.")
    parser.add_argument("--titanet-sha256", type=str, default=DEFAULT_TITANET_SHA256, help="Expected SHA-256 for TitaNet.")
    parser.add_argument("--wespeaker-path", type=Path, default=DEFAULT_WESPEAKER_PATH, help="Path to WeSpeaker ONNX checkpoint.")
    parser.add_argument("--wespeaker-sha256", type=str, default=DEFAULT_WESPEAKER_SHA256, help="Expected SHA-256 for WeSpeaker.")
    parser.add_argument("--output-pilot-dir", type=Path, default=None, help="Optional isolated pilot workspace for secondary output copies.")
    return parser.parse_args()


def load_audio_waveforms(rec_id: str, repo_root: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    manifest_path = repo_root / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec_entry = None
    for entry in data:
        if entry.get("rec_id") == rec_id:
            rec_entry = entry
            break
    if rec_entry is None:
        raise FileNotFoundError(f"Recording '{rec_id}' not found in Stage 1 manifest: {manifest_path}")

    dfn_rel = rec_entry["dfn_wav"]
    mono_rel = rec_entry["mono_wav"]
    dfn_path = repo_root / dfn_rel
    mono_path = repo_root / mono_rel

    if not dfn_path.exists():
        raise FileNotFoundError(f"DFN audio not found at {dfn_path}")
    if not mono_path.exists():
        raise FileNotFoundError(f"Mono audio not found at {mono_path}")

    dfn_wav, sr_dfn = sf.read(str(dfn_path), dtype="float32")
    mono_wav, sr_mono = sf.read(str(mono_path), dtype="float32")

    if sr_dfn != 16000 or sr_mono != 16000:
        raise ValueError(f"Sample rate mismatch: DFN={sr_dfn}, Mono={sr_mono}, expected 16000 Hz")

    return dfn_wav, mono_wav, 16000


def run_pipeline_for_conditions(
    recordings: List[str],
    conditions: List[str],
    titanet_path: Path,
    titanet_sha256: str,
    wespeaker_path: Path,
    wespeaker_sha256: str,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    output_pilot_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, PipelineExecutionStats]]:
    repo_root = get_repo_root()
    wespeaker_path = Path(wespeaker_path)
    if wespeaker_path.is_symlink():
        wespeaker_path = wespeaker_path.resolve()
    titanet_path = Path(titanet_path)
    if titanet_path.is_symlink():
        titanet_path = titanet_path.resolve()
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated()
    vad_adapter = VADAdapterRemediated()
    overlap_loader = OverlapPosteriorLoaderStrict(execution_mode=execution_mode)

    if execution_mode == ExecutionMode.PRODUCTION:
        clustering_backend = NeMoProductionClusteringBackend(allow_inference=True)
        eval_adapter = ProductionEvaluationAdapterRemediated()
    else:
        clustering_backend = SyntheticSpectralClusteringBackend()
        eval_adapter = SyntheticEvaluationAdapter()

    orchestrator = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering_backend,
        evaluation_adapter=eval_adapter,
        overlap_loader=overlap_loader,
        titanet_checkpoint_path=titanet_path,
        titanet_checkpoint_sha256=titanet_sha256,
        wespeaker_checkpoint_path=wespeaker_path,
        wespeaker_checkpoint_sha256=wespeaker_sha256,
        execution_mode=execution_mode,
    )

    all_results: Dict[str, Dict[str, PipelineExecutionStats]] = {}

    for rec_id in recordings:
        dfn_wav, mono_wav, sr = load_audio_waveforms(rec_id, repo_root)
        ref_rttm = repo_root / "research_stages" / "stage1_data_prep" / "annotations_rttm" / f"{rec_id}.rttm"
        ref_transcript = repo_root / "research_stages" / "stage1_data_prep" / "annotations_json" / f"{rec_id}.json"
        aligned_json = repo_root / "research_stages" / "stage2_baseline" / "whisperx" / f"{rec_id}_aligned.json"

        all_results[rec_id] = {}
        for cond_id in conditions:
            cond = get_condition(cond_id)
            # E20 uses M0 (no overlap); E21 uses M4-DFN
            selected_stream = "dfn" if cond.overlap_config.method_id == "M4_conservative_abstention" else "mono"

            t0 = time.time()
            stats = orchestrator.process_recording(
                condition_id=cond_id,
                recording_id=rec_id,
                dfn_waveform=dfn_wav,
                mono_waveform=mono_wav,
                sample_rate=sr,
                reference_rttm_path=ref_rttm if ref_rttm.exists() else None,
                aligned_words_json=aligned_json if aligned_json.exists() else None,
                reference_transcript_json=ref_transcript if ref_transcript.exists() else None,
                overlap_stream=selected_stream,
            )
            elapsed = time.time() - t0
            stats.metadata = getattr(stats, "metadata", {})
            stats.metadata["elapsed_sec"] = elapsed
            all_results[rec_id][cond_id] = stats

            # If isolated pilot workspace requested, copy canonical RTTM there too
            if output_pilot_dir and stats.output_rttm_path:
                src_rttm = Path(stats.output_rttm_path)
                dest_dir = output_pilot_dir / cond_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_rttm = dest_dir / f"{rec_id}.rttm"
                dest_rttm.write_bytes(src_rttm.read_bytes())

    return all_results


def main() -> int:
    args = parse_args()
    if args.mode == "validate":
        validate_registry()
        print("[PASS] Condition registry valid.")
        return 0

    if args.mode == "run" and not args.authorize_inference:
        print("ERROR: Real inference in 'run' mode requires explicit '--authorize-inference'.", file=sys.stderr)
        return 1

    mode = ExecutionMode.PRODUCTION if args.mode == "run" else ExecutionMode.SYNTHETIC
    results = run_pipeline_for_conditions(
        recordings=args.recordings,
        conditions=args.conditions,
        titanet_path=args.titanet_path,
        titanet_sha256=args.titanet_sha256,
        wespeaker_path=args.wespeaker_path,
        wespeaker_sha256=args.wespeaker_sha256,
        execution_mode=mode,
        output_pilot_dir=args.output_pilot_dir,
    )
    print(f"[SUCCESS] Processed {len(results)} recordings.")
    for rec_id, conds in results.items():
        print(f"\n=================== Summary for {rec_id} ===================")
        for cond_id, s in conds.items():
            print(f"Condition: {cond_id} (VAD: {s.vad_id}, Overlap: {s.overlap_method})")
            print(f"  Output RTTM: {s.output_rttm_path}")
            print(f"  Estimated K: {s.estimated_k}")
            if s.evaluation_metrics:
                print(f"  Evaluation Metrics:")
                for k, v in s.evaluation_metrics.items():
                    print(f"    {k}: {v}")
            if hasattr(s, "metadata") and "elapsed_sec" in s.metadata:
                print(f"  Elapsed Time: {s.metadata['elapsed_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
