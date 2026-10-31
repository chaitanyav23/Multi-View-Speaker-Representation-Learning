#!/usr/bin/env python3
"""
Stage 1b — DeepFilterNet3 Audio Enhancement & Complete Stage 1 Validation.

Strictly disables all fallback mechanisms: if DeepFilterNet3 fails or is missing,
raises an explicit RuntimeError. Validates all 24 generated WAV files (8 recordings x
3 variants: 16k_mono, dfn, normalized) and generates STAGE1_DATA_PREPARATION_REPORT.md.
"""

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from run_stage1_prep import (
    CANONICAL_RECORDINGS,
    REPO_ROOT,
    STAGE1_DIR,
    probe_audio_file,
    sha256_file,
    validate_wav_signal,
)


def run_dfn3_strict(input_wav: Path, output_wav: Path, df_bin: str, env: Dict[str, str]) -> None:
    if not os.path.exists(df_bin):
        raise RuntimeError(f"STRICT FAILURE: deepFilter binary not found at {df_bin}. Fallback is strictly disabled.")
    tmp_dir = output_wav.parent / "_dfn_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [df_bin, str(input_wav), "--output-dir", str(tmp_dir), "--no-suffix"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"STRICT FAILURE: deepFilter failed on {input_wav} (exit {res.returncode}):\n{res.stderr}\n{res.stdout}"
        )
    produced = tmp_dir / input_wav.name
    if not produced.exists():
        raise RuntimeError(f"STRICT FAILURE: deepFilter did not produce {produced}")
    shutil.move(str(produced), str(output_wav))
    shutil.rmtree(tmp_dir)


def main() -> None:
    venv_dir = REPO_ROOT / "research_stages" / ".venv_phase20"
    df_bin = str(venv_dir / "bin" / "deepFilter")
    cache_root = REPO_ROOT / "research_stages" / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["HF_HOME"] = str(cache_root / "huggingface")
    env["TORCH_HOME"] = str(cache_root / "torch")

    manifest_path = STAGE1_DIR / "manifests" / "recording_manifest.json"
    manifest: List[Dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))

    val_path = STAGE1_DIR / "reports" / "audio_validation_report.json"
    validations: List[Dict[str, Any]] = json.loads(val_path.read_text(encoding="utf-8"))
    validations = [v for v in validations if v.get("audio_variant") != "dfn"]

    for entry in manifest:
        rec_id = entry["rec_id"]
        stem = entry["original_stem"]
        mp3_path = REPO_ROOT / entry["source_mp3"]
        mp3_info = probe_audio_file(mp3_path)

        mono_wav = REPO_ROOT / entry["mono_wav"]
        dfn_wav = mono_wav.parent / f"{rec_id}_dfn.wav"

        if not dfn_wav.exists():
            print(f"[Stage 1b] Running DeepFilterNet3 on {rec_id} ({stem})...")
            run_dfn3_strict(mono_wav, dfn_wav, df_bin, env)

        val_dfn = validate_wav_signal(dfn_wav, mp3_info["duration_sec"])
        val_dfn["rec_id"] = rec_id
        val_dfn["audio_variant"] = "dfn"
        if not val_dfn["validation_passed"]:
            raise RuntimeError(f"STRICT FAILURE: Audio validation failed for {dfn_wav}: {val_dfn}")
        validations.append(val_dfn)
        entry["dfn_wav"] = str(dfn_wav.relative_to(REPO_ROOT))
        entry["dfn_wav_sha256"] = val_dfn["sha256"]

    # Sort validations by rec_id and variant
    variant_order = {"16k_mono": 0, "dfn": 1, "normalized": 2}
    validations.sort(key=lambda x: (x["rec_id"], variant_order.get(x["audio_variant"], 9)))

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(validations, indent=2) + "\n", encoding="utf-8")

    with (STAGE1_DIR / "manifests" / "recording_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    with (STAGE1_DIR / "reports" / "audio_validation_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(validations[0].keys()))
        w.writeheader()
        w.writerows(validations)

    print(f"Stage 1b complete: {len(validations)} WAV files validated (all_passed={all(v['validation_passed'] for v in validations)}).")


if __name__ == "__main__":
    main()
