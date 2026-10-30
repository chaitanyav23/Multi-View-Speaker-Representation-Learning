from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ENVIRONMENTS = ("CAFE", "CAR", "HOME", "REVERB", "STREET")
SNR_LEVELS = (15, 10, 5, 0, -5, -10)
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".sph"}
LIBRISPEECH_DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
QUT_NOISE_URLS = (
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip",
    "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/"
    "resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip",
)

MANIFEST_FIELDS = [
    "file_id",
    "output_path",
    "speech_path",
    "noise_path",
    "environment",
    "snr_db",
    "sample_rate",
    "num_samples",
    "duration_sec",
    "noise_offset",
    "speech_rms",
    "noise_rms",
    "noise_scale",
    "final_gain",
    "peak_before_gain",
    "seed",
]


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "qut-vad-dataset-builder/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    os.replace(partial, destination)
    return destination


def safe_extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe archive member: {member.name}")
            handle.extractall(destination, filter="data")
        return
    raise ValueError(f"Unsupported archive: {archive}")


def acquire_librispeech(destination: Path, archive: Path | None = None) -> Path:
    archive = archive or destination.parent / "downloads" / "dev-clean.tar.gz"
    if not archive.exists():
        print(f"Downloading LibriSpeech dev-clean to {archive}")
        download_file(LIBRISPEECH_DEV_CLEAN_URL, archive)
    safe_extract_archive(archive, destination)
    return destination


def acquire_qut_noise(
    destination: Path,
    archives: list[Path] | None = None,
    urls: list[str] | None = None,
    repository_url: str | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archives = list(archives or [])
    if repository_url:
        repo = destination / "repository"
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", repository_url, str(repo)], check=True)
    for url in urls or []:
        filename = url.rsplit("/", 1)[-1].split("?", 1)[0] or f"qut_{len(archives)}.zip"
        target = destination.parent / "downloads" / filename
        if not target.exists():
            download_file(url, target)
        archives.append(target)
    for archive in archives:
        safe_extract_archive(archive, destination)
    return destination


def discover_audio(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def group_noise_files(root: Path) -> dict[str, list[Path]]:
    grouped = {environment: [] for environment in ENVIRONMENTS}
    for path in discover_audio(root):
        upper = str(path.relative_to(root)).upper()
        matches = [environment for environment in ENVIRONMENTS if environment in upper]
        if len(matches) == 1:
            grouped[matches[0]].append(path)
    missing = [environment for environment, paths in grouped.items() if not paths]
    if missing:
        raise RuntimeError(
            f"No audio found for QUT environments: {missing}. "
            "Keep CAFE/CAR/HOME/REVERB/STREET in archive directory or file names."
        )
    return grouped


def load_mono_16k(path: Path) -> np.ndarray:
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, sample_rate // divisor)
    if mono.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return np.asarray(mono, dtype=np.float64)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def stable_seed(speech_path: Path, environment: str, seed: int) -> int:
    digest = hashlib.sha256(f"{speech_path}|{environment}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def noise_segment(noise: np.ndarray, length: int, offset: int) -> np.ndarray:
    if noise.size == 0:
        raise ValueError("Noise signal is empty")
    indices = (np.arange(length, dtype=np.int64) + offset) % noise.size
    return noise[indices]


def mix_signals(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, offset: int
) -> tuple[np.ndarray, dict[str, float]]:
    speech = speech - np.mean(speech)
    selected_noise = noise_segment(noise - np.mean(noise), speech.size, offset)
    speech_rms = rms(speech)
    noise_rms = rms(selected_noise)
    if speech_rms <= 1e-9:
        raise ValueError("Speech RMS is zero")
    if noise_rms <= 1e-9:
        raise ValueError("Noise RMS is zero")
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    noise_scale = target_noise_rms / noise_rms
    mixture = speech + selected_noise * noise_scale
    peak = float(np.max(np.abs(mixture)))
    final_gain = min(1.0, 0.98 / peak) if peak > 0 else 1.0
    mixture *= final_gain
    return mixture, {
        "speech_rms": speech_rms,
        "noise_rms": noise_rms,
        "noise_scale": noise_scale,
        "final_gain": final_gain,
        "peak_before_gain": peak,
    }


def estimate_output_bytes(speech_files: list[Path]) -> int:
    total_samples = 0
    for path in speech_files:
        info = sf.info(path)
        total_samples += math.ceil(info.frames * 16000 / info.samplerate)
    return total_samples * len(ENVIRONMENTS) * len(SNR_LEVELS) * 2


def _write_one(
    speech_path: Path,
    speech_index: int,
    grouped_noise: dict[str, list[Path]],
    output_root: Path,
    seed: int,
    overwrite: bool,
) -> list[dict]:
    speech = load_mono_16k(speech_path)
    speech_id = speech_path.stem
    rows = []
    for environment in ENVIRONMENTS:
        candidates = grouped_noise[environment]
        noise_path = candidates[speech_index % len(candidates)]
        noise = load_mono_16k(noise_path)
        pair_seed = stable_seed(speech_path, environment, seed)
        offset = pair_seed % noise.size
        for snr_db in SNR_LEVELS:
            file_id = f"{speech_id}__{environment.lower()}__snr_{snr_db:+d}"
            output = output_root / environment.lower() / f"snr_{snr_db:+d}" / f"{file_id}.wav"
            mixture, values = mix_signals(speech, noise, snr_db, offset)
            output.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not output.exists():
                sf.write(output, mixture, 16000, subtype="PCM_16")
            rows.append(
                {
                    "file_id": file_id,
                    "output_path": str(output.resolve()),
                    "speech_path": str(speech_path),
                    "noise_path": str(noise_path),
                    "environment": environment,
                    "snr_db": snr_db,
                    "sample_rate": 16000,
                    "num_samples": speech.size,
                    "duration_sec": speech.size / 16000,
                    "noise_offset": offset,
                    **values,
                    "seed": seed,
                }
            )
    return rows


def synthesize_dataset(
    speech_root: Path,
    noise_root: Path,
    output_root: Path,
    manifest_path: Path,
    workers: int = 1,
    seed: int = 2026,
    limit: int | None = None,
    overwrite: bool = False,
    min_free_gb: float = 10.0,
) -> list[dict]:
    speech_files = discover_audio(speech_root)
    if limit:
        speech_files = speech_files[:limit]
    if not speech_files:
        raise RuntimeError(f"No clean speech audio found under {speech_root}")
    grouped_noise = group_noise_files(noise_root)
    estimated = estimate_output_bytes(speech_files)
    free = shutil.disk_usage(output_root.parent if output_root.parent.exists() else "/").free
    reserve = int(min_free_gb * 2**30)
    if estimated + reserve > free:
        raise RuntimeError(
            f"Estimated PCM output is {estimated / 2**30:.1f} GiB but only {free / 2**30:.1f} GiB is free. "
            f"Increase disk or lower --limit; {min_free_gb:.0f} GiB reserve requested."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (path, index, grouped_noise, output_root, seed, overwrite)
        for index, path in enumerate(speech_files)
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, result in enumerate(executor.map(lambda args: _write_one(*args), tasks), start=1):
            rows.extend(result)
            if index % 25 == 0 or index == len(tasks):
                print(f"Mixed {index}/{len(tasks)} clean files ({len(rows)} outputs)")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)
    run_info = {
        "speech_files": len(speech_files),
        "noise_environments": list(ENVIRONMENTS),
        "snr_levels": list(SNR_LEVELS),
        "expected_outputs": len(speech_files) * len(ENVIRONMENTS) * len(SNR_LEVELS),
        "manifest": str(manifest_path.resolve()),
    }
    manifest_path.with_suffix(".json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return rows
