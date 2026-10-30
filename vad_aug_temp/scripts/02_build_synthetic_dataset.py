#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.synthesis import synthesize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix clean speech with QUT-NOISE at six SNR levels.")
    parser.add_argument("--noise-root", type=Path, default=Path("/data/noise/qut_noise"))
    parser.add_argument("--speech-root", type=Path, default=Path("/data/speech/clean"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/output/qut_synthetic"))
    parser.add_argument("--manifest", type=Path, default=Path("/data/output/qut_synthetic_manifest.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = synthesize_dataset(
        args.speech_root,
        args.noise_root,
        args.output_root,
        args.manifest,
        args.workers,
        args.seed,
        args.limit,
        args.overwrite,
        args.min_free_gb,
    )
    print(f"Created {len(rows)} mixtures")


if __name__ == "__main__":
    main()
