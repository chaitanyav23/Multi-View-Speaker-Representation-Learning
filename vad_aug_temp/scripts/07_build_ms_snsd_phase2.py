#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vad_eval.ms_snsd_phase2 import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MS-SNSD clean/noisy VAD validation subset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("MS-SNSD/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("MS-SNSD/phase2"))
    parser.add_argument("--manifest", type=Path, default=Path("MS-SNSD/phase2/manifest.csv"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = build_dataset(args.dataset_root, args.output_root, args.manifest, args.limit, args.seed)
    print(f"Wrote {len(rows)} clean/noisy entries")


if __name__ == "__main__":
    main()
