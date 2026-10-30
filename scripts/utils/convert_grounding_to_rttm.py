import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding-dir", default="experiments/phase4_grounded_enhancement/grounding_master")
    parser.add_argument("--output-dir", default="experiments/phase5_audit/oracle_rttms")
    args = parser.parse_args()

    grounding_dir = Path(args.grounding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for g_path in grounding_dir.glob("*.json"):
        with open(g_path, "r") as f:
            data = json.load(f)
        
        stem = g_path.stem.replace("_grounding", "")
        rttm_path = output_dir / f"{stem}.rttm"
        
        with open(rttm_path, "w") as f:
            for seg in data["segments"]:
                duration = seg["end"] - seg["start"]
                # NeMo uses the 4th field for start, 5th for duration
                # We use 'speech' as the speaker label for oracle VAD
                f.write(f"SPEAKER {stem} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> speech <NA>\n")
        print(f"Created Oracle RTTM: {rttm_path}")

if __name__ == "__main__":
    main()
