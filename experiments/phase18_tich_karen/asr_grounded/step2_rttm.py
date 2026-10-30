import json
from pathlib import Path

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen/asr_grounded")
    GROUNDING_PATH = BASE_DIR / "grounding" / "asr_derived_grounding.json"
    RTTM_PATH = BASE_DIR / "grounding" / "asr_derived_oracle.rttm"

    with open(GROUNDING_PATH) as f:
        data = json.load(f)

    with open(RTTM_PATH, "w") as f:
        for seg in data["segments"]:
            start = seg["start"]
            duration = round(seg["end"] - seg["start"], 3)
            # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <lookahead> <speaker> <conf> <unused>
            f.write(f"SPEAKER rich_karen_normalized 1 {start:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

    print(f"Created ASR-derived oracle RTTM at {RTTM_PATH}")

if __name__ == "__main__":
    main()
