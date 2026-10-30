import json
from pathlib import Path

BASE_DIR = Path("experiments/phase17_bwc_no_norm")
GROUNDING_PATH = BASE_DIR / "grounding" / "bwc_grounding.json"
RTTM_PATH = BASE_DIR / "grounding" / "bwc_oracle.rttm"

with open(GROUNDING_PATH) as f:
    data = json.load(f)

with open(RTTM_PATH, "w") as f:
    for seg in data["segments"]:
        start = seg["start"]
        duration = round(seg["end"] - seg["start"], 3)
        # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <lookahead> <speaker> <conf> <unused>
        f.write(f"SPEAKER bwc_repaired 1 {start:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

print(f"Created oracle RTTM at {RTTM_PATH}")
