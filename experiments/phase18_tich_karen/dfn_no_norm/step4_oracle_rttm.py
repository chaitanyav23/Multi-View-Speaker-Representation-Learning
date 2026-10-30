import json
from pathlib import Path

BASE_DIR = Path("experiments/phase18_tich_karen/dfn_no_norm")
GROUNDING_PATH = BASE_DIR / "grounding" / "rich_karen_grounding.json"
RTTM_PATH = BASE_DIR / "grounding" / "rich_karen_oracle.rttm"

with open(GROUNDING_PATH) as f:
    data = json.load(f)

with open(RTTM_PATH, "w") as f:
    for seg in data["segments"]:
        start = seg["start"]
        duration = round(seg["end"] - seg["start"], 3)
        # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <lookahead> <speaker> <conf> <unused>
        f.write(f"SPEAKER rich_karen_dfn_no_norm 1 {start:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

print(f"Created oracle RTTM at {RTTM_PATH}")
