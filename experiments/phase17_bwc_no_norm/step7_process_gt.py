import re
import json
from pathlib import Path

def parse_timestamp(ts):
    parts = ts.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    return float(parts[0])

def main():
    BASE_DIR = Path("experiments/phase17_bwc_no_norm")
    GT_PATH = Path("ground_truth/bwc.txt")
    GT_JSON_DIR = Path("ground_truth_json")
    GT_RTTM_DIR = Path("ground_truth_rttm")
    
    GT_JSON_DIR.mkdir(exist_ok=True)
    GT_RTTM_DIR.mkdir(exist_ok=True)

    with open(GT_PATH, "r") as f:
        lines = f.readlines()

    segments = []
    
    # Regex for "Speaker N [HH:MM:SS - HH:MM:SS]: Text"
    line_re = re.compile(r"(Speaker \d+) \[([\d:]+) - ([\d:]+)\]: (.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = line_re.match(line)
        if match:
            speaker = match.group(1).replace(" ", "_")
            start = parse_timestamp(match.group(2))
            end = parse_timestamp(match.group(3))
            text = match.group(4)
            
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text.strip()
            })

    # Save JSON
    with open(GT_JSON_DIR / "bwc.json", "w") as f:
        json.dump({"file": "bwc", "segments": segments}, f, indent=2)

    # Save RTTM
    with open(GT_RTTM_DIR / "bwc.rttm", "w") as f:
        for s in segments:
            duration = s["end"] - s["start"]
            f.write(f"SPEAKER bwc 1 {s['start']:.3f} {duration:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")

    print(f"Processed {len(segments)} segments from GT.")
    print(f"Saved to {GT_JSON_DIR}/bwc.json and {GT_RTTM_DIR}/bwc.rttm")

if __name__ == "__main__":
    main()
