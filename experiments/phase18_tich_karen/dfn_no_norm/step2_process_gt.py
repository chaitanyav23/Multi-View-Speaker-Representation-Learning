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
    BASE_DIR = Path("experiments/phase18_tich_karen/dfn_no_norm")
    GT_PATH = Path("ground_truth/rich_karen.txt")
    
    (BASE_DIR / "grounding").mkdir(exist_ok=True)

    with open(GT_PATH, "r") as f:
        lines = f.readlines()

    segments = []
    header_re = re.compile(r"(Speaker_\d+) \[([\d:]+) - ([\d:]+)\]")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        match = header_re.match(line)
        if match:
            speaker = match.group(1)
            start = parse_timestamp(match.group(2))
            end = parse_timestamp(match.group(3))
            
            i += 1
            text = ""
            while i < len(lines) and not header_re.match(lines[i].strip()) and lines[i].strip():
                text += lines[i].strip() + " "
                i += 1
            
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text.strip()
            })
        else:
            i += 1

    # Save JSON in grounding (for evaluation)
    with open(BASE_DIR / "grounding" / "rich_karen_gt.json", "w") as f:
        json.dump({"file": "rich_karen", "segments": segments}, f, indent=2)

    # Save RTTM in grounding (for evaluation)
    with open(BASE_DIR / "grounding" / "rich_karen_gt.rttm", "w") as f:
        for s in segments:
            duration = s["end"] - s["start"]
            f.write(f"SPEAKER rich_karen 1 {s['start']:.3f} {duration:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")

    print(f"Processed {len(segments)} segments from GT.")
    print(f"Saved to {BASE_DIR}/grounding/rich_karen_gt.json and rich_karen_gt.rttm")

if __name__ == "__main__":
    main()
