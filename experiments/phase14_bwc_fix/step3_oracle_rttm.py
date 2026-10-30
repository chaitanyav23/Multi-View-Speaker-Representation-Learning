import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_rttm", type=str, required=True)
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_rttm = Path(args.output_rttm)

    with open(input_json) as f:
        data = json.load(f)

    file_id = Path(data["file"]).stem

    output_rttm.parent.mkdir(parents=True, exist_ok=True)

    with open(output_rttm, "w") as f:
        for seg in data["segments"]:
            start = seg["start"]
            duration = round(seg["end"] - seg["start"], 3)
            # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <lookahead> <speaker> <conf> <unused>
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

    print(f"Created oracle RTTM at {output_rttm}")

if __name__ == "__main__":
    main()
