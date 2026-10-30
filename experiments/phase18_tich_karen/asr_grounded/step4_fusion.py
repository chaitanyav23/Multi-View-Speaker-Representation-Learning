import json
from pathlib import Path

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen/asr_grounded")
    WORDS_PATH = BASE_DIR / "outputs" / "aligned_words_initial.json"
    RTTM_PATH = BASE_DIR / "diarization" / "asr_grounded_msdd.rttm"
    OUTPUT_TRANSCRIPT = BASE_DIR / "outputs" / "asr_grounded_transcript.json"

    with open(WORDS_PATH, "r") as f:
        words = json.load(f)

    # Load RTTM segments
    diarization = []
    with open(RTTM_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                diarization.append({
                    "start": float(parts[3]),
                    "end": float(parts[3]) + float(parts[4]),
                    "speaker": parts[7]
                })

    # Simple Fusion
    transcript = []
    for w in words:
        if w["start"] is None: continue
        best_speaker = "UNKNOWN"
        for d in diarization:
            if d["start"] <= w["start"] <= d["end"]:
                best_speaker = d["speaker"]
                break
        transcript.append({"word": w["word"], "start": w["start"], "end": w["end"], "speaker": best_speaker})

    # Grouping
    grouped = []
    if transcript:
        curr_speaker = transcript[0]["speaker"]
        curr_seg = {"speaker": curr_speaker, "start": transcript[0]["start"], "end": transcript[0]["end"], "text": transcript[0]["word"]}
        for i in range(1, len(transcript)):
            w = transcript[i]
            if w["speaker"] == curr_speaker and (w["start"] - curr_seg["end"] < 1.0):
                curr_seg["text"] += " " + w["word"]
                curr_seg["end"] = w["end"]
            else:
                grouped.append(curr_seg)
                curr_speaker = w["speaker"]
                curr_seg = {"speaker": curr_speaker, "start": w["start"], "end": w["end"], "text": w["word"]}
        grouped.append(curr_seg)

    with open(OUTPUT_TRANSCRIPT, "w") as f:
        json.dump(grouped, f, indent=2)

    print(f"Fusion complete. Transcript saved to {OUTPUT_TRANSCRIPT}")

if __name__ == "__main__":
    main()
