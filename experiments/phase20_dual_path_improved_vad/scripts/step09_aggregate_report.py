import argparse
from pathlib import Path

from phase19_common import BASE_DIR, read_json, setup_logging, write_json


def metric_or_empty(path):
    return read_json(path) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description="Phase 20 Step 9: aggregate metrics and final report.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    logger = setup_logging("step09_aggregate_report")
    m = args.base_dir / "metrics"
    all_metrics = {
        "pathA_vad": {k: metric_or_empty(m / "pathA_vad_metrics.json").get(k) for k in ["precision", "recall", "f1", "false_alarm", "miss"]},
        "pathB_vad": {k: metric_or_empty(m / "pathB_vad_metrics.json").get(k) for k in ["precision", "recall", "f1", "false_alarm", "miss"]},
        "asr": {k: metric_or_empty(m / "asr_metrics.json").get(k) for k in ["wer", "substitutions", "insertions", "deletions", "reference_words", "hypothesis_words"]},
        "diarization": {k: metric_or_empty(m / "diarization_metrics.json").get(k) for k in ["der", "miss", "false_alarm", "confusion", "jer"]},
        "fusion": {k: metric_or_empty(m / "fusion_metrics.json").get(k) for k in ["cpwer", "sawer", "wder", "speaker_accuracy", "word_attribution_error", "timeline_accuracy"]},
    }
    write_json(m / "all_metrics.json", all_metrics)
    report = args.base_dir / "reports" / "final_report.md"
    lines = [
        "# Phase 20 Dual Path Improved VAD Final Report",
        "",
        "## Pipeline Diagram",
        "",
        "```text",
        "raw_audio/bwc.wav",
        "  -> audio/bwc_repaired.wav (mono 16 kHz)",
        "    -> Path A: DeepFilterNet -> improved Pyannote VAD -> NeMo MSDD/TitaNet -> diarization eval",
        "    -> Path B: -20 LUFS loudnorm -> improved Pyannote VAD -> WhisperX large-v3 alignment -> ASR eval",
        "    -> maximum-overlap fusion -> speaker attributed transcript -> fusion eval",
        "```",
        "",
        "## Stage-Wise Metrics",
        "",
    ]
    for section, vals in all_metrics.items():
        lines.append(f"### {section}")
        for key, val in vals.items():
            lines.append(f"- `{key}`: {val}")
        lines.append("")
    lines.extend([
        "## Observations",
        "",
        "- Path A is optimized for diarization by applying DeepFilterNet before VAD and MSDD.",
        "- Path B is optimized for transcription by loudness normalizing before VAD and WhisperX.",
        "- Phase 20 replaces Phase 19 VAD post-processing with probability-guided recursive min-cut splitting.",
        "- Fusion assigns each aligned word to the diarization speaker with the largest temporal overlap.",
        "",
        "## Strengths",
        "",
        "- Every stage writes an auditable intermediate artifact.",
        "- VAD, ASR, diarization, and fusion metrics are stored as JSON and Markdown reports.",
        "",
        "## Weaknesses",
        "",
        "- cpWER and SA-WER are approximations built from available speaker-attributed segments.",
        "- Speaker attribution metrics depend on the timing quality of both WhisperX alignment and the reference transcript.",
        "",
        "## Comparison Table",
        "",
        "| Component | Primary Metric | Value |",
        "|---|---:|---:|",
        f"| Path A VAD | F1 | {all_metrics['pathA_vad'].get('f1')} |",
        f"| Path B VAD | F1 | {all_metrics['pathB_vad'].get('f1')} |",
        f"| ASR | WER | {all_metrics['asr'].get('wer')} |",
        f"| Diarization | DER | {all_metrics['diarization'].get('der')} |",
        f"| Fusion | cpWER | {all_metrics['fusion'].get('cpwer')} |",
    ])
    report.write_text("\n".join(lines))
    logger.info("Saved aggregate metrics to %s", m / "all_metrics.json")
    logger.info("Saved final report to %s", report)


if __name__ == "__main__":
    main()
