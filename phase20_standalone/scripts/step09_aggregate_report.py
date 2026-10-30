import argparse

from common import BASE_DIR, add_file_arg, read_json, recording_paths, setup_logging, stage_metadata, write_json


def metric_or_empty(path):
    return read_json(path) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 20 Step 9: aggregate metrics and final report.")
    add_file_arg(parser)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step09_aggregate_report")
    all_metrics = {
        "pathA_vad": {k: metric_or_empty(paths.path_a_vad_metrics).get(k) for k in ["precision", "recall", "f1", "false_alarm", "miss"]},
        "pathB_vad": {k: metric_or_empty(paths.path_b_vad_metrics).get(k) for k in ["precision", "recall", "f1", "false_alarm", "miss"]},
        "asr": {k: metric_or_empty(paths.asr_metrics).get(k) for k in ["wer", "substitutions", "insertions", "deletions", "reference_words", "hypothesis_words"]},
        "diarization": {k: metric_or_empty(paths.diar_metrics).get(k) for k in ["der", "miss", "false_alarm", "confusion", "jer"]},
        "fusion": {k: metric_or_empty(paths.fusion_metrics).get(k) for k in ["cpwer", "sawer", "speaker_accuracy", "word_attribution_error", "timeline_accuracy"]},
    }
    write_json(paths.all_metrics, all_metrics)
    write_json(paths.pipeline_metrics, all_metrics)
    lines = [
        "# Phase 20 Dual Path Improved VAD Final Report",
        "",
        "## Pipeline Diagram",
        "",
        "```text",
        f"raw_audio/{paths.name}.wav",
        f"  -> audio/{paths.name}_repaired.wav (mono 16 kHz)",
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
        "- cpWER follows concatenated minimum-permutation speaker assignment; SA-WER scores mapped speaker-attributed transcripts.",
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
    report_text = "\n".join(lines)
    paths.final_report.write_text(report_text)
    paths.final_pipeline_report.write_text(report_text)
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step09_aggregate_report.json", stage_metadata("aggregate_report", [paths.path_a_vad_metrics, paths.path_b_vad_metrics, paths.asr_metrics, paths.diar_metrics, paths.fusion_metrics], [paths.all_metrics, paths.pipeline_metrics, paths.final_report, paths.final_pipeline_report], {"sections": list(all_metrics.keys())}))
    logger.info("Saved aggregate metrics to %s", paths.all_metrics)
    logger.info("Saved final report to %s", paths.final_report)


if __name__ == "__main__":
    main()
