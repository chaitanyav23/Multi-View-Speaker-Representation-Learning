import argparse
import json
from pathlib import Path

from common import BASE_DIR, add_file_arg, read_json, recording_paths, setup_logging, stage_metadata, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE19 = PROJECT_ROOT / "phase19_reference"


def safe_json(path):
    return read_json(path) if path.exists() else {}


def vad_summary(path: Path):
    data = safe_json(path)
    durations = [round(s["end"] - s["start"], 3) for s in data.get("segments", [])]
    return {
        "raw_segments": data.get("raw_count"),
        "merged_segments": data.get("merged_count"),
        "removed_short_segments": data.get("removed_short_segments", 0),
        "recursive_min_cut_splits": len(data.get("split_operations", [])),
        "average_chunk_duration": data.get("average_segment_duration"),
        "maximum_chunk_duration": data.get("maximum_segment_duration"),
        "all_chunks_lte_30_seconds": all(d <= 30.0 for d in durations),
        "duration_histogram_before": data.get("pre_split_duration_histogram"),
        "duration_histogram_after": data.get("post_split_duration_histogram"),
        "final_segments": len(data.get("segments", [])),
    }


def metric_delta(phase19_metrics, phase20_metrics):
    out = {}
    for group, vals in phase20_metrics.items():
        out[group] = {}
        for key, value20 in vals.items():
            value19 = phase19_metrics.get(group, {}).get(key)
            delta = None
            if isinstance(value19, (int, float)) and isinstance(value20, (int, float)):
                delta = value20 - value19
            out[group][key] = {"phase19": value19, "phase20": value20, "delta": delta}
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_report(path, comparison):
    lines = [
        "# Phase 20 vs Phase 19 Validation Comparison",
        "",
        "Phase 20 keeps the Phase 19 dual-path pipeline unchanged except for VAD post-processing.",
        "The new VAD retains Pyannote frame probabilities and uses recursive probability-guided min-cut splitting for long speech regions.",
        "",
        "## VAD Processing Statistics",
        "",
    ]
    for path_name in ["pathA", "pathB"]:
        lines.append(f"### {path_name}")
        lines.append("")
        lines.append("| Statistic | Phase 19 | Phase 20 |")
        lines.append("|---|---:|---:|")
        p19 = comparison["vad"][path_name]["phase19"]
        p20 = comparison["vad"][path_name]["phase20"]
        for key in [
            "raw_segments",
            "merged_segments",
            "removed_short_segments",
            "recursive_min_cut_splits",
            "average_chunk_duration",
            "maximum_chunk_duration",
            "all_chunks_lte_30_seconds",
            "final_segments",
        ]:
            lines.append(f"| {key} | {fmt(p19.get(key))} | {fmt(p20.get(key))} |")
        lines.append("")
        lines.append("Before splitting histogram:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(p20.get("duration_histogram_before"), indent=2))
        lines.append("```")
        lines.append("")
        lines.append("After splitting histogram:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(p20.get("duration_histogram_after"), indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Metric Comparison")
    lines.append("")
    for group, vals in comparison["metrics"].items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Metric | Phase 19 | Phase 20 | Delta |")
        lines.append("|---|---:|---:|---:|")
        for key, payload in vals.items():
            lines.append(f"| {key} | {fmt(payload['phase19'])} | {fmt(payload['phase20'])} | {fmt(payload['delta'])} |")
        lines.append("")
    lines.extend(
        [
            "## Discussion",
            "",
            comparison["discussion"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def discussion(comparison):
    metrics = comparison["metrics"]
    observations = []
    for path_name in ["pathA", "pathB"]:
        p20 = comparison["vad"][path_name]["phase20"]
        if p20.get("all_chunks_lte_30_seconds"):
            observations.append(f"{path_name} satisfied the <= 30 second final chunk constraint.")
        else:
            observations.append(f"{path_name} did not satisfy the <= 30 second final chunk constraint.")
    p19_asr = metrics.get("asr", {}).get("wer", {}).get("phase19")
    p20_asr = metrics.get("asr", {}).get("wer", {}).get("phase20")
    p19_der = metrics.get("diarization", {}).get("der", {}).get("phase19")
    p20_der = metrics.get("diarization", {}).get("der", {}).get("phase20")
    p19_cpwer = metrics.get("fusion", {}).get("cpwer", {}).get("phase19")
    p20_cpwer = metrics.get("fusion", {}).get("cpwer", {}).get("phase20")
    if isinstance(p19_asr, (int, float)) and isinstance(p20_asr, (int, float)):
        observations.append("ASR WER improved." if p20_asr < p19_asr else "ASR WER did not improve.")
    if isinstance(p19_der, (int, float)) and isinstance(p20_der, (int, float)):
        observations.append("Diarization DER improved." if p20_der < p19_der else "Diarization DER did not improve.")
    if isinstance(p19_cpwer, (int, float)) and isinstance(p20_cpwer, (int, float)):
        observations.append("Fusion cpWER improved." if p20_cpwer < p19_cpwer else "Fusion cpWER did not improve.")
    return " ".join(observations)


def main():
    parser = argparse.ArgumentParser(description="Phase 20 validation: compare improved VAD experiment with Phase 19.")
    add_file_arg(parser)
    parser.add_argument("--phase19-dir", type=Path, default=PHASE19)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step10_compare_phase19")
    phase19_metrics = safe_json(args.phase19_dir / "metrics" / "all_metrics.json")
    phase20_metrics = safe_json(paths.all_metrics)
    comparison = {
        "vad": {
            "pathA": {"phase19": vad_summary(args.phase19_dir / "grounding" / "pathA_vad.json"), "phase20": vad_summary(paths.path_a_vad)},
            "pathB": {"phase19": vad_summary(args.phase19_dir / "grounding" / "pathB_vad.json"), "phase20": vad_summary(paths.path_b_vad)},
        },
        "metrics": metric_delta(phase19_metrics, phase20_metrics),
    }
    comparison["discussion"] = discussion(comparison)
    write_json(paths.phase19_comparison_json, comparison)
    write_report(paths.phase20_vs_phase19_report, comparison)
    write_report(paths.phase19_comparison_report, comparison)
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step10_compare_phase19.json", stage_metadata("phase19_comparison", [args.phase19_dir / "metrics" / "all_metrics.json", paths.all_metrics], [paths.phase19_comparison_json, paths.phase20_vs_phase19_report, paths.phase19_comparison_report], {"groups": list(comparison["metrics"].keys())}))
    logger.info("Saved comparison report to %s", paths.phase20_vs_phase19_report)


if __name__ == "__main__":
    main()
