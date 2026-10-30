from __future__ import annotations

from pathlib import Path

from .common import environment_info, read_csv
from .config import (
    EXCEL_DIR,
    FRAME_SHIFT_SEC,
    METADATA_PATH,
    METRICS_DIR,
    MIN_DURATION_OFF,
    MIN_DURATION_ON,
    MODEL_NAME,
    PLOTS_DIR,
    REPORTS_DIR,
    SAMPLE_RATE,
)
from .evaluate import select_best


def load_results() -> dict[str, list[dict]]:
    paths = {
        "overall": METRICS_DIR / "overall_metrics.csv",
        "snr": METRICS_DIR / "metrics_by_snr.csv",
        "noise": METRICS_DIR / "metrics_by_noise_type.csv",
        "file": METRICS_DIR / "file_level_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing metrics files: {missing}. Run 04_evaluate_metrics.py first.")
    return {name: read_csv(path) for name, path in paths.items()}


def _numeric_frame(frame, exclude: tuple[str, ...] = ()):
    import pandas as pd

    for column in frame.columns:
        if column not in exclude:
            frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def generate_plots(results: dict[str, list[dict]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    overall = _numeric_frame(pd.DataFrame(results["overall"]), ("test_id", "model_name", "mode"))
    generated = []

    def save(name: str) -> None:
        path = PLOTS_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        generated.append(path)

    for metric, name, ylabel in (
        ("f1", "f1_by_threshold.png", "F1"),
        ("false_alarm_seconds", "false_alarm_seconds_by_threshold.png", "False alarm seconds"),
        ("recall", "recall_by_threshold.png", "Recall"),
    ):
        for mode, group in overall.groupby("mode"):
            plt.plot(group["threshold"], group[metric], marker="o", label=mode)
        plt.xlabel("Threshold")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.3)
        save(name)

    raw = overall[overall["mode"] == "raw"]
    plt.plot(raw["threshold"], raw["precision"], marker="o", label="precision")
    plt.plot(raw["threshold"], raw["recall"], marker="o", label="recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)
    save("precision_recall_by_threshold.png")

    best = select_best(results["overall"])
    for source, group_field, filename in (
        ("snr", "snr", "metrics_by_snr.png"),
        ("noise", "noise_type", "metrics_by_noise_type.png"),
    ):
        frame = _numeric_frame(pd.DataFrame(results[source]), ("test_id", "mode", group_field))
        ids = [row["test_id"] for row in best]
        view = frame[frame["test_id"].isin(ids)]
        for mode, group in view.groupby("mode"):
            plt.plot(group[group_field].astype(str), group["f1"], marker="o", label=mode)
        plt.xlabel(group_field.replace("_", " ").title())
        plt.ylabel("F1")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.grid(alpha=0.3)
        save(filename)

    comparison = overall.pivot(index="threshold", columns="mode", values="false_alarm_seconds")
    comparison.plot(kind="bar")
    plt.ylabel("False alarm seconds")
    plt.xlabel("Threshold")
    save("raw_vs_postprocessed_comparison.png")
    return generated


def _test_config(overall: list[dict]) -> list[dict]:
    return [
        {
            "test_id": row["test_id"],
            "model_name": MODEL_NAME,
            "threshold": row["threshold"],
            "mode": row["mode"],
            "min_duration_on": MIN_DURATION_ON if row["mode"] == "postprocessed" else "NA",
            "min_duration_off": MIN_DURATION_OFF if row["mode"] == "postprocessed" else "NA",
            "frame_shift_sec": FRAME_SHIFT_SEC,
            "audio_sample_rate": SAMPLE_RATE,
            "notes": "Min duration filtering" if row["mode"] == "postprocessed" else "Raw predictions",
        }
        for row in overall
    ]


def _error_analysis(file_rows: list[dict], best: list[dict]) -> list[dict]:
    best_ids = {row["test_id"] for row in best}
    output = []
    for row in file_rows:
        if row["test_id"] not in best_ids:
            continue
        false_alarm = float(row["false_alarm_seconds"])
        missed = float(row["missed_speech_seconds"])
        total_error = false_alarm + missed
        if total_error < 0.01:
            dominant = "good"
        elif false_alarm > missed * 1.25:
            dominant = "false_positive_heavy"
        elif missed > false_alarm * 1.25:
            dominant = "miss_speech_heavy"
        else:
            dominant = "balanced_error"
        output.append(
            {
                "file_id": row["file_id"],
                "noise_type": row["noise_type"],
                "snr": row["snr"],
                "threshold": row["threshold"],
                "mode": row["mode"],
                "false_alarm_seconds": false_alarm,
                "missed_speech_seconds": missed,
                "dominant_error": dominant,
                "notes": "",
            }
        )
    return sorted(output, key=lambda row: row["false_alarm_seconds"] + row["missed_speech_seconds"], reverse=True)


def _summary_rows(results: dict[str, list[dict]], best: list[dict]) -> list[dict]:
    post = next(row for row in best if row["mode"] == "postprocessed")
    worst_noise = max(results["noise"], key=lambda row: float(row["detection_error_rate"]))
    worst_snr = max(results["snr"], key=lambda row: float(row["detection_error_rate"]))
    sections = {
        "Objective": "Measure MarbleNet speech detection robustness in noisy audio.",
        "Dataset Used": "QUT-NOISE-TIMIT",
        "Model Tested": MODEL_NAME,
        "Best Threshold": f"{post['threshold']} (postprocessed; constraint satisfied={post['constraint_satisfied']})",
        "Overall Result": (
            f"Precision={float(post['precision']):.4f}, recall={float(post['recall']):.4f}, "
            f"F1={float(post['f1']):.4f}, false alarms={float(post['false_alarm_seconds']):.2f}s."
        ),
        "Worst Noise Type": f"{worst_noise['noise_type']} at test {worst_noise['test_id']}.",
        "Worst SNR": f"{worst_snr['snr']} at test {worst_snr['test_id']}.",
        "Post-processing Impact": "Compare raw and postprocessed rows at the selected thresholds.",
        "Key Finding": post["summary"],
        "Recommendation": (
            "Use the selected postprocessed threshold for a representative pilot; validate on bodycam audio."
            if post["constraint_satisfied"]
            else "Do not deploy as configured; the minimum recall requirement was not met."
        ),
        "Next Steps": "Validate on in-domain bodycam recordings and tune post-processing.",
    }
    return [{"section": key, "summary": value} for key, value in sections.items()]


def generate_excel(results: dict[str, list[dict]], plot_paths: list[Path]) -> Path:
    import pandas as pd

    metadata = read_csv(METADATA_PATH)
    best = select_best(results["overall"])
    run_info = environment_info()
    run_info.update({"dataset_path": str(METADATA_PATH), "model_name": MODEL_NAME, "notes": ""})
    durations = sum(float(row["duration_sec"]) for row in metadata)
    dataset_summary = [
        {
            "total_files": len(metadata),
            "total_duration_hours": durations / 3600,
            "noise_types": ", ".join(sorted({row["noise_type"] for row in metadata})),
            "snr_values": ", ".join(sorted({row["snr"] for row in metadata})),
            "processed_audio_format": "16 kHz mono WAV PCM (or compliant source referenced in place)",
            "frame_shift_sec": FRAME_SHIFT_SEC,
            "label_format": "interval CSV; speech/non_speech",
        }
    ]
    best_rows = [
        {
            "model_name": MODEL_NAME,
            "mode": row["mode"],
            "best_threshold": row["threshold"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "false_positive_rate": row["false_positive_rate"],
            "false_alarm_seconds": row["false_alarm_seconds"],
            "missed_speech_seconds": row["missed_speech_seconds"],
            "detection_error_rate": row["detection_error_rate"],
            "selection_rule": row["selection_rule"],
            "summary": row["summary"],
        }
        for row in best
    ]
    workbook = EXCEL_DIR / "marblenet_qut_vad_evaluation.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        pd.DataFrame([{"field": key, "value": value} for key, value in run_info.items()]).to_excel(
            writer, sheet_name="Run_Info", index=False
        )
        pd.DataFrame(dataset_summary).to_excel(writer, sheet_name="Dataset_Summary", index=False)
        pd.DataFrame(_test_config(results["overall"])).to_excel(writer, sheet_name="Test_Config", index=False)
        pd.DataFrame(results["overall"]).to_excel(writer, sheet_name="Overall_Metrics", index=False)
        pd.DataFrame(results["snr"]).to_excel(writer, sheet_name="Metrics_By_SNR", index=False)
        pd.DataFrame(results["noise"]).to_excel(writer, sheet_name="Metrics_By_Noise_Type", index=False)
        pd.DataFrame(results["file"]).to_excel(writer, sheet_name="File_Level_Metrics", index=False)
        pd.DataFrame(best_rows).to_excel(writer, sheet_name="Best_Threshold_Summary", index=False)
        pd.DataFrame(_error_analysis(results["file"], best)).to_excel(writer, sheet_name="Error_Analysis", index=False)
        pd.DataFrame(_summary_rows(results, best)).to_excel(writer, sheet_name="Final_Summary", index=False)
        workbook_obj = writer.book
        plot_sheet = workbook_obj.add_worksheet("Plots")
        writer.sheets["Plots"] = plot_sheet
        for index, path in enumerate(plot_paths):
            plot_sheet.insert_image(index * 22, 0, str(path), {"x_scale": 0.65, "y_scale": 0.65})
    return workbook


def generate_markdown(results: dict[str, list[dict]]) -> Path:
    best = select_best(results["overall"])
    summary = _summary_rows(results, best)
    env = environment_info()
    lines = [
        "# NVIDIA MarbleNet VAD Evaluation on QUT-NOISE-TIMIT",
        "",
        "## 1. Objective",
        "Evaluate noisy-speech VAD accuracy, with emphasis on false alarms.",
        "",
        "## 2. VM/hardware used",
        f"- GPU: {env['gpu_name']}",
        f"- CPU count: {env['cpu_count']}",
        f"- RAM: {env['ram_gb']} GiB",
        "",
        "## 3. Dataset used",
        "QUT-NOISE-TIMIT, normalized to 16 kHz mono with 10 ms reference frames.",
        "",
        "## 4. Model tested",
        MODEL_NAME,
        "",
        "## 5. Evaluation method",
        "Frame-level confusion counts; raw and duration-filtered predictions.",
        "",
        "## 6. Thresholds tested",
        "0.30 through 0.80 in increments of 0.05.",
        "",
        "## 7. Metrics calculated",
        "Precision, recall, F1, false-positive/negative rates, false alarm seconds, missed speech seconds, and detection error rate.",
        "",
        "## 8. Overall results",
    ]
    for row in best:
        lines.append(
            f"- {row['mode']}: threshold {row['threshold']}, precision {float(row['precision']):.4f}, "
            f"recall {float(row['recall']):.4f}, F1 {float(row['f1']):.4f}, "
            f"false alarms {float(row['false_alarm_seconds']):.2f}s."
        )
    headings = (
        ("9. Best threshold", "Best Threshold"),
        ("10. Results by SNR", "Worst SNR"),
        ("11. Results by noise type", "Worst Noise Type"),
        ("12. Raw vs post-processed comparison", "Post-processing Impact"),
        ("13. Key false positive patterns", "Key Finding"),
        ("14. Recommendation", "Recommendation"),
        ("15. Next steps", "Next Steps"),
    )
    summaries = {row["section"]: row["summary"] for row in summary}
    for heading, key in headings:
        lines.extend(["", f"## {heading}", summaries[key]])
    path = REPORTS_DIR / "final_vad_evaluation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
