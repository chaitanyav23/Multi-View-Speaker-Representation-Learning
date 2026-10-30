from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw_qut_noise_timit"
PROCESSED_DATA = DATA_ROOT / "processed_qut"
AUDIO_DIR = PROCESSED_DATA / "audio"
LABEL_DIR = PROCESSED_DATA / "labels"
METADATA_PATH = PROCESSED_DATA / "metadata.csv"

OUTPUTS = PROJECT_ROOT / "outputs"
PROBABILITIES_DIR = OUTPUTS / "probabilities"
PREDICTIONS_DIR = OUTPUTS / "predictions"
METRICS_DIR = OUTPUTS / "metrics"
PLOTS_DIR = OUTPUTS / "plots"
EXCEL_DIR = OUTPUTS / "excel"
LOGS_DIR = OUTPUTS / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

FRAME_SHIFT_SEC = 0.01
SAMPLE_RATE = 16000
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]
MIN_DURATION_ON = 0.20
MIN_DURATION_OFF = 0.30
MODEL_NAME = "vad_marblenet"


def ensure_directories() -> None:
    for path in (
        RAW_DATA,
        AUDIO_DIR,
        LABEL_DIR,
        PROBABILITIES_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        EXCEL_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
