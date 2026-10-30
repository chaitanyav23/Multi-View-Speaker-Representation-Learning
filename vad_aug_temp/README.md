# MarbleNet VAD Evaluation

This repository evaluates NVIDIA NeMo MarbleNet VAD on QUT-NOISE-TIMIT at
10 ms resolution. It sweeps thresholds from 0.30 to 0.80, compares raw and
duration-filtered predictions, and generates CSV metrics, plots, Excel, and a
Markdown recommendation.

## Prerequisites

- Python 3.10 through 3.12
- NVIDIA driver and a CUDA-enabled PyTorch build
- ffmpeg when source audio is not already 16 kHz mono WAV
- QUT-NOISE-TIMIT audio plus matching `.phn`, `.wrd`, `.rttm`, or interval
  `.csv` annotations

Create the environment:

```bash
conda env create -f environment.yml
conda activate marblenet_vad_eval
python scripts/01_setup_check.py
```

If Conda is unavailable, install Python 3.10, ffmpeg, and then:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset placement

Extract or mount the dataset under:

```text
data/raw_qut_noise_timit/
```

Audio and its annotation should have the same stem, for example:

```text
data/raw_qut_noise_timit/.../sample.wav
data/raw_qut_noise_timit/.../sample.phn
```

Compliant 16 kHz mono WAV files are referenced in place to save disk.
Other audio is converted into `data/processed_qut/audio/`.

## Small validation run

Run 10 to 20 files before the full evaluation:

```bash
python scripts/02_prepare_dataset.py --limit 20
python scripts/03_run_marblenet_vad.py --limit 20
python scripts/04_evaluate_metrics.py --limit 20
python scripts/05_generate_excel_report.py
python scripts/06_generate_summary_report.py
```

Inspect `data/processed_qut/metadata.csv`, several prediction CSVs, and the
initial workbook before proceeding.

## Full run

Rebuild complete metadata, then resume inference. Existing probability files
are skipped automatically.

```bash
python scripts/02_prepare_dataset.py
python scripts/03_run_marblenet_vad.py
python scripts/04_evaluate_metrics.py
python scripts/05_generate_excel_report.py
python scripts/06_generate_summary_report.py
```

Use `--overwrite` on inference only when probabilities must be regenerated.

## Annotation formats

- TIMIT `.phn`/`.wrd`: sample-indexed intervals; silence phones are excluded.
- RTTM: `SPEAKER` intervals.
- CSV: columns `start,end,label`, where speech labels are `speech`, `1`, or
  `true`.

Files with no annotation are recorded in
`data/processed_qut/skipped_unlabelled.txt`; evaluation requires real timing
labels and does not infer ground truth from the noisy waveform.

## Build the synthetic QUT-NOISE benchmark

The synthetic builder uses LibriSpeech `dev-clean` and the five QUT-NOISE
environments. Defaults follow the `/data` layout:

```text
/data/noise/qut_noise/
/data/speech/clean/
/data/output/qut_synthetic/
```

Download and extract both sources:

```bash
VADtesting/bin/python scripts/00_download_synthetic_assets.py
```

Some QUT servers reject non-browser downloads with HTTP 403. In that case,
download these six archives from the official QUT-NOISE page:

```text
QUT_NOISE.zip
QUT_NOISE_CAFE.zip
QUT_NOISE_CAR.zip
QUT_NOISE_HOME.zip
QUT_NOISE_REVERB.zip
QUT_NOISE_STREET.zip
```

Put them in `/data/noise/downloads/`, then extract them with:

```bash
VADtesting/bin/python scripts/00_download_synthetic_assets.py \
  --skip-speech \
  --qut-archive /data/noise/downloads/QUT_NOISE.zip \
  --qut-archive /data/noise/downloads/QUT_NOISE_CAFE.zip \
  --qut-archive /data/noise/downloads/QUT_NOISE_CAR.zip \
  --qut-archive /data/noise/downloads/QUT_NOISE_HOME.zip \
  --qut-archive /data/noise/downloads/QUT_NOISE_REVERB.zip \
  --qut-archive /data/noise/downloads/QUT_NOISE_STREET.zip
```

Build and validate the complete matrix:

```bash
VADtesting/bin/python scripts/02_build_synthetic_dataset.py
VADtesting/bin/python scripts/02_validate_synthetic_dataset.py
```

For LibriSpeech `dev-clean`, the expected total is:

```text
2,703 clean files x 5 environments x 6 SNRs = 81,090 WAV files
```

The builder is deterministic, skips existing WAV files unless `--overwrite`
is supplied, and writes `/data/output/qut_synthetic_manifest.csv`. The
validator checks five audio specifications, reconstructs one isolated
speech/noise pair per SNR, checks 20 files for clipping, and enforces the
exact file-count equation.

After validation succeeds, register the mixtures for the MarbleNet pipeline:

```bash
VADtesting/bin/python scripts/02_register_synthetic_dataset.py
```

These utterance-level mixes are labelled as speech for their full duration.
They measure misses under noise, but do not contain dedicated noise-only
intervals for measuring VAD false alarms. Add padded noise-only regions or a
sequence-construction protocol when false-positive evaluation is required.

## Noise-only false-positive evaluation

`evaluate_vad_on_noise.py` runs MarbleNet only. It does not invoke ASR,
diarization, speaker clustering, or transcription.

Run the requested threshold sweep:

```bash
VADtesting/bin/python evaluate_vad_on_noise.py \
  --input-list data/noise/downloads/noise_files.txt \
  --output-dir outputs/noise_false_positive \
  --thresholds 0.4,0.5,0.6,0.7 \
  --min-speech-duration-ms 150 \
  --min-silence-duration-ms 200 \
  --pre-padding-ms 200 \
  --post-padding-ms 400 \
  --denoise false \
  --device cuda
```

Frame probabilities are cached, so changing thresholds does not rerun the
model. Each threshold produces a per-file CSV, text and JSON summaries, and
segment JSON files. A sweep also produces:

```text
outputs/noise_false_positive/vad_noise_fp_threshold_comparison.csv
```

DeepFilterNet is not installed in the current environment. `--denoise true`
therefore fails explicitly instead of silently reporting raw audio as
denoised.

The small `QUT_NOISE.zip` archive contains protocol code, documentation, and
checksums only. Noise audio is stored in the separate environment archives
such as `QUT_NOISE_STREET.zip`; at least one valid environment archive is
required for this evaluation.

## MS-SNSD Phase 1 benchmark

MS-SNSD is the active noise-only benchmark. Clone and inventory it with:

```bash
git clone https://github.com/microsoft/MS-SNSD.git MS-SNSD/dataset
VADtesting/bin/python scripts/list_ms_snsd_noise.py
```

Run the complete 72-configuration MarbleNet sweep:

```bash
VADtesting/bin/python evaluate_vad_on_noise.py \
  --input-list MS-SNSD/dataset/ms_snsd_noise_files.txt \
  --output-dir MS-SNSD/outputs/noise_vad \
  --grid \
  --thresholds 0.4,0.5,0.6,0.7 \
  --min-speech-values 150,250,400 \
  --min-silence-values 200,400,700 \
  --pre-padding-ms 200 \
  --post-padding-values 200,400 \
  --denoise false \
  --device cuda

VADtesting/bin/python scripts/analyze_ms_snsd_vad_results.py
```

Primary outputs:

```text
MS-SNSD/outputs/noise_vad/vad_noise_false_positive_report.csv
MS-SNSD/outputs/noise_vad/vad_noise_false_positive_summary.csv
MS-SNSD/outputs/noise_vad/vad_noise_false_positive_by_noise_type.csv
MS-SNSD/outputs/noise_vad/vad_noise_false_positive_summary_strict.csv
MS-SNSD/outputs/noise_vad/MS_SNSD_VAD_RECOMMENDATION.md
```

MS-SNSD includes speech-bearing noise categories such as Babble and
NeighborSpeaking. The strict summary excludes explicit speech-bearing
categories so actual speech is not mislabeled as a VAD false positive.

## MS-SNSD Phase 2 speech validation

Build the deterministic 200-utterance clean/noisy validation set, run the
three Phase 1 candidates, and generate the decision report:

```bash
VADtesting/bin/python scripts/07_build_ms_snsd_phase2.py --limit 200
VADtesting/bin/python scripts/08_evaluate_ms_snsd_phase2.py --device cuda
VADtesting/bin/python scripts/09_report_ms_snsd_phase2.py
```

The set contains clean audio plus 20, 10, 5, 0, and -5 dB mixtures. Explicit
speech-bearing noise categories are excluded. Reference VAD labels are
derived from clean-waveform energy and are proxy labels, not manual
annotations.

Primary outputs:

```text
MS-SNSD/outputs/phase2/phase2_file_metrics.csv
MS-SNSD/outputs/phase2/phase2_summary_by_snr.csv
MS-SNSD/outputs/phase2/phase2_candidate_decision.csv
MS-SNSD/outputs/phase2/MS_SNSD_PHASE2_RECOMMENDATION.md
```
