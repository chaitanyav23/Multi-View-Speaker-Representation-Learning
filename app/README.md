# Video Transcription with Confidence Analytics Platform (`app/`)

**Course Project:** CS698B — Conversational Intelligence Platform  
**Specification:** Video Transcription with Confidence Analytics  

---

## 1. Overview

The `app/` directory contains a standalone, fully decoupled web application for conversational speech intelligence. It provides:
1. **Pre-recorded Media Ingestion & ETLS:** Upload MP4, MOV, WAV, MP3, or M4A files, with automated FFmpeg extraction, 16 kHz mono PCM WAV conversion, EBU R128 loudness normalization, and SHA-256 deduplication.
2. **Conversational Speech Intelligence Engine:** High-recall VAD (Pyannote 3.1), recursive chunk slicing (< 30s), Faster-Whisper `large-v3` transcription (`condition_on_previous_text=False`), Wav2Vec2 forced alignment, NeMo Oracle MSDD speaker diarization, and word-level midpoint fusion.
3. **Calibrated Confidence Subsystem:** Feature extraction (acoustic likelihood, word duration, overlap, energy), empirical calibration via Isotonic Regression / Temperature Scaling, and calibration diagnostics (Brier Score, ECE, Reliability Diagrams).
4. **Adaptive Feedback Loop:** User corrections categorized by error type (acoustic vs diarization), with automated extraction of domain vocabulary to dynamically boost Whisper's `initial_prompt`.
5. **Accessible Web Dashboard:** Modern Streamlit frontend with Okabe-Ito colourblind-safe palettes, multi-sensory symbols (`▲`, `■`, `●`), audio playback, interactive turn editor, and CSV/JSON/SRT/WebVTT export.
6. **Decoupled REST API:** Independent FastAPI server with versioned `/api/v1` endpoints and automated Swagger UI documentation at `/docs`.
7. **Zero Outside Dependencies:** The entire application runs self-contained from `app/` without importing or referencing any parent repository files.

---

## 2. Architecture & Component Interaction

```text
Streamlit Frontend (Port 8501)
       │
       ▼ (HTTP / REST API v1)
FastAPI Backend Server (Port 8000)
       │
       ├── Application Services (Upload, Hashing, Background Job Queue, Recovery)
       ├── SQLite Persistence Layer (storage/app.db)
       │
       ├── Speech Intelligence Engine (app/engine/)
       │      ├── Preprocessing (16kHz mono WAV + EBU R128 norm)
       │      ├── Voice Activity Detection (Pyannote 3.1)
       │      ├── ASR Transcription (Faster-Whisper large-v3)
       │      ├── Phoneme/Word Alignment (Wav2Vec2)
       │      ├── Speaker Diarization (NeMo Oracle MSDD + TitaNet)
              │      └── Midpoint Speaker Attribution Fusion (collar tolerance)
       │
       └── Adaptive Subsystem (app/adaptive/)
              ├── Confidence Calibrator (Isotonic Regression, Brier, ECE)
              └── Feedback Loop & Domain Vocabulary Booster
```

---

## 3. Quickstart & Installation

### Step 1: Create and Activate Virtual Environment
```bash
cd app
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 2: Install Core Dependencies
```bash
pip install -e .
pip install -e .[dev]
```

---

## 4. Running the Application

### Option A: Run the FastAPI Backend Server
```bash
# From inside app/
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### Option B: Run the Streamlit Frontend Dashboard
```bash
# From inside app/
streamlit run frontend/streamlit_app.py --server.port 8501
```
- Open your browser at: `http://localhost:8501`

### Option C: Run Benchmark Evaluation CLI on Benchmark Audio
```bash
# Run real Phase 20 inference on benchmark recordings against ground truth:
python scripts/run_benchmarks.py --model tiny --limit 1
# Or run across the full corpus:
python scripts/run_benchmarks.py --model large-v3
```

---

## 5. Running the Automated Test Battery

The application comes with a comprehensive pytest suite covering static independence, schema serialization, confidence math, API endpoints, and end-to-end user workflows:

```bash
# Run all tests from inside app/
pytest -v
```

---

## 6. REST API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status and engine configuration. |
| `POST` | `/api/v1/media/upload` | Upload & validate media file (MP4, MOV, WAV, MP3). |
| `GET` | `/api/v1/media` | List all ingested recordings and metadata. |
| `GET` | `/api/v1/media/{id}/stream` | Direct audio streaming for playback player. |
| `POST` | `/api/v1/transcriptions` | Dispatch speech transcription job (returns `job_id`). |
| `GET` | `/api/v1/transcriptions/{id}/status`| Poll job execution progress (0.0 to 1.0) and stage. |
| `GET` | `/api/v1/transcriptions/{id}` | Retrieve full canonical `TranscriptResult` JSON. |
| `GET` | `/api/v1/transcriptions/{id}/export`| Export transcript as `txt`, `json`, `srt`, or `vtt`. |
| `GET` | `/api/v1/analytics/{id}` | Retrieve confidence tiers, ECE, Brier score, and reliability diagram. |
| `POST` | `/api/v1/feedback` | Submit manual user correction for adaptive vocabulary. |
| `GET` | `/api/v1/feedback` | List recorded user revisions. |
| `GET` | `/api/v1/feedback/vocabulary` | Retrieve accumulated verified domain terminology. |
