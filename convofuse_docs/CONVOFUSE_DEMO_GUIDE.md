# ConvoFuse — Demonstration & Operations Guide

**Project**: `CS787_Project`  
**Framework**: `ConvoFuse`  

---

## 1. Prerequisites & Environment Verification

1. **Local Ollama Server (`DiarizationLM-Gemma-4-E4B-v1` Q4_K_M)**:
   ```bash
   curl -s http://localhost:11434/api/version
   ollama list | grep diarizationlm-gemma
   ```
2. **Python Environments**:
   - Research & CLI environment: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`
   - Phase20 Application environment: `/home/priyanka/anaconda3/envs/cs698b_phase20/bin/python`

---

## 2. Running the ConvoFuse CLI (`R01` Demo & `R01`–`R08` Corpus Benchmark)

### A. Single-Recording R01 Demonstration (Stage 5B.4 E20 + DiarizationLM-Gemma LG-TPST)
```bash
cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --recording R01_sara_sharif \
  --condition E20 \
  --output-dir convofuse_outputs/r01_demo
```
Inspect the generated readable transcript:
```bash
cat convofuse_outputs/r01_demo/R01_sara_sharif/R01_sara_sharif__transcript.txt
```

### B. Sequential R01–R08 Corpus Benchmark
```bash
cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --all-recordings \
  --condition auto \
  --output-dir convofuse_outputs/r01_r08_benchmark
```

---

## 3. Running the Integrated FastAPI Backend & Streamlit Frontend (Self-Contained in Repository B)

### Step 1: Start the FastAPI Backend
```bash
cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app
/home/priyanka/anaconda3/envs/cs698b_phase20/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Start the Streamlit Frontend
```bash
cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app
/home/priyanka/anaconda3/envs/cs698b_phase20/bin/streamlit run frontend/streamlit_app.py --server.port 8501
```

### Step 3: Trigger & Inspect ConvoFuse via REST API or UI
- **In Streamlit (`http://127.0.0.1:8501`)**:
  1. Select **"🔬 ConvoFuse Research Replay (R01–R08)"** (or upload a new audio/video file under **"📤 Upload New Audio / Video"**).
  2. Choose `R01_sara_sharif` (or any recording `R01`–`R08`), set Diarization Condition to `auto` (`E20` for R01, `Condition_J` for R02–R08), keep **Locality-Gated TPST (LG-TPST)** enabled, and click **"🚀 Run ConvoFuse Pipeline (Dual-Encoder + DiarizationLM-Gemma)"**.
  3. View the chronological speaker turns `[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX: ...` in **📝 Transcript & Review** and inspect the full TPST telemetry, Speaker Summary Index, word-level diff table, and 15-stage manifest in **🔬 ConvoFuse Conversational Intelligence**.
- **Via cURL**:
  ```bash
  # Trigger R01 ConvoFuse job synchronously
  curl -s -X POST http://127.0.0.1:8000/api/v1/convofuse/jobs \
    -H "Content-Type: application/json" \
    -d '{"recording_id": "R01", "execution_mode": "FROZEN_INPUT_REPLAY", "condition_preference": "E20", "sync": true}' | jq .

  # Download readable chronological transcript TXT
  curl -s "http://127.0.0.1:8000/api/v1/convofuse/R01_sara_sharif/export?format=txt"
  ```
