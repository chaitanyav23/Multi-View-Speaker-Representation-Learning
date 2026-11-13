# ConvoFuse — Phase20 FastAPI & Streamlit Application Integration Report

**Repository (Self-Contained Target)**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`  
**Application Root**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app`  
**Backend Service**: [`app/backend/services/convofuse_service.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/backend/services/convofuse_service.py)  
**REST Endpoints**: [`app/backend/api/v1/endpoints/convofuse.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/backend/api/v1/endpoints/convofuse.py)  
**Frontend Client**: [`app/frontend/api_client.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/frontend/api_client.py)  
**Streamlit Dashboard**: [`app/frontend/streamlit_app.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/frontend/streamlit_app.py)  
**Integration Test Suite**: [`app/tests/test_convofuse_api.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/tests/test_convofuse_api.py)  

---

## 1. Backward Compatibility with Phase20 Application (Part 6.1)

The complete original Phase20 `app/` directory was copied recursively into `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/` (verified across all 36 directories and 191 files via SHA-256 hashes), while restoring `/home/priyanka/chaitanya/Video-Transcription-with-Confidence-Analytics/app/` to its untouched original Phase20 state.

All original Phase20 endpoints (`/api/v1/health`, `/api/v1/media`, `/api/v1/transcriptions`, `/api/v1/analytics`, `/api/v1/feedback`, `/api/v1/plugins`, `/api/v1/recordings/{id}/context`, `/api/v1/jobs/{id}`) and SQLite tables (`recordings`, `jobs`, `transcripts`, `feedback`, `vocabulary`, `plugin_runs`) remain 100% intact inside Repository B's `app/`.

`to_phase20_transcript_result_dict` converts every `ConvoFuseTranscriptBundle` into a schema-valid Phase20 `TranscriptResult` while embedding the full ConvoFuse bundle under `summary["convofuse"]`.

---

## 2. Dedicated ConvoFuse FastAPI Endpoints (Part 6.2)

Registered under `/api/v1/convofuse` in [`app/backend/api/v1/router.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app/backend/api/v1/router.py):

| HTTP Method & Path | Description |
| :--- | :--- |
| `GET /api/v1/convofuse/benchmarks` | Lists available benchmark recordings (`R01`–`R08`) and default Stage 5 conditions. |
| `POST /api/v1/convofuse/jobs` | Dispatches a ConvoFuse job in `FROZEN_INPUT_REPLAY` or `FULL_PROCESSING` mode (async via `JobService` or synchronous via `sync=true`). |
| `GET /api/v1/convofuse/jobs/{id}/status` | Returns job progress, stage, and the 15-stage `stage_history` audit trail. |
| `GET /api/v1/convofuse/{id}/metadata` | Returns the processing manifest, task-specific audio routing, and input SHA-256 verification. |
| `GET /api/v1/convofuse/{id}/transcript` | Returns the complete `ConvoFuseTranscriptBundle` including `readable_transcript_txt`. |
| `GET /api/v1/convofuse/{id}/words` | Returns word-level JSON with baseline vs refined speakers, overlap metadata, and TPST decisions. |
| `GET /api/v1/convofuse/{id}/turns` | Returns chronological turn-level JSON (`turns` and `baseline_turns`). |
| `GET /api/v1/convofuse/{id}/speakers` | Returns the speaker summary index and RTTM-to-canonical speaker mapping. |
| `GET /api/v1/convofuse/{id}/evaluation` | Returns post-inference evaluation metrics and baseline-vs-refined comparison. |
| `GET /api/v1/convofuse/{id}/export` | Downloads `txt`, `json`, `words`, `turns`, `speakers`, `comparison`, `manifest`, or `evaluation`. |

---

## 3. Streamlit Frontend Evolution (Part 6.3)

1. **Workflow Mode Selector**: Adds `"🔬 ConvoFuse Research Replay (R01–R08)"` alongside `"🎯 Select Benchmark Audio Recording"`, `"📤 Upload New Audio / Video"`, and `"📂 Past Ingested Recordings"`.
2. **Primary Chronological Conversational View**: In `📝 Transcript & Review`, displays `[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX: ...` turns with `⚠️ [OVERLAP]`, `✨ [+N TPST Refined]`, and `🛡️ [N Rejected]` badges above the multi-sensory word-level confidence inspector.
3. **Dedicated `🔬 ConvoFuse Conversational Intelligence` Tab**: Displays TPST decision metrics, post-inference WDER/cpWER deltas, the Speaker Summary Index table, complete readable transcript, word-level baseline-vs-refined inspection table, and the 15-stage processing manifest.
