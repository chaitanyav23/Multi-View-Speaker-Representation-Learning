"""Streamlit Web Application: Video Transcription with Confidence Analytics."""
from __future__ import annotations

import time
from typing import Any
import altair as alt
import pandas as pd
import requests
import streamlit as st

from frontend.api_client import APIClient
from frontend.components.charts import (
    render_confidence_histogram,
    render_confidence_scatter,
    render_reliability_diagram,
    render_segment_confidence_bar,
    render_speech_timeline,
)
from frontend.components.tables import render_accessible_tables
from frontend.components.transcript_view import render_accessible_transcript

st.set_page_config(
    page_title="Multi-View-Speaker-Representation-Learning",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------- Visual Palettes & Tiers
PALETTES = {
    "Okabe-Ito (Colourblind-Safe)": {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2"},
    "Blue / Orange / Purple (Colourblind-Safe)": {"low": "#E66100", "mid": "#5D3A9B", "high": "#005AB5"},
    "Greyscale (High Contrast / Print)": {"low": "#000000", "mid": "#666666", "high": "#AAAAAA"},
}
TIER_LABEL = {"low": "Low confidence", "mid": "Medium confidence", "high": "High confidence"}
TIER_SHAPE = {"low": "triangle-up", "mid": "square", "high": "circle"}
TIER_SYMBOL = {"low": "▲", "mid": "■", "high": "●"}
TIERS = ["low", "mid", "high"]


def assign_tier(conf: float, low_t: float, high_t: float) -> str:
    if conf < low_t:
        return "low"
    elif conf < high_t:
        return "mid"
    return "high"


# ---------------------------------------------------------------- Sidebar Controls
st.sidebar.title("⚙️ Control Panel")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")
client = APIClient(api_url)

# Backend Health Status
try:
    health = client.health()
    st.sidebar.success(f"Backend Connected ({health.get('app', 'API')})")
except Exception:
    st.sidebar.error("Backend Disconnected. Start FastAPI backend with: uvicorn backend.main:app")

st.sidebar.markdown("---")
st.sidebar.subheader("Accessibility & View Settings")
palette_name = st.sidebar.selectbox("Colour Scheme", list(PALETTES))
colors = PALETTES[palette_name]
font_px = st.sidebar.slider("Transcript Text Size (px)", 14, 32, 18)
low_t = st.sidebar.slider("'Low Confidence' Below", 0.0, 1.0, 0.30, 0.05)
high_t = st.sidebar.slider("'High Confidence' From", 0.0, 1.0, 0.60, 0.05)
if high_t < low_t:
    st.sidebar.warning("High threshold is lower than low threshold.")
    high_t = low_t

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 20 Speech Engine")
st.sidebar.markdown("• **ASR**: Faster-Whisper `large-v3`\n• **VAD**: Pyannote / Silero 16kHz\n• **Attribution**: Maximum-Overlap Midpoint Fusion\n• **Calibration**: Isotonic Regression (ECE < 0.05)")
domain_prompt = st.sidebar.text_input("Contextual Vocabulary Prompt (Optional)", value="")


# ---------------------------------------------------------------- Main Header
st.title("🎙️ Multi-View-Speaker-Representation-Learning")
st.markdown(
    "**ConvoFuse Conversational Intelligence Framework (Optional Demo UI)** | "
    "Dual-Path Speaker Representation Fusion, Calibrated Confidence, and LG-TPST Generative Refinement."
)

# Workflow Selection
action_tab = st.radio(
    "Choose Mode:",
    [
        "🎯 Select Benchmark Audio Recording",
        "📤 Upload New Audio / Video",
        "📂 Past Ingested Recordings",
        "🔬 ConvoFuse Research Replay (R01–R08)",
    ],
    horizontal=True,
)

# Reset active selection when workflow mode changes to prevent stale data cross-talk
if "active_mode" in st.session_state and st.session_state["active_mode"] != action_tab:
    st.session_state["recording_id"] = None
    st.session_state["job_id"] = None
    st.session_state["filename"] = None
st.session_state["active_mode"] = action_tab

current_recording_id: str | None = None
current_filename: str | None = None

if action_tab == "🎯 Select Benchmark Audio Recording":
    from pathlib import Path
    st.info("Benchmark Mode: Select one of the verified project benchmark recordings.")

    # Locate candidate benchmark directories
    possible_dirs = [
        Path("../raw_audio"),
        Path("raw_audio"),
        Path("../preprocessed_B_normalized"),
        Path("storage/uploads"),
    ]
    benchmark_files = []
    for d in possible_dirs:
        if d.exists():
            for p in d.glob("*.*"):
                if p.suffix.lower() in (".mp3", ".wav", ".mp4", ".m4a") and not p.name.startswith("."):
                    benchmark_files.append(p)

    if benchmark_files:
        benchmark_map = {f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)": p for p in benchmark_files}
        chosen_label = st.selectbox("Choose Benchmark Audio:", list(benchmark_map.keys()))
        chosen_path = benchmark_map[chosen_label]

        if st.button("Run Phase 20 Pipeline on Selected Recording", type="primary"):
            with st.spinner(f"Ingesting {chosen_path.name}..."):
                try:
                    file_bytes = chosen_path.read_bytes()
                    upload_res = client.upload_media(file_bytes, chosen_path.name)
                    rec_id = upload_res["recording_id"]
                    st.session_state["recording_id"] = rec_id
                    st.session_state["filename"] = upload_res["filename"]

                    # Trigger real transcription job and explicitly track job_id
                    job_res = client.trigger_transcription(rec_id, use_mock=False, initial_prompt=domain_prompt)
                    st.session_state["job_id"] = job_res.get("job_id")
                    st.success(f"Job submitted for {chosen_path.name}! Polling real inference progress...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to submit benchmark: {e}")
    else:
        st.warning("No benchmark audio found in raw_audio/. Please upload a file via the Upload tab.")

elif action_tab == "📤 Upload New Audio / Video":
    uploaded_file = st.file_uploader(
        "Upload Media File (.mp4, .mov, .wav, .mp3, .m4a)",
        type=["mp4", "mov", "wav", "mp3", "m4a", "aac", "flac"],
    )
    if uploaded_file is not None:
        if st.button("Upload and Process Recording", type="primary"):
            with st.spinner("Uploading and validating media..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    upload_res = client.upload_media(file_bytes, uploaded_file.name)
                    rec_id = upload_res["recording_id"]
                    st.session_state["recording_id"] = rec_id
                    st.session_state["filename"] = upload_res["filename"]

                    # Trigger real transcription job and explicitly track job_id
                    job_res = client.trigger_transcription(rec_id, use_mock=False, initial_prompt=domain_prompt)
                    st.session_state["job_id"] = job_res.get("job_id")
                    st.success("Recording uploaded and Phase 20 pipeline dispatched!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

elif action_tab == "📂 Past Ingested Recordings":
    try:
        recordings = client.list_recordings()
        if recordings:
            rec_options = {f"{r['filename']} ({r['recording_id'][:8]}...) - {r['duration']:.1f}s": r["recording_id"] for r in recordings}
            rec_labels = list(rec_options.keys())

            # Synchronize selector index with active session state if present
            current_id = st.session_state.get("recording_id")
            default_idx = 0
            if current_id:
                for idx, r_id in enumerate(rec_options.values()):
                    if r_id == current_id:
                        default_idx = idx
                        break

            selected_label = st.selectbox("Select Recording", rec_labels, index=default_idx, key="past_recording_selector")
            if selected_label:
                selected_rec_id = rec_options[selected_label]
                # If selection switched to a different recording, clear active job state
                if st.session_state.get("recording_id") != selected_rec_id:
                    st.session_state["recording_id"] = selected_rec_id
                    st.session_state["filename"] = selected_label.split(" (")[0]
                    st.session_state["job_id"] = None
                elif not st.session_state.get("filename"):
                    st.session_state["filename"] = selected_label.split(" (")[0]
        else:
            st.info("No past recordings found. Upload a recording or run the sample demo.")
    except Exception as e:
        st.warning(f"Could not load recording list: {e}")

elif action_tab == "🔬 ConvoFuse Research Replay (R01–R08)":
    st.info(
        "ConvoFuse Frozen-Input Replay Mode: Reuses verified Stage 1 dual-path audio, Stage 2 WhisperX "
        "aligned words, Stage 5 dual-encoder (TitaNet-Large + WeSpeaker ResNet293-LM) affinity-fused RTTM, "
        "and Stage 5B.2 overlap posteriors, then executes live ground-truth-isolated DiarizationLM-Gemma "
        "generative refinement and TPST reconciliation."
    )
    cf_benchmarks = [
        "R01_sara_sharif",
        "R02_dennis_skinner",
        "R03_disabled_parking",
        "R04_piers_robinson",
        "R05_southwest_pilot",
        "R06_rich_karen",
        "R07_bwc_cheap_shot",
        "R08_uk_parliament",
    ]
    c_cf1, c_cf2, c_cf3 = st.columns(3)
    chosen_cf_rec = c_cf1.selectbox("Select Benchmark Recording (R01–R08):", cf_benchmarks, key="cf_benchmark_selector")
    chosen_cf_cond = c_cf2.selectbox(
        "Diarization Condition:",
        ["auto", "E20", "Condition_J", "Condition_K"],
        key="cf_condition_selector",
    )
    chosen_cf_lg = c_cf3.checkbox("Enable Locality-Gated TPST (LG-TPST)", value=True, key="cf_lg_checkbox")

    if st.button("🚀 Run ConvoFuse Pipeline (Dual-Encoder + DiarizationLM-Gemma)", type="primary"):
        with st.spinner(f"Running ConvoFuse pipeline on {chosen_cf_rec}..."):
            try:
                job_res = client.trigger_convofuse_job(
                    recording_id=chosen_cf_rec,
                    execution_mode="FROZEN_INPUT_REPLAY",
                    condition_preference=chosen_cf_cond,
                    enable_refinement=True,
                    use_locality_gating=chosen_cf_lg,
                    sync=False,
                )
                st.session_state["recording_id"] = job_res.get("recording_id", chosen_cf_rec)
                st.session_state["filename"] = f"{chosen_cf_rec}.wav"
                st.session_state["job_id"] = job_res.get("job_id")
                st.success(f"ConvoFuse job submitted for {chosen_cf_rec}!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to submit ConvoFuse job: {exc}")

current_recording_id = st.session_state.get("recording_id")
current_job_id = st.session_state.get("job_id")
current_filename = st.session_state.get("filename", "recording.wav")

if not current_recording_id:
    st.info("Please load a sample recording or upload media above to view results.")
    st.stop()

# ---------------------------------------------------------------- Transcript-First Resolution
transcript_res = None
try:
    transcript_res = client.get_transcript(current_recording_id)
except requests.exceptions.HTTPError as http_err:
    st.error(f"Failed to retrieve transcript: {http_err}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while retrieving transcript: {exc}")
    st.stop()

# If transcript is not yet available, check whether a processing job exists
if not transcript_res:
    # Use explicit job_id if tracked, otherwise fall back to recording_id
    poll_target = current_job_id if current_job_id else current_recording_id
    job_status = None
    try:
        job_status = client.get_job_status(poll_target)
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Failed to query job status: {http_err}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error checking job status: {exc}")
        st.stop()

    if job_status is not None:
        status_val = job_status.get("status", "unknown")
        progress_val = float(job_status.get("progress", 0.0))
        stage_val = job_status.get("stage", "idle")

        if status_val in ("queued", "processing"):
            st.warning(f"⏳ **Processing in progress:** {stage_val}")
            st.progress(progress_val)
            time.sleep(1.0)
            st.rerun()
        elif status_val == "failed":
            st.error(f"❌ Processing failed: {job_status.get('error')}")
            st.stop()
        elif status_val == "completed":
            # Retry fetching transcript once job completes
            transcript_res = client.get_transcript(current_recording_id)
            if not transcript_res:
                time.sleep(0.5)
                st.rerun()
    else:
        # Ingested recording has no active job and no completed transcript
        st.info("ℹ️ **Not transcribed yet.** This recording has been ingested but has not been transcribed.")
        if st.button("Run Phase 20 Pipeline on this Recording", type="primary"):
            with st.spinner("Submitting transcription job..."):
                try:
                    job_res = client.trigger_transcription(current_recording_id, use_mock=False, initial_prompt=domain_prompt)
                    st.session_state["job_id"] = job_res.get("job_id")
                    st.success("Transcription job submitted! Polling progress...")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to submit transcription job: {exc}")
        st.stop()

if not transcript_res:
    st.info("No transcript found yet.")
    st.stop()

# Clear active job_id once transcript is loaded to avoid stale task ID persistence
if st.session_state.get("job_id"):
    st.session_state["job_id"] = None

# Load DataFrames
raw_words = transcript_res.get("words", [])
raw_segs = transcript_res.get("segments", [])

words_df = pd.DataFrame(raw_words)
if not words_df.empty:
    words_df["duration"] = words_df["end"] - words_df["start"]
    words_df["tier"] = words_df["confidence"].apply(lambda c: assign_tier(c, low_t, high_t))
    words_df["Confidence level"] = words_df["tier"].map(TIER_LABEL)
    words_df["Marker"] = words_df["tier"].map(TIER_SYMBOL)

segs_df = pd.DataFrame([{k: v for k, v in s.items() if k != "words"} for s in raw_segs])
if not segs_df.empty:
    segs_df["segment"] = [f"Turn {i + 1}" for i in range(len(segs_df))]
    segs_df["duration"] = segs_df["end"] - segs_df["start"]
    segs_df["n_words"] = [len(s.get("words", [])) for s in raw_segs]

# Scales
color_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[colors[t] for t in TIERS])
shape_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[TIER_SHAPE[t] for t in TIERS])

# ---------------------------------------------------------------- Summary Cards
st.header("1. Executive Summary")
total_dur = float(transcript_res.get("duration", 0.0))
n_words = len(words_df)
mean_conf = float(words_df["confidence"].mean()) if not words_df.empty else 0.0
n_low = int((words_df["tier"] == "low").sum()) if not words_df.empty else 0
speakers_count = len(transcript_res.get("speakers", []))
wpm = (n_words / (total_dur / 60.0)) if total_dur > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Media Duration", f"{total_dur:.1f} s")
c2.metric("Total Words", n_words)
c3.metric("Avg Confidence", f"{mean_conf:.1%}")
c4.metric("Low-Confidence Words", f"{n_low} ({n_low / max(1, n_words):.0%})")
c5.metric("Speakers Detected", speakers_count)

summary_data = transcript_res.get("summary", {})
if summary_data.get("diarization_blocker"):
    st.info(f"ℹ️ **Diarization Notice:** {summary_data['diarization_blocker']} (Status: `{summary_data.get('diarization_status')}`)")

# Audio playback player
stream_url = client.get_stream_url(current_recording_id)
st.subheader("🎧 Media Audio Playback")
st.audio(stream_url)

# ---------------------------------------------------------------- Main Dashboard Tabs
tab_tx, tab_convofuse, tab_charts, tab_data, tab_adaptive, tab_export, tab_plugin = st.tabs([
    "📝 Transcript & Review",
    "🔬 ConvoFuse Conversational Intelligence",
    "📊 Confidence Analytics",
    "📋 Accessible Data Tables",
    "🧠 Adaptive Intelligence",
    "💾 Export Transcripts",
    "🧩 Conversational Intelligence (Plugin)",
])

# ---------------------------------------------------------------- Tab 1: Transcript & Review
with tab_tx:
    convofuse_payload = summary_data.get("convofuse") if isinstance(summary_data, dict) else None
    if isinstance(convofuse_payload, dict) and convofuse_payload.get("turns"):
        st.subheader("Chronological Conversational Transcript (ConvoFuse)")
        st.caption(
            f"Mode: `{convofuse_payload.get('execution_mode')}` | "
            f"Diarization: `{convofuse_payload.get('diarization_condition')}` | "
            f"Refinement: `{convofuse_payload.get('refinement_model')}`"
        )
        for turn_obj in convofuse_payload.get("turns", []):
            ovl_badge = " ⚠️ `[OVERLAP]`" if turn_obj.get("has_overlapping_speech") else ""
            acc_cnt = int(turn_obj.get("accepted_refinement_count", 0))
            rej_cnt = int(turn_obj.get("rejected_refinement_count", 0))
            ref_badge = ""
            if acc_cnt > 0:
                ref_badge += f" ✨ `[+{acc_cnt} TPST Refined]`"
            if rej_cnt > 0:
                ref_badge += f" 🛡️ `[{rej_cnt} Rejected]`"
            st.markdown(
                f"**{turn_obj.get('formatted_header', turn_obj.get('speaker_id'))}**{ovl_badge}{ref_badge}: "
                f"{turn_obj.get('text', '')}"
            )
        st.markdown("---")

    st.subheader("Word-Level Transcript with Multi-Sensory Confidence Markers")

    def on_submit_correction(turn_id: str, old_t: str, new_t: str, new_spk: str):
        client.submit_feedback(
            recording_id=current_recording_id,
            turn_id=turn_id,
            original_text=old_t,
            corrected_text=new_t,
            corrected_speaker=new_spk,
        )

    render_accessible_transcript(
        segs_df=segs_df,
        words_df=words_df,
        colors=colors,
        tier_label=TIER_LABEL,
        tier_symbol=TIER_SYMBOL,
        low_t=low_t,
        high_t=high_t,
        font_px=font_px,
        on_submit_correction=on_submit_correction,
    )

# ---------------------------------------------------------------- Tab 1B: ConvoFuse Conversational Intelligence
with tab_convofuse:
    st.header("🔬 ConvoFuse — Dual-Path Diarization & DiarizationLM-Gemma Refinement")
    convofuse_payload = summary_data.get("convofuse") if isinstance(summary_data, dict) else None
    if not isinstance(convofuse_payload, dict):
        st.info("This recording has not been processed through the ConvoFuse refinement layer yet.")
        if st.button("🚀 Upgrade Recording with ConvoFuse (DiarizationLM-Gemma TPST)", key="upgrade_convofuse_btn"):
            with st.spinner("Running ConvoFuse pipeline..."):
                try:
                    job_res = client.trigger_convofuse_job(
                        recording_id=current_recording_id,
                        execution_mode="FULL_PROCESSING",
                        enable_refinement=True,
                        use_locality_gating=True,
                        sync=True,
                    )
                    st.success("ConvoFuse processing completed!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"ConvoFuse execution failed: {exc}")
    else:
        st.markdown(
            f"**Execution Mode:** `{convofuse_payload.get('execution_mode')}` | "
            f"**Diarization Condition:** `{convofuse_payload.get('diarization_condition')}` | "
            f"**Refinement Model:** `{convofuse_payload.get('refinement_model')}`"
        )
        comp_info = convofuse_payload.get("baseline_vs_refined_comparison", {})
        ref_sum = comp_info.get("refinement_summary", {}).get("summary_counts", {})
        eval_rep = convofuse_payload.get("evaluation_report") or {}

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accepted Corrections", int(ref_sum.get("accepted_corrections", 0)))
        k2.metric("Rejected Proposals", int(ref_sum.get("rejected_proposals", 0)))
        k3.metric("Abstained Words", int(ref_sum.get("abstained_words", 0)))
        k4.metric("Unchanged Words", int(ref_sum.get("unchanged_words", 0)))

        if eval_rep.get("has_reference_annotations"):
            st.subheader("Post-Inference Evaluation (Baseline vs Refined)")
            base_m = eval_rep.get("baseline_metrics", {})
            ref_m = eval_rep.get("refined_metrics", {})
            deltas = eval_rep.get("metric_deltas", {})
            e1, e2, e3 = st.columns(3)
            e1.metric(
                "Timestamp WDER (Assigned)",
                f"{float(ref_m.get('wder_timestamp', {}).get('wder_assigned', 0.0))*100:.2f}%",
                delta=f"{float(deltas.get('delta_wder_timestamp_assigned', 0.0))*100:+.2f} pp",
                delta_color="inverse",
            )
            e2.metric(
                "Lexical WDER",
                f"{float(ref_m.get('wder_lexical', {}).get('wder_lexical', 0.0))*100:.2f}%",
                delta=f"{float(deltas.get('delta_wder_lexical', 0.0))*100:+.2f} pp",
                delta_color="inverse",
            )
            e3.metric(
                "cpWER",
                f"{float(ref_m.get('cpwer', 0.0))*100:.2f}%",
                delta=f"{float(deltas.get('delta_cpwer', 0.0))*100:+.2f} pp",
                delta_color="inverse",
            )

        st.subheader("Speaker Summary Index")
        spk_list = convofuse_payload.get("speaker_summaries", [])
        if spk_list:
            st.dataframe(pd.DataFrame(spk_list), use_container_width=True)

        st.subheader("Complete Readable Conversational Transcript")
        st.code(convofuse_payload.get("readable_transcript_txt", ""), language="text")

        st.subheader("Word-Level Baseline vs Refined Inspection")
        cf_words = convofuse_payload.get("words", [])
        if cf_words:
            flat_words = [
                {
                    "word_index": w.get("word_index"),
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "baseline_speaker": w.get("baseline_speaker_id"),
                    "refined_speaker": w.get("refined_speaker_id"),
                    "decision": w.get("refinement_decision"),
                    "reason": w.get("reconciliation_reason"),
                    "overlap": w.get("overlap_metadata", {}).get("is_overlapping_speech", False),
                    "overlap_prob_max": w.get("overlap_metadata", {}).get("frame_overlap_prob_max"),
                }
                for w in cf_words
            ]
            st.dataframe(pd.DataFrame(flat_words), use_container_width=True)

        with st.expander("🔍 Processing Manifest & 15-Stage Audit Trail", expanded=False):
            st.json(convofuse_payload.get("processing_manifest", {}))

# ---------------------------------------------------------------- Tab 2: Confidence Analytics
with tab_charts:
    st.header("Confidence Analytics & Reliability Visualizations")

    # Scatter
    st.subheader("1. Word Confidence Over Time")
    st.caption("Each point represents a recognized word. Triangles are low, squares medium, circles high confidence.")
    render_confidence_scatter(words_df, color_scale, shape_scale)

    # Histogram
    st.subheader("2. Distribution of Confidence Scores")
    st.caption("Histogram showing frequency of words across confidence intervals.")
    render_confidence_histogram(words_df, color_scale)

    # Speech Timeline
    st.subheader("3. Conversational Speech Turns & Silences")
    st.caption("Bars represent active speech turns. Intervening gaps represent silences.")
    render_speech_timeline(segs_df, colors["high"])

    # Segment Bar
    st.subheader("4. Average Confidence Per Turn")
    render_segment_confidence_bar(segs_df, colors["high"])

    # Reliability Diagram
    st.subheader("5. Calibration Reliability Diagram & Metrics")
    st.caption("Compares predicted confidence levels against observed empirical accuracy (Isotonic Regression).")
    try:
        analytics = client.get_analytics(current_recording_id)
        m1, m2 = st.columns(2)
        m1.metric("Brier Score", f"{analytics.get('brier_score', 0.0):.4f}")
        m2.metric("Expected Calibration Error (ECE)", f"{analytics.get('ece', 0.0):.4f}")
        render_reliability_diagram(analytics.get("reliability_diagram", []))
    except Exception as e:
        st.info(f"Calibration metrics unavailable: {e}")

# ---------------------------------------------------------------- Tab 3: Data Tables
with tab_data:
    render_accessible_tables(segs_df, words_df)

# ---------------------------------------------------------------- Tab 4: Adaptive Intelligence
with tab_adaptive:
    st.header("Adaptive Feedback and Terminology Boosting")
    st.markdown(
        "The system extracts verified domain vocabulary from user manual corrections and injects them "
        "as contextual prompts into subsequent Faster-Whisper recognition runs."
    )
    vocab = client.get_vocabulary()
    st.subheader(f"Accumulated Domain Vocabulary ({len(vocab)} terms)")
    if vocab:
        st.write(", ".join([f"`{v}`" for v in vocab]))
    else:
        st.info("No domain terms accumulated yet. Edit any turn in the Transcript tab to add corrections.")

    st.subheader("Dynamic Whisper Contextual Prompt Preview")
    prompt_preview = f"Domain terms: {', '.join(vocab[:25])}." if vocab else "None"
    st.code(prompt_preview, language="text")

# ---------------------------------------------------------------- Tab 5: Export
with tab_export:
    st.header("Download & Export Formatted Transcripts")
    st.markdown("Select an export format for the finalized speaker-attributed transcript:")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)

    # JSON
    json_bytes = client.get_export_url(current_recording_id, "json")
    try:
        res_json = requests.get(json_bytes).text
        col_e1.download_button("📄 Download JSON", res_json, f"{current_filename}.json", "application/json")
    except Exception:
        col_e1.button("📄 Download JSON", disabled=True)

    # TXT
    txt_url = client.get_export_url(current_recording_id, "txt")
    try:
        res_txt = requests.get(txt_url).text
        col_e2.download_button("📝 Download TXT", res_txt, f"{current_filename}.txt", "text/plain")
    except Exception:
        col_e2.button("📝 Download TXT", disabled=True)

    # SRT
    srt_url = client.get_export_url(current_recording_id, "srt")
    try:
        res_srt = requests.get(srt_url).text
        col_e3.download_button("🎬 Download SubRip (SRT)", res_srt, f"{current_filename}.srt", "text/plain")
    except Exception:
        col_e3.button("🎬 Download SubRip (SRT)", disabled=True)

    # VTT
    vtt_url = client.get_export_url(current_recording_id, "vtt")
    try:
        res_vtt = requests.get(vtt_url).text
        col_e4.download_button("🌐 Download WebVTT", res_vtt, f"{current_filename}.vtt", "text/vtt")
    except Exception:
        col_e4.button("🌐 Download WebVTT", disabled=True)

# ---------------------------------------------------------------- Tab 6: Conversational Intelligence (Plugin)
with tab_plugin:
    st.header("🧩 Conversational Intelligence & Plugin Analytics")
    st.caption("Host integration point for third-party conversational analytics plugins.")

    try:
        plugins = client.list_plugins()
    except Exception as exc:
        st.warning(f"Unable to reach plugin registry: {exc}")
        plugins = []

    if not plugins:
        st.info("No plugin installed / No analysis available for this recording.")
        st.markdown(
            """
            **Developer Integration Guide:**
            External student teams can integrate custom conversational intelligence plugins (summarization,
            topic extraction, action-item tracking, or interaction dynamics) directly into this host platform.

            - Implement `BasePlugin` (`manifest` property and `run(context, parameters)` method).
            - Register via `PluginRegistry.register(plugin)`.
            - Read the Developer Guide: `app/docs/developer_api_integration_guide.md`.
            """
        )
    else:
        st.subheader(f"Registered Plugins ({len(plugins)})")
        plugin_options = {p["name"]: p for p in plugins}
        selected_plugin_name = st.selectbox(
            "Select Plugin Extension:",
            options=list(plugin_options.keys()),
            key="selected_plugin_selector",
        )
        selected_manifest = plugin_options[selected_plugin_name]
        selected_plugin_id = selected_manifest["plugin_id"]

        with st.expander("ℹ️ Plugin Manifest Metadata", expanded=False):
            st.markdown(f"**Plugin ID:** `{selected_manifest['plugin_id']}`")
            st.markdown(f"**Version:** `{selected_manifest.get('version', '1.0.0')}`")
            st.markdown(f"**Author:** {selected_manifest.get('author', 'Unknown')}")
            st.markdown(f"**Description:** {selected_manifest.get('description', '')}")
            caps = selected_manifest.get("capabilities", [])
            if caps:
                st.markdown(f"**Capabilities:** {', '.join([f'`{c}`' for c in caps])}")

        # Check for existing results
        existing_result = None
        try:
            existing_result = client.get_plugin_result(selected_plugin_id, current_recording_id)
        except Exception:
            existing_result = None

        col_run1, col_run2 = st.columns([1, 3])
        with col_run1:
            run_btn = st.button("🚀 Run Plugin Analysis", key=f"run_plugin_{selected_plugin_id}")

        if run_btn:
            with st.spinner(f"Executing {selected_plugin_name}..."):
                try:
                    res = client.run_plugin(selected_plugin_id, current_recording_id)
                    st.success(f"Analysis completed in {res.get('execution_time_seconds', 0.0):.3f}s")
                    existing_result = res
                except Exception as exc:
                    st.error(f"Plugin execution failed: {exc}")

        if existing_result:
            st.subheader("Analysis Results")
            st.caption(
                f"Executed at {existing_result.get('created_at', 'unknown')} "
                f"(Duration: {existing_result.get('execution_time_seconds', 0.0):.3f}s)"
            )
            status = existing_result.get("status", "completed")
            if status == "completed":
                payload = existing_result.get("result_json", {})
                st.json(payload)
            else:
                st.error(f"Plugin reported status: {status}. Error: {existing_result.get('error', 'Unknown error')}")
        else:
            st.info(f"No analysis available for recording '{current_filename}'. Click 'Run Plugin Analysis' above.")
