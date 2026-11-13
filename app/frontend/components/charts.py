"""Altair Visualization Charts with Accessibility Support."""
from __future__ import annotations

from typing import Any
import altair as alt
import pandas as pd
import streamlit as st


def render_confidence_scatter(words_df: pd.DataFrame, color_scale: alt.Scale, shape_scale: alt.Scale) -> None:
    """Render word-level confidence scatter plot across the audio timeline."""
    if words_df.empty:
        st.info("No words to display in scatter plot.")
        return

    scatter = (
        alt.Chart(words_df)
        .mark_point(filled=True, size=70, opacity=0.85)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)"),
            y=alt.Y("confidence:Q", title="Confidence (0 to 1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Confidence level:N", scale=color_scale, legend=alt.Legend(title="Level")),
            shape=alt.Shape("Confidence level:N", scale=shape_scale),
            tooltip=["word", "start", "end", "confidence", "speaker", "Confidence level"],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)


def render_confidence_histogram(words_df: pd.DataFrame, color_scale: alt.Scale) -> None:
    """Render distribution of confidence scores."""
    if words_df.empty:
        st.info("No words to display in histogram.")
        return

    hist = (
        alt.Chart(words_df)
        .mark_bar()
        .encode(
            x=alt.X("confidence:Q", bin=alt.Bin(step=0.1), title="Confidence range"),
            y=alt.Y("count():Q", title="Number of words"),
            color=alt.Color("Confidence level:N", scale=color_scale, legend=alt.Legend(title="Level")),
            tooltip=["count()"],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)


def render_speech_timeline(segs_df: pd.DataFrame, high_color: str) -> None:
    """Render continuous speech turns and gaps along time axis."""
    if segs_df.empty:
        st.info("No segments to display in timeline.")
        return

    timeline = (
        alt.Chart(segs_df)
        .mark_bar(color=high_color, stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)"),
            x2="end:Q",
            y=alt.Y("segment:N", sort=None, title=None),
            tooltip=["segment", "speaker", "start", "end", "duration", "n_words"],
        )
        .properties(height=max(120, 32 * len(segs_df) + 30))
    )
    st.altair_chart(timeline, use_container_width=True)


def render_segment_confidence_bar(segs_df: pd.DataFrame, high_color: str) -> None:
    """Render average confidence per segment bar chart."""
    if segs_df.empty:
        return

    bar = (
        alt.Chart(segs_df)
        .mark_bar(color=high_color, stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("segment:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("avg_confidence:Q", title="Average confidence", scale=alt.Scale(domain=[0, 1])),
            tooltip=["segment", "speaker", alt.Tooltip("avg_confidence:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(bar, use_container_width=True)


def render_reliability_diagram(diagram_data: list[dict[str, Any]]) -> None:
    """Render calibration reliability diagram comparing confidence to observed accuracy."""
    if not diagram_data:
        st.info("No calibration data available.")
        return

    df = pd.DataFrame(diagram_data)
    # Perfect calibration line
    line_df = pd.DataFrame({"conf": [0.0, 1.0], "acc": [0.0, 1.0]})

    line = (
        alt.Chart(line_df)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x=alt.X("conf:Q", title="Predicted Confidence"), y=alt.Y("acc:Q", title="Empirical Accuracy"))
    )

    bars = (
        alt.Chart(df)
        .mark_bar(opacity=0.6, color="#0072B2")
        .encode(
            x=alt.X("bin_mid:Q", title="Predicted Confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy:Q", title="Empirical Accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["bin_mid", "confidence", "accuracy", "count"],
        )
    )

    combined = (bars + line).properties(height=300)
    st.altair_chart(combined, use_container_width=True)
