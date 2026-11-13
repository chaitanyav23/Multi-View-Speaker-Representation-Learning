"""Accessible Data Tables and CSV Export Components."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_accessible_tables(segs_df: pd.DataFrame, words_df: pd.DataFrame) -> None:
    """Render accessible data tables offering full non-visual tabular alternatives to charts."""
    st.header("Data Tables (Screen-Reader & Tabular Alternative)")

    # Segments Table
    st.subheader("1. Conversational Turns & Segments")
    if not segs_df.empty:
        cols_to_show = [c for c in ["turn_id", "speaker", "start", "end", "duration", "n_words", "avg_confidence", "text"] if c in segs_df.columns]
        st.dataframe(segs_df[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("No segment records available.")

    # Words Table
    st.subheader("2. Word-Level Confidence & Timings")
    if not words_df.empty:
        c1, c2 = st.columns([2, 1])
        only_low = c1.checkbox("Filter: Show only low-confidence words", key="chk_only_low")
        selected_spk = c2.selectbox("Filter by speaker", ["All"] + sorted(words_df["speaker"].unique().tolist()))

        filtered = words_df.copy()
        if only_low:
            filtered = filtered[filtered["tier"] == "low"]
        if selected_spk != "All":
            filtered = filtered[filtered["speaker"] == selected_spk]

        word_cols = [c for c in ["word", "start", "end", "duration", "confidence", "raw_confidence", "speaker", "Confidence level"] if c in filtered.columns]
        st.dataframe(filtered[word_cols], use_container_width=True, hide_index=True)

        # Download CSV
        csv_bytes = filtered[word_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Words as CSV",
            data=csv_bytes,
            file_name="attributed_words_confidence.csv",
            mime="text/csv",
        )
    else:
        st.info("No word records available.")
