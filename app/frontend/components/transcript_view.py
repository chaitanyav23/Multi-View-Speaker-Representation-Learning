"""Accessible Transcript View and Interactive Review Components."""
from __future__ import annotations

import html
from typing import Callable, Optional
import pandas as pd
import streamlit as st


def render_accessible_transcript(
    segs_df: pd.DataFrame,
    words_df: pd.DataFrame,
    colors: dict[str, str],
    tier_label: dict[str, str],
    tier_symbol: dict[str, str],
    low_t: float,
    high_t: float,
    font_px: int = 18,
    on_submit_correction: Optional[Callable[[str, str, str, str], None]] = None,
) -> None:
    """Render word-level transcript with multi-sensory accessibility markers."""
    st.markdown(
        f"- {tier_symbol['low']} **Low** (below {low_t:.2f}): bold, double underline, marked with ▲\n"
        f"- {tier_symbol['mid']} **Medium**: dotted underline, marked with ■\n"
        f"- {tier_symbol['high']} **High** (from {high_t:.2f}): plain text"
    )

    view = st.radio(
        "Transcript Format",
        ["Styled (Multi-sensory Colour + Symbols)", "Plain Text (Best for Screen Readers)"],
        horizontal=True,
    )

    for _, seg in segs_df.iterrows():
        seg_id = seg.get("turn_id", seg.get("segment", "turn"))
        speaker = seg.get("speaker", "Speaker")
        start_t = seg["start"]
        end_t = seg["end"]

        st.subheader(f"🗣️ {speaker} [{start_t:.1f}s – {end_t:.1f}s]")

        sw = words_df[(words_df["start"] >= start_t - 1e-4) & (words_df["end"] <= end_t + 1e-4)]

        if view.startswith("Styled"):
            parts = []
            for _, w in sw.iterrows():
                txt = html.escape(str(w["word"]))
                t = w["tier"]
                title = f"{tier_label[t]}: {w['confidence']:.2f} (Speaker: {w.get('speaker', speaker)})"
                if t == "low":
                    style = f"font-weight:700;color:{colors['low']};text-decoration:underline double;"
                    txt += "▲"
                elif t == "mid":
                    style = f"color:{colors['mid']};text-decoration:underline dotted;"
                    txt += "■"
                else:
                    style = ""
                parts.append(f'<span title="{title}" style="{style}">{txt}</span>')

            st.markdown(
                f'<p style="font-size:{font_px}px;line-height:1.8">{" ".join(parts)}</p>',
                unsafe_allow_html=True,
            )
        else:
            plain_words = []
            for _, w in sw.iterrows():
                plain_words.append(f"[{w['word']}?]" if w["tier"] == "low" else str(w["word"]))
            plain_text = " ".join(plain_words)
            st.markdown(
                f'<p style="font-size:{font_px}px;line-height:1.8">{html.escape(plain_text)}</p>',
                unsafe_allow_html=True,
            )
            st.caption("Words in [brackets with ?] represent low-confidence recognition.")

        # Interactive Correction Expander
        with st.expander(f"Edit / Correct {speaker}'s turn ({start_t:.1f}s)"):
            turn_text = seg.get("text", " ".join(sw["word"].tolist()))
            c_text, c_spk = st.columns([3, 1])
            new_text = c_text.text_area("Correct text", value=turn_text, key=f"edit_text_{seg_id}")
            new_spk = c_spk.text_input("Speaker name", value=speaker, key=f"edit_spk_{seg_id}")

            if st.button("Save Correction", key=f"btn_save_{seg_id}"):
                if on_submit_correction:
                    on_submit_correction(seg_id, turn_text, new_text, new_spk)
                    st.success("Correction saved and registered for adaptive learning!")
                    st.rerun()
