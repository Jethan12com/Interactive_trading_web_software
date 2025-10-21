import streamlit as st
import pandas as pd

def render_signal_viewer(signals_df):
    st.header("📊 Signal Viewer")

    if signals_df.empty:
        st.info("No signals available.")
        return

    # -----------------------------
    # Filtering options
    # -----------------------------
    unique_patterns = sorted(signals_df["pattern_id"].unique())
    pattern_filter = st.multiselect("🔎 Filter by Pattern", unique_patterns, default=[])

    filtered_df = signals_df.copy()
    if pattern_filter:
        filtered_df = filtered_df[filtered_df["pattern_id"].isin(pattern_filter)]

    # -----------------------------
    # Sorting options
    # -----------------------------
    sort_order = st.radio(
        "📈 Sort by Confidence Score",
        options=["High → Low", "Low → High"],
        index=0,
        horizontal=True
    )

    if sort_order == "High → Low":
        filtered_df = filtered_df.sort_values(by="adjusted_signal_score", ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by="adjusted_signal_score", ascending=True)

    # -----------------------------
    # Color-coded pattern tags
    # -----------------------------
    def style_pattern(pattern_str: str) -> str:
        parts = pattern_str.split(" + ")
        styled_parts = []
        for p in parts:
            if "Breakout" in p:
                styled_parts.append(f"<span style='color:green; font-weight:bold'>{p}</span>")
            elif "Breakdown" in p:
                styled_parts.append(f"<span style='color:red; font-weight:bold'>{p}</span>")
            else:
                styled_parts.append(f"<span style='color:blue;'>{p}</span>")
        return " + ".join(styled_parts)

    # -----------------------------
    # Color-coded signal direction
    # -----------------------------
    def style_signal(signal: str) -> str:
        if signal == "BUY":
            return f"<span style='color:green; font-weight:bold'>{signal}</span>"
        elif signal == "SELL":
            return f"<span style='color:red; font-weight:bold'>{signal}</span>"
        return signal

    # -----------------------------
    # Color-coded risk level
    # -----------------------------
    def style_risk(risk: str) -> str:
        if risk == "Low":
            return f"<span style='color:green; font-weight:bold'>{risk}</span>"
        elif risk == "Moderate":
            return f"<span style='color:orange; font-weight:bold'>{risk}</span>"
        elif risk == "High":
            return f"<span style='color:red; font-weight:bold'>{risk}</span>"
        return risk

    # Apply styles
    filtered_df = filtered_df.copy()
    filtered_df["pattern_id"] = filtered_df["pattern_id"].apply(style_pattern)
    filtered_df["signal"] = filtered_df["signal"].apply(style_signal)
    filtered_df["risk_level"] = filtered_df["risk_level"].apply(style_risk)

    # -----------------------------
    # Render styled + sorted table
    # -----------------------------
    st.markdown(
        filtered_df[
            [
                "timestamp",
                "user_id",
                "pair",
                "signal",
                "adjusted_signal_score",
                "pattern_id",
                "session",
                "risk_level",
                "tp",
                "sl",
                "capital",
                "position_size",
                "activation_time",
                "expiration_time",
                "outcome",
            ]
        ].to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
