import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------
# Delivery & Session Analytics
# -----------------------------
def render_delivery_analytics(signals_df, users_list):
    st.header("📊 Delivery & Session Analytics")
    if signals_df.empty:
        st.info("No signals available for analytics.")
        return

    # Session Counts
    session_counts = signals_df["session"].value_counts().reset_index()
    session_counts.columns = ["Session","Signals"]
    st.plotly_chart(px.bar(session_counts, x="Session", y="Signals", title="Signals per Session"))

    # Confidence Distribution
    st.plotly_chart(px.histogram(signals_df, x="adjusted_signal_score", nbins=10, title="Confidence Distribution"))

    # Per-User Performance with Capital
    st.subheader("Per-User Performance by Capital")
    users_df = pd.DataFrame(users_list)
    if not signals_df.empty and not users_df.empty:
        merged_df = signals_df.merge(users_df[["user_id","capital"]], on="user_id")
        per_user = merged_df.groupby(["user_id","capital"]).agg({"pnl": "sum", "adjusted_signal_score": "mean"}).reset_index()
        per_user["pnl"] = per_user["pnl"].fillna(0)
        st.dataframe(per_user)
        st.plotly_chart(px.bar(per_user, x="user_id", y="pnl", color="capital", title="PnL by User and Capital"))

    # Per-Pair Performance
    st.subheader("Per-Pair Performance")
    per_pair = signals_df.groupby("pair").agg({"pnl": "sum", "adjusted_signal_score": "mean"}).reset_index()
    st.dataframe(per_pair)
    st.plotly_chart(px.bar(per_pair, x="pair", y="pnl", title="PnL per Pair", color="pnl"))


# -----------------------------
# Alerts Conversion Analytics
# -----------------------------
def render_alerts_analytics(log_file="data/logs/pre_activation_alerts.csv"):
    st.header("⏳ Pre-Activation Alerts Analytics")

    if not os.path.exists(log_file):
        st.info("No alerts have been logged yet.")
        return

    alerts_df = pd.read_csv(log_file)
    if alerts_df.empty:
        st.info("No alerts to display.")
        return

    # Conversion Performance
    if "outcome" in alerts_df.columns:
        st.subheader("🎯 Conversion Performance")
        total_alerts = len(alerts_df)
        wins = (alerts_df["outcome"] == "Win").sum()
        losses = (alerts_df["outcome"] == "Loss").sum()
        pending = (alerts_df["outcome"] == "Pending").sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wins", wins)
        with col2:
            st.metric("Losses", losses)
        with col3:
            st.metric("Pending", pending)

        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            st.progress(win_rate / 100)
            st.write(f"Win Rate: **{win_rate:.2f}%**")
    else:
        st.warning("Outcome tracking not enabled yet (waiting for journal_evaluator).")

    # Distribution by user/pair
    st.subheader("📈 Alerts Distribution")
    alerts_per_user = alerts_df["user_id"].value_counts().reset_index()
    alerts_per_user.columns = ["user_id", "alerts"]

    alerts_per_pair = alerts_df["pair"].value_counts().reset_index()
    alerts_per_pair.columns = ["pair", "alerts"]

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(alerts_per_user.set_index("user_id"))
    with col2:
        st.bar_chart(alerts_per_pair.set_index("pair"))
