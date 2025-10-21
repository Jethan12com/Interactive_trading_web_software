import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from modules.metrics_aggregator import MetricsAggregator
from modules.notifier import TelegramNotifier

# ==========================================================
# MAIN DASHBOARD ENTRY
# ==========================================================
def render_dashboard():
    st.title("📊 CoPilot Performance Dashboard 🚀")

    tabs = st.tabs(["📈 Live Performance", "📊 Historical Analytics"])
    with tabs[0]:
        render_live_performance()
    with tabs[1]:
        render_historical_analytics()


# ==========================================================
# LIVE PERFORMANCE + ALERTS
# ==========================================================
def render_live_performance():
    st.subheader("🚀 Live Performance Monitor")

    # --- Sidebar controls ---
    st.sidebar.header("Live Settings")
    module_filter = st.sidebar.selectbox(
        "Select Module",
        ["All", "rl_trader", "signal_engine", "journal_evaluator"],
        index=0
    )
    days = st.sidebar.slider("Days of History", 1, 60, 7)
    refresh_rate = st.sidebar.slider("Auto-Refresh (seconds)", 5, 60, 15)
    auto_refresh = st.sidebar.toggle("Enable Auto-Refresh", True)
    summary_only = st.sidebar.toggle("Summary-Only Mode", False)

    # --- Alert configuration ---
    st.sidebar.markdown("### ⚠️ Alert Settings")
    enable_alerts = st.sidebar.checkbox("Enable Alerts", True)
    drawdown_limit = st.sidebar.number_input("Max Drawdown (%)", 1, 50, 10)
    loss_limit = st.sidebar.number_input("Max Single Loss ($)", 10, 5000, 500)

    if module_filter == "All":
        module_filter = None

    aggregator = MetricsAggregator()
    notifier = TelegramNotifier()

    placeholder_summary = st.empty()
    placeholder_charts = st.empty()
    placeholder_leaderboard = st.empty()
    placeholder_alerts = st.empty()

    last_alert_time = None
    cooldown = 120  # seconds between alerts

    # --- Load metrics ---
    summary, df = aggregator.summarize_logs(prefix_filter=module_filter, days=days)

    # --- Summary section ---
    with placeholder_summary.container():
        st.subheader(f"Summary (Last {days} Days)")
        if not summary:
            st.warning("No metrics found for selected module/time range.")
        else:
            cols = st.columns(4)
            cols[0].metric("💰 Total PnL", f"{summary.get('total_pnl', 0):.2f}")
            cols[1].metric("🏆 Win Rate", f"{(summary.get('win_rate', 0) or 0) * 100:.1f}%")
            cols[2].metric("📈 Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
            cols[3].metric("⚙️ Expectancy", f"{summary.get('expectancy', 0):.2f}")

    # --- Early exit if summary-only mode ---
    if summary_only:
        st.info("🧭 Summary-Only Mode enabled — charts and alerts are paused for performance.")
        if auto_refresh:
            time.sleep(refresh_rate)
            st.experimental_rerun()
        return

    # --- Equity curve chart ---
    with placeholder_charts.container():
        if not df.empty and "timestamp" in df.columns and "pnl" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df["equity"] = df["pnl"].cumsum()
            st.markdown("### 💹 Equity Curve")
            st.line_chart(df.set_index("timestamp")[["equity"]], use_container_width=True)
        else:
            st.info("No data for equity curve.")

    # --- Leaderboard section ---
    with placeholder_leaderboard.container():
        if not df.empty and "pair" in df.columns:
            st.markdown("### 🏁 Leaderboard")
            pair_stats = df.groupby("pair")["pnl"].agg(["sum", "mean", "count"])
            pair_stats.columns = ["Total PnL", "Avg PnL", "Trades"]
            pair_stats = pair_stats.sort_values("Total PnL", ascending=False)
            st.dataframe(pair_stats.head(10).style.format({"Total PnL": "{:.2f}", "Avg PnL": "{:.2f}"}))
            st.markdown(f"Best performer: **{pair_stats.index[0]}** 💚")
            st.markdown(f"Worst performer: **{pair_stats.index[-1]}** 💔")

    # --- Alert section ---
    with placeholder_alerts.container():
        alerts = []
        if not df.empty and "equity" in df.columns:
            current_equity = df["equity"].iloc[-1]
            peak_equity = df["equity"].max()
            drawdown = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
            max_loss = df["pnl"].min()

            if drawdown > drawdown_limit:
                alerts.append(f"⚠️ Drawdown alert: {drawdown:.2f}% exceeds {drawdown_limit}%")
            if abs(max_loss) > loss_limit:
                alerts.append(f"🔴 Large loss detected: {max_loss:.2f} > ${loss_limit}")

        if alerts:
            st.error("\n".join(alerts))
            if enable_alerts:
                now = time.time()
                if not last_alert_time or (now - last_alert_time) > cooldown:
                    notifier.send_alert("\n".join(alerts))
                    last_alert_time = now
        else:
            st.success("✅ All systems nominal.")

    # --- Auto-refresh ---
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()


# ==========================================================
# HISTORICAL ANALYTICS DASHBOARD
# ==========================================================
def render_historical_analytics():
    st.subheader("📊 Historical Analytics Dashboard")

    st.sidebar.header("Analytics Filters")
    module_filter = st.sidebar.selectbox(
        "Select Module",
        ["All", "rl_trader", "signal_engine", "journal_evaluator"],
        index=0,
        key="hist_filter"
    )
    days = st.sidebar.slider("Days of History", 1, 90, 30, key="hist_days")

    if module_filter == "All":
        module_filter = None

    aggregator = MetricsAggregator()

    @st.cache_data(ttl=300)
    def load_metrics(prefix_filter, days):
        return aggregator.summarize_logs(prefix_filter=prefix_filter, days=days)

    summary, df = load_metrics(module_filter, days)

    # --- Summary Cards ---
    st.subheader(f"Summary (Last {days} Days)")
    if summary:
        cols = st.columns(4)
        cols[0].metric("💰 Total PnL", f"{summary.get('total_pnl', 0):.2f}")
        cols[1].metric("🏆 Win Rate", f"{(summary.get('win_rate', 0) or 0) * 100:.1f}%")
        cols[2].metric("📈 Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
        cols[3].metric("⚙️ Expectancy", f"{summary.get('expectancy', 0):.2f}")
    else:
        st.warning("No metrics found for selected module/time range.")
        return

    if df.empty:
        st.info("No detailed data available.")
        return

    # --- Equity Curve ---
    if "timestamp" in df.columns and "pnl" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values("timestamp")
        df_sorted["equity"] = df_sorted["pnl"].cumsum()
        st.markdown("### 💹 Equity Curve")
        st.line_chart(df_sorted.set_index("timestamp")[["equity"]], use_container_width=True)

    # --- Module Comparison ---
    if "prefix" in df.columns and "pnl" in df.columns:
        st.markdown("#### ⚖️ Module Comparison (PnL by Source)")
        module_perf = df.groupby("prefix")["pnl"].sum().reset_index()
        module_perf.columns = ["Module", "Total PnL"]
        st.bar_chart(module_perf.set_index("Module"), use_container_width=True)

    # --- Win/Loss Distribution ---
    if "win" in df.columns:
        st.markdown("#### 📊 Win vs Loss Distribution")
        win_dist = df["win"].astype(int).value_counts().rename_axis("Outcome").reset_index(name="Count")
        st.bar_chart(win_dist.set_index("Outcome"), use_container_width=True)

    # --- Reward/Risk Scatter ---
    if "reward" in df.columns and "risk" in df.columns:
        st.markdown("#### 🎯 Reward vs Risk Scatter")
        scatter_data = df[["reward", "risk"]].dropna()
        st.scatter_chart(scatter_data, use_container_width=True)

    # --- Export ---
    st.markdown("---")
    st.download_button(
        "📥 Export Data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"copilot_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )