import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules.user_management import UserManagement
import pytz

# =====================================================
# LOAD SIGNAL DATA
# =====================================================
def load_signal_data():
    """Load all signal logs from data/logs/ into a single DataFrame."""
    log_dir = "data/logs"
    signals = []
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(log_dir, file))
                    signals.append(df)
                except Exception as e:
                    st.warning(f"Error loading {file}: {e}")
    return pd.concat(signals, ignore_index=True) if signals else pd.DataFrame()


# =====================================================
# METRIC CALCULATIONS
# =====================================================
def compute_success_rate(signals_df):
    if signals_df.empty:
        return pd.DataFrame(columns=["date", "success_rate"])
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    grouped = signals_df.groupby(signals_df['timestamp'].dt.date)
    success = grouped.apply(lambda x: (x['outcome'] == 'Hit TP').mean() * 100)
    return pd.DataFrame({"date": success.index, "success_rate": success.values})


def compute_win_loss_by_symbol(signals_df):
    if signals_df.empty:
        return pd.DataFrame(columns=["pair", "wins", "losses"])
    grouped = signals_df.groupby('pair')
    df = pd.DataFrame({
        "wins": grouped.apply(lambda x: (x['outcome'] == 'Hit TP').sum()),
        "losses": grouped.apply(lambda x: (x['outcome'] == 'Hit SL').sum())
    }).reset_index()
    return df


def compute_user_performance(signals_df, users):
    if signals_df.empty or not users:
        return pd.DataFrame(columns=["User", "Win Rate", "PnL"])
    rows = []
    for user in users:
        user_data = users[user]
        user_signals = signals_df[signals_df['user_id'] == user_data['user_id']]
        if user_signals.empty:
            continue
        wins = (user_signals['outcome'] == 'Hit TP').sum()
        total = len(user_signals)
        win_rate = (wins / total * 100) if total > 0 else 0
        pnl = (
            user_signals[user_signals['outcome'] == 'Hit TP']['position_size'].sum()
            - user_signals[user_signals['outcome'] == 'Hit SL']['position_size'].sum()
        )
        rows.append({
            "User": user_data.get('name', 'Unknown'),
            "Win Rate": round(win_rate, 2),
            "PnL": round(pnl, 2)
        })
    return pd.DataFrame(rows)


def compute_volatility_index(signals_df):
    if signals_df.empty:
        return pd.DataFrame(columns=["pair", "volatility"])
    grouped = signals_df.groupby('pair')
    df = pd.DataFrame({
        "volatility": grouped.apply(lambda x: (x['tp'] - x['sl']).abs().mean())
    }).reset_index()
    return df


def compute_pattern_discovery(signals_df):
    if signals_df.empty or 'pattern_id' not in signals_df.columns:
        return pd.DataFrame(columns=["Pattern ID", "Frequency", "Success Probability"])
    grouped = signals_df.groupby('pattern_id')
    rows = []
    for pid, group in grouped:
        freq = len(group)
        success = (group['outcome'] == 'Hit TP').mean() * 100
        rows.append({
            "Pattern ID": pid,
            "Frequency": freq,
            "Success Probability": round(success, 2)
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="Frequency", ascending=False)


def compute_user_growth_trend(user_management: UserManagement):
    """Compute user creation trend over time."""
    users_dict = user_management.list_users()
    if not users_dict:
        return pd.DataFrame(columns=["Date", "Total Users"])
    df = pd.DataFrame(users_dict)
    df["created_on"] = pd.to_datetime(df["created_on"], errors="coerce")
    df = df.dropna(subset=["created_on"])
    trend = df.groupby(df["created_on"].dt.date).size().cumsum()
    return pd.DataFrame({"Date": trend.index, "Total Users": trend.values})


def compute_signal_volume_by_session(signals_df):
    if signals_df.empty or "session" not in signals_df.columns:
        return pd.DataFrame(columns=["date", "session", "count"])
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    grouped = signals_df.groupby([signals_df['timestamp'].dt.date, "session"]).size().reset_index(name="count")
    return grouped.rename(columns={"timestamp": "date"})


# =====================================================
# PNL / Equity Curves
# =====================================================
def compute_daily_pnl(signals_df):
    if signals_df.empty or "position_size" not in signals_df.columns:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl"])
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    signals_df['pnl'] = signals_df.apply(
        lambda x: x['position_size'] if x['outcome'] == 'Hit TP' else (-x['position_size'] if x['outcome'] == 'Hit SL' else 0),
        axis=1
    )
    daily_pnl = signals_df.groupby(signals_df['timestamp'].dt.date)['pnl'].sum().reset_index()
    daily_pnl.rename(columns={"timestamp": "date", "pnl": "daily_pnl"}, inplace=True)
    daily_pnl['cumulative_pnl'] = daily_pnl['daily_pnl'].cumsum()
    return daily_pnl


def compute_user_equity_curves(signals_df, users):
    if signals_df.empty or not users:
        return pd.DataFrame()
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    signals_df['pnl'] = signals_df.apply(
        lambda x: x['position_size'] if x['outcome'] == 'Hit TP'
        else (-x['position_size'] if x['outcome'] == 'Hit SL' else 0), axis=1
    )
    user_curves = {}
    for user_data in users.values():
        user_id = user_data['user_id']
        user_signals = signals_df[signals_df['user_id'] == user_id]
        if user_signals.empty:
            continue
        daily = user_signals.groupby(user_signals['timestamp'].dt.date)['pnl'].sum()
        user_curves[user_data.get('name', 'Unknown')] = daily.cumsum()
    if not user_curves:
        return pd.DataFrame()
    equity_df = pd.DataFrame(user_curves).fillna(method='ffill').fillna(0)
    equity_df.index.name = "date"
    return equity_df


def compute_combined_equity_curves(signals_df, users):
    user_eq = compute_user_equity_curves(signals_df, users)
    if user_eq.empty:
        return pd.DataFrame()
    user_eq['Total PnL'] = user_eq.sum(axis=1)
    return user_eq


# =====================================================
# STREAMLIT RENDER
# =====================================================
def render_analytics_dashboard(user_management: UserManagement):
    st.title("📊 CoPilot Analytics Dashboard")

    # Load data
    signals_df = load_signal_data()
    users = user_management.list_users()

    # ---- Metrics summary ----
    total_signals = len(signals_df)
    avg_conf = round(signals_df['adjusted_signal_score'].mean(), 2) if not signals_df.empty else 0
    win_rate = round((signals_df['outcome'] == 'Hit TP').mean() * 100, 2) if not signals_df.empty else 0
    active_users = sum(1 for u in users.values() if u.get('status') == 'Active') if users else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", total_signals)
    col2.metric("Avg Confidence", f"{avg_conf}%")
    col3.metric("Win Rate", f"{win_rate}%")
    col4.metric("Active Users", active_users)

    st.divider()

    # ---- Success rate over time ----
    st.subheader("📈 Signal Success Rate Over Time")
    success_df = compute_success_rate(signals_df)
    st.line_chart(success_df.set_index("date")) if not success_df.empty else st.info("No signals to plot yet.")

    # ---- Win/Loss by Symbol ----
    st.subheader("⚔️ Win/Loss by Symbol")
    wl_df = compute_win_loss_by_symbol(signals_df)
    st.bar_chart(wl_df.set_index("pair")[["wins", "losses"]]) if not wl_df.empty else st.info("No win/loss data available.")

    # ---- User Performance ----
    st.subheader("👤 User Performance Summary")
    perf_df = compute_user_performance(signals_df, users)
    if not perf_df.empty:
        st.dataframe(perf_df, use_container_width=True)
        st.bar_chart(perf_df.set_index("User")[["PnL", "Win Rate"]])
    else:
        st.info("No user performance data found.")

    # ---- Volatility Index ----
    st.subheader("🌪️ Volatility Index (per symbol)")
    vol_df = compute_volatility_index(signals_df)
    st.bar_chart(vol_df.set_index("pair")) if not vol_df.empty else st.info("No volatility data yet.")

    # ---- Pattern Discovery ----
    st.subheader("🧠 Pattern Discovery Tracker")
    pattern_df = compute_pattern_discovery(signals_df)
    st.dataframe(pattern_df, use_container_width=True) if not pattern_df.empty else st.info("No patterns detected yet.")

    # ---- User Growth Trend ----
    st.subheader("📊 User Growth Over Time")
    growth_df = compute_user_growth_trend(user_management)
    st.line_chart(growth_df.set_index("Date")) if not growth_df.empty else st.info("No user growth data available yet.")

    # ---- Signal Volume by Session ----
    st.subheader("🌍 Signal Volume by Market Session")
    session_df = compute_signal_volume_by_session(signals_df)
    if not session_df.empty:
        pivot = session_df.pivot(index="date", columns="session", values="count").fillna(0)
        st.area_chart(pivot)
    else:
        st.info("No session-based signal data found.")

    # ---- Daily PnL / Equity Curve ----
    st.subheader("💰 Daily PnL / Equity Curve")
    pnl_df = compute_daily_pnl(signals_df)
    st.line_chart(pnl_df.set_index("date")[["daily_pnl", "cumulative_pnl"]]) if not pnl_df.empty else st.info("No PnL data available yet.")

    # ---- Per-User Equity Curves ----
    st.subheader("👥 Per-User Equity Curves")
    equity_df = compute_user_equity_curves(signals_df, users)
    st.line_chart(equity_df) if not equity_df.empty else st.info("No user PnL data available yet.")

    # ---- Combined Equity Curves ----
    st.subheader("📊 Combined Equity Curve (Total + Users)")
    combined_eq = compute_combined_equity_curves(signals_df, users)
    st.line_chart(combined_eq) if not combined_eq.empty else st.info("No equity curve data available yet.")