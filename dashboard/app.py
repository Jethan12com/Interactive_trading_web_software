import streamlit as st
import pandas as pd
import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
import uuid
import plotly.graph_objs as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from modules.config_manager import ConfigManager
from modules.signal_logger import SignalLogger
from modules.data_provider import MultiProviderDataProvider
from modules.security.encryption_helper import EncryptionHelper
from modules.notifier import TelegramNotifier
from modules.user_management import UserManagement
from modules.rate_limiter import RateLimiter

# Logger Setup
def setup_logger(name: str, log_file: str, to_console: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z")
    formatter.converter = lambda *args: datetime.now(pytz.timezone('Africa/Lagos')).timetuple()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger("AdminDashboard", "logs/app.log", to_console=True)

# Plotting Utility for Price and Signals
def plot_price_with_signals(df, signals_df=None, title=None, min_confidence=0.2):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    if signals_df is not None and not signals_df.empty:
        for _, sig in signals_df.iterrows():
            if sig.get('model_confidence', 0) >= min_confidence:
                fig.add_trace(go.Scatter(
                    x=[sig['timestamp']],
                    y=[sig['price']],
                    mode='markers',
                    marker=dict(color='green' if sig['action'] == 'buy' else 'red', size=10),
                    name=f"{sig['action'].capitalize()} Signal"
                ))
    if title:
        fig.update_layout(title=title)
    return fig

# Admin Dashboard
def main():
    st.set_page_config(page_title="Admin Trading Dashboard", layout="wide")
    
    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60_000, key="dashboard_refresh")
    
    # Admin Authentication
    config_manager = ConfigManager()
    admin_id = config_manager.get_config("telegram").get("admin_telegram_id")
    if "admin_authenticated" not in st.session_state:
        password = st.text_input("Enter Admin Password", type="password")
        if password != "admin_password":  # Replace with Vault-based auth
            st.error("Invalid password")
            st.stop()
        st.session_state["admin_authenticated"] = True
        st.session_state["admin_id"] = admin_id

    st.title("Admin Trading Dashboard")
    user_manager = UserManagement()
    signal_logger = SignalLogger()
    data_provider = MultiProviderDataProvider(config_manager)
    rate_limiter = RateLimiter()
    pairs = config_manager.get_pairs()
    notifier = TelegramNotifier()
    try:
        notifier.initialize_from_vault()
    except Exception as e:
        st.error(f"Failed to initialize Telegram notifier: {e}")
        st.stop()

    # Sidebar
    st.sidebar.header("Dashboard Controls")
    selected_pairs = st.sidebar.multiselect("Select Assets", list(pairs.keys()), default=list(pairs.keys())[:3])
    min_conf = st.sidebar.slider("Min Signal Confidence (%)", 0, 100, 20) / 100.0
    st.sidebar.header("Analytics Filters")
    date_range = st.sidebar.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"], index=0)
    selected_tiers = st.sidebar.multiselect("Select Tiers", ["Free", "Premium", "Pro"], default=["Free", "Premium", "Pro"])
    selected_user = st.sidebar.selectbox("Select User (Optional)", ["All Users"] + list(user_manager.get_users().keys()), index=0)
    selected_strategy = st.sidebar.multiselect("Select Strategy", ["PPO", "DQN", "MA Crossover", "RSI Mean Reversion"], default=["PPO", "DQN", "MA Crossover", "RSI Mean Reversion"])
    selected_ab_group = st.sidebar.multiselect("Select A/B Group", ["A", "B"], default=["A", "B"])
    st.sidebar.header("Export Options")
    export_format = st.sidebar.selectbox("Export Format", ["CSV", "PNG"], index=0)
    if st.sidebar.button("Export Analytics"):
        if export_format == "CSV":
            st.session_state["export_data"] = True
        elif export_format == "PNG":
            st.session_state["export_png"] = True

    # Date Range Mapping
    date_ranges = {
        "Last 7 Days": timedelta(days=7),
        "Last 30 Days": timedelta(days=30),
        "Last 90 Days": timedelta(days=90),
        "All Time": None
    }
    start_date = None if date_range == "All Time" else (datetime.now(pytz.timezone('Africa/Lagos')) - date_ranges[date_range]).isoformat()

    # User Management Section
    st.subheader("Manage Users")
    user_action = st.sidebar.selectbox("User Action", ["Add User", "Edit User", "Delete User", "Rotate Keys"])
    if user_action == "Add User":
        with st.form("add_user_form"):
            telegram_id = st.text_input("Telegram ID")
            username = st.text_input("Username")
            tier = st.selectbox("Tier", ["Free", "Premium", "Pro"])
            account_type = st.selectbox("Account Type", ["Standard", "Demo", "Live"])
            risk_profile = st.selectbox("Risk Profile", ["Low", "Medium", "High"])
            time_zone = st.selectbox("Timezone", ["Africa/Lagos", "America/New_York", "Europe/London", "Asia/Tokyo"])
            active_sessions = st.multiselect("Active Sessions", ["London", "NewYork", "Tokyo"], default=["London"])
            capital = st.number_input("Capital", min_value=0.0, value=1000.0)
            preferred_pairs = st.multiselect("Preferred Pairs", list(pairs.keys()), default=list(pairs.keys())[:2])
            expiry_days = st.number_input("Account Expiry (Days)", min_value=1, value=30)
            submit = st.form_submit_button("Add User")
            if submit:
                token = str(uuid.uuid4())
                expiry = datetime.now(pytz.timezone("Africa/Lagos")) + timedelta(hours=24)
                expiry_date = datetime.now(pytz.timezone("Africa/Lagos")) + timedelta(days=expiry_days)
                user_data = {
                    "telegram_id": telegram_id,
                    "username": username,
                    "tier": tier,
                    "account_type": account_type,
                    "risk_profile": risk_profile,
                    "time_zone": time_zone,
                    "active_sessions": active_sessions,
                    "capital": capital,
                    "preferred_pairs": preferred_pairs,
                    "status": "Pending",
                    "activation_token": token,
                    "activation_token_expiry": expiry.isoformat(),
                    "expiry_date": expiry_date.isoformat(),
                    "role": "user",
                    "is_active": False
                }
                try:
                    user_manager.add_user(user_data)
                    activation_link = f"https://t.me/{config_manager.get_config('telegram').get('bot_name', 'YourBotName')}?start={token}"
                    st.success(f"User added! Activation link: {activation_link} (Expires: {expiry})")
                    notifier.send_message(f"✅ New user added: {telegram_id} (@{username}, Tier: {tier})\nActivation link: {activation_link}")
                except Exception as e:
                    st.error(f"Failed to add user: {e}")
                    notifier.send_message(f"🚨 Failed to add user {telegram_id}: {e}")

    elif user_action == "Edit User":
        users = user_manager.get_users()
        telegram_id = st.selectbox("Select User", list(users.keys()))
        if telegram_id:
            user = users[telegram_id]
            with st.form("edit_user_form"):
                username = st.text_input("Username", value=user["username"])
                tier = st.selectbox("Tier", ["Free", "Premium", "Pro"], index=["Free", "Premium", "Pro"].index(user["tier"]))
                account_type = st.selectbox("Account Type", ["Standard", "Demo", "Live"], index=["Standard", "Demo", "Live"].index(user["account_type"]))
                risk_profile = st.selectbox("Risk Profile", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(user["risk_profile"]))
                time_zone = st.selectbox("Timezone", ["Africa/Lagos", "America/New_York", "Europe/London", "Asia/Tokyo"], index=["Africa/Lagos", "America/New_York", "Europe/London", "Asia/Tokyo"].index(user["time_zone"]))
                active_sessions = st.multiselect("Active Sessions", ["London", "NewYork", "Tokyo"], default=user["active_sessions"])
                capital = st.number_input("Capital", min_value=0.0, value=user["capital"])
                preferred_pairs = st.multiselect("Preferred Pairs", list(pairs.keys()), default=user["preferred_pairs"])
                expiry_date = st.date_input("Expiry Date", value=datetime.fromisoformat(user["expiry_date"]) if user["expiry_date"] else datetime.now(pytz.timezone("Africa/Lagos")) + timedelta(days=30))
                regenerate_token = st.checkbox("Regenerate Activation Token")
                submit = st.form_submit_button("Update User")
                if submit:
                    updates = {
                        "username": username,
                        "tier": tier,
                        "account_type": account_type,
                        "risk_profile": risk_profile,
                        "time_zone": time_zone,
                        "active_sessions": active_sessions,
                        "capital": capital,
                        "preferred_pairs": preferred_pairs,
                        "expiry_date": expiry_date.isoformat()
                    }
                    if regenerate_token:
                        token = str(uuid.uuid4())
                        expiry = datetime.now(pytz.timezone("Africa/Lagos")) + timedelta(hours=24)
                        updates["activation_token"] = token
                        updates["activation_token_expiry"] = expiry.isoformat()
                        updates["is_active"] = False
                        updates["status"] = "Pending"
                        activation_link = f"https://t.me/{config_manager.get_config('telegram').get('bot_name', 'YourBotName')}?start={token}"
                        st.success(f"New activation link: {activation_link}")
                        notifier.send_message(f"🔄 User {telegram_id} updated with new activation link: {activation_link}")
                    try:
                        user_manager.update_user(telegram_id, updates)
                        st.success(f"User {telegram_id} updated!")
                        notifier.send_message(f"✅ User {telegram_id} (@{username}, Tier: {tier}) updated")
                    except Exception as e:
                        st.error(f"Failed to update user: {e}")
                        notifier.send_message(f"🚨 Failed to update user {telegram_id}: {e}")

    elif user_action == "Delete User":
        users = user_manager.get_users()
        telegram_id = st.selectbox("Select User", list(users.keys()))
        if st.button("Delete User"):
            try:
                user_manager.delete_user(telegram_id)
                st.success(f"User {telegram_id} deleted!")
                notifier.send_message(f"🗑️ User {telegram_id} deleted")
            except Exception as e:
                st.error(f"Failed to delete user: {e}")
                notifier.send_message(f"🚨 Failed to delete user {telegram_id}: {e}")

    elif user_action == "Rotate Keys":
        st.subheader("Key Rotation")
        new_key_id = st.text_input("New Key ID", value=f"key_{datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y%m%d')}")
        old_key_id = st.selectbox("Old Key ID to Re-encrypt", list(user_manager.encryption_helper.active_keys.keys()))
        if st.button("Rotate and Re-encrypt"):
            try:
                user_manager.encryption_helper.rotate_key(new_key_id)
                user_manager.reencrypt_users(old_key_id, new_key_id)
                signals_df = signal_logger.get_signal_history() or pd.read_csv("logs/signals.csv") if os.path.exists("logs/signals.csv") else pd.DataFrame()
                if not signals_df.empty:
                    reencrypted_signals = user_manager.encryption_helper.reencrypt_data(old_key_id, new_key_id, signals_df.to_dict("records"))
                    pd.DataFrame(reencrypted_signals).to_csv("logs/signals.csv", index=False)
                st.success(f"Rotated to key {new_key_id} and re-encrypted data")
                notifier.send_message(f"🔐 Key rotation completed to {new_key_id}")
            except Exception as e:
                st.error(f"Key rotation failed: {e}")
                notifier.send_message(f"🚨 Key rotation failed: {e}")

    # Signal History
    st.subheader("Signal History")
    signals_df = signal_logger.get_signal_history(
        user_id=selected_user if selected_user != "All Users" else None,
        start_date=start_date
    )
    if not signals_df.empty:
        signals_df["user_id"] = signals_df.apply(
            lambda row: user_manager.encryption_helper.decrypt_text(row["user_id"], row.get("user_key_id", list(user_manager.encryption_helper.active_keys.keys())[0])) if row["user_id"] else "", axis=1)
        if selected_tiers:
            signals_df = signals_df[signals_df["user_tier"].isin(selected_tiers)] if "user_tier" in signals_df.columns else signals_df
        if selected_strategy:
            signals_df = signals_df[signals_df["strategy"].isin(selected_strategy)]
        if selected_ab_group:
            signals_df = signals_df[signals_df["ab_group"].isin(selected_ab_group)]
    if signals_df.empty:
        st.warning("No signals found for the selected filters")
    else:
        st.dataframe(signals_df[["signal_id", "pair", "strategy", "action", "price", "timestamp", "user_id", "delivery_status", "candlestick_signal", "anomaly_score", "divergence_signal", "ab_group"]])
        if "export_data" in st.session_state and st.session_state["export_data"]:
            csv = signals_df.to_csv(index=False)
            st.download_button("Download Signal History CSV", csv, "signal_history.csv", "text/csv")
            st.session_state["export_data"] = False

    # Signal Plots
    for pair in selected_pairs:
        pair_signals = signals_df[signals_df["pair"] == pair]
        if pair_signals.empty:
            st.info(f"No signals for {pair}")
            continue
        data = asyncio.run(data_provider.fetch_historical([pair], start_date=(datetime.now(pytz.timezone('Africa/Lagos')) - timedelta(days=1)).strftime("%Y-%m-%d"), end_date=datetime.now(pytz.timezone('Africa/Lagos')).strftime("%Y-%m-%d"), interval="5m"))
        if data.empty:
            st.info(f"No data for {pair}")
            continue
        fig = plot_price_with_signals(data[data["pair"] == pair], pair_signals, title=f"{pair} Price & Signals", min_confidence=min_conf)
        st.plotly_chart(fig, use_container_width=True)

    # User Activity
    st.subheader("User Activity")
    users = user_manager.get_users()
    user_data = [{
        "telegram_id": user["telegram_id"],
        "username": user["username"],
        "tier": user["tier"],
        "max_signals": user["max_signals"],
        "session_limit": user["session_limit"],
        "capital": user["capital"],
        "active_sessions": ", ".join(user["active_sessions"]),
        "status": user["status"],
        "signal_count": len(signals_df[signals_df["user_id"] == user["telegram_id"]])
    } for user in users.values()]
    if selected_tiers:
        user_data = [u for u in user_data if u["tier"] in selected_tiers]
    if selected_user != "All Users":
        user_data = [u for u in user_data if u["telegram_id"] == selected_user]
    user_df = pd.DataFrame(user_data)
    if user_df.empty:
        st.warning("No users found for the selected filters")
    else:
        st.dataframe(user_df)
        if "export_data" in st.session_state and st.session_state["export_data"]:
            csv = user_df.to_csv(index=False)
            st.download_button("Download User Activity CSV", csv, "user_activity.csv", "text/csv")
            st.session_state["export_data"] = False

    # User Analytics Section
    st.subheader("User Analytics")

    # Tier Distribution Pie Chart
    if not user_df.empty:
        st.write("### Tier Distribution")
        tier_counts = user_df["tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_tiers = {
            "type": "pie",
            "data": {
                "labels": tier_counts["Tier"].tolist(),
                "datasets": [{
                    "data": tier_counts["Count"].tolist(),
                    "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c"],
                    "borderColor": "#ffffff",
                    "borderWidth": 2
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "top"},
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { return context.label + ': ' + context.parsed + ' users (' + (context.parsed / context.dataset.data.reduce((a, b) => a + b, 0) * 100).toFixed(1) + '%)'; }"
                        }
                    }
                }
            }
        }
        st.write("Chart showing distribution of users across subscription tiers.")
        st.markdown("```chartjs\n" + json.dumps(fig_tiers, indent=2) + "\n```")
        if "export_png" in st.session_state and st.session_state["export_png"]:
            st.write("Export PNG functionality requires client-side rendering (not supported in Streamlit). Use screenshot instead.")
            st.session_state["export_png"] = False

    # Signal Performance by Strategy (Bar Chart)
    if not signals_df.empty:
        st.write("### Signal Performance by Strategy")
        performance_metrics = signal_logger.analyze_signal_performance(
            user_id=selected_user if selected_user != "All Users" else None,
            start_date=start_date,
            strategy=selected_strategy[0] if len(selected_strategy) == 1 else None,
            ab_group=selected_ab_group[0] if len(selected_ab_group) == 1 else None
        )
        if performance_metrics and 'strategy_breakdown' in performance_metrics:
            strategy_data = pd.DataFrame(performance_metrics['strategy_breakdown']).transpose().reset_index()
            strategy_data.columns = ['strategy', 'count', 'average_pnl']
            strategy_data['win_rate'] = [
                signal_logger.analyze_signal_performance(strategy=strategy, user_id=selected_user if selected_user != "All Users" else None, start_date=start_date).get('win_rate', 0)
                for strategy in strategy_data['strategy']
            ]
            if not strategy_data.empty:
                fig_strategy = {
                    "type": "bar",
                    "data": {
                        "labels": strategy_data["strategy"].tolist(),
                        "datasets": [
                            {
                                "label": "Win Rate (%)",
                                "data": (strategy_data["win_rate"] * 100).tolist(),
                                "backgroundColor": "#1f77b4",
                                "borderColor": "#1f77b4",
                                "borderWidth": 1
                            },
                            {
                                "label": "Average PnL ($)",
                                "data": strategy_data["average_pnl"].tolist(),
                                "backgroundColor": "#ff7f0e",
                                "borderColor": "#ff7f0e",
                                "borderWidth": 1
                            }
                        ]
                    },
                    "options": {
                        "responsive": True,
                        "scales": {
                            "x": {"title": {"display": True, "text": "Strategy"}},
                            "y": {"title": {"display": True, "text": "Value"}, "beginAtZero": True}
                        },
                        "plugins": {
                            "legend": {"position": "top"},
                            "tooltip": {
                                "callbacks": {
                                    "label": "function(context) { return context.dataset.label + ': ' + (context.dataset.label.includes('Win Rate') ? context.parsed.y.toFixed(1) + '%' : '$' + context.parsed.y.toFixed(2)); }"
                                }
                            }
                        }
                    }
                }
                st.write("Bar chart showing win rate and average PnL by strategy.")
                st.markdown("```chartjs\n" + json.dumps(fig_strategy, indent=2) + "\n```")
                if "export_data" in st.session_state and st.session_state["export_data"]:
                    csv = strategy_data.to_csv(index=False)
                    st.download_button("Download Strategy Performance CSV", csv, "strategy_performance.csv", "text/csv")
                    st.session_state["export_data"] = False
            else:
                st.warning("No strategy performance data for the selected filters")

    # TP/SL Hit Rate Pie Chart
    if not signals_df.empty and performance_metrics:
        st.write("### TP/SL Hit Rate")
        tp_sl_data = {
            "Hit TP/SL": performance_metrics.get('tp_sl_hit_rate', 0) * 100,
            "Missed TP/SL": (1 - performance_metrics.get('tp_sl_hit_rate', 0)) * 100
        }
        fig_tp_sl = {
            "type": "pie",
            "data": {
                "labels": list(tp_sl_data.keys()),
                "datasets": [{
                    "data": list(tp_sl_data.values()),
                    "backgroundColor": ["#2ca02c", "#d62728"],
                    "borderColor": "#ffffff",
                    "borderWidth": 2
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "top"},
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { return context.label + ': ' + context.parsed.toFixed(1) + '%'; }"
                        }
                    }
                }
            }
        }
        st.write("Pie chart showing the proportion of signals hitting TP/SL targets.")
        st.markdown("```chartjs\n" + json.dumps(fig_tp_sl, indent=2) + "\n```")
        if "export_data" in st.session_state and st.session_state["export_data"]:
            csv = pd.DataFrame(tp_sl_data.items(), columns=["Status", "Percentage"]).to_csv(index=False)
            st.download_button("Download TP/SL Hit Rate CSV", csv, "tp_sl_hit_rate.csv", "text/csv")
            st.session_state["export_data"] = False

    # Pattern Impact Bar Chart
    if not signals_df.empty and performance_metrics and 'pattern_metrics' in performance_metrics:
        st.write("### Pattern Impact")
        pattern_data = pd.DataFrame(performance_metrics['pattern_metrics']).transpose().reset_index()
        pattern_data.columns = ['pattern', 'win_rate', 'average_pnl', 'count']
        if not pattern_data.empty:
            fig_pattern = {
                "type": "bar",
                "data": {
                    "labels": pattern_data["pattern"].tolist(),
                    "datasets": [
                        {
                            "label": "Win Rate (%)",
                            "data": (pattern_data["win_rate"] * 100).tolist(),
                            "backgroundColor": "#1f77b4",
                            "borderColor": "#1f77b4",
                            "borderWidth": 1
                        },
                        {
                            "label": "Average PnL ($)",
                            "data": pattern_data["average_pnl"].tolist(),
                            "backgroundColor": "#ff7f0e",
                            "borderColor": "#ff7f0e",
                            "borderWidth": 1
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "scales": {
                        "x": {"title": {"display": True, "text": "Pattern Type"}},
                        "y": {"title": {"display": True, "text": "Value"}, "beginAtZero": True}
                    },
                    "plugins": {
                        "legend": {"position": "top"},
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return context.dataset.label + ': ' + (context.dataset.label.includes('Win Rate') ? context.parsed.y.toFixed(1) + '%' : '$' + context.parsed.y.toFixed(2)); }"
                            }
                        }
                    }
                }
            }
            st.write("Bar chart showing win rate and average PnL for high-confidence patterns (candlestick, anomaly, divergence).")
            st.markdown("```chartjs\n" + json.dumps(fig_pattern, indent=2) + "\n```")
            if "export_data" in st.session_state and st.session_state["export_data"]:
                csv = pattern_data.to_csv(index=False)
                st.download_button("Download Pattern Impact CSV", csv, "pattern_impact.csv", "text/csv")
                st.session_state["export_data"] = False
        else:
            st.warning("No pattern performance data for the selected filters")

    # Signal Activity Heatmap
    if not signals_df.empty:
        st.write("### Signal Activity Heatmap")
        signals_df["hour"] = pd.to_datetime(signals_df["timestamp"]).dt.hour
        heatmap_data = signals_df.groupby(["hour", "user_id"]).size().reset_index(name="count")
        heatmap_data = heatmap_data.merge(user_df[["telegram_id", "username", "active_sessions"]], left_on="user_id", right_on="telegram_id")
        if selected_user != "All Users":
            heatmap_data = heatmap_data[heatmap_data["user_id"] == selected_user]
        if not heatmap_data.empty:
            session_mapping = {
                "London": list(range(8, 17)),
                "NewYork": list(range(13, 22)),
                "Tokyo": list(range(0, 9))
            }
            heatmap_data["session"] = heatmap_data["hour"].apply(
                lambda h: next((s for s, hours in session_mapping.items() if h in hours), "Other")
            )
            heatmap_pivot = heatmap_data.pivot_table(index="hour", columns="session", values="count", aggfunc="sum", fill_value=0)
            fig_heatmap = {
                "type": "heatmap",
                "data": {
                    "x": heatmap_pivot.columns.tolist(),
                    "y": heatmap_pivot.index.tolist(),
                    "z": heatmap_pivot.values.tolist(),
                    "colorscale": [
                        [0, "#f7fbff"],
                        [0.5, "#6baed6"],
                        [1, "#08306b"]
                    ]
                },
                "options": {
                    "responsive": True,
                    "scales": {
                        "x": {"title": {"display": True, "text": "Session"}},
                        "y": {"title": {"display": True, "text": "Hour of Day"}}
                    },
                    "plugins": {
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return 'Hour ' + context.parsed.y + ', ' + context.parsed.x + ': ' + context.parsed.z + ' signals'; }"
                            }
                        }
                    }
                }
            }
            st.write("Heatmap showing signal activity by hour and trading session.")
            st.markdown("```chartjs\n" + json.dumps(fig_heatmap, indent=2) + "\n```")
            if "export_data" in st.session_state and st.session_state["export_data"]:
                csv = heatmap_pivot.to_csv()
                st.download_button("Download Heatmap Data CSV", csv, "signal_heatmap.csv", "text/csv")
                st.session_state["export_data"] = False
        else:
            st.warning("No signal activity data for the selected filters")

    # Rate Limit Usage Bar Chart
    if not user_df.empty:
        st.write("### Rate Limit Usage")
        rate_limit_data = []
        for user in user_df.to_dict("records"):
            telegram_id = user["telegram_id"]
            signal_count = user["signal_count"]
            max_signals = user["max_signals"]
            rate_limit_data.append({
                "username": user["username"],
                "tier": user["tier"],
                "signal_count": signal_count,
                "max_signals": max_signals,
                "usage_percent": (signal_count / max_signals * 100) if max_signals > 0 else 0
            })
        rate_limit_df = pd.DataFrame(rate_limit_data)
        if selected_user != "All Users":
            rate_limit_df = rate_limit_df[rate_limit_df["username"] == users[selected_user]["username"]]
        if not rate_limit_df.empty:
            fig_rate_limit = {
                "type": "bar",
                "data": {
                    "labels": rate_limit_df["username"].tolist(),
                    "datasets": [
                        {
                            "label": "Signals Used",
                            "data": rate_limit_df["signal_count"].tolist(),
                            "backgroundColor": "#1f77b4",
                            "borderColor": "#1f77b4",
                            "borderWidth": 1
                        },
                        {
                            "label": "Max Signals",
                            "data": rate_limit_df["max_signals"].tolist(),
                            "backgroundColor": "#ff7f0e",
                            "borderColor": "#ff7f0e",
                            "borderWidth": 1
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "scales": {
                        "x": {"title": {"display": True, "text": "Username"}},
                        "y": {"title": {"display": True, "text": "Signal Count"}, "beginAtZero": True}
                    },
                    "plugins": {
                        "legend": {"position": "top"},
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return context.dataset.label + ': ' + context.parsed.y + ' (' + (context.parsed.y / context.dataset.data[context.dataIndex] * 100).toFixed(1) + '% of max)'; }"
                            }
                        }
                    }
                }
            }
            st.write("Bar chart comparing signals used vs. maximum allowed signals per user, by tier.")
            st.markdown("```chartjs\n" + json.dumps(fig_rate_limit, indent=2) + "\n```")
            if "export_data" in st.session_state and st.session_state["export_data"]:
                csv = rate_limit_df.to_csv(index=False)
                st.download_button("Download Rate Limit Data CSV", csv, "rate_limit.csv", "text/csv")
                st.session_state["export_data"] = False
        else:
            st.warning("No rate limit data for the selected filters")

if __name__ == "__main__":
    main()