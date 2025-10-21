# admin_access_logs.py
import os
import socket
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

class AdminLogViewer:
    def __init__(self, log_path, config_manager=None, user_manager=None, notifier=None, email_notifier=None):
        self.log_path = log_path
        self.config_manager = config_manager
        self.user_manager = user_manager
        self.notifier = notifier
        self.email_notifier = email_notifier

        # Initialize daily digest
        if "last_digest_date" not in st.session_state:
            st.session_state["last_digest_date"] = None
        self.send_daily_digest_if_needed()

    def send_daily_digest_if_needed(self):
        today = datetime.today().date()
        if st.session_state["last_digest_date"] != today:
            self.send_daily_failed_login_digest()
            st.session_state["last_digest_date"] = today

    def send_daily_failed_login_digest(self):
        if not os.path.exists(self.log_path):
            return
        try:
            logs_df = pd.read_csv(
                self.log_path, sep=" - ", header=None,
                names=["timestamp","level","message"], engine="python"
            )
            logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
            since = datetime.now() - timedelta(days=1)
            failed_keywords = ["invalid credentials","non-admin role tried"]
            failed_logs = logs_df[
                (logs_df["timestamp"] >= since) &
                (logs_df["message"].str.lower().str.contains("|".join(failed_keywords)))
            ]
            if failed_logs.empty:
                return
            summary = "\n".join(
                f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {row['message']}"
                for _, row in failed_logs.iterrows()
            )
            msg = f"📢 *Daily Failed Login Digest*\n\nTotal Failed Logins: {len(failed_logs)}\n\n{summary}"

            sent = False
            if self.notifier:
                try:
                    self.notifier.send_message(msg)
                    sent = True
                except: pass

            if not sent and self.email_notifier:
                try:
                    plain_msg = msg.replace("*","").replace("`","")
                    self.email_notifier.send_email(
                        to=["security@example.com"],
                        subject="Daily Failed Login Digest",
                        body=plain_msg
                    )
                except: pass

        except Exception as e:
            st.error(f"Failed to generate daily digest: {e}")

    def get_user_role(self):
        return st.session_state.get("user_role","guest")

    def render_access_logs(self):
        st.subheader("📜 Admin Access Logs (Live Feed + Filters)")

        # Auto-refresh every 30 seconds
        st_autorefresh(interval=30*1000, key="live_admin_feed")

        if not os.path.exists(self.log_path):
            st.info("No admin access logs yet.")
            return

        try:
            logs_df = pd.read_csv(
                self.log_path, sep=" - ", header=None,
                names=["timestamp","level","message"], engine="python"
            )
            logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
        except Exception as e:
            st.error(f"Failed to load logs: {e}")
            return

        role = self.get_user_role()

        # -------------------------------
        # LIVE ADMIN ACTIVITY FEED
        # -------------------------------
        st.markdown("### ⚡ Live Admin Activity Feed (Last 50 Events)")
        feed_df = logs_df.sort_values("timestamp", ascending=False).head(50)

        def highlight_activity(row):
            msg = row["message"].lower()
            if "invalid credentials" in msg or "non-admin role tried" in msg:
                return ["background-color: #FFCCCC"]*len(row)
            elif "login_success" in msg:
                return ["background-color: #CCFFCC"]*len(row)
            return [""]*len(row)

        st.dataframe(feed_df.style.apply(highlight_activity, axis=1), use_container_width=True)
        st.caption("Green = Successful login, Red = Failed/Unauthorized login")

        # -------------------------------
        # FILTERED LOGS (Search, Date, Level)
        # -------------------------------
        col1,col2,col3 = st.columns([2,2,2])
        with col1: search_user = st.text_input("🔍 Search Username/Email").strip().lower()
        with col2: start_date = st.date_input("Start Date", datetime.now().date().replace(day=1))
        with col3: end_date = st.date_input("End Date", datetime.now().date())
        level_filter = st.multiselect("Log Level Filter", options=logs_df["level"].unique().tolist(), default=logs_df["level"].unique().tolist())
        failed_toggle = st.checkbox("🚨 Failed Login Only", value=False) if role=="superadmin" else False

        filtered = logs_df[logs_df["level"].isin(level_filter)]
        filtered = filtered[
            (logs_df["timestamp"].dt.date >= start_date) &
            (logs_df["timestamp"].dt.date <= end_date)
        ]
        if search_user:
            filtered = filtered[filtered["message"].str.lower().str.contains(search_user)]
        if failed_toggle:
            failed_keywords = ["invalid credentials","non-admin role tried"]
            filtered = filtered[filtered["message"].str.lower().str.contains("|".join(failed_keywords))]

        st.dataframe(filtered.tail(100).style.apply(highlight_activity, axis=1), use_container_width=True)
        st.caption(f"Showing {len(filtered.tail(100))} entries.")

        # Superadmin controls
        if role=="superadmin":
            from io import StringIO
            csv_buffer = StringIO()
            filtered.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Filtered Logs CSV", csv_buffer.getvalue(),
                               file_name=f"admin_access_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")