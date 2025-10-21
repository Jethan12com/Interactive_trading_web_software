# run_dashboard.py
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import yaml
import logging
from streamlit_echarts import st_echarts

from modules.signal_engine import HybridSignalEngine
from modules.user_management import UserManagement
from dashboard.components.analytics import render_analytics_dashboard

# =====================================================
# Logging Setup
# =====================================================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'dashboard.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TF warnings

# =====================================================
# Load Configuration
# =====================================================
def load_config():
    try:
        with open('config/settings.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Could not load config: {e}")
        return {}

# =====================================================
# Main Dashboard
# =====================================================
def main():
    st.title("📊 CoPilot Admin Dashboard")
    
    # Load configuration
    config = load_config()
    
    # Initialize modules
    signal_engine = HybridSignalEngine()
    user_mgmt = UserManagement()

    # ----------------------
    # Admin Login
    # ----------------------
    st.sidebar.title("🔐 Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if user_mgmt.authenticate_admin(username, password):
            st.sidebar.success("✅ Logged in successfully!")
            
            # Render analytics dashboard
            render_analytics_dashboard(user_mgmt)
            
            # ----------------------
            # Signal Engine Metrics
            # ----------------------
            st.header("📈 Signal Engine Metrics")
            try:
                signals_df = pd.read_csv('data/logs/signal_log.csv')
                outcomes_df = pd.read_csv('data/reports/outcomes.csv')
                logging.info(f"Loaded {len(signals_df)} signals and {len(outcomes_df)} outcomes")
                
                signal_engine.train_ml_model(signals_df, outcomes_df)
                metrics = signal_engine.ml_model.evaluate()
                
                st.metric("F1-Score", f"{metrics['f1_score']:.2f}")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Expectancy", f"{metrics['expectancy']:.2f}")
                logging.info(f"Computed metrics: {metrics}")
            except Exception as e:
                st.error(f"Failed to load signals or compute metrics: {e}")
                logging.error(f"Metrics computation error: {e}")

            # ----------------------
            # Session Analysis Chart
            # ----------------------
            st.subheader("📊 Session Analysis")
            try:
                session_counts = signals_df['session'].value_counts().to_dict() if not signals_df.empty else {}
                if session_counts:
                    options = {
                        "xAxis": {"type": "category", "data": list(session_counts.keys())},
                        "yAxis": {"type": "value"},
                        "series": [{"data": list(session_counts.values()), "type": "bar"}]
                    }
                    st_echarts(options=options)
                    logging.info("Rendered session analysis chart")
                else:
                    st.info("No session data available to plot.")
            except Exception as e:
                st.error(f"Failed to render chart: {e}")
                logging.error(f"Chart rendering error: {e}")

        else:
            st.sidebar.error("❌ Invalid username or password.")
    else:
        st.info("Please login with your admin credentials to access the dashboard.")


if __name__ == "__main__":
    main()