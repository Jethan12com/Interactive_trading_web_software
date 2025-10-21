CoPilot: Research + Signal Distribution Lab
Overview
CoPilot is a modular, hybrid signal generation and distribution system designed for trading research and signal delivery. It combines machine learning (ML), reinforcement learning (RL), and rule-based logic to generate trading signals, delivered to users via Telegram based on their subscription tier, account type, session preferences, and risk profile. The admin-only dashboard provides analytics, research tools, portfolio optimization, and backtesting capabilities.
Key Features

Signal Engine: Hybrid ML/RL/rule-based signal generation with multi-timeframe analysis, volatility forecasting, and alternative data (X and news sentiment).
User Management: Admin can add/remove/archive users; users receive customized signals via Telegram.
Dashboard: Streamlit-based, admin-only interface for analytics, backtesting, portfolio optimization, and research.
Monitoring: Prometheus and Grafana for real-time metrics (signal generation, RL performance, API usage).
Scalability: Dockerized deployment with batched data fetches, adaptive RL retraining, and high-frequency trading optimizations.

System Architecture
copilot/
│── modules/
│   ├── signal_engine.py        # Hybrid signal generation with user feedback
│   ├── user_management.py      # User add/remove/archive
│   ├── signal_logger.py        # Signal and delivery logging
│   ├── journal_evaluator.py    # Trade logging, adaptive RL retraining
│   ├── utils.py               # Market sessions, sizing, helpers
│   ├── notifier.py            # Telegram bot delivery
│   ├── pattern_discovery.py   # Multi-timeframe candlestick, anomaly, divergence, sentiment
│   ├── ml_model.py            # ML state preparation
│   ├── reinforcement_trader.py # Hybrid PPO/DQN RL with automated model selection
│   ├── data_provider.py       # Multi-provider data fetch
│   ├── config_manager.py      # Configuration management
│   ├── backtester.py          # Backtesting logic
│── dashboard/
│   ├── app.py                 # Streamlit dashboard with portfolio optimization
│── logs/                      # Signal, trade, and delivery logs
│── data/                      # Historical data storage
│── tests/                     # Unit tests
│── grafana/provisioning/      # Grafana dashboards and alerts
│── Dockerfile                 # Docker configuration
│── docker-compose.yml         # Multi-container setup
│── prometheus.yml             # Prometheus configuration
│── stress_test.py             # Stress testing script
│── deploy.sh                  # Production deployment script
│── requirements.txt           # Dependencies
│── README.md                  # This file

Setup Instructions

Prerequisites:

Docker and Docker Compose installed.
Environment variables: VAULT_TOKEN, TELEGRAM_BOT_TOKEN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SESSION_SECRET, XAI_API_KEY, FINNHUB_API_KEY, GRAFANA_ADMIN_PASSWORD.
Python 3.10+ for local development.


Clone Repository:
git clone <repository-url>
cd copilot


Set Environment Variables:Create a .env file:
VAULT_TOKEN=your-vault-token
TELEGRAM_BOT_TOKEN=your-telegram-token
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
SESSION_SECRET=your-session-secret
XAI_API_KEY=your-xai-api-key
FINNHUB_API_KEY=your-finnhub-api-key
GRAFANA_ADMIN_PASSWORD=your-grafana-password


Install Dependencies (for local dev):
pip install -r requirements.txt


Run Locally:
python run_live.py & streamlit run app.py --server.port 8501


Run with Docker:
docker-compose up --build


Dashboard: http://localhost:8501
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (login: admin/)



Production Deployment

Prepare Environment:

Ensure .env file is populated with all required variables.
Verify AWS ECS or equivalent cloud service is configured for Docker deployment.


Run Deployment Script:
chmod +x deploy.sh
./deploy.sh


Monitoring Setup:

Prometheus scrapes metrics at copilot:8000 (configured in prometheus.yml).
Grafana dashboards (grafana/provisioning/dashboards/copilot_dashboard.json) visualize:
Signal generation rate
RL model performance (win rate, Sharpe ratio)
API call success/failure


Alerts for:
Win rate < 50% (10-minute duration)
API failure rate > 5% (5-minute duration)





User Guide
Admin Guide

Access Dashboard: Login via Google OAuth at http://<host>:8501.
Pages:
AI Management: Train RL models (PPO/DQN) for specific pairs with customizable timesteps.
Research: Run multi-pair backtests with customizable date ranges; view equity curves and metrics.
Analytics: Monitor overall performance (win rate, PnL, Sharpe ratio) and A/B test results.
Portfolio Optimization: Select pairs and optimize allocation using Markowitz mean-variance; view pie charts and risk metrics.
Archive: Review historical signals and trades.


Monitoring: Access Grafana (http://<host>:3000) to view dashboards and configure Telegram alerts.

End-User Guide

Signal Delivery: Receive trading signals via Telegram bot, customized by:
Subscription Tier: Free (demo, delayed), Standard (real-time, limited sessions), Premium (full sessions), VIP (experimental signals).
Account Type: Demo or Live.
Risk Profile: Low, Medium, Aggressive.
Trading Session: London, New York, Asia.


Feedback: Interact with the Telegram bot to indicate whether signals were acted upon, influencing future signal prioritization.
Access: No dashboard access; signals delivered solely via Telegram.

API References
xAI Sentiment API

Endpoint: https://api.x.ai/v1/sentiment
Method: GET
Parameters:
pair: Trading pair (e.g., BTC/USD)
start_date: ISO format (e.g., 2025-10-09T00:00:00Z)
end_date: ISO format


Headers: Authorization: Bearer <XAI_API_KEY>
Response:{
  "sentiments": [
    {
      "timestamp": "2025-10-09T00:00:00Z",
      "sentiment_score": 0.7  // -1 (bearish) to 1 (bullish)
    }
  ]
}


Usage: Integrated in pattern_discovery.py for real-time sentiment in RL state.

Finnhub News Sentiment API

Endpoint: https://finnhub.io/api/v1/news-sentiment
Method: GET
Parameters:
symbol: Base asset (e.g., BTC for BTC/USD)
from: ISO format
to: ISO format


Headers: X-Finnhub-Token: <FINNHUB_API_KEY>
Response:{
  "sentiment": [
    {
      "timestamp": "2025-10-09T00:00:00Z",
      "sentiment_score": 0.5  // -1 (bearish) to 1 (bullish)
    }
  ]
}


Usage: Combined with X sentiment in pattern_discovery.py.

Telegram Bot API

Endpoint: Configured via TELEGRAM_BOT_TOKEN in notifier.py
Usage: Sends signals to users and collects feedback (acted/ignored).

Development Notes

Multi-Timeframe Analysis: pattern_discovery.py analyzes 1H, 4H, D1 signals, weighted 20%/30%/50% for RL state.
Portfolio Optimization: app.py uses Markowitz mean-variance to balance risk across pairs.
RL Model Selection: reinforcement_trader.py switches between PPO/DQN based on win rate and Sharpe ratio.
User Feedback: signal_engine.py adjusts signal weights based on user actions (acted: +10%, ignored: -10%).
Stress Testing: stress_test.py simulates 100 signals/second; optimized cache and batch sizes.
CI/CD: GitHub Actions pipeline (cicd.yml) runs tests and deploys to AWS ECS.

Testing
Run unit tests:
pytest tests/ --cov=modules

Run stress test:
python stress_test.py

Troubleshooting

API Errors: Verify XAI_API_KEY and FINNHUB_API_KEY in .env.
Docker Issues: Ensure all environment variables are set and ports (8501, 9090, 3000) are available.
Dashboard Access: Check Google OAuth credentials and SESSION_SECRET.
Signal Delivery: Confirm TELEGRAM_BOT_TOKEN and bot configuration.

Future Improvements

Add support for additional technical indicators in pattern_discovery.py.
Implement dynamic risk adjustment in run_live.py based on portfolio metrics.
Enhance Grafana alerts with email/SMS notifications.
