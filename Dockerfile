# =========================================
# 🚀 Dockerfile for CoPilot (Pipeline + Dashboard)
# =========================================
FROM python:3.12-slim AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# -----------------------------------------
# 🧩 System dependencies
# -----------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------
# 📦 Install Python dependencies
# -----------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------------------
# 📁 Copy project source
# -----------------------------------------
COPY . /app

# -----------------------------------------
# 🧱 Ensure important directories exist
# -----------------------------------------
RUN mkdir -p /app/logs /app/data /app/models /app/grafana /app/prometheus

# -----------------------------------------
# 🌐 Environment defaults (override in compose/heroku)
# -----------------------------------------
ENV VAULT_ADDR=http://127.0.0.1:8200
ENV VAULT_TOKEN=dev-token
ENV DB_HOST=localhost
ENV DB_PORT=5432
ENV DB_NAME=copilot
ENV DB_USER=postgres
ENV DB_PASSWORD=password
ENV PYTHONUNBUFFERED=1

# -----------------------------------------
# 🌐 Exposed ports
# 8000 : API / Prometheus metrics
# 8501 : Streamlit dashboard
# -----------------------------------------
EXPOSE 8000 8501

# -----------------------------------------
# 🏁 Default command
# Runs pipeline + dashboard together
# -----------------------------------------
CMD ["sh", "-c", "python run_live.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]