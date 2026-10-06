# syntax=docker/dockerfile:1
# The Wheat Advisor app: Streamlit pages, the YOLO detector and the LangGraph workflow.
# Built by compose.yaml (CPU PyTorch) or with compose.gpu.yaml added (CUDA PyTorch for an NVIDIA GPU).
FROM python:3.12-slim

# CPU wheels by default; compose.gpu.yaml swaps in a CUDA index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=60 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    OLLAMA_HOST=http://ollama:11434

# OpenCV (pulled in by Ultralytics) needs these system libraries. Nothing else is installed, to keep the attack surface small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before code, so a code change does not reinstall them. The pip cache lives in a build-cache
# mount: a failed or repeated build reuses what was already downloaded, and none of it ends up in the image.
COPY requirements-app.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
 && pip install -r requirements-app.txt

COPY frontend/ frontend/

# Run as a normal user. Weights are mounted at /app/weights; the advisor cache must be writable.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/weights /app/frontend/advisor/cache /tmp/ultralytics /tmp/matplotlib \
 && chown -R app:app /app /tmp/ultralytics /tmp/matplotlib
USER app
WORKDIR /app/frontend

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]

# Listen on every interface inside the container; compose.yaml limits the host side to 127.0.0.1.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
