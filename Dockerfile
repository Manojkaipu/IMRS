# CPU image for the Streamlit app. Mount your dataset and runs directory:
#   docker build -t imrs .
#   docker run -p 8501:8501 -v $PWD/data:/app/data -v $PWD/runs:/app/runs imrs
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    IMRS_DATA_ROOT=/app/data/caltech256_proc \
    IMRS_RUNS_DIR=/app/runs

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"
COPY app ./app

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
