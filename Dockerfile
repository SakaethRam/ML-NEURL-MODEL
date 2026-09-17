# NEURL Engine — CPU image
# PyG's optional binary extensions must match the installed torch
# build exactly, hence the explicit two-step install below rather than
# a bare `pip install -r requirements.txt` for the ML stack.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, matched version.
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "run_pipeline.py"]
