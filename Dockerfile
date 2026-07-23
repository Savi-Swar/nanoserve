# nanoserve — reproducible CPU benchmark environment.
#
# Build and pull the graphs out (results/ is bind-mounted so the PNGs land on
# your host):
#
#   docker build -t nanoserve .
#   docker run --rm -v $(pwd)/results:/app/results nanoserve
#
# The default CMD runs the full CPU repro: `make all` (memory + bench + plot).
#
# GPU runs are NOT built here: swap the base image for an NVIDIA CUDA runtime
# (e.g. nvidia/cuda:12.4.1-runtime with Python) and launch with `--gpus all`,
# then `make bench DEVICE=cuda`. CPU is the default so this builds anywhere.

FROM python:3.12-slim

# build-essential is only needed if a wheel has to compile from source; keep it
# so `pip install` never fails on a source-only dependency, then move on.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the (slow) pip layer is cached across code edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then the project itself.
COPY . .

# Full CPU reproduction: regenerate every artifact and graph.
CMD ["make", "all"]
