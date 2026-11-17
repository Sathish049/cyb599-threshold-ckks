# For macOS / Windows (OpenFHE's Python wheel targets Linux x86_64 only).
#   docker build -t tckks .
#   docker run --rm -v "$PWD/results:/app/results" tckks            # full run
#   docker run --rm -v "$PWD/results:/app/results" tckks --quick    # smoke run
# On Apple Silicon add:  --platform linux/amd64  to both commands.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN rm -rf .venv && chmod +x run.sh
ENTRYPOINT ["./run.sh"]
