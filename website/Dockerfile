FROM python:3.11-slim

# Install system dependencies + Rust toolchain (needed for subtr_actor)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Install CPU-only PyTorch (much smaller than full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR $HOME/app

# Copy application code
COPY --chown=user:user . .

# Create temp directories
RUN mkdir -p tmp_replays tmp_csvs

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
