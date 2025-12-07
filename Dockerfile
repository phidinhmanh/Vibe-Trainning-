FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2 (optional, for clustering)
RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
    && tar xvfz mmseqs-linux-avx2.tar.gz \
    && mv mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs mmseqs-linux-avx2.tar.gz \
    || echo "MMseqs2 installation skipped (optional)"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY Makefile .

# Create data directories
RUN mkdir -p data/raw data/processed experiments resources

# Default command
CMD ["python", "-m", "src.models.infer", "--help"]
