# Use Python 3.12 slim as base
FROM python:3.12-slim

# Include Deno JS runtime for yt-dlp YouTube EJS support
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install openwakeword >=0.6.0 without legacy tflite-runtime (uses onnxruntime on Linux Python 3.12+)
RUN pip install --no-cache-dir --no-deps "openwakeword>=0.6.0"
# Install latest yt-dlp with all features for YouTube Music
RUN pip install --no-cache-dir -U --pre "yt-dlp[default]"
# Pre-download and verify openwakeword ONNX models as root before dropping to appuser
RUN python -c "from openwakeword.utils import download_models; from openwakeword.model import Model; download_models(['hey_jarvis']); Model(wakeword_models=['hey_jarvis'], inference_framework='onnx')"

# Copy application code
COPY *.py .
COPY cookies.txt .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "ai_assistant.py"]