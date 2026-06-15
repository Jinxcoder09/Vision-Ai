# ── Build Stage: Frontend ──────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy dependencies manifest
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source code
COPY frontend/ ./

# Build Next.js as a production static export
ENV NODE_ENV=production
RUN npm run build

# ── Build Stage: Backend & Runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies needed by PaddleOCR, OpenCV, soundfile, ffmpeg, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install PaddlePaddle (CPU version)
RUN pip install --no-cache-dir paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html || \
    pip install --no-cache-dir paddlepaddle

# Copy and install python backend requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Copy compiled frontend static files from Stage 1 into backend/static
COPY --from=frontend-builder /app/frontend/out ./static

# Create logs directory
RUN mkdir -p logs

# Expose server port (Render routes HTTP traffic to this port dynamically)
EXPOSE 8000

# Health check (verifies backend is healthy and responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the uvicorn backend on the Render assigned $PORT (defaults to 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --loop asyncio"]
