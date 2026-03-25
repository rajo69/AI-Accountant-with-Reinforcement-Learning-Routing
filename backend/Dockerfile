FROM python:3.12-slim

# System dependencies required by WeasyPrint (PDF generation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fontconfig \
    fonts-dejavu-core \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[ai,docs]"

# Copy application code
COPY . .

# Run database migrations then start the server
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
