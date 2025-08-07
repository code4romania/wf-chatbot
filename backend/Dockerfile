# Start from the official slim-variant Python 3.11
FROM python:3.11-slim

# Safety & caching flags
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps if you need any (e.g. for torch)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install only what you need
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bring in your app code
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
