# FROM python:3.10-slim

# WORKDIR /app

# COPY . .

# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# EXPOSE 8000

# CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]