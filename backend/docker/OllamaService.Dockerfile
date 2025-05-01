# Use Python 3.9 slim as the base image for a smaller footprint
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_PORT=8000 \
    OLLAMA_URL="http://ollama-service:11434/api/generate"

# Install dependencies with specific versions to avoid compatibility issues
RUN pip install --no-cache-dir \
    flask==2.0.1 \
    werkzeug==2.0.1 \
    requests==2.28.1 \
    pyyaml==6.0

# Copy the application code
COPY ./api.py .

# Create a directory for configuration files
RUN mkdir -p /app/config

# Expose the port the app runs on
EXPOSE ${SERVICE_PORT}

# Command to run the application
CMD ["python", "api.py"]