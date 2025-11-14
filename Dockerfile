# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy requirements first (for better caching)
COPY requirements.txt .

# Fresh install of Gradio 6 as recommended by Gradio team
# Install in a single command to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "gradio[mcp,oauth]==6.0.0.dev4" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app.py"]
