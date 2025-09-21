# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (for faiss, torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 7860

# Start Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
