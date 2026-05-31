# Use official Python 3.12 base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Default command: run the ML pipeline
CMD ["python", "src/pipeline.py"]
