# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc-dev \
    libz-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml file
COPY pyproject.toml .

# Create a constraints file
RUN echo "numcodecs==0.10.2" > constraints.txt

# Install build dependencies and the project
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build && \
    pip install --no-cache-dir -c constraints.txt . && \
    pip install --no-cache-dir -e .

# Copy the entire project directory
COPY . /app

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the application using python main.py
CMD ["python", "api/main.py"]