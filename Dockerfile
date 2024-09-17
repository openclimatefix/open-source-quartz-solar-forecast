# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory (including quartz_solar_forecast)
COPY . /app

# Copy the pyproject.toml file
COPY pyproject.toml .

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install the quartz_solar_forecast package normally
RUN pip install . --verbose

# Expose port 8000 and 8501 to the outside world
EXPOSE 8000 8501

# The CMD will be provided by docker-compose.yml
CMD ["sh", "-c", "$CMD"]