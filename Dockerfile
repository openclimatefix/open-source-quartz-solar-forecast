# Use an official Python runtime as a parent image
FROM python:3.11-slim

# install extra requirements
RUN apt-get clean
RUN apt-get update -y
RUN apt-get install gcc g++ git -y 

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory (including quartz_solar_forecast)
COPY pyproject.toml . 

# Install the quartz_solar_forecast package in editable mode
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv sync

# Copy the entire project directory (including quartz_solar_forecast)
COPY . /app

# Capture git version and set as environment variable
RUN GIT_VERSION=$(git describe --tags 2>/dev/null || echo "0.0.0") && \
    echo "export QUARTZ_SOLAR_FORECAST_VERSION=${GIT_VERSION}" > /app/git_version.sh

# Expose port 8000 to the outside world
EXPOSE 8000

# add api/v1 to python path
ENV PYTHONPATH="${PYTHONPATH}:/app/api/v1"

# Run the application using python main.py
# Note you can override this if you want to load v0
CMD ["/bin/bash", "-c", "source /app/git_version.sh && uv run uvicorn api.v1.api:app --host 0.0.0.0 --port 8000"]

