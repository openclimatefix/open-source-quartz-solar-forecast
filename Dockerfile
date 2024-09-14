# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml file
COPY pyproject.toml .

# Install the project and its dependencies
RUN pip install --no-cache-dir .
RUN pip install quartz-solar-forecast

# Install wait-for-it script
ADD https://github.com/vishnubob/wait-for-it/raw/master/wait-for-it.sh /usr/local/bin/wait-for-it
RUN chmod +x /usr/local/bin/wait-for-it

# Copy the entire project directory
COPY . /app

# Expose ports 8000 (API) and 8501 (Streamlit) to the outside world
EXPOSE 8000 8501

# The CMD will be provided by docker-compose.yml
CMD ["sh", "-c", "wait-for-it open-meteo-api:8080 -- $CMD"]