# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory (including quartz_solar_forecast)
COPY . /app

# Install the quartz_solar_forecast package in editable mode
RUN pip install -e .

# Expose port 8000 and 8501 to the outside world
EXPOSE 8000 8501

# The CMD will be provided by docker-compose.yml
CMD ["sh", "-c", "$CMD"]