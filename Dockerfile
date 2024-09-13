# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory (including quartz_solar_forecast)
COPY . /app

# Install the quartz_solar_forecast package in editable mode
RUN pip install -e .

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the application using python main.py
CMD ["python", "api/main.py"]
