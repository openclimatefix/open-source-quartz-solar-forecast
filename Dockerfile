# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install git unzip g++ gcc libgeos++-dev libproj-dev proj-data proj-bin -y

RUN apt-get update && \
    apt-get install git unzip g++ gcc libgeos++-dev libproj-dev proj-data proj-bin -y

# Copy the pyproject.toml file
COPY pyproject.toml .

# Copy the entire project directory
COPY . /app

# Install requirements
RUN conda install python=3.12
RUN conda install -c conda-forge xesmf esmpy h5py pytorch-cpu=2.3.1 torchvision -y
RUN pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cpu

# Install build dependencies and the project
RUN pip install --no-cache-dir -e .

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the application using python main.py
CMD ["python", "api/main.py"]