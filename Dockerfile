# Use an official Python runtime as a parent image
FROM python:3.11

# Install necessary build tools and curl
RUN apt-get update && apt-get install -y build-essential curl

# Install Rust using rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install the project and its dependencies
RUN pip install .

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the application using python main.py
CMD ["python", "api/main.py"]
