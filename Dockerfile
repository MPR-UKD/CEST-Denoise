# Start from a base image
FROM python:3.10-slim-buster

# Working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory in the Docker image
COPY . /app

# This makes sure Python finds your custom modules
ENV PYTHONPATH "${PYTHONPATH}:/app"
