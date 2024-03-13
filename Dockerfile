# Use the official Python image as a base image
FROM python:3.9-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install gcc and python3-dev
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory and other necessary files into the container
COPY src/ ./src/

# Expose port 8050 for the Dash app (based on the Makefile content)
EXPOSE 8050

# Specify the default command to run the Dash app
CMD ["python", "src/app.py"]
