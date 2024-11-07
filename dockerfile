FROM --platform=linux/amd64 selenium/standalone-chrome:latest

USER root

# Install Python and pip
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to selenium user
# USER seluser

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./tiktok-uploader/ install_tiktok_uploader.bash ./

# Create and use a virtual environment
# RUN python3 -m venv venv
RUN python3 -m pip install -r requirements.txt --break-system-packages
RUN ./install_tiktok_uploader.bash

# Copy the rest of the application code
COPY . .

# Run the main.py script
# CMD ["python3", "main.py"]