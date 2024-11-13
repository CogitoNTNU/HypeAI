FROM python:3.9

ENV GECKO_DRIVER_PATH=/usr/bin/geckodriver
ENV MOZ_HEADLESS=1

USER root

# Install Python and pip
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    firefox-esr \
    && wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz \
    && tar -xzf geckodriver-v0.33.0-linux64.tar.gz \
    && mv geckodriver /usr/bin/geckodriver \
    && chmod +x /usr/bin/geckodriver \
    && rm geckodriver-v0.33.0-linux64.tar.gz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
COPY tiktok-uploader/ ./tiktok-uploader/
COPY install_tiktok_uploader.bash .

# Create and use a virtual environment
RUN python3 -m pip install -r requirements.txt

RUN pip install hatch

RUN cd tiktok-uploader && \
 hatch build && \
 python3 -m pip install -e . && \
 cd ..
# RUN ./install_tiktok_uploader.bash

# Copy the rest of the application code
COPY . .


# Run the main.py script
CMD ["python3", "main.py"]