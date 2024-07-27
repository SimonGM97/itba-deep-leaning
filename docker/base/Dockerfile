# Use a python base image
FROM python:3.10.8 
# as build-image

# Switch to /tmp dir
WORKDIR /tmp

# Upgrade pip
RUN pip install --upgrade pip

# Copy & install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install dask[dataframe]
RUN pip install "dask[dataframe]"

# Copy specified files to install PyTradeX
COPY PyTradeX PyTradeX
COPY README.md .
COPY LICENSE .
COPY setup.py .

# Install PyTradeX library
RUN pip install --no-cache-dir .

# Switch back to /app dir
WORKDIR /app

# Copy required files & directories into the container at /app
COPY .aws /root/.aws
COPY config config
COPY schemas schemas

# [remove all stopped containers] yes | docker container prune
# docker rmi base_image_dev:v1.0.0
# docker build -t base_image_dev:v1.0.0 -f docker/base/Dockerfile .
# docker container run -it base_image_dev:v1.0.0 bash

# docker build --platform linux/amd64 -t base_image_dev:v1.0.0 -f docker/base/Dockerfile .