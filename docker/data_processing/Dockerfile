# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY scripts/data_processing/data_processing.py .

# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "data_processing.py" ]

# [remove all stopped containers] yes | docker container prune
# docker rmi data_processing_image_dev:v1.0.0
# docker build -t data_processing_image_dev:v1.0.0 -f docker/data_processing/Dockerfile .
# docker run -d --name data_processing_container_dev data_processing_image_dev:v1.0.0