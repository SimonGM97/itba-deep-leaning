# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY scripts/data_warehousing/data_warehousing.py .

# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "data_warehousing.py" ]

# [remove all stopped containers] yes | docker container prune
# docker rmi data_warehousing_image_dev:v1.0.0
# docker build -t data_warehousing_image_dev:v1.0.0 -f docker/data_warehousing/Dockerfile .
# docker run -d --name data_warehousing_container_dev data_warehousing_image_dev:v1.0.0