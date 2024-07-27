# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY scripts/modeling/modeling.py .

# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "modeling.py" ]

# [remove all stopped containers] yes | docker container prune
# docker rmi modeling_image_dev:v1.0.0
# docker build -t modeling_image_dev:v1.0.0 -f docker/modeling/Dockerfile .
# docker run -d --name modeling_container_dev modeling_image_dev:v1.0.0