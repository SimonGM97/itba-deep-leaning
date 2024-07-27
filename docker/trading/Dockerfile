# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY scripts/trading/trading.py .

# Pass the name of the function handler as an argument to the runtime
CMD [ "python", "trading.py" ]

# [remove all stopped containers] yes | docker container prune
# docker rmi trading_image_dev:v1.0.0
# docker build -t trading_image_dev:v1.0.0 -f docker/trading/Dockerfile .
# docker run -d --name trading_container_dev trading_image_dev:v1.0.0