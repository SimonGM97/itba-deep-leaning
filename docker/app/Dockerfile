# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY app.py .

# Pass the name of the function handler as an argument to the runtime
CMD [ "streamlit run app.py" ]

# [remove all stopped containers] yes | docker container prune
# docker rmi app_image_dev:v1.0.0
# docker build -t app_image_dev:v1.0.0 -f docker/app/Dockerfile .
# docker run -d --name app_container_dev app_image_dev:v1.0.0