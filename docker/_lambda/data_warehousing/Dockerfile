# Use a base image compatible with arm64 architecture
# latest
FROM lambda_base_image_dev:v1.0.0

# Install awslambdaric
RUN pip install awslambdaric

# Copy required script file into the container at /app
COPY scripts/data_warehousing/data_warehousing.py .

# Use a slim version of the base Python image to reduce the final image size
# FROM python:3.10.8-slim

# Define workdir
# WORKDIR /app

# Copy in the built dependencies
# COPY --from=build-image /app /app

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
CMD [ "data_warehousing.lambda_handler" ]

# docker rm -f $(docker ps -aq)
# docker rmi lambda_data_warehousing_image_dev:v1.0.0
# docker build --platform linux/amd64 -t lambda_data_warehousing_image_dev:v1.0.0 -f docker/_lambda/data_warehousing/Dockerfile .
# docker run --platform linux/amd64 -d -v ~/.aws-lambda-rie:/aws-lambda -p 9000:8080 \
#     --entrypoint /aws-lambda/aws-lambda-rie \
#     --name data_warehousing_container_dev \
#     lambda_data_warehousing_image_dev:v1.0.0 \    
#     /usr/local/bin/python -m awslambdaric data_warehousing.lambda_handler