# Use a python base image
FROM lambda_base_image_dev:v1.0.0

# Install awslambdaric
RUN pip install awslambdaric

# Copy required script file into the container at /app
COPY dummy_lambda.py .

# Use a slim version of the base Python image to reduce the final image size
# FROM python:3.10.8-slim

# Define workdir
# WORKDIR /app

# Copy in the built dependencies
# COPY --from=build-image /app /app

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
CMD [ "dummy_lambda.lambda_handler" ]
