#!/bin/bash
# export ECR_REPOSITORY_NAME=ecr_repository_name
# export ECR_REPOSITORY_URI=ecr_repository_name
# export REGION=sa-east-1

# chmod +x ./scripts/bash/02_image_building.sh
# ./scripts/bash/02_image_building.sh

# Set variables
ECR_REPOSITORY_NAME=pytradex-ecr
ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
REGION=sa-east-1
VERSION=v1.0.0
ENV=dev

# Clean running containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Make scripts executable
chmod +x app.py
chmod +x ./scripts/trading/trading.py
chmod +x ./scripts/data_processing/data_processing.py
chmod +x ./scripts/modeling/modeling.py
chmod +x ./scripts/data_warehousing/data_warehousing.py
chmod +x dummy_lambda.py

# Build Docker images
docker build -t base_image_${ENV}:${VERSION} -f docker/base/Dockerfile .
# docker build -t app_image_${ENV}:${VERSION} -f docker/app/Dockerfile .
docker build -t trading_image_${ENV}:${VERSION} -f docker/trading/Dockerfile .
docker build -t data_processing_image_${ENV}:${VERSION} -f docker/data_processing/Dockerfile .
docker build -t modeling_image_${ENV}:${VERSION} -f docker/modeling/Dockerfile .
docker build -t data_warehousing_image_${ENV}:${VERSION} -f docker/data_warehousing/Dockerfile .

# Build lambda Docker images
docker build --platform linux/amd64 -t lambda_base_image_${ENV}:${VERSION} -f docker/base/Dockerfile .
# docker build --platform linux/amd64 -t lambda_app_image_${ENV}:${VERSION} -f docker/_lambda/app/Dockerfile .
docker build --platform linux/amd64 -t lambda_trading_image_${ENV}:${VERSION} -f docker/_lambda/trading/Dockerfile .
docker build --platform linux/amd64 -t lambda_data_processing_image_${ENV}:${VERSION} -f docker/_lambda/data_processing/Dockerfile .
docker build --platform linux/amd64 -t lambda_modeling_image_${ENV}:${VERSION} -f docker/_lambda/modeling/Dockerfile .
docker build --platform linux/amd64 -t lambda_data_warehousing_image_${ENV}:${VERSION} -f docker/_lambda/data_warehousing/Dockerfile .
# docker build --platform linux/amd64 -t lambda_test_image_${ENV}:${VERSION} -f docker/_lambda/test/Dockerfile .

# Log-in to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URI}

# Tag lambda docker images
# docker tag lambda_app_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_app_image_${ENV}_${VERSION}
docker tag lambda_trading_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_trading_image_${ENV}_${VERSION}
docker tag lambda_data_processing_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_processing_image_${ENV}_${VERSION}
docker tag lambda_modeling_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_modeling_image_${ENV}_${VERSION}
docker tag lambda_data_warehousing_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_warehousing_image_${ENV}_${VERSION}
# docker tag lambda_test_image_${ENV}:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_test_image_${ENV}_${VERSION}

# Push images to AWS ECR repository
# docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_app_image_${ENV}_${VERSION}
docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_trading_image_${ENV}_${VERSION}
docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_processing_image_${ENV}_${VERSION}
docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_modeling_image_${ENV}_${VERSION}
docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_warehousing_image_${ENV}_${VERSION}
# docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_test_image_${ENV}_${VERSION}

# Delete untagged images from ECR
IMAGES_TO_DELETE=$( aws ecr list-images --region $REGION --repository-name ${ECR_REPOSITORY_NAME} --filter "tagStatus=UNTAGGED" --query 'imageIds[*]' --output json )
aws ecr batch-delete-image --region $REGION --repository-name ${ECR_REPOSITORY_NAME} --image-ids "$IMAGES_TO_DELETE" || true

# List AWS ECR images
# aws ecr list-images --region $REGION --repository-name ${ECR_REPOSITORY_NAME}