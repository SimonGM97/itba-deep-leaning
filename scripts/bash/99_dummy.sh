#!/bin/bash
# export ECR_REPOSITORY_NAME=pytradex-ecr
# export ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
# export REGION=sa-east-1

# chmod +x ./scripts/bash/02_image_pulling.sh
# ./scripts/bash/02_image_pulling.sh

# Set repository variables
ECR_REPOSITORY_NAME=pytradex-ecr
ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
REGION=sa-east-1
VERSION=v1.0.0

# Clean containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Log-in to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Pull images from AWS ECR repository
docker pull $ECR_REPOSITORY_NAME/data_processing_image:$VERSION
docker pull $ECR_REPOSITORY_NAME/model_tuning_image:$VERSION
docker pull $ECR_REPOSITORY_NAME/model_updating_image:$VERSION

docker pull $ECR_REPOSITORY_NAME/drift_monitoring_image:$VERSION
# docker pull $ECR_REPOSITORY_NAME/model_serving_image:$VERSION
# docker pull $ECR_REPOSITORY_NAME/inference_image:$VERSION

docker pull $ECR_REPOSITORY_NAME/trading_image:$VERSION
docker pull $ECR_REPOSITORY_NAME/app_image:$VERSION


# Get the source folder and bucket name from the user
dev_folder="/Users/simongarciamorillo/Library/CloudStorage/OneDrive-Personal/Documents/BetterTradeGroup/PyTradeX/pytradex-dev"
dev_bucket="pytradex-dev"

prod_folder="/Users/simongarciamorillo/Library/CloudStorage/OneDrive-Personal/Documents/BetterTradeGroup/PyTradeX/pytradex-prod"
prod_bucket="pytradex-prod"

# Upload DEV folders to the dev S3 bucket
# aws s3 cp "$dev_folder"/backup s3://"$dev_bucket"/backup --recursive
# aws s3 cp "$dev_folder"/data_processing s3://"$dev_bucket"/data_processing --recursive
aws s3 cp "$dev_folder"/modeling s3://"$dev_bucket"/modeling --recursive
aws s3 cp "$dev_folder"/trading s3://"$dev_bucket"/trading --recursive
aws s3 cp "$dev_folder"/utils s3://"$dev_bucket"/utils --recursive

# # Upload PROD folders to the prod S3 bucket
# aws s3 cp "$prod_folder"/backup s3://"$prod_bucket"/backup --recursive
# aws s3 cp "$prod_folder"/data_processing s3://"$prod_bucket"/data_processing --recursive
# aws s3 cp "$prod_folder"/modeling s3://"$prod_bucket"/modeling --recursive
# aws s3 cp "$prod_folder"/trading s3://"$prod_bucket"/trading --recursive
# aws s3 cp "$prod_folder"/utils s3://"$prod_bucket"/utils --recursive