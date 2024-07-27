#!/bin/bash
# export ECR_REPOSITORY_NAME=ecr_repository_name
# export ECR_REPOSITORY_URI=ecr_repository_name
# export REGION=sa-east-1

# chmod +x ./scripts/bash/03_lambda_functions_building.sh
# ./scripts/bash/03_lambda_functions_building.sh

# Set variables
ECR_REPOSITORY_NAME=pytradex-ecr
ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
LAMBDA_ROLE_ARN=arn:aws:iam::097866913509:role/lambda_role
LAMBDA_LOGS_ARN=arn:aws:logs:sa-east-1:097866913509:log-group:/aws/lambda
REGION=sa-east-1
VERSION=v1.0.0
INTERVALS=60min
ENV=dev

# Define LOGGING_CONFIG
# LOGGING_CONFIG='{"CloudWatchLogsArn":"arn:aws:logs:sa-east-1:097866913509:log-group:/aws/lambda/data_processing_dev:*","LogRetentionInDays":1,"ApplicationLogLevel":"INFO"}'

# """
# TRADING
# """
# Delete trading lambda function
aws lambda delete-function --function-name trading-${ENV}

# Create new trading lambda function
aws lambda create-function \
    --function-name trading-${ENV} \
    --role ${LAMBDA_ROLE_ARN} \
    --package-type Image \
    --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_trading_image_${ENV}_${VERSION} \
    --description "Trading lambda function." \
    --architecture x86_64 \
    --memory-size 1536 \
    --timeout 900 \
    --tags environment=${ENV},intervals=${INTERVALS} \
    --region ${REGION} \
    --logging-config LogFormat=JSON,ApplicationLogLevel=INFO,SystemLogLevel=INFO,LogGroup=/aws/lambda/trading-dev \
    --publish

# TEST EVENT:
# {
#     "account_id": 0,
#     "workflow": "trading_round"
# }

# """
# DATA PROCESSING
# """
# Delete data-processing lambda function
aws lambda delete-function --function-name data-processing-${ENV}

# Create new data-processing lambda function
aws lambda create-function \
    --function-name data-processing-${ENV} \
    --role ${LAMBDA_ROLE_ARN} \
    --package-type Image \
    --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_processing_image_${ENV}_${VERSION} \
    --description "Data processing lambda function." \
    --architecture x86_64 \
    --memory-size 2048 \
    --timeout 900 \
    --tags environment=${ENV},intervals=${INTERVALS} \
    --region ${REGION} \
    --logging-config LogFormat=JSON,ApplicationLogLevel=INFO,SystemLogLevel=INFO,LogGroup=/aws/lambda/data-processing-dev \
    --publish

# TEST EVENT:
# {
#     "workflow": "trading_round",
#     "forced_intervals": null
# }

# """
# MODELING
# """
# Delete modeling lambda function
aws lambda delete-function --function-name modeling-${ENV}

# Create new modeling lambda function
aws lambda create-function \
    --function-name modeling-${ENV} \
    --role ${LAMBDA_ROLE_ARN} \
    --package-type Image \
    --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_modeling_image_${ENV}_${VERSION} \
    --description "Modeling lambda function." \
    --architecture x86_64 \
    --memory-size 2048 \
    --timeout 900 \
    --tags environment=${ENV},intervals=${INTERVALS} \
    --region ${REGION} \
    --logging-config LogFormat=JSON,ApplicationLogLevel=INFO,SystemLogLevel=INFO,LogGroup=/aws/lambda/modeling-dev \
    --publish

# TEST EVENT:
# {
#     "workflow": "trading_round"
# }

# """
# DATA WAREHOUSING
# """
# Delete data-warehousing lambda function
aws lambda delete-function --function-name data-warehousing-${ENV}

# Create new data-warehousing lambda function
aws lambda create-function \
    --function-name data-warehousing-${ENV} \
    --role ${LAMBDA_ROLE_ARN} \
    --package-type Image \
    --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_data_warehousing_image_${ENV}_${VERSION} \
    --description "Data warehousing lambda function." \
    --architecture x86_64 \
    --memory-size 1024 \
    --timeout 900 \
    --tags environment=${ENV},intervals=${INTERVALS} \
    --region ${REGION} \
    --logging-config LogFormat=JSON,ApplicationLogLevel=INFO,SystemLogLevel=INFO,LogGroup=/aws/lambda/data-warehousing-dev \
    --publish

# TEST EVENT:
# {
#     "workflow": "trading_round"
# }

# # """
# # TEST
# # """
# # Delete trading lambda function
# aws lambda delete-function --function-name test-${ENV}

# # Create new trading lambda function
# aws lambda create-function \
#     --function-name test-${ENV} \
#     --role ${LAMBDA_ROLE_ARN} \
#     --package-type Image \
#     --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_test_image_${ENV}_${VERSION} \
#     --description "Test lambda function." \
#     --architecture x86_64 \
#     --memory-size 128 \
#     --timeout 30 \
#     --tags environment=${ENV},intervals=${INTERVALS} \
#     --region ${REGION} \
#     --logging-config LogFormat=JSON,ApplicationLogLevel=INFO,SystemLogLevel=INFO,LogGroup=/aws/lambda/test-dev \
#     --publish


# COMPLETE PARAMETERS
# aws lambda create-function \
#     --function-name <value> \
#     [--runtime <value>] \
#     --role <value> \
#     [--handler <value>] \
#     [--code <value>] \
#     [--description <value>] \
#     [--timeout <value>] \
#     [--memory-size <value>] \
#     [--publish | --no-publish] \
#     [--vpc-config <value>] \
#     [--package-type <value>] \
#     [--dead-letter-config <value>] \
#     [--environment <value>] \
#     [--kms-key-arn <value>] \
#     [--tracing-config <value>] \
#     [--tags <value>] \
#     [--layers <value>] \
#     [--file-system-configs <value>] \
#     [--image-config <value>] \
#     [--code-signing-config-arn <value>] \
#     [--architectures <value>] \
#     [--ephemeral-storage <value>] \
#     [--snap-start <value>] \
#     [--logging-config <value>] \
#     [--zip-file <value>] \
#     [--cli-input-json <value>] \
#     [--generate-cli-skeleton <value>] \
#     [--debug] \
#     [--endpoint-url <value>] \
#     [--no-verify-ssl] \
#     [--no-paginate] \
#     [--output <value>] \
#     [--query <value>] \
#     [--profile <value>] \
#     [--region <value>] \
#     [--version <value>] \
#     [--color <value>] \
#     [--no-sign-request] \
#     [--ca-bundle <value>] \
#     [--cli-read-timeout <value>] \
#     [--cli-connect-timeout <value>]