#!/bin/bash

# This script does:
#   - Get stored model from GCS and upload to Vertex AI
#   - Create endpoint
#   - Deploy model to endpoint

TS=$(date +"%Y_%m_%d-%H%M")

export VERSION=`python3 setup.py --version`
export PROJECT_ID='ihr-vertex-pipelines'
export MY_REGION='europe-west4' # :flag-nl:
export MODEL_NAME="text_model_${VERSION}_${TS}"
export ENDPOINT_NAME='text_endpoint_'$TS
export MODEL_LOCATION='gs://'${PROJECT_ID}'/'${VERSION}'/batch=8192/epochs=15/model'

echo "Deploying model $MODEL_NAME"

python3 deploy/deploy_model_to_endpoint.py