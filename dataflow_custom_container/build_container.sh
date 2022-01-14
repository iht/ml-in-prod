PROJECT=ihr-vertex-pipelines
REPO=europe-west4-docker.pkg.dev/ihr-vertex-pipelines/dataflow-containers
TAG=latest
IMAGE_URI=$REPO/ml-in-prod-container:$TAG
gcloud builds submit . --tag $IMAGE_URI