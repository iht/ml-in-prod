PROJECT=SOME_PROJECT
REGION=europe-west4
REPO=$REGION-docker.pkg.dev/$PROJECT/dataflow-containers
TAG=latest
IMAGE_URI=$REPO/ml-in-prod-container:$TAG
gcloud builds submit . --tag $IMAGE_URI
