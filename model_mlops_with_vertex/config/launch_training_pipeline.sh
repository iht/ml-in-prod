PROJECT=ihr-vertex-pipelines
REGION=europe-west4
MACHINE_TYPE=n1-standard-4
SERVICE_ACCOUNT=ml-in-prod-vertex-sa@ihr-vertex-pipelines.iam.gserviceaccount.com
TENSORBOAD=projects/237148598933/locations/europe-west4/tensorboards/3505278253721976832
CONTAINER_IMAGE=europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest

VERSION=`python3 setup.py --version`

DATA_LOCATION=gs://ihr-vertex-pipelines/data/tf_record_with_text/
TFT_LOCATION=gs://ihr-vertex-pipelines/data/tf_record_with_text/transform_fn

EPOCHS=15
BATCH_SIZE=8192

PKG_GCS_LOCATION=gs://ihr-vertex-pipelines/dist/my_first_ml_model-$VERSION.tar.gz
PYTHON_MODULE_NAME=trainer.task

BASE_OUTPUT_DIR=gs://ihr-vertex-pipelines/0.13/batch=$BATCH_SIZE/epochs=$EPOCHS/

python3 setup.py sdist

LOCAL_PACKAGE=dist/my_first_ml_model-$VERSION.tar.gz

pip3 install "$LOCAL_PACKAGE"

python -m pipeline.custom_trainer_pipeline \
  --job-name training-pipeline-$VERSION \
  --project $PROJECT \
  --region $REGION \
  --package-gcs-location $PKG_GCS_LOCATION \
  --python-module-name $PYTHON_MODULE_NAME \
  --base-output-dir $BASE_OUTPUT_DIR \
  --service-account $SERVICE_ACCOUNT \
  --tensorboard $TENSORBOAD \
  --worker-type $MACHINE_TYPE \
  --container-image $CONTAINER_IMAGE \
  --data-location $DATA_LOCATION \
  --tft-location $TFT_LOCATION \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE
