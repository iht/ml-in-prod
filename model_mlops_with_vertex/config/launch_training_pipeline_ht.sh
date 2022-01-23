#
#  Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

PROJECT=ihr-vertex-pipelines
REGION=europe-west4
STAGING_BUCKET=gs://ihr-vertex-pipelines/tmp/
MACHINE_TYPE=n1-standard-4
SERVICE_ACCOUNT=ml-in-prod-vertex-sa@ihr-vertex-pipelines.iam.gserviceaccount.com
TENSORBOAD=projects/237148598933/locations/europe-west4/tensorboards/3505278253721976832
CONTAINER_IMAGE=europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest

VERSION=`python3 setup.py --version`

DATA_LOCATION=gs://ihr-vertex-pipelines/data/tf_record_with_text/
TFT_LOCATION=gs://ihr-vertex-pipelines/data/tf_record_with_text/transform_fn


PKG_GCS_LOCATION=gs://ihr-vertex-pipelines/dist/my_first_ml_model-$VERSION.tar.gz
PYTHON_MODULE_NAME=trainer.task

BASE_OUTPUT_DIR=gs://ihr-vertex-pipelines/ht/$VERSION

python3 setup.py sdist

LOCAL_PACKAGE=dist/my_first_ml_model-$VERSION.tar.gz

pip3 install "$LOCAL_PACKAGE"

JOB_NAME=training-pipeline-$VERSION

echo Running job $JOB_NAME

python -m pipeline.custom_training_pipeline_ht \
  --job-name $JOB_NAME \
  --project $PROJECT \
  --region $REGION \
  --staging-bucket $STAGING_BUCKET \
  --package-gcs-location $PKG_GCS_LOCATION \
  --python-module-name $PYTHON_MODULE_NAME \
  --base-output-dir $BASE_OUTPUT_DIR \
  --service-account $SERVICE_ACCOUNT \
  --tensorboard $TENSORBOAD \
  --worker-type $MACHINE_TYPE \
  --container-image $CONTAINER_IMAGE \
  --data-location $DATA_LOCATION \
  --tft-location $TFT_LOCATION