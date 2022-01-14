INPUT_DATA=gs://ihr-vertex-pipelines/data/aclImdb/
OUTPUT_DATA=gs://ihr-vertex-pipelines/data/prepared/

REGION=europe-west4
PROJECT=ihr-vertex-pipelines
SERVICE_ACCOUNT=ml-in-prod-dataflow-sa@ihr-vertex-pipelines.iam.gserviceaccount.com
TEMP_LOCATION=gs://ihr-vertex-pipelines/tmp/
SUBNETWORK=regions/$REGION/subnetworks/default

CONTAINER=$REGION-docker.pkg.dev/$PROJECT/dataflow-containers/ml-in-prod-container

VERSION=`python setup.py --version`

python setup.py sdist

EXTRA_PACKAGE=dist/my_first_ml_model-$VERSION.tar.gz

pip install "$EXTRA_PACKAGE"

python run_preprocess.py \
  --runner=DataflowRunner \
  --region=$REGION \
  --project=$PROJECT \
  --temp_location=$TEMP_LOCATION \
  --no_use_public_ips \
  --subnetwork=$SUBNETWORK \
  --service_account_email=$SERVICE_ACCOUNT \
  --experiments=use_runner_v2 \
  --sdk_container_image=$CONTAINER \
  --extra_package=$EXTRA_PACKAGE \
  --data-location=$INPUT_DATA \
  --output-location=$OUTPUT_DATA
