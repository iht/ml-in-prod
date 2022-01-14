INPUT_DATA=gs://ihr-vertex-pipelines/data/aclImdb/
OUTPUT_DATA=gs://ihr-vertex-pipelines/data/prepared/

REGION=europe-west4
PROJECT=ihr-vertex-pipelines
SERVICE_ACCOUNT=ml-in-prod-dataflow-sa@ihr-vertex-pipelines.iam.gserviceaccount.com
TEMP_LOCATION=gs://ihr-vertex-pipelines/tmp/
NETWORK=default

python -m pipeline.preprocess_pipeline \
  --runner=DataflowRunner \
  --region=$REGION \
  --project=$PROJECT \
  --temp_location=$TEMP_LOCATION \
  --network=$NETWORK \
  --service_account_email=$SERVICE_ACCOUNT \
  --setup_file=./setup.py \
  --data-location=$INPUT_DATA \
  --output-location=$OUTPUT_DATA
