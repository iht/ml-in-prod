EPOCHS=30
BATCH_SIZE=2048

gcloud beta ai custom-jobs \
  --region=europe-west4 \
  --display-name=my-first-model \
  --config=training_config.yaml
