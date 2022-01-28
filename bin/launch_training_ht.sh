gcloud ai hp-tuning-jobs create \
  --region=europe-west4 \
  --display-name=ht-my-first-model \
  --max-trial-count=56 \
  --parallel-trial-count=8 \
  --config=./config/training_config_ht.yaml
