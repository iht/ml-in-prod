EPOCHS=30
BATCH_SIZE=2048

gcloud ai-platform jobs submit training mnist_ht_`date +"%s"` \
  --python-version 3.7 \
  --runtime-version 1.15 \
  --scale-tier BASIC \
  --package-path ./trainer \
  --module-name trainer.task \
  --region europe-west1 \
  --job-dir gs://ihr-ml-in-prod/tmp/ \
  --config ./bin/hyper.yaml


