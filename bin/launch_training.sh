EPOCHS=30
BATCH_SIZE=2048

gcloud ai-platform jobs submit training mnist_`date +"%s"` \
  --python-version 3.7 \
  --runtime-version 2.3 \
  --scale-tier BASIC \
  --package-path ./trainer \
  --module-name trainer.task \
  --region europe-west1 \
  --job-dir gs://ihr-ml-in-prod/tmp/ \
  -- \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --output-bucket ihr-ml-in-prod \
  --output-path models

