gcloud ai-platform jobs submit training mnist_ht_`date +"%s"` \
  --python-version 3.7 \
  --runtime-version 2.3 \
  --scale-tier BASIC \
  --package-path ./trainer \
  --module-name trainer.task_hypertune \
  --region europe-west1 \
  --job-dir gs://ihr-ml-in-prod/tmp/ \
  --config ./bin/hyper.yaml


