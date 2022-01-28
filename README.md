# Machine Learning in Production using Google Cloud Platform

The goal of this project is to serve as an accelerator for the deployment of production-ready machine learning pipelines.
The use case is NLP sentiment analysis binary classification (positive/negative), trained and evaluated on reviews from [IMDB](https://imdb.com).

It provides:

* CI/CD pipeline to deploy as new binary to Google Cloud Storage using Cloud Build
* DataFlow pipeline to ingest and preprocess training data
* Vertex pipeline to train a new model
* Deployment of the Vertex model

## CI/CD pipeline

A Cloud Build trigger launches the build process which creates a dist package of the pipeline code and stores it in Google Cloud Storage.

This process is triggered by commits being pushed to the `pipelines` branch.

## DataFlow preprocessing

The Beam pipeline to be executed on DataFlow is launched from the `launch_preprocess.sh` script which provides the necessary configuration:
Project in which to run, input/output data locations, etc. The script calls `run_preprocess.py`. The actual
processing is defined in `pipeline/preprocess_pipeline.py`.

To execute it:

```
$ gcloud auth application-default login
$ cd models
$ venv ...
$ source activate
$ pip3 install -r requirements
$ bin/launch_preprocess.sh
```

Functionally you could say there are three pipelines. First for the training data:

* "Read train set": `read_set.py` loads the files with reviews from `pos` and `neg` subdirectories and builds a PCollection dataset that has a label column with 1 for a positive review, and 0 for a negative review, and one column for the review text.
* "Anlyz. and Transf.": `preprocess_pipeline.py preprocessing_fn` uses Tensorflow Transform to build n-grams, calculate the TF-IDF and outputs records as `[label (pos or neg), ngram token index, weight]`
* "TrainToExamples": Convert the result to `TFRecord`
* "Write Train Data": Output to Google Cloud Storage

For the test data:

* "Read test set"
* "Transform test" takes the raw test dataset and transform function from "Analyz. and Transf." that was applied to the training dataset as well, and applies it to the test data
* "TestToExamples": Convert the result to `TFRecord`
* "Write Test Data": Output to Google Cloud Storage

Finally we store the transform function to reuse it later for inference:

* "Write Transform fn" to Google Cloud Storage

This last step is so that we can correctly tokenize and transform any review that we want to run the model on, to see if it's a positive or negative review.

## Training on Vertex AI

We see signficant performance improvement when we use input data in the binary `TFRecord` format, but this means that in Vertex we must use "Custom Jobs" 
rather than "Training Pipelines". Launch using 

`$ bin/launch_training.sh`

As suggested in the output from this command, now you can:

* Check the job status: `gcloud ai custom-jobs stream-logs projects/237148598933/locations/europe-west4/customJobs/3627616514498101248`
* Or stream the logs while it's training: `gcloud ai custom-jobs stream-logs projects/237148598933/locations/europe-west4/customJobs/3627616514498101248`

Note that depending on various factors, it's not uncommon that a training job remains pending for 10 minutes.

## Hyperparameter Tuning

To tune the hyperparameters according to the specification in `training_config_ht.yaml`:

`$ bin/launch_training_ht.sh`
