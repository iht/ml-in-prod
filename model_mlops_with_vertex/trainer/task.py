#   Copyright 2022 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse
import logging
import os
import sys
from typing import Optional, List, Tuple
from urllib.parse import urlparse

import hypertune
import tensorflow as tf
import tensorflow_transform as tft
from google.cloud import storage
from google.cloud.storage import Blob
from keras import Model, activations, models, losses, metrics
from keras import layers
from keras.layers import TextVectorization
from keras.optimizer_v2.rmsprop import RMSProp
from keras.type.types import Layer
from schemas.imdb_instance import LABEL_COLUMN, TEXT_COLUMN
from tensorflow_transform import TFTransformOutput

MAX_TOKENS = 20000
HIDDEN_DIM = 16


def _training_input_fn(file_pattern: str,
                       tf_transform_output: tft.TFTransformOutput,
                       batch_size: int) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    filenames = _get_load_paths(file_pattern)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda r: tf.io.parse_single_example(r, transformed_feature_spec))
    dataset = dataset.map(lambda d: (d[TEXT_COLUMN], d[LABEL_COLUMN]))
    dataset = dataset.batch(batch_size)

    return dataset


def _read_tfrecords(data_location: str,
                    tft_location: str,
                    batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_location = os.path.join(data_location, "train/")
    test_location = os.path.join(data_location, "test/")

    tf_transform_output: TFTransformOutput = tft.TFTransformOutput(tft_location)

    train_ds = _training_input_fn(train_location, tf_transform_output, batch_size=batch_size)
    test_ds = _training_input_fn(test_location, tf_transform_output, batch_size=batch_size)

    return train_ds, test_ds


def _get_load_paths(file_pattern: str) -> List[str]:
    storage_client = storage.Client()
    url_parts = urlparse(file_pattern)
    bucket = url_parts.hostname
    location = url_parts.path
    output: List[Blob] = storage_client.list_blobs(bucket, prefix=location[1:])
    paths = [f"gs://{b.bucket.name}/{b.name}" for b in output]

    return paths


def _get_save_paths(job_dir: Optional[str]) -> Tuple[str, str]:
    if job_dir:
        logging.info("Running in local")
        logs_dir = os.path.join(job_dir, "logs")
        models_dir = os.path.join(job_dir, "models")
    else:
        logging.info("Running in Vertex")
        logs_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        models_dir = os.environ.get('AIP_MODEL_DIR')

    logging.info(f"Tensorboard logs will be written to {logs_dir}")
    logging.info(f"Model will be written to {models_dir}")

    return logs_dir, models_dir


def train_and_evaluate(data_location: str,
                       tft_location: str,
                       batch_size: int,
                       epochs: int,
                       max_tokens: int,
                       hidden_dim: int,
                       job_dir: Optional[str]):
    logs_dir, models_dir = _get_save_paths(job_dir)

    train_ds, test_ds = _read_tfrecords(data_location=data_location,
                                        tft_location=tft_location,
                                        batch_size=batch_size)

    x_text_train = train_ds.map(lambda text, label: text)

    vectorizer: TextVectorization = layers.TextVectorization(ngrams=2, max_tokens=max_tokens, output_mode="multi_hot")
    vectorizer.adapt(x_text_train)

    train_ds = train_ds.map(lambda x, y: (vectorizer(x), y))
    test_ds = test_ds.map(lambda x, y: (vectorizer(x), y))

    model = _build_model(max_tokens, hidden_dim)

    # Show model summary in log for debugging purposes
    model.summary(print_fn=logging.info)

    logging.info(f"Writing TensorBoard logs to {logs_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1)

    model.fit(train_ds, epochs=epochs, callbacks=[tensorboard_callback], validation_data=test_ds)

    # Evaluate metrics and write to log
    loss, acc = model.evaluate(test_ds)
    logging.info(f"LOSS: {loss:.4f}")
    logging.info(f"ACCURACY: {acc:.4f}")

    # Publish metrics (for hyperparam. tuning)
    metric_tag = "kschool_accuracy"
    logging.info(f"Writing accuracy with hypertune")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=metric_tag,
        metric_value=acc,
        global_step=epochs)

    # Write model artifact
    model_path = os.path.join(models_dir, "saved_model")
    logging.info(f"Writing model to {model_path}")
    model.save(model_path)

    # Write vectorizer. Wrap it in a model for easy I/O
    vectorizer_path = os.path.join(models_dir, "vectorizer")
    logging.info(f"Writing vectorizer to {vectorizer_path}")
    vect_model = tf.keras.models.Sequential()
    vect_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    vect_model.add(vectorizer)
    vect_model.save(vectorizer_path)


def _build_model(max_tokens: int, hidden_dim: int) -> Model:
    # TextVectorization cannot be used as layer at the same time as Tensorboard
    # https://github.com/keras-team/keras/issues/15163

    inputs: Layer = layers.Input(shape=(max_tokens,))
    x: Layer = layers.Dense(hidden_dim, activation=activations.relu)(inputs)
    x: Layer = layers.Dropout(0.5)(x)
    outputs: Layer = layers.Dense(1, activation=activations.sigmoid)(x)
    model: Model = models.Model(inputs, outputs, name="my-kschool-model")
    model.compile(optimizer=RMSProp(), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--tft-location', default=None, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--max-tokens', default=MAX_TOKENS, type=int)
    parser.add_argument('--hidden-dim', default=HIDDEN_DIM, type=int)
    parser.add_argument('--log', default='INFO', required=False)
    parser.add_argument('--job-dir', default=None, required=False)

    args = parser.parse_args()

    loglevel = getattr(logging, args.log.upper())
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(data_location=args.data_location,
                       tft_location=args.tft_location,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       max_tokens=args.max_tokens,
                       hidden_dim=args.hidden_dim,
                       job_dir=args.job_dir)
