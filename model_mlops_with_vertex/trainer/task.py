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
from typing import Optional, List

import hypertune
import tensorflow as tf
import tensorflow_transform as tft
from keras import Model, activations, models, losses, metrics
from keras import layers
from keras.layers import TextVectorization
from keras.optimizer_v2.rmsprop import RMSProp
from keras.type.types import Layer
from schemas.imdb_instance import LABEL_COLUMN, TEXT_COLUMN
from tensorflow_transform import TFTransformOutput

MAX_TOKENS = 20000
HIDDEN_DIM = 16
VALIDATION_SPLIT = 0.2

DATASET_SIZE = 25000

NUM_PARALLEL_CALLS = 4  # for performance when transforming text data (assuming 4 vCPUs in worker)


def _training_input_fn(file_pattern: List[str],
                       tf_transform_output: tft.TFTransformOutput,
                       batch_size: int) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    dataset: tf.data.Dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec)

    dataset = dataset.map(lambda d: (d[TEXT_COLUMN], d[LABEL_COLUMN]))

    return dataset


def _read_tfrecords(data_location: str,
                    tft_location: str,
                    batch_size: int) -> (tf.data.Dataset, tf.data.Dataset):
    train_location = os.path.join(data_location, "train/*")
    test_location = os.path.join(data_location, "test/*")

    tf_transform_output: TFTransformOutput = tft.TFTransformOutput(tft_location)

    train_ds = _training_input_fn([train_location], tf_transform_output, batch_size=batch_size)
    test_ds = _training_input_fn([test_location], tf_transform_output, batch_size=batch_size)

    return train_ds, test_ds


def get_save_paths(job_dir: Optional[str]) -> (str, str):
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
                       dataset_size: int,
                       max_tokens: int,
                       hidden_dim: int,
                       job_dir: Optional[str]):
    logs_dir, models_dir = get_save_paths(job_dir)

    train_ds, test_ds = _read_tfrecords(data_location=data_location,
                                        tft_location=tft_location,
                                        batch_size=batch_size)

    x_text_train = train_ds.map(lambda x, y: x)

    vectorizer: TextVectorization = layers.TextVectorization(ngrams=2, max_tokens=max_tokens, output_mode="multi_hot")
    vectorizer.adapt(x_text_train, steps=epochs)
    train_ds = train_ds.map(lambda x, y: (vectorizer(x), y))
    test_ds = test_ds.map(lambda x, y: (vectorizer(x), y))

    model = _build_model(max_tokens, hidden_dim)

    # Show model summary in log for debugging purposes
    model.summary(print_fn=logging.info)

    logging.info(f"Writing TensorBoard logs to {logs_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1)

    model.fit(train_ds, epochs=epochs, callbacks=[tensorboard_callback], steps_per_epoch=dataset_size // epochs)

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

    # Write vectorizer
    vectorizer_path = os.path.join(models_dir, "vectorizer")
    logging.info(f"Writing vectorizer to {vectorizer_path}")
    model.save(vectorizer_path)


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
    parser.add_argument('--full-dataset-size', type=int, default=DATASET_SIZE)
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
                       dataset_size=args.full_dataset_size,
                       max_tokens=args.max_tokens,
                       hidden_dim=args.hidden_dim,
                       job_dir=args.job_dir)
