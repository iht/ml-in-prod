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
import glob
import logging
import os
import sys
from typing import Tuple, Optional

import hypertune
import tensorflow as tf
from keras import Model, activations, models, losses, metrics
from keras import layers
from keras.layers import TextVectorization
from keras.optimizer_v2.rmsprop import RMSProp
from keras.type.types import Layer
from schemas.imdb_instance import SCHEMA

from . import __version__

MAX_TOKENS = 20000
HIDDEN_DIM = 16
VALIDATION_SPLIT = 0.2

NUM_PARALLEL_CALLS = 4  # for performance when transforming text data (assuming 4 vCPUs in worker)


def _read_tfrecords(data_location: str,
                    batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:  # , tf.data.Dataset]:
    train_location = os.path.join(data_location, "train/train_data-00000-of-00001.tfrecord")
    test_location = os.path.join(data_location, "test/test_data-00000-of-00001.tfrecord")

    logging.info(f"Reading training data from {train_location}")
    all_train_ds: tf.data.Dataset = tf.data.TFRecordDataset([train_location])
    logging.info(f"Reading test data from {test_location}")
    test_ds: tf.data.Dataset = tf.data.TFRecordDataset([test_location])

    all_train_ds = all_train_ds.batch(batch_size)
    all_train_ds = all_train_ds.map(lambda r: tf.io.parse_example(r, SCHEMA))
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda r: tf.io.parse_example(r, SCHEMA))

    return all_train_ds, test_ds


def get_save_paths(job_dir: Optional[str]):
    if job_dir:
        logging.info("Running in local")
        logs_dir = os.path.join(job_dir, "logs")
        models_dir = os.path.join(job_dir, "models")
    else:
        logging.info("Running in Vertex")
        logs_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        models_dir = os.environ.get('AIP_MODEL_DIR')

    logging.info(f"Tensorboard logs will be written to {logs_dir}")
    logging.info(f"Models written will be to {models_dir}")

    return logs_dir, models_dir


def train_and_evaluate(data_location: str,
                       batch_size: int,
                       epochs: int,
                       max_tokens: int,
                       hidden_dim: int,
                       num_parallel_calls: int,
                       job_dir: Optional[str]):
    logs_dir, models_dir = get_save_paths(job_dir)

    train_ds, test_ds = _read_tfrecords(data_location=data_location,
                                        batch_size=batch_size)

    vectorizer: TextVectorization = TextVectorization(ngrams=2, max_tokens=MAX_TOKENS, output_mode="multi_hot")
    vectorizer.adapt(train_ds.map(lambda r: r['text']))
    train_ds: tf.data.Dataset = train_ds.map(lambda r: (vectorizer(r['text']), r['target']),
                                             num_parallel_calls=num_parallel_calls)

    test_ds: tf.data.Dataset = test_ds.map(lambda r: (vectorizer(r['text']), r['target']),
                                           num_parallel_calls=num_parallel_calls)

    model = _build_model(max_tokens, hidden_dim)

    model.summary(print_fn=logging.info)

    logging.info(f"Writing TB logs to {logs_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1)

    model.fit(train_ds.cache(), epochs=epochs, callbacks=[tensorboard_callback])

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

    # Write text vectorizer (to avoid training-inference skew)
    vectorizer_path = os.path.join(models_dir, "vectorizer_fn")
    logging.info(f"Writing text vectorizer to {vectorizer_path}")
    vectorizer_model = models.Sequential()
    vectorizer_model.add(layers.Input(shape=(1,), dtype=tf.string))
    vectorizer_model.add(vectorizer)
    vectorizer_model.save(vectorizer_path)


def _build_model(max_tokens, hidden_dim) -> Model:
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
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--max-tokens', default=MAX_TOKENS, type=int)
    parser.add_argument('--hidden-dim', default=HIDDEN_DIM, type=int)
    parser.add_argument('--num-parallel-calls', default=NUM_PARALLEL_CALLS, type=int)
    parser.add_argument('--log', default='INFO', required=False)
    parser.add_argument('--job-dir', default=None, required=False)

    args = parser.parse_args()

    loglevel = getattr(logging, args.log.upper())
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(data_location=args.data_location,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       max_tokens=args.max_tokens,
                       hidden_dim=args.hidden_dim,
                       num_parallel_calls=args.num_parallel_calls,
                       job_dir=args.job_dir)
