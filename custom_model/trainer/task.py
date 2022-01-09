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
import random
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import Model, activations, models, losses, metrics
from keras import layers
from keras.layers import TextVectorization
from keras.optimizer_v2.rmsprop import RMSProp
from keras.type.types import Layer
from tensorflow.keras import utils

from . import __version__

MAX_TOKENS = 20000
HIDDEN_DIM = 16
VALIDATION_SPLIT = 0.2

NUM_PARALLEL_READS = 128  # for performance when reading from GCS
NUM_PARALLEL_CALLS = 4  # for performance when transforming text data (assuming 4 vCPUs in worker)


def _read_data_with_keras_utils(data_location: str,
                                batch_size: int,
                                validation_split: float) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_location = os.path.join(data_location, "train/")
    test_location = os.path.join(data_location, "test/")

    seed = random.randrange(1000)

    train_ds: tf.data.Dataset = utils.text_dataset_from_directory(train_location,
                                                                  batch_size=batch_size,
                                                                  validation_split=validation_split,
                                                                  seed=seed,
                                                                  subset='training')
    validation_ds: tf.data.Dataset = utils.text_dataset_from_directory(train_location,
                                                                       batch_size=batch_size,
                                                                       validation_split=validation_split,
                                                                       seed=seed,
                                                                       subset='validation')
    test_ds: tf.data.Dataset = utils.text_dataset_from_directory(test_location,
                                                                 batch_size=batch_size)

    return train_ds, validation_ds, test_ds


def _read_positive_and_negative(data_location: str, num_parallel_reads: int) -> Tuple[tf.data.Dataset, int]:
    pos_location = os.path.join(data_location, "pos/*.txt")
    neg_location = os.path.join(data_location, "neg/*.txt")
    pos_filenames = glob.glob(pos_location)
    neg_filenames = glob.glob(neg_location)

    pos_cardinality = len(pos_filenames)
    logging.info(f"Found {pos_cardinality} pos instances")
    neg_cardinality = len(neg_filenames)
    logging.info(f"Found {neg_cardinality} neg instances")

    pos_ds: tf.data.Dataset = tf.data.TextLineDataset(pos_filenames, num_parallel_reads=num_parallel_reads)
    pos_ds = pos_ds.map(lambda t: (t, 1))

    neg_ds: tf.data.Dataset = tf.data.TextLineDataset(neg_filenames, num_parallel_reads=num_parallel_reads)
    neg_ds = neg_ds.map(lambda t: (t, 0))

    concat_ds: tf.data.Dataset = pos_ds.concatenate(neg_ds)

    return concat_ds, pos_cardinality + neg_cardinality


def _read_using_parallel_reads(data_location: str,
                               batch_size: int,
                               validation_split: float,
                               num_parallel_reads: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_location = os.path.join(data_location, "train/")

    logging.info(f"Reading training data from {train_location}")
    all_train_ds, cardinality = _read_positive_and_negative(train_location, num_parallel_reads=num_parallel_reads)
    num_training_samples: int = int(cardinality * validation_split)
    all_train_ds = all_train_ds.shuffle(cardinality + 1)  # essential for random validation selection!
    train_ds = all_train_ds.take(num_training_samples)
    validation_ds = all_train_ds.skip(num_training_samples)

    test_location = os.path.join(data_location, "test/")
    logging.info(f"Reading test data from {test_location}")
    test_ds, _ = _read_positive_and_negative(test_location, num_parallel_reads=num_parallel_reads)

    train_ds = train_ds.cache().batch(batch_size=batch_size)
    validation_ds = validation_ds.cache().batch(batch_size=batch_size)
    test_ds = test_ds.cache().batch(batch_size=batch_size)

    return train_ds, validation_ds, test_ds


def train_and_evaluate(data_location: str,
                       batch_size: int,
                       epochs: int,
                       job_dir: str,
                       validation_split: float,
                       max_tokens: int,
                       hidden_dim: int,
                       use_parallel_reads: bool,
                       num_parallel_reads: int,
                       num_parallel_calls: int):
    if use_parallel_reads:
        logging.info("Reading with parallel reads")
        train_ds, validation_ds, test_ds = _read_using_parallel_reads(data_location=data_location,
                                                                      batch_size=batch_size,
                                                                      validation_split=validation_split,
                                                                      num_parallel_reads=num_parallel_reads)
    else:
        logging.info("Reading with Keras utils")
        train_ds, validation_ds, test_ds = _read_data_with_keras_utils(data_location=data_location,
                                                                       batch_size=batch_size,
                                                                       validation_split=validation_split)

    vectorizer: TextVectorization = TextVectorization(ngrams=2, max_tokens=MAX_TOKENS, output_mode="multi_hot")
    vectorizer.adapt(train_ds.map(lambda x, _: x))
    train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (vectorizer(x), y),
                                             num_parallel_calls=num_parallel_calls)
    validation_ds: tf.data.Dataset = validation_ds.map(lambda x, y: (vectorizer(x), y),
                                                       num_parallel_calls=num_parallel_calls)
    test_ds: tf.data.Dataset = test_ds.map(lambda x, y: (vectorizer(x), y),
                                           num_parallel_calls=num_parallel_calls)

    model = _build_model(max_tokens, hidden_dim)

    model.summary(print_fn=logging.info)

    model.fit(train_ds.cache(), epochs=epochs, validation_data=validation_ds.cache())

    # Evaluate metrics and write to log
    loss, acc = model.evaluate(test_ds)
    logging.info(f"LOSS: {loss:.4f}")
    logging.info(f"ACCURACY: {acc:.4f}")

    # Publish metrics (for hyperparam. tuning)
    metric_tag = "kschool_accuracy"
    eval_path = os.path.join(job_dir, metric_tag)
    logging.info(f"Writing accuracy to {eval_path}")
    writer = tf.summary.create_file_writer(eval_path)
    with writer.as_default():
        tf.summary.scalar(name=metric_tag, step=epochs, data=acc)
    writer.flush()

    # Write model artifact
    model_path = os.path.join(job_dir, "saved_model")
    logging.info(f"Writing model to {model_path}")
    model.save(model_path)

    # Write text vectorizer (to avoid training-inference skew)
    vectorizer_path = os.path.join(job_dir, "text_vectorizer_weights.npy")
    logging.info(f"Writing text vectorizer to {vectorizer_path}")
    np.save(vectorizer_path, vectorizer.get_weights())


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
    parser.add_argument('--job-dir', default=None, required=True)
    parser.add_argument('--max-tokens', default=MAX_TOKENS, type=int)
    parser.add_argument('--hidden-dim', default=HIDDEN_DIM, type=int)
    parser.add_argument('--validation-split', default=VALIDATION_SPLIT, type=float)
    parser.add_argument('--num-parallel-calls', default=NUM_PARALLEL_CALLS, type=int)
    parser.add_argument('--num-parallel-reads', default=NUM_PARALLEL_READS, type=int)
    parser.add_argument('--parallel-reads', action='store_true', required=False)
    parser.add_argument('--log', default='INFO', required=False)

    args = parser.parse_args()

    loglevel = getattr(logging, args.log.upper())
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    base_job_dir = args.job_dir
    version_job_dir = os.path.join(base_job_dir, __version__, f"batch={args.batch_size}", f"epochs={args.epochs}")

    train_and_evaluate(data_location=args.data_location,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       job_dir=version_job_dir,
                       max_tokens=args.max_tokens,
                       hidden_dim=args.hidden_dim,
                       validation_split=args.validation_split,
                       num_parallel_calls=args.num_parallel_calls,
                       num_parallel_reads=args.num_parallel_reads,
                       use_parallel_reads=args.parallel_reads)
