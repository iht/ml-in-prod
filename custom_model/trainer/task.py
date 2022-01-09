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
import random
import sys

import tensorflow as tf
from keras import Model, activations, models, losses, metrics
from keras import layers
from keras.layers import TextVectorization
from keras.optimizer_v2.rmsprop import RMSProp
from keras.type.types import Layer
from tensorflow.keras import utils

MAX_TOKENS = 20000
HIDDEN_DIM = 16


def train_and_evaluate(data_location, batch_size, epochs, job_dir, max_tokens, hidden_dim):
    train_location = os.path.join(data_location, "train/")
    test_location = os.path.join(data_location, "test/")

    seed = random.randrange(1000)

    train_ds: tf.data.Dataset = utils.text_dataset_from_directory(train_location,
                                                                  batch_size=batch_size,
                                                                  validation_split=0.2,
                                                                  seed=seed,
                                                                  subset='training')
    validation_ds: tf.data.Dataset = utils.text_dataset_from_directory(train_location,
                                                                       batch_size=batch_size,
                                                                       validation_split=0.2,
                                                                       seed=seed,
                                                                       subset='validation')
    test_ds: tf.data.Dataset = utils.text_dataset_from_directory(test_location,
                                                                 batch_size=batch_size)

    vectorizer: TextVectorization = TextVectorization(ngrams=2, max_tokens=MAX_TOKENS, output_mode="multi_hot")
    vectorizer.adapt(train_ds.map(lambda x, _: x))
    train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=4)
    validation_ds: tf.data.Dataset = validation_ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=4)
    test_ds: tf.data.Dataset = test_ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=4)

    model = _build_model(max_tokens, hidden_dim)

    model.summary(print_fn=logging.info)

    model.fit(train_ds.cache(), epochs=epochs, validation_data=validation_ds)

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
    parser.add_argument('--log', default='INFO', required=False)

    args = parser.parse_args()

    loglevel = getattr(logging, args.log.upper())
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(data_location=args.data_location,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       job_dir=args.job_dir,
                       max_tokens=args.max_tokens,
                       hidden_dim=args.hidden_dim)
