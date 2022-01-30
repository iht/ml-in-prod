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

import logging
import os
from typing import Optional, List

import hypertune
import tensorflow as tf
import tensorflow_transform as tft
import tfx.v1 as tfx
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

    dataset = tf.data.TFRecordDataset(file_pattern)
    dataset = dataset.map(lambda r: tf.io.parse_single_example(r, transformed_feature_spec))
    dataset = dataset.map(lambda d: (d[TEXT_COLUMN], d[LABEL_COLUMN]))
    dataset = dataset.batch(batch_size)

    return dataset


def _read_tfrecords(data_location: str,
                    tft_location: str,
                    batch_size: int) -> (tf.data.Dataset, tf.data.Dataset):
    train_location = os.path.join(data_location, "train/train_data-00000-of-00001.tfrecord")
    test_location = os.path.join(data_location, "test/test_data-00000-of-00001.tfrecord")

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


def run_fn(fn_args: tfx.components.FnArgs):
    logs_dir, models_dir = get_save_paths(job_dir)

    train_ds, test_ds = _read_tfrecords(data_location=data_location,
                                        tft_location=tft_location,
                                        batch_size=batch_size)

    x_text_train = train_ds.map(lambda x, y: x)

    vectorizer: TextVectorization = layers.TextVectorization(ngrams=2, max_tokens=max_tokens, output_mode="multi_hot")
    vectorizer.adapt(x_text_train)
    train_ds = train_ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=4)
    test_ds = test_ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=4)

    model = _build_model(max_tokens, hidden_dim)

    # Show model summary in log for debugging purposes
    model.summary(print_fn=logging.info)

    logging.info(f"Writing TensorBoard logs to {logs_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1)

    model.fit(train_ds, epochs=epochs, callbacks=[tensorboard_callback])

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
    inputs: Layer = layers.Input(shape=(max_tokens,))
    x: Layer = layers.Dense(hidden_dim, activation=activations.relu)(inputs)
    x: Layer = layers.Dropout(0.5)(x)
    outputs: Layer = layers.Dense(1, activation=activations.sigmoid)(x)
    model: Model = models.Model(inputs, outputs, name="my-kschool-model")
    model.compile(optimizer=RMSProp(), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

    return model
