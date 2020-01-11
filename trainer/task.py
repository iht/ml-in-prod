"""Main module for training."""

from .model import build_model
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.keras.datasets import mnist
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.utils import to_categorical

import argparse
import logging.config
import os
import tensorflow as tf


LOGGER = logging.getLogger()


def _transform_x(x):
  return x / 255.0


def _transform_y(y):
  return to_categorical(y)


def train_and_evaluate(epochs, batch_size, job_dir):
  """Train and evaluate the model."""
  model = build_model()

  # Download data
  train_data, test_data = mnist.load_data()
  x_train, y_train = train_data
  x_test, y_test = test_data

  # Transform data
  x_train_transf = _transform_x(x_train)
  x_test_transf = _transform_x(x_test)
  y_train_transf = _transform_y(y_train)
  y_test_transf = _transform_y(y_test)

  # Fit the model
  model.compile(optimizer=optimizers.Adam(),
                loss=losses.categorical_crossentropy,
                metrics=[metrics.categorical_accuracy])
  model.fit(x_train_transf,
            y_train_transf,
            epochs=epochs,
            batch_size=batch_size)

  # Evaluate the model
  myloss, myacc = model.evaluate(x_test_transf, y_test_transf)
  LOGGER.info('Test loss value: %.4f' % myloss)
  LOGGER.info('Test accuracy: %.4f' % myacc)
  # TODO: Publish metric for hypertuning
  metric_tag = 'accKSchool'
  summ_value = Summary.Value(tag=metric_tag,
                             simple_value=myacc)
  summary = Summary(value=[summ_value])
  eval_path = os.path.join(job_dir, metric_tag)
  LOGGER.info('Writing metric to %s' % eval_path)
  summary_writer = tf.summary.FileWriter(eval_path)
  summary_writer.add_summary(summary)
  summary_writer.flush()

  # TODO: Serialize and store the model


if '__main__' == __name__:
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, required=True)
  parser.add_argument('--batch-size', type=int, required=True)
  parser.add_argument('--job-dir', default=None, required=False)

  args = parser.parse_args()
  epochs = args.epochs
  batch_size = args.batch_size
  job_dir = args.job_dir

  train_and_evaluate(epochs, batch_size, job_dir)
