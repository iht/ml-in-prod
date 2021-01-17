"""The model that will be trainined."""
import logging.config


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations


LOGGER = logging.getLogger()


def build_model():
  """Create a neural network."""
  LOGGER.info("Building model")

  m = models.Sequential()

  m.add(layers.Input((28, 28,), name='my_input_layer'))
  m.add(layers.Flatten())
  m.add(layers.Dense(256, activation=activations.relu))
  m.add(layers.Dense(128, activation=activations.relu))
  m.add(layers.Dense(64, activation=activations.relu))
  m.add(layers.Dense(32, activation=activations.relu))
  m.add(layers.Dense(10, activation=activations.softmax))

  return m
