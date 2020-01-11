"""Model to be trained."""

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations


def build_model():
  """Create and return a Tensorflow Keras model."""
  m = models.Sequential()
  m.add(layers.Input(shape=(28, 28)))
  m.add(layers.Flatten())
  m.add(layers.Dense(256, activation=activations.relu))
  m.add(layers.Dense(128, activation=activations.relu))
  m.add(layers.Dense(64, activation=activations.relu))
  m.add(layers.Dense(10, activation=activations.softmax))

  return m
