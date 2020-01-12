"""Test for build_model."""

from trainer.model import build_model


def test_build_model():
  """Test the build_model function."""
  m = build_model()

  # If the model changes, let's make the test fail,
  # so the developer is aware of this test
  assert len(m.layers) == 6

  # Check the input shape to the model
  assert m.layers[0].input_shape == (None, 28, 28)

  # Check the output shape to the model
  assert m.layers[-1].output_shape == (None, 10)
