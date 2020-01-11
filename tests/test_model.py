"""Tests for the model module."""

from trainer import model


def test_build_model():
  """Test the build_model function."""
  m = model.build_model()
  assert m is not None
  assert m.layers is not None
  assert m.layers != []
