"""Test for simple_main."""
from trainer.sample_functions import add, divide, main, multiply, subtract


def test_add():
    """Test add."""
    assert add(1, 2) == 3


def test_subtract():
    """Test substract."""
    assert subtract(22, 2) == 20


def test_multiply():
    """Test multiply."""
    assert multiply(7, 9) == 63


def test_divide():
    """Test divide."""
    assert divide(10, 2) == 5


def test_main(caplog):
    """Capture logging output and check that is being generated."""
    main()
    assert len(caplog.records) >= 1
