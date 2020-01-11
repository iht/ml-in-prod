"""A simple main file to showcase the template."""

import logging.config

"""
This module is an example for a single Python application with some
top level functions. The tests directory includes some unitary tests
for these functions.

This is one of two main files samples included in this
template. Please feel free to remove this, or the other
(sklearn_main.py), or adapt as you need.
"""


def add(x, y):
    """Add the given parameters.

    :param x: x value
    :type x: int
    :param y: y value
    :type y: int
    :return: the sum of x and y
    :rtype: int
    """
    return x + y


def subtract(x, y):
    """Substract the given parameters.

    :param x: x value
    :type x: int
    :param y: y value
    :type y: int
    :return: the diff of x and y
    :rtype: int
    """
    return x - y


def multiply(x, y):
    """Multiply the given parameters.

    :param x: x value
    :type x: int
    :param y: y value
    :type y: int
    :return: the multiplication of x and y
    :rtype: int
    """
    return x * y


def divide(x, y):
    """Divide the given parameters.

    :param x: x value
    :type x: int
    :param y: y value
    :type y: int
    :return: the division of x by y
    :rtype: int
    """
    return x / y


def main():
    """Entry point for your module."""
    logger = logging.getLogger()
    x1 = 2
    y1 = 3
    logger.info('Realizando suma')
    logger.debug('{x} + {y} = '.format(x=x1, y=y1) + str(add(x1, y1)))
    logger.info('Realizando resta')
    logger.debug('{x} - {y} = '.format(x=x1, y=y1) + str(subtract(x1, y1)))
    logger.info('Realizando multiplicación')
    logger.debug('{x} * {y} = '.format(x=x1, y=y1) + str(multiply(x1, y1)))
    logger.info('Realizando división')
    logger.debug('{x} / {y} = '.format(x=x1, y=y1) + str(divide(x1, y1)))
    logger.debug('TERMINADO')
    logger.critical('FAIL')


if __name__ == "__main__":
    main()
