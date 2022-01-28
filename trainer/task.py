"""A simple main file to showcase the template."""

import logging


def train_and_evaluate(some_arg: str):
    pass


if __name__ == "__main__":
    import argparse

    parser.add_argument('--some-arg', default=None, required=True)

    args = parser.parse_args()

    loglevel = 'INFO'
    logging.basicConfig(stream=sys.stdout, level=loglevel)

    train_and_evaluate(args.some_arg)
