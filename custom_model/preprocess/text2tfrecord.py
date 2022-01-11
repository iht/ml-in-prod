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

import argparse
import glob
import logging
import os
from typing import Dict

import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example


def _create_tf_example(text: str, label: int) -> Example:
    tf_example: Example = tf.train.Example(features=tf.train.Features(feature={
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
    return tf_example


def _get_filenames(data_location: str) -> Dict:
    pos_location = os.path.join(data_location, "pos/*.txt")
    neg_location = os.path.join(data_location, "neg/*.txt")
    pos_filenames = glob.glob(pos_location)
    neg_filenames = glob.glob(neg_location)

    pos_cardinality = len(pos_filenames)
    logging.info(f"Found {pos_cardinality} pos instances")
    neg_cardinality = len(neg_filenames)
    logging.info(f"Found {neg_cardinality} neg instances")

    return {'neg': neg_filenames, 'pos': pos_filenames}


def _write_tfrecords(fns_dict: Dict, output_dir: str, labels_dict: Dict):
    with tf.io.TFRecordWriter(output_dir) as writer:
        for label, filenames in fns_dict.items():
            for fn in filenames:
                with open(fn, 'r') as f:
                    data = f.read().strip()
                num_label = labels_dict[label]
                example: Example = _create_tf_example(data, num_label)
                writer.write(example.SerializeToString())


def generate_tfrecord_files(data_location: str, output_dir: str):
    labels_dict = {'neg': 0, 'pos': 1}

    # Generate train TF records
    train_tfrecord_path = os.path.join(output_dir, "imbd-train.tfrecords")
    train_location = os.path.join(data_location, "train")
    fns = _get_filenames(train_location)
    _write_tfrecords(fns, train_tfrecord_path, labels_dict)

    # Generate test TF records
    test_tfrecord_path = os.path.join(output_dir, "imbd-test.tfrecords")
    test_location = os.path.join(data_location, "test")
    fns = _get_filenames(test_location)
    _write_tfrecords(fns, test_tfrecord_path, labels_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--output-dir', required=True, default=None)

    args = parser.parse_args()

    generate_tfrecord_files(data_location=args.data_location,
                            output_dir=args.output_dir)