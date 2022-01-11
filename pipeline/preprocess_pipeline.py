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
import os
import tempfile
from typing import Dict

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam import PCollection
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

# Just an identity for the moment
from tfx_bsl.cc.tfx_bsl_extension.coders import RecordBatchToExamples


def preprocessing_fn(inputs):
    outputs = inputs.copy()
    return outputs


def run_pipeline(argv, data_location: str, output_location: str):
    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec({
            'label': tf.io.FixedLenFeature([], tf.int64),
            'text': tf.io.FixedLenFeature([], tf.string),
        }))

    train_location = os.path.join(data_location, "train")
    train_pos_location = os.path.join(train_location, "pos/*.txt")
    train_neg_location = os.path.join(train_location, "neg/*.txt")

    with beam.Pipeline(argv=argv) as p:
        with tft_beam.Context(temp_dir=tempfile.mktemp()):
            pos_txt: PCollection[str] = p | "Read train data pos" >> beam.io.ReadFromText(train_pos_location)
            neg_txt: PCollection[str] = p | "Read train data neg" >> beam.io.ReadFromText(train_neg_location)
            pos_dicts: PCollection[Dict] = pos_txt | "Pos2Example" >> beam.Map(lambda t: {'label': 1, 'text': t})
            neg_dicts: PCollection[Dict] = neg_txt | "Neg2Example" >> beam.Map(lambda t: {'label': 1, 'text': t})
            raw_data: PCollection[Dict] = (pos_dicts, neg_dicts) | beam.Flatten()

            transformed_dataset, transform_fn = (raw_data,
                                                 metadata) | "Analyz. and Transf." >> \
                                                tft_beam.AnalyzeAndTransformDataset(
                                                    preprocessing_fn,
                                                    output_record_batches=True)

            transformed_data, _ = transformed_dataset
            output_location_train = os.path.join(output_location, 'train')
            tf_examples = transformed_data | "ToExamples" >> beam.FlatMapTuple(
                lambda batch, _: RecordBatchToExamples(batch))
            tf_examples | "Write train data" >> beam.io.WriteToTFRecord(output_location_train,
                                                                        file_name_suffix='.tfrecord')

            transform_fn_loc = os.path.join(output_location, 'transform_fn/')
            transform_fn | "Write transform fn" >> tft_beam.WriteTransformFn(transform_fn_loc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--output-location', default=None, required=True)

    known_args, others = parser.parse_known_args()
    run_pipeline(others, known_args.data_location, known_args.output_location)
