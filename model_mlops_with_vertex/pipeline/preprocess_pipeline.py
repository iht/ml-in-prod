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

import os

import apache_beam as beam
import tensorflow_transform.beam as tft_beam
import tensorflow as tf
from apache_beam import PCollection
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions

from tfx_bsl.coders.example_coder import RecordBatchToExamples

from .read_set import ReadSetTransform, TypeOfDataSet

MAX_TOKENS = 20000


def preprocessing_just_text_fn(inputs):
    return inputs.copy()


def run_pipeline(argv, data_location: str, output_location: str):
    opts = PipelineOptions(argv)
    gcp_opts = opts.view_as(GoogleCloudOptions)

    with beam.Pipeline(options=opts) as p, tft_beam.Context(temp_dir=gcp_opts.temp_location):
        raw_data_train = p | "Read train set" >> ReadSetTransform(data_location=data_location,
                                                                  data_set=TypeOfDataSet.TRAIN)

        transf_train_ds, transform_fn = (raw_data_train, ReadSetTransform.metadata) | "Analyz. and Transf." >> \
                                        tft_beam.AnalyzeAndTransformDataset(preprocessing_just_text_fn,
                                                                            output_record_batches=True)

        transformed_train, _ = transf_train_ds  # Ignore metadata (not required for RecordBatch)

        raw_data_test = p | "Read test set" >> ReadSetTransform(data_location=data_location,
                                                                data_set=TypeOfDataSet.TEST)

        raw_dataset_test = (raw_data_test, ReadSetTransform.metadata)

        # Apply the same transform from train set to test set
        transf_test_ds = (raw_dataset_test, transform_fn) | "Transform test" >> tft_beam.TransformDataset(
            output_record_batches=True)

        transformed_test, _ = transf_test_ds  # Ignore metadata

        output_location_train = os.path.join(output_location, 'train/train_data')
        train_tf_examples: PCollection[tf.train.Example] = transformed_train | "TrainToExamples" >> beam.FlatMapTuple(
            lambda batch, _: RecordBatchToExamples(batch))

        train_tf_examples | "Write train data" >> beam.io.tfrecordio.WriteToTFRecord(output_location_train,
                                                                                     num_shards=1,
                                                                                     file_name_suffix='.tfrecord')

        output_location_test = os.path.join(output_location, 'test/test_data')
        test_tf_examples = transformed_test | "TestToExamples" >> beam.FlatMapTuple(
            lambda batch, _: RecordBatchToExamples(batch))
        test_tf_examples | "Write test data" >> beam.io.WriteToTFRecord(output_location_test, num_shards=1,
                                                                        file_name_suffix='.tfrecord')

        transform_fn_loc = os.path.join(output_location, 'transform_fn/')
        transform_fn | "Write transform fn" >> tft_beam.WriteTransformFn(transform_fn_loc)
