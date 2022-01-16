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

import enum
import os.path

from typing import Dict

import apache_beam as beam
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

from schemas.imdb_instance import SCHEMA


class TypeOfDataSet(enum.Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class ReadSetTransform(beam.PTransform):
    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(SCHEMA))

    def __init__(self, data_location: str, data_set: TypeOfDataSet):
        self._data_location: str = data_location
        self._data_set: TypeOfDataSet = data_set

        if data_set == TypeOfDataSet.TRAIN:
            self._data_set_name = "train"
        elif data_set == TypeOfDataSet.TEST:
            self._data_set_name = "test"
        elif data_set == TypeOfDataSet.VALIDATION:
            self._data_set_name = "validation"
        else:
            raise RuntimeError(f"Unknown type of data set f{str(data_set)}")

        self._data_set_location = os.path.join(data_location, self._data_set_name)
        self._pos_location = os.path.join(self._data_set_location, "pos/*.txt")
        self._neg_location = os.path.join(self._data_set_location, "neg/*.txt")

        super().__init__()

    def expand(self, p: beam.Pipeline) -> beam.PCollection[Dict]:
        pos_txt: beam.PCollection[str] = p | "Read pos txt" >> beam.io.ReadFromText(self._pos_location)
        neg_txt: beam.PCollection[str] = p | "Read neg txt" >> beam.io.ReadFromText(self._neg_location)

        pos_dicts: beam.PCollection[Dict] = pos_txt | "txt2dict pos" >> beam.Map(lambda t: {'target': 1, 'text': t})
        neg_dicts: beam.PCollection[Dict] = neg_txt | "txt2dict neg" >> beam.Map(lambda t: {'target': 0, 'text': t})

        all_dicts: beam.PCollection[Dict] = (pos_dicts, neg_dicts) | "Fuse pos and neg" >> beam.Flatten()

        return all_dicts
