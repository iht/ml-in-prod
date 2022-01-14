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

from typing import Dict


from tfx.orchestration.pipeline import Pipeline
from tfx.v1.extensions import google_cloud_ai_platform

SERVICE_ACCOUNT = 'ml-in-prod-sa@ihr-vertex-pipelines.iam.gserviceaccount.com'
TENSORBOARD = ' projects/237148598933/locations/europe-west4/tensorboards/8364662251654742016'
MODEL_VERSION = '0.9+2.g860ee7f'
OUTPUT_DIRECTORY_PREFIX = 'gs://ihr-vertex-pipelines/job_dir/'


def _get_training_config(service_account: str,
                         tensorboard: str,
                         output_directory_prefix: str,
                         model_version: str,
                         epochs: int,
                         batch_size: int) -> Dict:
    job_dir = os.path.join(output_directory_prefix, model_version, f"batch={batch_size}", f"epochs={epochs}")
    job_dir_fuse = job_dir.replace("gs://", "/gcs/")

    vertex_job_spec = {
        'serviceAccount': service_account,
        'tensorboard': tensorboard,
        'baseOutputDirectory': {'outputUriPrefix': job_dir},
        'workerPoolSpecs': {'machineSpec': {'machineType': 'n1-standard-4'}, 'replicaCount': 1},
        'pythonPackageSpec': {
            'executorImageUri': 'europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest',
            'packageUris': f'gs://ihr-vertex-pipelines/dist/my_first_ml_model-{model_version}.tar.gz',
            'pythonModule': 'trainer.task',
            'args': ['--data-location=/gcs/ihr-vertex-pipelines/data/aclImdb/', f'--epochs={epochs}',
                     f'--batch-size={batch_size}',
                     f'--job-dir={job_dir_fuse}', '--parallel-reads']
        }
    }

    return vertex_job_spec


def create_vertex_pipeline(pipeline_name: str,
                           input_dir: str,
                           pipeline_root: str,
                           data_root: str,
                           module_file: str,
                           serving_model: str,
                           project_id: str,
                           region: str,
                           ) -> Pipeline:

    trainer = google_cloud_ai_platform.Trainer()
