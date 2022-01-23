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
from typing import List

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


def create_training_pipeline(project: str,
                             region: str,
                             job_name: str,
                             staging_bucket: str,
                             python_package_gcs_uri: str,
                             python_module_name: str,
                             module_args: List[str],
                             container_uri: str,
                             service_account: str,
                             tensorboard: str,
                             base_output_dir: str,
                             machine_type: str):
    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type": machine_type
        }},
        {"replica_count": 1},
        {"python_package_spec": {
            "executor_image_uri": container_uri,
            "package_uris": python_package_gcs_uri,
            "python_module": python_module_name,
            "args": module_args
        }}]

    python_job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir)

    ht_job = aiplatform.HyperparameterTuningJob(
        display_name='hp-test',
        custom_job=python_job,
        metric_spec={
            'kschool_accuracy': 'maximize',
        },
        parameter_spec={
            'epochs': hpt.IntegerParameterSpec(min=10, max=50, scale='linear'),
            'batch_size': hpt.DiscreteParameterSpec(
                values=[128, 256, 512, 1024, 2048],
                scale='linear')
        },
        max_trial_count=128,
        parallel_trial_count=8,
    )

    ht_job.run(service_account=service_account, tensorboard=tensorboard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--job-name', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--staging-bucket', required=True)
    parser.add_argument('--package-gcs-location', required=True)
    parser.add_argument('--python-module-name', required=True)
    parser.add_argument('--base-output-dir', required=True)
    parser.add_argument('--service-account', required=True)
    parser.add_argument('--tensorboard', required=True)
    parser.add_argument('--worker-type', required=True)
    parser.add_argument('--container-image', required=True)
    parser.add_argument('--data-location', default=None, required=True)
    parser.add_argument('--tft-location', default=None, required=True)

    args = parser.parse_args()

    module_args = [f'--data-location={args.data_location}',
                   f'--tft-location={args.tft_location}']

    create_training_pipeline(job_name=args.job_name,
                             project=args.project,
                             region=args.region,
                             staging_bucket=args.staging_bucket,
                             python_package_gcs_uri=args.package_gcs_location,
                             python_module_name=args.python_module_name,
                             module_args=module_args,
                             base_output_dir=args.base_output_dir,
                             service_account=args.service_account,
                             tensorboard=args.tensorboard,
                             machine_type=args.worker_type,
                             container_uri=args.container_image)
