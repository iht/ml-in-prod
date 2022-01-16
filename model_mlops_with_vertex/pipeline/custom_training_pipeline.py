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

from typing import List, Optional, Union

from google.cloud import aiplatform


def create_training_pipeline(project: str,
                             location: str,
                             staging_bucket: str,
                             job_name: str,
                             python_package_gcs_uri: str,
                             python_module_name: str,
                             container_uri: str,
                             service_account: str,
                             tensorboard: str, base_output_dir: str, machine_type: str, replica_count: int):
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomPythonPackageTrainingJob(display_name=job_name,
                                                    python_package_gcs_uri=python_package_gcs_uri,
                                                    python_module_name=python_module_name,
                                                    container_uri=container_uri)

    job.run(service_account=service_account,
            tensorboard=tensorboard,
            base_output_dir=base_output_dir,
            machine_type=machine_type,
            replica_count=replica_count)
