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

from tfx import v1 as tfx

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

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


def _create_pipeline(pipeline_name: str,
                     pipeline_root: str,
                     data_root: str,
                     module_file: str,
                     endpoint_name: str,
                     project_id: str,
                     region: str,
                     service_account: str,
                     tensorboard: str,
                     output_directory_prefix: str,
                     model_version: str,
                     epochs: int,
                     batch_size: int) -> tfx.dsl.Pipeline:
    # Brings data into the pipeline or otherwise joins/converts training data.
    input_config = tfx.proto.Input(splits=[
        tfx.proto.example_gen_pb2.Input.Split(name='train', pattern='train/*'),
        tfx.proto.example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
    ])
    example_gen = tfx.components.ImportExampleGen(input_base=data_root, input_config=input_config)

    vertex_job_spec = _get_training_config(service_account,
                                           tensorboard,
                                           output_directory_prefix,
                                           model_version,
                                           epochs,
                                           batch_size)

    # Trains a model using Vertex AI Training.
    # NEW: We need to specify a Trainer for GCP with related configs.
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                vertex_job_spec,
        })

    # Configuration for pusher.
    vertex_serving_spec = {
        'project_id': project_id,
        'endpoint_name': endpoint_name,
        # Remaining argument is passed to aiplatform.Model.deploy()
        # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
        # for the detail.
        #
        # Machine type is the compute resource to serve prediction requests.
        # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
        # for available machine types and acccerators.
        'machine_type': 'n1-standard-4',
    }

    # Vertex AI provides pre-built containers with various configurations for
    # serving.
    # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    # for available container images.
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'

    # NEW: Pushes the model to Vertex AI.
    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
                serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
                vertex_serving_spec,
        })

    components = [
        example_gen,
        trainer,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components)


if __name__ == '__main__':
    PIPELINE_NAME = "test"
    PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=PIPELINE_DEFINITION_FILE)
    _ = runner.run(
        _create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root="",
            data_root="",
            module_file="",
            endpoint_name="",
            project_id="",
            region="",
            service_account="",
            tensorboard="",
            output_directory_prefix="",
            model_version="",
            epochs=0,
            batch_size=0))

    aiplatform.init(project="", location="")

    job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                    display_name=PIPELINE_NAME)
    job.run(sync=False)
