# This script does:
#   - Get stored model from GCS and upload to Vertex AI
#   - Create endpoint
#   - Deploy model to endpoint


from typing import Dict, Optional, Sequence, Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform import explain

import tensorflow as tf
import numpy as np

from keras import layers
from keras.layers import TextVectorization
from keras.models import load_model

project_id='ihr-vertex-pipelines'
my_region='europe-west4' # :flag-nl:
model_name='model_text_jan27'
endpoint_name='raw_model_endpoint'
model_location='gs://ihr-vertex-pipelines/0.13+14.gb9a60f7/batch=8192/epochs=15/model'


def upload_model(
    project: str,
    location: str,
    display_name: str,
    serving_container_image_uri: str,
    artifact_uri: Optional[str] = None,
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    description: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
    instance_schema_uri: Optional[str] = None,
    parameters_schema_uri: Optional[str] = None,
    prediction_schema_uri: Optional[str] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
    sync: bool = True,
):

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        instance_schema_uri=instance_schema_uri,
        parameters_schema_uri=parameters_schema_uri,
        prediction_schema_uri=prediction_schema_uri,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model


def create_endpoint(
    project: str, display_name: str, location: str,
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint.create(
        display_name=display_name, project=project, location=my_region,
    )

    print(endpoint.display_name)
    print(endpoint.resource_name)
    return endpoint


def deploy_model_with_dedicated_resources(
    project,
    location,
    model_name: str,
    machine_type: str,
    endpoint: Optional[aiplatform.Endpoint] = None,
    deployed_model_display_name: Optional[str] = None,
    traffic_percentage: Optional[int] = 0,
    traffic_split: Optional[Dict[str, int]] = None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
    metadata: Optional[Sequence[Tuple[str, str]]] = (),
    sync: bool = True,
):
    """
        model_name: A fully-qualified model resource name or model ID.
              Example: "projects/123/locations/us-central1/models/456" or
              "456" when project and location are initialized or passed.
    """

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model(model_name=model_name)

    # The explanation_metadata and explanation_parameters should only be
    # provided for a custom trained model and not an AutoML model.
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync, # Whether to execute this method synchronously
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model




# Upload model
model = upload_model(
    project=project_id,
    location=my_region,
    display_name=model_name,
    serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest',
    artifact_uri=model_location + '/saved_model/'
)

# Create an endpoint
endpoint = create_endpoint(project_id, endpoint_name, my_region)
endpoint_id = endpoint.resource_name.split('/')[-1]

# Deploy model to endpoint
deployed_model = deploy_model_with_dedicated_resources(
    project_id, 
    my_region, 
    model.resource_name, 
    'n1-standard-4', 
    endpoint, 
    traffic_percentage=100)