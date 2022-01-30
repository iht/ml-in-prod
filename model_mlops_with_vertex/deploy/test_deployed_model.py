from google.cloud import aiplatform
from google.cloud.aiplatform import explain

import tensorflow as tf
import numpy as np

from keras import layers
from keras.layers import TextVectorization
from keras.models import load_model


version = '0.13+14.gb9a60f7'
endpoint_id = '6538047982476984320'

project_id = 'ihr-vertex-pipelines'
my_region = 'europe-west4'

model_location = f'gs://{project_id}/{version}/batch=8192/epochs=15/model'


def endpoint_predict(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

print('\tFetching endpoint')
endpoint = aiplatform.Endpoint(endpoint_id)

print('\tLoading Vectorization')
restored_model = load_model(model_location + '/vectorizer/')
restored_vectorizer = restored_model.layers[0]
restored_vectorizer.get_config()

# read data and prepare instances to predict
print('\tLoading test data')
testdata = 'gs://ihr-vertex-pipelines/data/prepared/test/test_data-00000-of-00001.tfrecord'
raw_dataset = tf.data.TFRecordDataset([testdata])

# the numpy types are confusing when you have to pass through JSON, so go to basic python types
n=5
print(f'\tCalling prediction endpoint with {n} examples')
instances_lists = [[float(y) for y in x.numpy()] for x in raw_dataset.map(restored_vectorizer).take(n)]
pred = endpoint_predict(project_id, my_region, instances_lists, endpoint_id)

#print(f'{pred}')
