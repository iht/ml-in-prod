#
#  Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
gcloud ai hp-tuning-jobs create \
  --region=europe-west4 \
  --display-name=ht-my-first-model \
  --max-trial-count=56 \
  --parallel-trial-count=8 \
  --config=./config/training_config_ht.yaml