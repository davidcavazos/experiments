# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
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

from typing import Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tfx.components.trainer.executor import TrainerFnArgs

from . import tensorflow_model as tfm

# TODO: Features and labels could be a YAML file

file_name = "pipeline.taxi_model"

inputs = {
    "trip_miles": tfm.Numeric(),
    "pickup_latitude": tfm.Numeric(buckets=10),
    "pickup_longitude": tfm.Numeric(buckets=10),
    "trip_start_hour": tfm.Categorical(24),
    "trip_start_day": tfm.Categorical(31),
    "trip_start_month": tfm.Categorical(12),
    # "payment_type": tfm.Categorical(1000, out_of_vocab_buckets=10),
}

outputs = {
    "big_tipper": tfm.Categorical(2),
}

deep_layers = {
    "big_tipper": [
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
    ],
}


def preprocess(values: Dict[str, tf.Tensor]):
    return tfm.preprocess(values, inputs, outputs)


def train(fn_args: TrainerFnArgs):
    return tfm.train_model(
        fn_args,
        inputs,
        outputs,
        model=lambda: tfm.wide_deep_model(inputs, outputs, deep_layers),
    )
