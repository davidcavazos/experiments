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
from tfx.components.trainer.executor import TrainerFnArgs

from . import tensorflow_model as tfm

file_name = "pipeline.penguin_model"

inputs = {
    "culmen_length_mm": tfm.Numeric(),
    "culmen_depth_mm": tfm.Numeric(),
    "flipper_length_mm": tfm.Numeric(),
    "body_mass_g": tfm.Numeric(),
}

outputs = {
    "species": tfm.Categorical(3),
}

layers = {
    "species": [
        keras.layers.Dense(8, activation="relu"),
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
        model=lambda: tfm.linear_model(inputs, outputs, layers),
    )
