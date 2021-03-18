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
"""Define LocalDagRunner to run the pipeline locally."""

import os
from absl import logging

from pipeline import pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import trainer_pb2


# TODO: have configs in a YAML file.


def run():
    LocalDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name="fishing-classifier",
            data_path="data",
            outputs_path="outputs",
            output_model_path="outputs/model",
            train_args=trainer_pb2.TrainArgs(num_steps=100),
            eval_args=trainer_pb2.EvalArgs(num_steps=15),
            eval_accuracy_threshold=0.6,
            metadata_connection_config=sqlite_metadata_connection_config(
                f"outputs/metadata.db"
            ),
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
