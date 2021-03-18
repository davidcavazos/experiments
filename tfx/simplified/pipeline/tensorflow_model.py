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

import tensorflow_model_analysis as tfma

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import reduce
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import concatenate
import tensorflow_transform as tft
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.utils import io_utils
from tfx_bsl.tfxio import dataset_options

DEFAULT_BATCH_SIZE = 20


@dataclass
class Feature:
    def transform(self, name, values):
        raise NotImplementedError(f"{type(self)}.transform")

    def input_layer(self, name):
        raise NotImplementedError(f"{type(self)}.input_layer")

    def feature_column(self, name):
        raise NotImplementedError(f"{type(self)}.feature_column")

    def output_layer(self, name):
        raise NotImplementedError(f"{type(self)}.output_layer")

    def loss(self):
        raise NotImplementedError(f"{type(self)}.loss")

    def metrics(self):
        raise NotImplementedError(f"{type(self)}.metrics")

    def eval_model_spec(self, name):
        return tfma.ModelSpec(label_key=name)

    def eval_slicing_spec(self):
        return tfma.SlicingSpec()

    def eval_metrics_spec(self):
        raise NotImplementedError(f"{type(self)}.eval_metrics_spec")

    @staticmethod
    def fill_in_missing(x):
        # TODO: support custom default values
        if not isinstance(x, tf.sparse.SparseTensor):
            return x

        return tf.squeeze(
            tf.sparse.to_dense(
                tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
                default_value="" if x.dtype == tf.string else 0,
            ),
            axis=1,
        )


@dataclass
class Numeric(Feature):
    buckets: Optional[int] = None

    def transform(self, name, values):
        value = Feature.fill_in_missing(values[name])
        if self.buckets is None:
            return tft.scale_to_z_score(value)
        return tft.bucketize(value, self.buckets)

    def input_layer(self, name):
        if self.buckets is None:
            return keras.layers.Input(shape=(), name=name, dtype=tf.float32)
        return keras.layers.Input(shape=(), name=name, dtype=tf.int32)

    def feature_column(self, name):
        if self.buckets is None:
            return tf.feature_column.numeric_column(name, shape=())
        # TODO: the default_value should point to an out_of_vocab_bucket
        # TODO: use embedding_column if the vocab_size is too large (?)
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                name,
                num_buckets=self.buckets,
                default_value=0,
            )
        )

    def output_layer(self, name):
        if self.buckets is None:
            raise NotImplementedError(f"{type(self)}.output_layer({name})")
        if self.buckets == 2:
            return tf.keras.layers.Dense(1, activation="sigmoid", name=name)
        return keras.layers.Dense(self.buckets, activation="softmax", name=name)


@dataclass
class Categorical(Feature):
    vocab_size: int
    out_of_vocab_buckets: int = 0

    def transform(self, name, values):
        value = Feature.fill_in_missing(values[name])
        return tft.compute_and_apply_vocabulary(
            value,
            top_k=self.vocab_size,
            num_oov_buckets=self.out_of_vocab_buckets,
        )

    def input_layer(self, name):
        return keras.layers.Input(shape=(), name=name, dtype=tf.int32)

    def feature_column(self, name):
        # TODO: the default_value should point to an out_of_vocab_bucket
        # TODO: use embedding_column if the vocab_size is too large (?)
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                name,
                num_buckets=self.vocab_size + self.out_of_vocab_buckets,
                default_value=0,
            )
        )

    def output_layer(self, name):
        if self.vocab_size + self.out_of_vocab_buckets == 2:
            return tf.keras.layers.Dense(1, activation="sigmoid", name=name)
        return keras.layers.Dense(
            self.vocab_size + self.out_of_vocab_buckets,
            activation="softmax",
            name=name,
        )

    def loss(self):
        if self.vocab_size + self.out_of_vocab_buckets == 2:
            return "binary_crossentropy"
        return "sparse_categorical_crossentropy"

    def metrics(self):
        if self.vocab_size + self.out_of_vocab_buckets == 2:
            return keras.metrics.BinaryAccuracy()
        return keras.metrics.SparseCategoricalAccuracy()

    def eval_metrics_spec(self, eval_accuracy_threshold):
        class_name = (
            "BinaryAccuracy"
            if self.vocab_size + self.out_of_vocab_buckets == 2
            else "SparseCategoricalAccuracy"
        )
        return tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name=class_name,
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": eval_accuracy_threshold}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": -1e-10},
                        ),
                    ),
                )
            ]
        )


@dataclass
class Features:
    dict: Dict[str, Feature]

    def map(self, f: Callable[[str, Feature], Any]):
        return {name: f(name, feature) for name, feature in self.dict.items()}

    def filter(self, f: Callable[[str, Feature], bool]):
        return Features(
            {name: feature for name, feature in self.dict.items() if f(name, feature)}
        )

    def names(self):
        return list(self.dict.keys())

    def transform(self, values: Dict[str, tf.Tensor]):
        return self.map(lambda name, feature: feature.transform(name, values))

    def input_layers(self):
        return self.map(lambda name, feature: feature.input_layer(name))

    def feature_columns(self):
        return self.map(lambda name, feature: feature.feature_column(name)).values()

    def dense_features_layer(self):
        return keras.layers.DenseFeatures(self.feature_columns())

    def loss(self):
        return self.map(lambda _, feature: feature.loss())

    def metrics(self):
        return self.map(lambda _, feature: feature.metrics())

    def eval_model_specs(self):
        return self.map(lambda name, feature: feature.eval_model_spec(name)).values()

    def eval_slicing_specs(self):
        return self.map(lambda _, feature: feature.eval_slicing_spec()).values()

    def eval_metrics_specs(self, eval_accuracy_threshold):
        return self.map(
            lambda _, feature: feature.eval_metrics_spec(eval_accuracy_threshold)
        ).values()

    def eval_config(self, eval_accuracy_threshold=0.6):
        return tfma.EvalConfig(
            model_specs=self.eval_model_specs(),
            slicing_specs=self.eval_slicing_specs(),
            metrics_specs=self.eval_metrics_specs(eval_accuracy_threshold),
        )


def preprocess(
    values: Dict[str, tf.Tensor],
    inputs: Dict[str, Feature],
    outputs: Dict[str, Feature],
):
    return Features({**inputs, **outputs}).transform(values)


def sequential_layers(*layers: List[keras.layers.Layer]) -> keras.layers.Layer:
    return reduce(
        lambda layer, result: result(layer),
        layers[1:],
        layers[0],
    )


def linear_model(
    inputs: Dict[str, Feature],
    outputs: Dict[str, Feature],
    layers: Dict[str, List[keras.layers.Layer]],
):
    input_layers = Features(inputs).input_layers()
    output_layers = Features(outputs).map(
        lambda name, feature: sequential_layers(
            input_layers,
            Features(inputs).dense_features_layer(),
            *layers[name],
            feature.output_layer(name),
        )
    )
    return keras.Model(input_layers, output_layers)


def wide_deep_model(
    inputs: Dict[str, Feature],
    outputs: Dict[str, Feature],
    deep_layers: Dict[str, List[keras.layers.Layer]],
):
    def is_numeric(name, feature):
        return isinstance(feature, Numeric) and feature.buckets is None

    def is_categorical(name, feature):
        return not is_numeric(name, feature)

    input_layers = Features(inputs).input_layers()

    def output_layer(name, feature):
        deep = sequential_layers(
            input_layers,
            Features(inputs).filter(is_categorical).dense_features_layer(),
            *deep_layers[name],
        )
        wide = sequential_layers(
            input_layers,
            Features(inputs).filter(is_categorical).dense_features_layer(),
        )
        return sequential_layers(
            keras.layers.concatenate([deep, wide]),
            feature.output_layer(name),
        )

    output_layers = Features(outputs).map(output_layer)
    return keras.Model(input_layers, output_layers)


def train_custom_model(
    fn_args: TrainerFnArgs,
    inputs: Dict[str, Feature],
    outputs: Dict[str, Feature],
    custom_model: Callable[
        [
            TrainerFnArgs,
            tf.data.Dataset,
            tf.data.Dataset,
        ],
        Any,
    ],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Callable[[TrainerFnArgs], None]:
    if fn_args.transform_output is None:
        tf_transform_output = None
        schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())
        feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
        for output_name in Features(outputs).names():
            feature_spec.pop(output_name)

        def transform_features(serialized_tf_examples):
            return tf.io.parse_example(serialized_tf_examples, feature_spec)

    else:
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
        tft_layer = tf_transform_output.transform_features_layer()
        schema = tf_transform_output.transformed_metadata.schema
        feature_spec = tf_transform_output.raw_feature_spec()
        for output_name in Features(outputs).names():
            feature_spec.pop(output_name)

        def transform_features(serialized_tf_examples):
            parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
            return tft_layer(parsed_features)

    def build_dataset(files):
        return (
            fn_args.data_accessor.tf_dataset_factory(
                files,
                dataset_options.TensorFlowDatasetOptions(batch_size),
                schema,
            )
            .map(
                lambda batch: (
                    Features(inputs).map(lambda name, _: batch[name]),
                    Features(outputs).map(lambda name, _: batch[name]),
                )
            )
            .repeat()
        )

    distributed_strategy = tf.distribute.MirroredStrategy()
    with distributed_strategy.scope():
        custom_model(
            fn_args=fn_args,
            train_dataset=build_dataset(fn_args.train_files),
            eval_dataset=build_dataset(fn_args.eval_files),
            transform_features=transform_features,
        )


def train_model(
    fn_args: TrainerFnArgs,
    inputs: Dict[str, Feature],
    outputs: Dict[str, Feature],
    model: Callable[[], keras.Model],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Callable[[TrainerFnArgs], None]:
    def build_model(
        fn_args,
        train_dataset,
        eval_dataset,
        transform_features,
    ):
        keras_model = model()

        keras_model.compile(
            optimizer=optimizer,
            loss=Features(outputs).loss(),
            metrics=Features(outputs).metrics(),
        )
        keras_model.summary(print_fn=logging.info)

        keras_model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=fn_args.model_run_dir,
                    update_freq="batch",
                )
            ],
        )

        @tf.function
        def serve_examples(serialized_tf_examples):
            return keras_model(transform_features(serialized_tf_examples))

        keras_model.save(
            fn_args.serving_model_dir,
            save_format="tf",
            signatures={
                "serving_default": serve_examples.get_concrete_function(
                    tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
                )
            },
        )

    train_custom_model(
        fn_args=fn_args,
        inputs=inputs,
        outputs=outputs,
        custom_model=build_model,
        batch_size=batch_size,
    )
