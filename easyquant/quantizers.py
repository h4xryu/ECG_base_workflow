"""Learnable quantizer 구현체 — LearnableScaleQuantizer, LearnableThresholdQuantizer."""

import tensorflow as tf
import tensorflow_model_optimization as tfmot

keras = tf.keras
qmod = tfmot.quantization.keras.quantizers


@keras.utils.register_keras_serializable(package="EasyQAT")
class LearnableScaleQuantizer(qmod.Quantizer):
    """학습 가능한 scale 파라미터를 사용하는 symmetric/asymmetric quantizer (STE 적용)."""

    def __init__(
        self,
        num_bits=8,
        symmetric=True,
        narrow_range=False,
        clip_ratio=6.0,
        init_scale=1.0,
        trainable_scale=True,
    ):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.clip_ratio = clip_ratio
        self.init_scale = init_scale
        self.trainable_scale = trainable_scale

    def build(self, tensor_shape, name, layer):
        del tensor_shape
        scale = layer.add_weight(
            name=f"{name}_scale",
            shape=(),
            initializer=keras.initializers.Constant(self.init_scale),
            trainable=self.trainable_scale,
        )
        return {"scale": scale}

    def __call__(self, inputs, training, weights, **kwargs):
        del training, kwargs

        scale = tf.nn.softplus(weights["scale"]) + 1e-8
        clip_value = self.clip_ratio * scale
        x = tf.clip_by_value(inputs, -clip_value, clip_value)

        if self.symmetric:
            qmin = -(2 ** (self.num_bits - 1)) + (1 if self.narrow_range else 0)
            qmax = (2 ** (self.num_bits - 1)) - 1
        else:
            qmin = 0
            qmax = (2 ** self.num_bits) - 1

        x_int = tf.round(x / scale)
        x_int = tf.clip_by_value(x_int, qmin, qmax)
        x_q = x_int * scale

        # STE
        return inputs + tf.stop_gradient(x_q - inputs)

    def get_config(self):
        return {
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
            "clip_ratio": self.clip_ratio,
            "init_scale": self.init_scale,
            "trainable_scale": self.trainable_scale,
        }


@keras.utils.register_keras_serializable(package="EasyQAT")
class LearnableThresholdQuantizer(qmod.Quantizer):
    """학습 가능한 threshold 구간을 누적합으로 생성하는 quantizer (STE 적용)."""

    def __init__(
        self,
        num_bits=4,
        init_interval=0.25,
        symmetric_output=False,
        trainable_input_scale=True,
        trainable_output_scale=True,
    ):
        self.num_bits = num_bits
        self.init_interval = init_interval
        self.symmetric_output = symmetric_output
        self.trainable_input_scale = trainable_input_scale
        self.trainable_output_scale = trainable_output_scale

    def build(self, tensor_shape, name, layer):
        del tensor_shape

        n_levels = 2 ** self.num_bits
        n_intervals = n_levels - 1

        raw_intervals = layer.add_weight(
            name=f"{name}_raw_intervals",
            shape=(n_intervals,),
            initializer=keras.initializers.Constant(self.init_interval),
            trainable=True,
        )

        input_scale = layer.add_weight(
            name=f"{name}_input_scale",
            shape=(),
            initializer="ones",
            trainable=self.trainable_input_scale,
        )

        output_scale = layer.add_weight(
            name=f"{name}_output_scale",
            shape=(),
            initializer="ones",
            trainable=self.trainable_output_scale,
        )

        return {
            "raw_intervals": raw_intervals,
            "input_scale": input_scale,
            "output_scale": output_scale,
        }

    def __call__(self, inputs, training, weights, **kwargs):
        del training, kwargs

        x = inputs * (tf.nn.softplus(weights["input_scale"]) + 1e-8)
        output_scale = tf.nn.softplus(weights["output_scale"]) + 1e-8

        intervals = tf.nn.softplus(weights["raw_intervals"]) + 1e-6
        thresholds = tf.cumsum(intervals)
        thresholds = thresholds - thresholds[0]

        max_abs = tf.stop_gradient(tf.reduce_max(tf.abs(x)) + 1e-6)
        thresholds = thresholds / (tf.reduce_max(thresholds) + 1e-6) * max_abs

        x_ex = tf.expand_dims(x, axis=-1)
        passed = tf.cast(x_ex >= thresholds, x.dtype)
        q_index = tf.reduce_sum(passed, axis=-1)

        if self.symmetric_output:
            center = (2 ** self.num_bits - 1) / 2.0
            x_q = (q_index - center) * output_scale
        else:
            x_q = q_index * output_scale

        return inputs + tf.stop_gradient(x_q - inputs)

    def get_config(self):
        return {
            "num_bits": self.num_bits,
            "init_interval": self.init_interval,
            "symmetric_output": self.symmetric_output,
            "trainable_input_scale": self.trainable_input_scale,
            "trainable_output_scale": self.trainable_output_scale,
        }
