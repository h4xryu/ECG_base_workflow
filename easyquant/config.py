"""TFMOT QuantizeConfig 어댑터 — LayerRule을 TFMOT API로 연결한다 (QAT 전용)."""

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .factory import QuantizerFactory
from .specs import LayerRule

keras = tf.keras
qkeras = tfmot.quantization.keras


@keras.utils.register_keras_serializable(package="EasyQAT")
class EasyQuantizeConfig(qkeras.QuantizeConfig):
    """LayerRule 하나를 TFMOT QuantizeConfig 인터페이스로 감싸는 어댑터."""

    WEIGHT_ATTRS = ["kernel", "depthwise_kernel", "pointwise_kernel", "bias"]
    ACT_ATTRS = ["activation"]

    def __init__(self, rule: LayerRule):
        self.rule = rule
        factory = QuantizerFactory()
        self.wq = factory(rule.weight_quantizer)
        self.aq = factory(rule.activation_quantizer)
        self.oq = factory(rule.output_quantizer)

    def get_weights_and_quantizers(self, layer):
        if self.rule.skip or self.wq is None:
            return []

        result = []
        for attr in self.WEIGHT_ATTRS:
            if not hasattr(layer, attr):
                continue
            if attr == "bias" and not self.rule.quantize_bias:
                continue
            value = getattr(layer, attr)
            if value is not None:
                result.append((value, self.wq))
        return result

    def get_activations_and_quantizers(self, layer):
        if self.rule.skip or self.aq is None:
            return []
        for attr in self.ACT_ATTRS:
            if hasattr(layer, attr):
                value = getattr(layer, attr)
                if value is not None:
                    return [(value, self.aq)]
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        idx = 0
        for attr in self.WEIGHT_ATTRS:
            if not hasattr(layer, attr):
                continue
            if attr == "bias" and not self.rule.quantize_bias:
                continue
            value = getattr(layer, attr)
            if value is not None:
                setattr(layer, attr, quantize_weights[idx])
                idx += 1

    def set_quantize_activations(self, layer, quantize_activations):
        if not quantize_activations:
            return
        for attr in self.ACT_ATTRS:
            if hasattr(layer, attr):
                setattr(layer, attr, quantize_activations[0])
                return

    def get_output_quantizers(self, layer):
        del layer
        return [] if self.oq is None else [self.oq]

    def get_config(self):
        # NOTE(easyquant): TFMOT serialization에서 rule 복원은 quantize_scope가 담당하므로
        # 여기서는 빈 dict를 반환해도 무방하다.
        return {}
