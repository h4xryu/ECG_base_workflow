"""TFMOT QuantizeConfig 어댑터 — LayerRule을 TFMOT API로 연결한다 (QAT 전용)."""

import importlib

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .factory import QuantizerFactory
from .specs import LayerRule, QuantizerSpec

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

    @staticmethod
    def _serialize_quantizer_spec(spec):
        if spec is None:
            return None
        return {"kind": spec.kind, "kwargs": spec.kwargs, "enabled": spec.enabled}

    @staticmethod
    def _deserialize_quantizer_spec(d):
        if d is None:
            return None
        return QuantizerSpec(kind=d["kind"], kwargs=d.get("kwargs", {}), enabled=d.get("enabled", True))

    @staticmethod
    def _serialize_type(t):
        return f"{t.__module__}.{t.__qualname__}"

    @staticmethod
    def _deserialize_type(s):
        module_path, _, class_name = s.rpartition(".")
        return getattr(importlib.import_module(module_path), class_name)

    def get_config(self):
        return {
            "rule": {
                "target_types": [self._serialize_type(t) for t in self.rule.target_types],
                "name_contains": list(self.rule.name_contains),
                "weight_quantizer": self._serialize_quantizer_spec(self.rule.weight_quantizer),
                "activation_quantizer": self._serialize_quantizer_spec(self.rule.activation_quantizer),
                "output_quantizer": self._serialize_quantizer_spec(self.rule.output_quantizer),
                "quantize_bias": self.rule.quantize_bias,
                "skip": self.rule.skip,
            }
        }

    @classmethod
    def from_config(cls, config):
        rc = config["rule"]
        target_types = []
        for type_str in rc.get("target_types", []):
            try:
                target_types.append(cls._deserialize_type(type_str))
            except (ImportError, AttributeError):
                pass
        rule = LayerRule(
            target_types=target_types,
            name_contains=rc.get("name_contains", []),
            weight_quantizer=cls._deserialize_quantizer_spec(rc.get("weight_quantizer")),
            activation_quantizer=cls._deserialize_quantizer_spec(rc.get("activation_quantizer")),
            output_quantizer=cls._deserialize_quantizer_spec(rc.get("output_quantizer")),
            quantize_bias=rc.get("quantize_bias", False),
            skip=rc.get("skip", False),
        )
        return cls(rule)
