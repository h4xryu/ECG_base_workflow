"""QATBuilder — Keras 모델에 QAT annotation을 붙이고 quantize_apply까지 수행한다."""

from typing import Sequence

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .config import EasyQuantizeConfig
from .matcher import RuleMatcher
from .quantizers import LearnableScaleQuantizer, LearnableThresholdQuantizer
from .specs import LayerRule

keras = tf.keras
qkeras = tfmot.quantization.keras


class QATBuilder:
    """rules/default_rule을 받아 Keras 모델을 QAT-ready 모델로 변환한다."""

    def __init__(self, rules: Sequence[LayerRule], default_rule: LayerRule):
        self.matcher = RuleMatcher(rules, default_rule)

    def annotate(self, layer):
        # Skip nested models and composite layers (layers that contain sub-layers)
        if isinstance(layer, keras.Model) or any(
            isinstance(l, keras.layers.Layer)
            for l in getattr(layer, '_layers', [])
        ):
            return layer

        rule = self.matcher(layer)
        if rule.skip:
            return layer

        needs_qat = any([
            rule.weight_quantizer is not None,
            rule.activation_quantizer is not None,
            rule.output_quantizer is not None,
        ])
        if not needs_qat:
            return layer

        return qkeras.quantize_annotate_layer(layer, EasyQuantizeConfig(rule))

    def build(self, model: keras.Model) -> keras.Model:
        annotated = keras.models.clone_model(model, clone_function=self.annotate)
        with qkeras.quantize_scope({
            "EasyQuantizeConfig": EasyQuantizeConfig,
            "LearnableScaleQuantizer": LearnableScaleQuantizer,
            "LearnableThresholdQuantizer": LearnableThresholdQuantizer,
        }):
            return qkeras.quantize_apply(annotated)
