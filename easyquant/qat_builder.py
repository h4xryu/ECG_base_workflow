"""QATBuilder — Keras 모델에 QAT annotation을 붙이고 quantize_apply까지 수행한다."""

from typing import Dict, Sequence, Type

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

    def __init__(
        self,
        rules: Sequence[LayerRule],
        default_rule: LayerRule,
        custom_objects: Dict[str, Type] = None,
    ):
        self.matcher = RuleMatcher(rules, default_rule)
        self.custom_objects = custom_objects or {}

    def annotate(self, layer):

        if isinstance(layer, keras.Model):
            return layer
        
        # Skip any layer that has trainable/non-trainable sublayers
        try:
            if hasattr(layer, '_layers') and len(layer._layers) > 0:
                return layer
            if hasattr(layer, 'sublayers') and len(layer.sublayers) > 0:
                return layer
        except:
            pass

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
        with keras.utils.custom_object_scope(self.custom_objects):
            annotated = keras.models.clone_model(
                model,
                clone_function=self.annotate,
            )
        scope = {
            "EasyQuantizeConfig": EasyQuantizeConfig,
            "LearnableScaleQuantizer": LearnableScaleQuantizer,
            "LearnableThresholdQuantizer": LearnableThresholdQuantizer,
            **self.custom_objects,
        }
        # Use custom_object_scope (outer) and quantize_scope (inner) to ensure
        # custom layers are available during both serialization and quantization
        with keras.utils.custom_object_scope(scope):
            with qkeras.quantize_scope(scope):
                return qkeras.quantize_apply(annotated)
