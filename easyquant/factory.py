"""QuantizerSpec → Quantizer 인스턴스 변환 팩토리."""

from typing import Optional

import tensorflow_model_optimization as tfmot

from .quantizers import LearnableScaleQuantizer, LearnableThresholdQuantizer
from .specs import QuantizerSpec

qmod = tfmot.quantization.keras.quantizers


class QuantizerFactory:
    """등록된 kind 문자열을 보고 Quantizer 인스턴스를 생성한다."""

    def __init__(self):
        self.registry = {
            "moving_average": qmod.MovingAverageQuantizer,
            "last_value": qmod.LastValueQuantizer,
            "learnable_scale": LearnableScaleQuantizer,
            "learnable_threshold": LearnableThresholdQuantizer,
        }

    def __call__(self, spec: Optional[QuantizerSpec]):
        if spec is None or not spec.enabled:
            return None
        if spec.kind not in self.registry:
            raise KeyError(
                f"Unknown quantizer kind: '{spec.kind}'. "
                f"Available: {list(self.registry)}"
            )
        return self.registry[spec.kind](**spec.kwargs)
