"""QAT/PTQ 양쪽에서 공유하는 설정 데이터클래스."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class QuantizerSpec:
    """단일 quantizer의 종류와 파라미터를 담는 컨테이너."""

    kind: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LayerRule:
    """레이어 매칭 조건과 적용할 quantizer 설정을 담는 컨테이너."""

    target_types: Sequence[type] = field(default_factory=list)
    name_contains: Sequence[str] = field(default_factory=list)

    weight_quantizer: Optional[QuantizerSpec] = None
    activation_quantizer: Optional[QuantizerSpec] = None
    output_quantizer: Optional[QuantizerSpec] = None

    quantize_bias: bool = False
    skip: bool = False
