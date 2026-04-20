"""레이어 → LayerRule 매핑 로직 (QAT/PTQ 공용)."""

from typing import Sequence

from .specs import LayerRule


class RuleMatcher:
    """rules 목록을 순서대로 검색해 레이어에 맞는 첫 번째 LayerRule을 반환한다."""

    def __init__(self, rules: Sequence[LayerRule], default_rule: LayerRule):
        self.rules = rules
        self.default_rule = default_rule

    def __call__(self, layer) -> LayerRule:
        for rule in self.rules:
            if self._match_type(layer, rule) or self._match_name(layer, rule):
                return rule
        return self.default_rule

    def _match_type(self, layer, rule: LayerRule) -> bool:
        return any(isinstance(layer, t) for t in rule.target_types)

    def _match_name(self, layer, rule: LayerRule) -> bool:
        return any(token in layer.name for token in rule.name_contains)
