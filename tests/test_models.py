import pytest
from pydantic import ValidationError

from src.models import TradeIntent


def fields(**overrides):
    values = {
        "action": "BUY", "symbol": "NVDA", "target_weight": 0.5, "confidence": 0.8,
        "holding_period_days": 20, "expected_alpha_vs_spy": 0.02, "expected_alpha_vs_qqq": 0.01,
        "thesis": ["t"], "risk_factors": ["r"], "invalidation_conditions": ["i"], "evidence_used": ["e"],
    }
    values.update(overrides)
    return values


def test_cash_must_have_zero_weight():
    with pytest.raises(ValidationError):
        TradeIntent(**fields(action="CASH", symbol=None, target_weight=0.2))


def test_switch_requires_both_symbols():
    with pytest.raises(ValidationError):
        TradeIntent(**fields(action="SWITCH"))
    intent = TradeIntent(**fields(action="SWITCH", current_symbol="MSFT", new_symbol="NVDA"))
    assert intent.symbol == "NVDA"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        TradeIntent(**fields(unexpected="no"))
