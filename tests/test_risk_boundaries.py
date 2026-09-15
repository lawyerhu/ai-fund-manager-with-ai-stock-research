from datetime import datetime, timedelta, timezone

import pytest

from src.models import PortfolioState, TradeIntent
from src.risk_engine import RiskEngine

CFG = {
    "hard_drawdown_limit": 0.25,
    "target_annualized_vol": 0.25,
    "absolute_max_weight": 1.0,
    "min_confidence_to_open": 0.55,
    "max_data_age_minutes": 30,
    "max_gap_pct": 0.08,
    "max_annualized_volatility": 1.0,
    "min_avg_dollar_volume": 500000,
    "event_risk": {"earnings_blackout_days": 3, "earnings_max_weight": 0.5, "macro_event_max_weight": 0.7, "reject_new_before_earnings": False},
    "drawdown_tiers": [
        {"drawdown": 0.10, "max_weight": 0.70},
        {"drawdown": 0.15, "max_weight": 0.50},
        {"drawdown": 0.20, "max_weight": 0.25},
        {"drawdown": 0.25, "max_weight": 0.00},
    ],
}


def fresh():
    return datetime.now(timezone.utc).isoformat()


def make_intent(action="BUY", weight=1.0, confidence=0.8):
    return TradeIntent(action=action, symbol=None if action == "CASH" else "NVDA", target_weight=0 if action == "CASH" else weight, confidence=confidence, holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01, thesis=["t"], risk_factors=["r"], invalidation_conditions=["i"], evidence_used=["e"])


def test_twenty_percent_boundary_triggers_tier_with_tolerance():
    portfolio = PortfolioState(equity=800.0000000000001, peak_equity=1000, cash=800.0000000000001)
    result = RiskEngine(CFG).evaluate(make_intent(), portfolio, 0.20, fresh())
    assert result.approved_weight == 0.25


@pytest.mark.parametrize("volatility", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_volatility_is_rejected(volatility):
    result = RiskEngine(CFG).evaluate(
        make_intent(), PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        volatility, fresh(),
    )
    assert not result.approved
    assert result.approved_weight == 0
    assert "volatility" in result.reason.lower()


def test_hard_halt_marks_risk_state_and_forces_cash():
    portfolio = PortfolioState(equity=750.0000000000001, peak_equity=1000, cash=750.0000000000001)
    result = RiskEngine(CFG).evaluate(make_intent(), portfolio, 0.20, fresh())
    assert result.forced_cash and result.risk_state == "RISK_HALTED"


def test_stale_data_is_rejected():
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    result = RiskEngine(CFG).evaluate(make_intent(), PortfolioState(equity=1000, peak_equity=1000, cash=1000), 0.20, old)
    assert not result.approved and "stale" in result.reason.lower()


def test_earnings_event_reduces_weight():
    result = RiskEngine(CFG).evaluate(make_intent(), PortfolioState(equity=1000, peak_equity=1000, cash=1000), 0.20, fresh(), stock_snapshot={"price": 100, "avg_dollar_volume": 1_000_000, "gap_pct": 0}, events={"days_to_earnings": 1})
    assert result.approved and result.approved_weight == 0.5


def test_abnormal_gap_and_unavailable_price_are_rejected():
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=1000)
    gap = RiskEngine(CFG).evaluate(make_intent(), portfolio, 0.2, fresh(), stock_snapshot={"price": 100, "avg_dollar_volume": 1_000_000, "gap_pct": 0.2})
    unavailable = RiskEngine(CFG).evaluate(make_intent(), portfolio, 0.2, fresh(), stock_snapshot={"price_available": False})
    assert not gap.approved
    assert not unavailable.approved


def test_cash_decision_is_always_allowed():
    result = RiskEngine(CFG).evaluate(make_intent("CASH"), PortfolioState(equity=500, peak_equity=1000, cash=500), None, None)
    assert result.approved and result.approved_weight == 0


def test_unknown_macro_data_uses_explicit_fail_closed_policy():
    cfg = {**CFG, "event_risk": {**CFG["event_risk"], "macro_unknown_policy": "reject_new"}}
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=1000)

    result = RiskEngine(cfg).evaluate(
        make_intent("BUY"), portfolio, 0.2, fresh(),
        stock_snapshot={"price": 100, "avg_dollar_volume": 1_000_000, "gap_pct": 0},
        events={"macro_event": None, "macro_event_supported": False},
    )

    assert not result.approved
    assert "unavailable" in result.reason.lower()


def test_switch_must_match_reconciled_current_position():
    switch = make_intent("BUY", weight=0.5).model_copy(update={"action": "SWITCH", "current_symbol": "META", "new_symbol": "NVDA"})
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=500, current_symbol="MSFT", current_quantity=5, current_weight=0.5)

    result = RiskEngine(CFG).evaluate(switch, portfolio, 0.20, fresh())

    assert result.approved is False
    assert "current_symbol" in result.reason


def test_cash_portfolio_cannot_open_stock_with_hold():
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=1000)

    result = RiskEngine(CFG).evaluate(make_intent("HOLD", weight=0.5), portfolio, 0.20, fresh())

    assert result.approved is False
    assert "HOLD requires" in result.reason


def test_hold_wrong_symbol_is_rejected():
    portfolio = PortfolioState(
        equity=1000, peak_equity=1000, cash=500,
        current_symbol="META", current_quantity=5, current_weight=0.5,
    )

    result = RiskEngine(CFG).evaluate(make_intent("HOLD", weight=0.5), portfolio, 0.20, fresh())

    assert result.approved is False
    assert "HOLD symbol" in result.reason


def test_low_confidence_cannot_open_exposure_through_hold():
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=1000)

    result = RiskEngine(CFG).evaluate(make_intent("HOLD", weight=0.5, confidence=0), portfolio, 0.20, fresh())

    assert result.approved is False
    assert result.approved_weight == 0


def test_unsupported_macro_data_applies_configured_cap():
    cfg = {**CFG, "event_risk": {**CFG["event_risk"], "macro_unknown_policy": "cap", "macro_unknown_max_weight": 0.5}}
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=1000)

    result = RiskEngine(cfg).evaluate(
        make_intent("BUY", weight=1.0), portfolio, 0.2, fresh(),
        stock_snapshot={"price": 100, "avg_dollar_volume": 1_000_000, "gap_pct": 0},
        events={"macro_event": None, "macro_event_supported": False},
    )

    assert result.approved is True
    assert result.approved_weight == 0.5
    assert result.risk_state == "REDUCED"
