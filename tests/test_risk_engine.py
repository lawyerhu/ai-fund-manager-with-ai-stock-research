from datetime import datetime, timezone
from src.models import TradeIntent, PortfolioState
from src.risk_engine import RiskEngine

CFG = {
    "hard_drawdown_limit": 0.25,
    "target_annualized_vol": 0.25,
    "absolute_max_weight": 1.0,
    "min_confidence_to_open": 0.55,
    "max_data_age_minutes": 30,
    "drawdown_tiers": [
        {"drawdown": 0.10, "max_weight": 0.70},
        {"drawdown": 0.15, "max_weight": 0.50},
        {"drawdown": 0.20, "max_weight": 0.25},
        {"drawdown": 0.25, "max_weight": 0.00},
    ],
}


def intent(weight=1.0, confidence=0.8):
    return TradeIntent(
        action="BUY", symbol="NVDA", target_weight=weight, confidence=confidence,
        holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02,
        thesis=["x"], risk_factors=["y"], invalidation_conditions=["z"], evidence_used=["e"]
    )


def fresh():
    return datetime.now(timezone.utc).isoformat()


def test_volatility_scales_weight():
    r = RiskEngine(CFG).evaluate(intent(), PortfolioState(equity=1000, peak_equity=1000, cash=1000), 0.50, fresh())
    assert r.approved
    assert r.approved_weight == 0.5


def test_drawdown_tier_caps_weight():
    p = PortfolioState(equity=800, peak_equity=1000, cash=800)
    r = RiskEngine(CFG).evaluate(intent(), p, 0.20, fresh())
    assert r.approved_weight == 0.25


def test_hard_drawdown_forces_cash():
    p = PortfolioState(equity=740, peak_equity=1000, cash=740)
    r = RiskEngine(CFG).evaluate(intent(), p, 0.20, fresh())
    assert not r.approved
    assert r.forced_cash
