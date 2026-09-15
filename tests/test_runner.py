from src.data_provider import MockDataProvider
from src.runner import StrategyRunner
from src.storage import SQLiteStore
from src.models import PortfolioState


def test_runner_persists_end_to_end_paper_cycle():
    cfg = {
        "portfolio": {"starting_equity": 1000.0},
        "agent": {"decision_horizon_days": 20},
        "execution": {"fractional_shares": True, "minimum_order_notional": 1.0},
        "risk": {
            "hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0,
            "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_gap_pct": 0.08,
            "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000,
            "drawdown_tiers": [{"drawdown": 0.2, "max_weight": 0.25}, {"drawdown": 0.25, "max_weight": 0}],
            "event_risk": {"earnings_blackout_days": 3, "earnings_max_weight": 0.5, "macro_event_max_weight": 0.7},
        },
    }
    with SQLiteStore(":memory:") as store:
        result = StrategyRunner(cfg, MockDataProvider(), store=store).run(use_llm=False)
        assert result["state"] == "FILLED"
        assert store.recent("llm_decisions", 1)
        assert store.recent("risk_decisions", 1)
        assert store.recent("order_records", 1)[0]["status"] == "FILLED"
        assert store.benchmark_history()["SPY"]


def test_weekend_query_records_last_common_benchmark_trading_date():
    data = MockDataProvider()
    data.now = "2026-08-30T12:00:00+00:00"  # Sunday
    data.benchmark_data = lambda days=20: {
        "days": days,
        "dates": ["2026-08-27", "2026-08-28"],
        "series": {"SPY": [100.0, 101.0], "QQQ": [100.0, 102.0]},
        "as_of": data.now,
        "source": "test",
    }
    with SQLiteStore(":memory:") as store:
        runner = StrategyRunner({"portfolio": {"starting_equity": 1000}, "agent": {"decision_horizon_days": 20}}, data, store=store)
        portfolio = PortfolioState(equity=1010, peak_equity=1010, cash=1010)

        runner._record_daily_performance(portfolio)
        runner._record_daily_performance(portfolio)

        rows = store.daily_performance_history()
        assert rows == [{"date": "2026-08-28", "ai_nav": 1010.0, "spy_close": 101.0, "qqq_close": 102.0}]
