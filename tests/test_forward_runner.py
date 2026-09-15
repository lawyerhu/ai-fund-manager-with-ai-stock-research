from datetime import datetime, timedelta, timezone

from src.forward_runner import ForwardPaperRunner
from src.data_provider import MockDataProvider
from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest
from src.runner import StrategyRunner
from src.storage import SQLiteStore

from test_safety_reconciliation import config


def test_weekly_forward_cycle_due_logic():
    runner = ForwardPaperRunner(None)
    now = datetime.now(timezone.utc)
    assert runner.is_due(None, now)
    assert not runner.is_due((now - timedelta(days=2)).isoformat(), now)
    assert runner.is_due((now - timedelta(days=7)).isoformat(), now)


def test_daily_risk_monitor_liquidates_at_hard_drawdown(tmp_path):
    database = tmp_path / "daily-risk.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        strategy = StrategyRunner(config(database), data, executor=broker, store=store)
        strategy.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 50.0

        result = ForwardPaperRunner(strategy).risk_monitor()
        actual = broker.reconcile({"NVDA": 50.0})

        assert result["state"] == "RISK_HALTED"
        assert result["portfolio"].risk_state == "RISK_HALTED"
        assert actual.positions == []
        assert actual.cash == 750.0
