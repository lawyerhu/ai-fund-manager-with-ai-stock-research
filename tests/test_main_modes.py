import pytest

from src.execution.ibkr import IBKRPaperExecutor
from src.execution.observe import ObserveExecutor
from src.execution.paper import LocalPaperExecutor
from src.main import build_executor
from src.storage import SQLiteStore


def test_trading_mode_selects_only_supported_paper_executors(monkeypatch):
    monkeypatch.delenv("BROKER_SOURCE", raising=False)
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    cfg = {
        "portfolio": {"starting_equity": 1000.0},
        "execution": {"mode": "local_paper", "fractional_shares": True, "minimum_order_notional": 1.0},
    }
    with SQLiteStore(":memory:") as store:
        monkeypatch.setenv("TRADING_MODE", "local_paper")
        assert isinstance(build_executor(cfg, store), LocalPaperExecutor)

        monkeypatch.setenv("TRADING_MODE", "OBSERVE")
        assert isinstance(build_executor(cfg, store), ObserveExecutor)

        monkeypatch.setenv("TRADING_MODE", "ibkr_paper")
        assert isinstance(build_executor(cfg, store), IBKRPaperExecutor)

        monkeypatch.setenv("TRADING_MODE", "ibkr_live")
        with pytest.raises(ValueError, match="Unsupported trading mode"):
            build_executor(cfg, store)


def test_observe_mode_is_read_only_for_an_approved_runner(tmp_path):
    from src.data_provider import MockDataProvider
    from src.runner import StrategyRunner
    from test_safety_reconciliation import config
    from src.models import TradeIntent

    database = tmp_path / "observe.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        executor = ObserveExecutor(starting_cash=1000.0)
        runner = StrategyRunner(config(database), data, executor=executor, store=store)
        result = runner.run(intent=TradeIntent(action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01, thesis=["observe"], risk_factors=["observe"], invalidation_conditions=["observe"], evidence_used=["observe"], decision_id="observe-decision"))

        assert result["state"] == "OBSERVED"
        assert executor.reconcile().cash == 1000.0
        assert executor.reconcile().positions == []
        assert "not sent" in store.recent("executions", 1)[0]["execution_json"]
        events = store.runtime_events()
        event_types = {event["event_type"] for event in events}
        assert {"AI_RUN_STARTED", "BROKER_RECONCILED", "RISK_EVALUATED", "WOULD_BUY", "AI_RUN_COMPLETED"} <= event_types
        hypothetical = next(event for event in events if event["event_type"] == "WOULD_BUY")
        assert hypothetical["run_id"]
        assert hypothetical["decision_id"] == "observe-decision"
        assert '"place_order_calls": 0' in hypothetical["metadata_json"]
        assert '"cancel_order_calls": 0' in hypothetical["metadata_json"]


def test_backend_service_factory_wires_scheduler_and_paper_mode(tmp_path, monkeypatch):
    from src.main import build_backend_service

    # Explicit test mode must not be overridden by a developer's local .env.
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("BROKER_SOURCE", raising=False)
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("FIRST_RUN_MODE", raising=False)

    cfg = {
        "portfolio": {"starting_equity": 1000.0, "database_path": str(tmp_path / "service.sqlite3")},
        "execution": {"mode": "LOCAL_PAPER", "fractional_shares": True, "minimum_order_notional": 1.0},
        "data": {"provider": "mock"},
        "agent": {"max_tool_rounds": 1},
        "risk": {"hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0, "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_gap_pct": 0.08, "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000, "drawdown_tiers": [], "event_risk": {}},
        "scheduler": {"enabled": False, "weekly_day": "monday", "weekly_time": "09:30", "daily_time": "16:00", "timezone": "Asia/Shanghai", "poll_seconds": 30},
    }
    service = build_backend_service(cfg, use_llm=False, trading_mode="LOCAL_PAPER")
    try:
        result = service.start()
        assert result["trading_mode"] == "LOCAL_PAPER"
        assert result["service_status"] == "ONLINE"
        assert result["scheduler_status"] == "STOPPED"
    finally:
        service.stop()
        service.store.close()


def test_completed_first_run_observe_does_not_disable_scheduler_on_restart(tmp_path, monkeypatch):
    from src.main import build_backend_service

    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("BROKER_SOURCE", raising=False)
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.setenv("FIRST_RUN_MODE", "FIRST_RUN_OBSERVE")
    cfg = {
        "portfolio": {"starting_equity": 1000.0, "database_path": str(tmp_path / "service.sqlite3")},
        "execution": {"mode": "LOCAL_PAPER", "fractional_shares": True, "minimum_order_notional": 1.0},
        "data": {"provider": "mock"},
        "agent": {"max_tool_rounds": 1},
        "risk": {"hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0, "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_gap_pct": 0.08, "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000, "drawdown_tiers": [], "event_risk": {}},
        "scheduler": {"enabled": True, "weekly_day": "monday", "weekly_time": "09:45", "daily_time": "16:05", "timezone": "America/New_York", "poll_seconds": 30},
    }
    with SQLiteStore(cfg["portfolio"]["database_path"]) as store:
        store.set_runtime("first_run_state", "OBSERVE_RESEARCH_COMPLETED")
        service = build_backend_service(cfg, use_llm=False, store=store, trading_mode="LOCAL_PAPER")
        try:
            result = service.start()
            assert service.first_run_observe is False
            assert result["scheduler_status"] == "RUNNING"
        finally:
            service.stop()
