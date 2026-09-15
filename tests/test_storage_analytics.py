import json
import sqlite3

import pytest

from src.analytics import compare_to_benchmarks, compute_metrics
from src.models import ExecutionReport, RiskDecision, TradeIntent
from src.storage import SQLiteStore


def intent():
    return TradeIntent(action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01, thesis=["t"], risk_factors=["r"], invalidation_conditions=["i"], evidence_used=["e"], model_name="test", decision_id="decision-storage")


def test_sqlite_persists_decision_risk_order_and_evidence():
    with SQLiteStore(":memory:") as store:
        trade_intent = intent()
        store.save_decision(trade_intent)
        store.save_evidence(trade_intent.decision_id, [{"tool": "get_universe", "arguments": {}, "result": [{"symbol": "NVDA"}]}])
        store.save_risk(trade_intent.decision_id, RiskDecision(approved=True, requested_weight=0.5, approved_weight=0.4, reason="ok"))
        store.save_state("FILLED", "test", trade_intent.decision_id)
        assert store.recent("llm_decisions", 1)[0]["decision_id"] == "decision-storage"
        assert store.recent("risk_decisions", 1)[0]["decision_id"] == "decision-storage"
        assert store.latest_state() == "FILLED"


def test_metrics_and_benchmark_alpha():
    metrics = compute_metrics([100, 105, 102, 110])
    comparison = compare_to_benchmarks([100, 105, 102, 110], [100, 102, 104, 106], [100, 103, 106, 109])
    assert metrics["total_return"] == pytest.approx(0.1)
    assert comparison["excess_return_vs_spy"] is not None
    assert "alpha_vs_spy" not in comparison


def test_sqlite_migrates_usage_and_broker_id_columns(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE llm_decisions (
            decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, model_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL, decision_json TEXT NOT NULL, latency_ms REAL,
            input_tokens INTEGER, output_tokens INTEGER, estimated_cost REAL
        );
        CREATE TABLE order_records (
            client_order_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL, order_json TEXT NOT NULL, status TEXT NOT NULL
        );
        CREATE TABLE model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER,
            output_tokens INTEGER, latency_ms REAL, estimated_cost REAL
        );
    """)
    connection.close()

    with SQLiteStore(database) as store:
        decision_columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(llm_decisions)")}
        order_columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(order_records)")}
        usage_columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(model_usage)")}

    assert {"cached_tokens", "reasoning_tokens"} <= decision_columns
    assert {"broker_order_id", "perm_id"} <= order_columns
    assert {"cached_tokens", "reasoning_tokens"} <= usage_columns


def test_daily_performance_observations_use_synchronized_dates():
    with SQLiteStore(":memory:") as store:
        store.save_daily_performance("2026-08-28", 1000, 640, 570)
        store.save_daily_performance("2026-08-29", 1010, 646, 581)
        store.save_daily_performance("2026-08-29", 1012, 647, 582)

        rows = store.daily_performance_history()

        assert [row["date"] for row in rows] == ["2026-08-28", "2026-08-29"]
        assert rows[-1] == {"date": "2026-08-29", "ai_nav": 1012.0, "spy_close": 647.0, "qqq_close": 582.0}


def test_execution_commission_is_persisted():
    with SQLiteStore(":memory:") as store:
        report = ExecutionReport(
            client_order_id="commission-order", decision_id="commission-decision", status="FILLED",
            filled_quantity=2, average_price=100, commission=1.25, fees=0.10, slippage=0.40,
        )
        store.save_execution(report)

        stored = json.loads(store.recent("executions", 1)[0]["execution_json"])

        assert stored["commission"] == 1.25
        assert stored["fees"] == 0.10
        assert stored["slippage"] == 0.40


def test_purge_old_runtime_events(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    for i in range(1500):
        store.save_runtime_event(
            component="test",
            event_type="INFO",
            message=f"event {i}",
            run_id="test-run",
        )
    # min_keep_rows=1000, keep_days=14 -> very recent events won't be deleted
    pruned = store.purge_old_runtime_events(keep_days=14, min_keep_rows=1000)
    assert pruned == 0
    # keep_days=0 with min_keep_rows=1000 should prune 500
    pruned = store.purge_old_runtime_events(keep_days=0, min_keep_rows=1000)
    assert pruned == 500
    events = store.runtime_events(limit=2000)
    assert len(events) == 1000
