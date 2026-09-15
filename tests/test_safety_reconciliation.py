from datetime import datetime, timedelta, timezone

import pytest

from src.dashboard_data import build_dashboard_payload
from src.data_provider import MockDataProvider
from src.execution.paper import LocalPaperExecutor
from src.models import ExecutionReport, OrderRequest, PortfolioState, TradeIntent, make_client_order_id
from src.runner import StrategyRunner
from src.storage import SQLiteStore


def config(database_path):
    return {
        "portfolio": {"starting_equity": 1000.0, "database_path": str(database_path), "max_positions": 1},
        "agent": {"decision_horizon_days": 20},
        "execution": {"fractional_shares": True, "minimum_order_notional": 1.0},
        "risk": {
            "hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0,
            "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_gap_pct": 0.08,
            "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000,
            "comparison_tolerance": 1e-9, "halt_recovery_drawdown": 0.20,
            "drawdown_tiers": [
                {"drawdown": 0.10, "max_weight": 0.70}, {"drawdown": 0.15, "max_weight": 0.50},
                {"drawdown": 0.20, "max_weight": 0.25}, {"drawdown": 0.25, "max_weight": 0.0},
            ],
            "event_risk": {"earnings_blackout_days": 3, "earnings_max_weight": 0.5, "macro_event_max_weight": 0.7},
        },
    }


def test_portfolio_and_peak_survive_restart(tmp_path):
    database = tmp_path / "restart.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 100.0
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        report = broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        assert report.status == "FILLED"
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        initial = runner.reconcile_portfolio()
        assert initial.equity == pytest.approx(1000.0)

    data.rows["NVDA"]["price"] = 60.0
    with SQLiteStore(database) as restarted_store:
        restarted_broker = LocalPaperExecutor(starting_cash=1000.0, store=restarted_store)
        restarted_runner = StrategyRunner(config(database), data, executor=restarted_broker, store=restarted_store)
        restored = restarted_runner.reconcile_portfolio()

        assert restored.equity == pytest.approx(800.0)
        assert restored.cash == pytest.approx(500.0)
        assert restored.current_quantity == pytest.approx(5.0)
        assert restored.peak_equity == pytest.approx(1000.0)
        assert restored.drawdown == pytest.approx(0.20)


def intent(action, symbol, weight, decision_id, current_symbol=None, new_symbol=None):
    return TradeIntent(
        action=action, symbol=symbol, target_weight=weight, confidence=0.8,
        holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01,
        thesis=["test"], risk_factors=["test"], invalidation_conditions=["test"], evidence_used=["test"],
        decision_id=decision_id, current_symbol=current_symbol, new_symbol=new_symbol,
    )


def test_hold_at_current_target_does_not_duplicate_buy(tmp_path):
    database = tmp_path / "hold.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.5, "hold-same"))
        actual = broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "FILLED"
        assert actual.cash == pytest.approx(500.0)
        assert actual.positions[0].quantity == pytest.approx(5.0)


def test_hold_target_reduction_sells_only_delta_and_updates_cash(tmp_path):
    database = tmp_path / "reduce.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=8, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.5, "hold-reduce"))
        actual = broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "FILLED"
        assert actual.cash == pytest.approx(500.0)
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert result["portfolio"].cash == pytest.approx(500.0)
        assert result["portfolio"].current_quantity == pytest.approx(5.0)


def test_small_hold_rebalance_below_configured_minimum_does_not_submit(tmp_path):
    class CapturingBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.strategy_submit_calls = 0

        def submit_order(self, request, **kwargs):
            if request.decision_id != "seed":
                self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "minimum-rebalance.sqlite3"
    cfg = config(database)
    cfg["execution"]["minimum_rebalance_notional"] = 1000.0
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = CapturingBroker(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(cfg, data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.49, "small-rebalance"))

        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == pytest.approx(5.0)
        assert broker.strategy_submit_calls == 0


def test_hold_rebalance_above_configured_minimum_still_submits(tmp_path):
    database = tmp_path / "rebalance-above-minimum.sqlite3"
    cfg = config(database)
    cfg["portfolio"]["starting_equity"] = 10000.0
    cfg["execution"]["minimum_rebalance_notional"] = 1000.0
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=10000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=80, reference_price=100))
        runner = StrategyRunner(cfg, data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.5, "large-rebalance"))

        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == pytest.approx(50.0)


def test_cash_really_liquidates_current_position(tmp_path):
    database = tmp_path / "cash.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("CASH", None, 0.0, "cash-all"))
        actual = broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "FILLED"
        assert actual.positions == []
        assert actual.cash == pytest.approx(1000.0)
        assert result["portfolio"].current_symbol is None
        assert result["portfolio"].cash == pytest.approx(1000.0)


def test_switch_sells_old_position_before_buying_new(tmp_path):
    database = tmp_path / "switch.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    data.rows["META"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("SWITCH", "META", 0.5, "switch-ok", current_symbol="NVDA", new_symbol="META"))
        actual = broker.reconcile({"NVDA": 100.0, "META": 100.0})

        assert result["state"] == "FILLED"
        assert len(actual.positions) == 1
        assert actual.positions[0].symbol == "META"
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert actual.cash == pytest.approx(500.0)


def test_switch_partial_sell_blocks_new_buy(tmp_path):
    class PartialSellExecutor(LocalPaperExecutor):
        def submit_order(self, request, fill_ratio=1.0, fill_price=None):
            if request.action == "SELL":
                fill_ratio = 0.5
            return super().submit_order(request, fill_ratio=fill_ratio, fill_price=fill_price)

    database = tmp_path / "switch-partial.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    data.rows["META"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = PartialSellExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("SWITCH", "META", 0.5, "switch-partial", current_symbol="NVDA", new_symbol="META"))
        actual = broker.reconcile({"NVDA": 100.0, "META": 100.0})

        assert result["state"] == "PARTIALLY_FILLED"
        assert len(actual.positions) == 1
        assert actual.positions[0].symbol == "NVDA"
        assert actual.positions[0].quantity == pytest.approx(2.5)


def test_actual_twenty_percent_drawdown_triggers_twenty_five_percent_cap(tmp_path):
    database = tmp_path / "drawdown-20.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 60.0

        result = runner.run(intent=intent("HOLD", "NVDA", 1.0, "drawdown-20"))
        persisted, _ = store.latest_portfolio()
        dashboard = build_dashboard_payload(store, config(database))

        assert result["risk"].approved_weight == pytest.approx(0.25)
        assert result["risk"].risk_state == "REDUCED"
        assert result["portfolio"].drawdown == pytest.approx(0.20)
        assert result["portfolio"].current_weight == pytest.approx(0.25)
        assert result["portfolio"].risk_state == "REDUCED"
        assert persisted.risk_state == "REDUCED"
        assert dashboard["status"]["risk_state"] == "REDUCED"


def test_fifteen_percent_drawdown_persists_reduced_state(tmp_path):
    database = tmp_path / "drawdown-15.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 70.0

        result = runner.run(intent=intent("HOLD", "NVDA", 1.0, "drawdown-15"))
        persisted, _ = store.latest_portfolio()
        dashboard = build_dashboard_payload(store, config(database))

        assert result["risk"].approved_weight == pytest.approx(0.50)
        assert result["risk"].risk_state == "REDUCED"
        assert result["portfolio"].risk_state == "REDUCED"
        assert persisted.risk_state == "REDUCED"
        assert dashboard["status"]["risk_state"] == "REDUCED"


def test_final_risk_decision_state_is_persisted_after_reconciliation(tmp_path):
    database = tmp_path / "risk-decision-state.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20, "earnings_days": 1})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.8, "event-reduced"))
        persisted, _ = store.latest_portfolio()

        assert result["risk"].risk_state == "REDUCED"
        assert result["portfolio"].risk_state == "REDUCED"
        assert persisted.risk_state == "REDUCED"
        assert store.get_runtime("risk_state") == "REDUCED"


def test_risk_recovery_records_state_event(tmp_path):
    database = tmp_path / "risk-recovery.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 100.0
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 70.0
        assert runner.reconcile_portfolio().risk_state == "REDUCED"

        data.rows["NVDA"]["price"] = 100.0
        recovered = runner.reconcile_portfolio()

        events = store.risk_state_history()
        assert recovered.risk_state == "NORMAL"
        assert events[-1]["previous_state"] == "REDUCED"
        assert events[-1]["risk_state"] == "NORMAL"


def test_actual_twenty_five_percent_drawdown_liquidates_and_halts(tmp_path):
    database = tmp_path / "drawdown-25.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 50.0

        result = runner.run(intent=intent("HOLD", "NVDA", 1.0, "drawdown-25"))
        actual = broker.reconcile({"NVDA": 50.0})
        persisted, _ = store.latest_portfolio()

        assert result["risk"].risk_state == "RISK_HALTED"
        assert actual.positions == []
        assert actual.cash == pytest.approx(750.0)
        assert result["portfolio"].risk_state == "RISK_HALTED"
        assert persisted.risk_state == "RISK_HALTED"


def test_restart_during_pending_order_does_not_duplicate(tmp_path):
    database = tmp_path / "pending-restart.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    client_id = make_client_order_id("pending-decision", "target", "NVDA", "BUY")
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        pending = broker.submit_order(OrderRequest(decision_id="pending-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100, client_order_id=client_id), fill_ratio=0)
        assert pending.status == "PENDING"

    with SQLiteStore(database) as restarted_store:
        restarted_broker = LocalPaperExecutor(starting_cash=1000.0, store=restarted_store)
        runner = StrategyRunner(config(database), data, executor=restarted_broker, store=restarted_store)
        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "pending-decision"))
        final = restarted_broker.fill_order(client_id)
        actual = restarted_broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "ORDER_PENDING"
        assert final.status == "FILLED"
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert actual.cash == pytest.approx(500.0)


def test_pending_previous_decision_blocks_new_buy(tmp_path):
    database = tmp_path / "global-order-gate.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(
            OrderRequest(decision_id="old-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100),
            fill_ratio=0,
        )
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "new-decision"))

        assert result["state"] == "ERROR"
        assert {order.decision_id for order in broker.current_orders()} == {"old-decision"}
        assert result["portfolio"].current_quantity == 0


def test_partial_previous_decision_blocks_new_buy(tmp_path):
    database = tmp_path / "global-partial-order-gate.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(
            OrderRequest(decision_id="old-partial", symbol="NVDA", action="BUY", quantity=5, reference_price=100),
            fill_ratio=0.4,
        )
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "new-after-partial"))

        assert result["state"] == "ERROR"
        assert broker.reconcile({"NVDA": 100.0}).positions[0].quantity == pytest.approx(2.0)


def test_pending_sell_blocks_new_exposure(tmp_path):
    database = tmp_path / "global-sell-order-gate.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        broker.submit_order(
            OrderRequest(decision_id="old-sell", symbol="NVDA", action="SELL", quantity=2, reference_price=100),
            fill_ratio=0,
        )
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.8, "new-exposure"))

        assert result["state"] == "ERROR"
        assert broker.reconcile({"NVDA": 100.0}).positions[0].quantity == pytest.approx(5.0)


def test_resolved_old_order_permits_new_decision(tmp_path):
    database = tmp_path / "resolved-order-gate.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        old = broker.submit_order(
            OrderRequest(decision_id="old-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100),
            fill_ratio=0,
        )
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        blocked = runner.run(intent=intent("BUY", "NVDA", 0.5, "blocked-decision"))

        cancelled = broker.cancel_order(old.client_order_id)
        allowed = runner.run(intent=intent("BUY", "NVDA", 0.5, "allowed-decision"))

        assert blocked["state"] == "ERROR"
        assert cancelled.status == "CANCELLED"
        assert allowed["state"] == "FILLED"
        assert allowed["portfolio"].current_quantity == pytest.approx(5.0)


def test_restart_during_partial_fill_does_not_duplicate(tmp_path):
    database = tmp_path / "partial-restart.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    client_id = make_client_order_id("partial-decision", "target", "NVDA", "BUY")
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        partial = broker.submit_order(OrderRequest(decision_id="partial-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100, client_order_id=client_id), fill_ratio=0.4)
        assert partial.status == "PARTIALLY_FILLED"

    with SQLiteStore(database) as restarted_store:
        restarted_broker = LocalPaperExecutor(starting_cash=1000.0, store=restarted_store)
        runner = StrategyRunner(config(database), data, executor=restarted_broker, store=restarted_store)
        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "partial-decision"))
        restarted_broker.fill_order(client_id)
        actual = restarted_broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "PARTIALLY_FILLED"
        assert result["portfolio"].cash == pytest.approx(800.0)
        assert result["portfolio"].current_quantity == pytest.approx(2.0)
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert actual.cash == pytest.approx(500.0)


def test_restart_after_broker_fill_before_sqlite_update_does_not_duplicate(tmp_path):
    database = tmp_path / "filled-restart.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    client_id = make_client_order_id("filled-decision", "target", "NVDA", "BUY")
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="filled-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100, client_order_id=client_id))

    with SQLiteStore(database) as restarted_store:
        restarted_broker = LocalPaperExecutor(starting_cash=1000.0, store=restarted_store)
        runner = StrategyRunner(config(database), data, executor=restarted_broker, store=restarted_store)
        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "filled-decision"))
        actual = restarted_broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "FILLED"
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert actual.cash == pytest.approx(500.0)


def test_restart_before_broker_submit_submits_once(tmp_path):
    database = tmp_path / "created-restart.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    client_id = make_client_order_id("created-decision", "target", "NVDA", "BUY")
    request = OrderRequest(decision_id="created-decision", symbol="NVDA", action="BUY", quantity=5, reference_price=100, client_order_id=client_id)
    with SQLiteStore(database) as store:
        store.save_order(request, status="CREATED")

    with SQLiteStore(database) as restarted_store:
        restarted_broker = LocalPaperExecutor(starting_cash=1000.0, store=restarted_store)
        runner = StrategyRunner(config(database), data, executor=restarted_broker, store=restarted_store)
        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "created-decision"))
        actual = restarted_broker.reconcile({"NVDA": 100.0})

        assert result["state"] == "FILLED"
        assert actual.positions[0].quantity == pytest.approx(5.0)
        assert actual.cash == pytest.approx(500.0)


def test_broker_state_overrides_stale_sqlite_and_updates_research_context(tmp_path):
    database = tmp_path / "broker-truth.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 100.0
    with SQLiteStore(database) as store:
        stale = PortfolioState(equity=9999, peak_equity=9999, cash=9999)
        store.save_portfolio(stale, [])
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=2, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        actual = runner.reconcile_portfolio()
        persisted, persisted_positions = store.latest_portfolio()

        assert actual.equity == pytest.approx(1000.0)
        assert actual.cash == pytest.approx(800.0)
        assert actual.current_quantity == pytest.approx(2.0)
        assert persisted == actual
        assert data.portfolio()["cash"] == actual.cash
        assert data.current_positions() == persisted_positions


def test_max_positions_one_is_automatically_enforced(tmp_path):
    database = tmp_path / "max-positions.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = data.rows["META"]["price"] = 100.0
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="one", symbol="NVDA", action="BUY", quantity=1, reference_price=100))
        broker.submit_order(OrderRequest(decision_id="two", symbol="META", action="BUY", quantity=1, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        state = runner.reconcile_portfolio()

        assert state.current_symbol == "META"
        assert len(broker.reconcile({"NVDA": 100.0, "META": 100.0}).positions) == 1


def test_unconfirmed_buy_cancellation_blocks_liquidation(tmp_path):
    class CancelTimeoutExecutor(LocalPaperExecutor):
        def cancel_order(self, client_order_id):
            request = self._orders[client_order_id]
            return ExecutionReport(client_order_id=client_order_id, decision_id=request.decision_id, status="TIMEOUT", message="cancel timeout")

    database = tmp_path / "cancel-timeout.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 100.0
    with SQLiteStore(database) as store:
        broker = CancelTimeoutExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="pending", symbol="NVDA", action="BUY", quantity=1, reference_price=100), fill_ratio=0)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("CASH", None, 0.0, "cash-cancel-timeout"))

        assert result["state"] == "ERROR"
        assert len(broker.current_orders()) == 1
        assert result["portfolio"].cash == pytest.approx(1000.0)


def test_closed_market_prevents_immediate_submission(tmp_path):
    class ClosedMarket:
        def is_open(self):
            return False

    database = tmp_path / "closed-market.sqlite3"
    cfg = config(database)
    cfg["execution"]["enforce_market_hours"] = True
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(cfg, data, executor=broker, store=store, market_clock=ClosedMarket())

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "closed-market"))

        assert result["state"] == "WAITING_FOR_MARKET"
        assert broker.current_orders() == []
        assert result["portfolio"].current_quantity == 0


def test_stale_decision_prevents_submission(tmp_path):
    database = tmp_path / "stale-decision.sqlite3"
    cfg = config(database)
    cfg["risk"]["max_decision_age_minutes_for_execution"] = 30
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    stale = intent("BUY", "NVDA", 0.5, "stale-decision").model_copy(
        update={"timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}
    )
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(cfg, data, executor=broker, store=store)

        result = runner.run(intent=stale)

        assert result["state"] == "REJECTED"
        assert "decision" in result["risk"].reason.lower()
        assert broker.current_orders() == []


def test_buy_notional_respects_cash_safety_buffer(tmp_path):
    database = tmp_path / "cash-buffer.sqlite3"
    cfg = config(database)
    cfg["execution"]["buy_cash_safety_factor"] = 0.98
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(cfg, data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 1.0, "cash-buffer"))

        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == pytest.approx(9.8)
        assert result["portfolio"].cash == pytest.approx(20.0)


def test_runner_rounds_ibkr_paper_buy_quantity_down_when_fractional_shares_disabled(tmp_path):
    database = tmp_path / "ibkr-whole-shares.sqlite3"
    cfg = config(database)
    cfg["execution"].update({"fractional_shares": False, "buy_cash_safety_factor": 0.98})
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(cfg, data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 1.0, "whole-share-buy"))

        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == 9
        assert result["portfolio"].cash == pytest.approx(100.0)


def test_switch_refreshes_cash_and_target_price_after_sell(tmp_path):
    class ChangingTargetPriceData(MockDataProvider):
        def __init__(self):
            super().__init__()
            self.meta_quote_calls = 0

        def stock_snapshot(self, symbol):
            if symbol.upper() == "META":
                self.meta_quote_calls += 1
                self.rows["META"]["price"] = 100.0 if self.meta_quote_calls == 1 else 200.0
            return super().stock_snapshot(symbol)

    database = tmp_path / "switch-refresh.sqlite3"
    data = ChangingTargetPriceData()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    data.rows["META"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("SWITCH", "META", 0.5, "switch-refresh", current_symbol="NVDA", new_symbol="META"))

        assert data.meta_quote_calls >= 2
        assert result["portfolio"].current_symbol == "META"
        assert result["portfolio"].current_quantity == pytest.approx(2.5)
        assert result["portfolio"].cash == pytest.approx(500.0)


def test_execution_quote_is_separate_from_research_daily_bar(tmp_path):
    class ExecutionQuotes:
        def get_quote(self, symbol):
            return {
                "symbol": symbol, "price": 110.0, "bid": 109.9, "ask": 110.1,
                "mid": 110.0, "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_status": "OPEN", "source": "IBKR", "data_type": "REALTIME",
            }

    database = tmp_path / "separate-quote.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store, quote_provider=ExecutionQuotes())

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "separate-quote"))

        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == pytest.approx(500 / 110)
        assert result["portfolio"].invested_value == pytest.approx(500.0)


def test_stale_execution_quote_blocks_order(tmp_path):
    class StaleQuotes:
        def get_quote(self, symbol):
            return {
                "symbol": symbol, "price": 110.0, "bid": 109.9, "ask": 110.1, "mid": 110.0,
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                "market_status": "OPEN", "source": "IBKR", "data_type": "REALTIME",
            }

    database = tmp_path / "stale-execution-quote.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        result = StrategyRunner(config(database), data, broker, store, quote_provider=StaleQuotes()).run(
            intent=intent("BUY", "NVDA", 0.5, "stale-execution-quote")
        )

        assert result["state"] == "REJECTED"
        assert "stale" in result["risk"].reason.lower()
        assert broker.current_orders() == []


def test_fresh_ibkr_quote_permits_risk_evaluation(tmp_path):
    class FreshIBKRQuotes:
        def get_quote(self, symbol):
            return {
                "symbol": symbol, "price": 125.0, "bid": 124.9, "ask": 125.1, "mid": 125.0,
                "timestamp": datetime.now(timezone.utc).isoformat(), "market_status": "OPEN",
                "source": "IBKR", "data_type": "REALTIME",
            }

    database = tmp_path / "fresh-ibkr-quote.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        result = StrategyRunner(config(database), data, broker, store, quote_provider=FreshIBKRQuotes()).run(
            intent=intent("BUY", "NVDA", 0.5, "fresh-ibkr-quote")
        )

        assert result["risk"].approved is True
        assert result["state"] == "FILLED"
        assert result["portfolio"].current_quantity == pytest.approx(4.0)


def test_sell_is_blocked_when_position_disappears_before_submit(tmp_path):
    class PositionDisappearsBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.strategy_submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                quantity = self._positions.pop("NVDA", 0.0)
                self._cash += quantity * self._prices["NVDA"]
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            if request.decision_id != "seed":
                self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "position-disappears.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = PositionDisappearsBroker(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.4, "sell-after-position-change"))

        assert result["state"] == "REJECTED"
        assert broker.strategy_submit_calls == 0


def test_sell_is_blocked_when_fresh_holding_is_smaller_than_order(tmp_path):
    class PositionShrinksBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.strategy_submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                removed = self._positions["NVDA"] - 0.5
                self._positions["NVDA"] = 0.5
                self._cash += removed * self._prices["NVDA"]
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            if request.decision_id != "seed":
                self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "position-shrinks.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = PositionShrinksBroker(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("HOLD", "NVDA", 0.4, "sell-after-position-shrink"))

        assert result["state"] == "REJECTED"
        assert broker.strategy_submit_calls == 0


def test_buy_is_blocked_when_position_changes_before_submit(tmp_path):
    class PositionAppearsBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.strategy_submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                self._positions["NVDA"] = 1.0
                self._prices["NVDA"] = 100.0
                self._average_costs["NVDA"] = 100.0
                self._cash -= 100.0
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "position-appears.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = PositionAppearsBroker(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "buy-after-position-change"))

        assert result["state"] == "REJECTED"
        assert broker.strategy_submit_calls == 0


def test_buy_is_blocked_when_another_symbol_appears_before_submit(tmp_path):
    class OtherPositionAppearsBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.strategy_submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                self._positions["META"] = 1.0
                self._prices["META"] = 100.0
                self._average_costs["META"] = 100.0
                self._cash -= 100.0
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "other-position-appears.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    data.rows["META"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = OtherPositionAppearsBroker(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "buy-with-other-position"))

        assert result["state"] == "REJECTED"
        assert broker.strategy_submit_calls == 0


def test_order_is_blocked_when_final_holdings_refresh_fails(tmp_path):
    class HoldingsUnavailableBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.strategy_submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                raise RuntimeError("broker holdings unavailable")
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            self.strategy_submit_calls += 1
            return super().submit_order(request, **kwargs)

    database = tmp_path / "holdings-unavailable.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = HoldingsUnavailableBroker(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        result = runner.run(intent=intent("BUY", "NVDA", 0.5, "holdings-unavailable"))

        assert result["state"] == "REJECTED"
        assert broker.strategy_submit_calls == 0
