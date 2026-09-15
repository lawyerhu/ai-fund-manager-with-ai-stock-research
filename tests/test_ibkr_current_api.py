import inspect
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.execution.ibkr import IBKRApp, IBKRConfig, IBKRPaperExecutor
from src.execution.observe import ReadOnlyBroker
from src.execution_quotes import IBKRExecutionQuoteProvider
from src.main import build_executor
from src.models import BrokerSnapshot, PositionSnapshot, TradeIntent
from src.ibkr_diagnostics import inspect_api_capabilities
from src.models import OrderRequest
from src.storage import SQLiteStore


def test_current_error_callback_preserves_error_time_and_local_receive_time():
    app = IBKRApp()
    app.error(17, 1735600000, 414, "invalid snapshot", '{"reason":"generic ticks"}')

    record = app.error_records[-1]
    assert record["request_id"] == 17
    assert record["error_time"] == 1735600000
    assert record["error_code"] == 414
    assert record["error_string"] == "invalid snapshot"
    assert record["advanced_order_reject_json"] == '{"reason":"generic ticks"}'
    assert record["local_received_at"]


def test_legacy_error_callback_remains_compatible():
    app = IBKRApp()
    app.error(17, 414, "invalid snapshot", '{"legacy":true}')

    record = app.error_records[-1]
    assert record["error_time"] is None
    assert record["error_code"] == 414
    assert record["error_string"] == "invalid snapshot"


def test_ibkr_reader_reset_race_is_contained():
    app = IBKRApp()
    app.serverVersion = lambda: None
    app.msg_queue.put(b"1\0")

    app.run()

    assert app.transport_connected is False
    assert any("connection reset" in record["error_string"] for record in app.error_records)


def test_ibkr_reset_during_reader_loop_is_contained(monkeypatch):
    from src.execution import ibkr
    app = IBKRApp()
    app.serverVersion = lambda: 200

    def reset_during_run(client):
        client.serverVersion = lambda: None
        raise TypeError("'>=' not supported between instances of 'NoneType' and 'int'")

    monkeypatch.setattr(ibkr.EClient, "run", reset_during_run)
    app.run()
    assert app.transport_connected is False
    assert any("connection reset" in record["error_string"] for record in app.error_records)


def test_ibkr_reader_does_not_hide_unrelated_typeerror(monkeypatch):
    from src.execution import ibkr
    app = IBKRApp()
    app.serverVersion = lambda: 200

    def broken_run(client):
        raise TypeError("unrelated parser error")

    monkeypatch.setattr(ibkr.EClient, "run", broken_run)
    with pytest.raises(TypeError, match="unrelated parser error"):
        app.run()


def test_current_commission_and_fees_callback_persists_actual_total_cost(tmp_path):
    class App:
        pass

    app = IBKRApp()
    app.commissionAndFeesReport(SimpleNamespace(execId="exec-1", commissionAndFees=2.34, currency="USD", realizedPNL=4.5))

    assert app.commission_values["exec-1"] == {
        "total_execution_cost": 2.34,
        "commission_and_fees": 2.34,
        "currency": "USD",
        "realized_pnl": 4.5,
    }


def test_ibkr_port_safety_uses_current_paper_defaults():
    assert IBKRConfig().port == 7947
    for port in (7496, 7946, 4001):
        with pytest.raises(ValueError, match="paper"):
            IBKRConfig(port=port).validate()
    for port in (7947, 4002):
        IBKRConfig(port=port).validate()


def test_capability_check_rejects_old_api_contract():
    class OldWrapper:
        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            pass

        def commissionReport(self, report):
            pass

    class OldClient:
        def cancelOrder(self, order_id):
            pass

    result = inspect_api_capabilities(OldWrapper, OldClient, None)

    assert result["ok"] is False
    assert any("errorTime" in item["message"] for item in result["checks"] if item["status"] == "FAIL")
    assert any("commissionAndFeesReport" in item["message"] for item in result["checks"] if item["status"] == "FAIL")
    assert any("OrderCancel" in item["message"] for item in result["checks"] if item["status"] == "FAIL")


def test_capability_check_accepts_current_api_contract():
    class CurrentWrapper:
        def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
            pass

        def commissionAndFeesReport(self, report):
            pass

    class CurrentClient:
        def cancelOrder(self, order_id, order_cancel):
            pass

    class OrderCancel:
        manualOrderIndicator = 0

    result = inspect_api_capabilities(CurrentWrapper, CurrentClient, OrderCancel)

    assert result["ok"] is True


def test_cancel_adapter_uses_current_order_cancel_and_never_business_direct_call(monkeypatch):
    class CurrentCancel:
        def __init__(self):
            self.manualOrderIndicator = None
            self.manualOrderCancelTime = None

    class App:
        next_order_id = 10
        def __init__(self):
            self.cancel_args = None
            self.order_values = {10: {"order_id": 10, "client_order_id": "cancel-me", "status": "Submitted", "filled": 0, "remaining": 1}}
        def cancelOrder(self, order_id, order_cancel):
            self.cancel_args = (order_id, order_cancel)
            self.order_values[order_id]["status"] = "Cancelled"
        def disconnect(self): pass

    from src.execution import ibkr
    monkeypatch.setattr(ibkr, "OrderCancel", CurrentCancel, raising=False)
    app = App()
    executor = IBKRPaperExecutor(IBKRConfig(), app_factory=lambda: app)
    executor.app = app
    executor.connected = True
    executor.broker_state_known = True
    executor._client_orders["cancel-me"] = 10
    executor._order_decisions["cancel-me"] = "decision"

    executor.cancel_order("cancel-me")

    assert app.cancel_args[0] == 10
    assert isinstance(app.cancel_args[1], CurrentCancel)
    assert app.cancel_args[1].manualOrderIndicator == 0


class _ReadOnlyBrokerStub:
    mode = "IBKR_PAPER"
    broker_source = "IBKR_PAPER"
    execution_mode = "PAPER"
    mutations_allowed = True

    def __init__(self, *args, **kwargs):
        self.connected = False
        self.broker_state_known = False
        self.place_calls = 0
        self.cancel_calls = 0

    def connect(self):
        self.connected = True
        self.broker_state_known = True
        return {"connected": True}

    def disconnect(self):
        self.connected = False
        self.broker_state_known = False

    def reconnect(self):
        self.disconnect()
        return self.connect()

    def position_symbols(self, timeout_seconds=None):
        return []

    def reconcile(self, prices=None, timeout_seconds=None):
        return BrokerSnapshot(equity=1000, cash=1000, positions=[], open_orders=[], source=self.mode)

    def account_summary(self):
        return {"TotalCashValue": "1000", "NetLiquidation": "1000"}

    def cash_balance(self):
        return 1000.0

    def current_positions(self):
        return []

    def positions(self):
        return []

    def current_orders(self):
        return []

    def get_execution_quote(self, symbol):
        return {"symbol": symbol.upper(), "price": 100.0, "bid": 99.0, "ask": 101.0, "mid": 100.0, "timestamp": datetime.now(timezone.utc).isoformat(), "market_status": "OPEN", "source": "IBKR", "data_type": "REALTIME"}

    def resolve_instrument(self, symbol):
        return symbol.upper()

    def order_status(self, client_order_id):
        return None

    def submit_order(self, request):
        self.place_calls += 1
        raise AssertionError("OBSERVE must not delegate submit_order")

    def await_order(self, client_order_id, timeout_seconds=None):
        return None

    def cancel_order(self, client_order_id):
        self.cancel_calls += 1
        raise AssertionError("OBSERVE must not delegate cancel_order")


def test_ibkr_paper_observe_connects_reads_state_and_uses_ibkr_quote(monkeypatch):
    import src.main as main

    created = {}

    def factory(*args, **kwargs):
        created["broker"] = _ReadOnlyBrokerStub(*args, **kwargs)
        return created["broker"]

    monkeypatch.setattr(main, "IBKRPaperExecutor", factory)
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("BROKER_SOURCE", raising=False)
    cfg = {"execution": {"mode": "OBSERVE"}, "portfolio": {"starting_equity": 1000}}
    with SQLiteStore(":memory:") as store:
        broker = build_executor(cfg, store, broker_source="IBKR_PAPER", execution_mode="OBSERVE")
        assert isinstance(broker, ReadOnlyBroker)
        broker.connect()
        assert broker.reconcile().equity == 1000
        assert broker.get_execution_quote("NVDA")["source"] == "IBKR"
        assert created["broker"].connected is True


def test_ibkr_observe_records_hypothetical_order_without_broker_mutation():
    raw = _ReadOnlyBrokerStub()
    broker = ReadOnlyBroker(raw)
    broker.connect()
    request = OrderRequest(decision_id="observe-decision", symbol="NVDA", action="BUY", quantity=2, reference_price=100)

    report = broker.submit_order(request)
    cancelled = broker.cancel_order(request.client_order_id)

    assert report.status == "SIMULATED"
    assert "WOULD_BUY" in report.message
    assert cancelled.status == "CANCELLED"
    assert raw.place_calls == 0
    assert raw.cancel_calls == 0


def test_ibkr_observe_quote_provider_uses_historical_only_while_market_closed():
    class ClosedClock:
        def is_open(self):
            return False

        def latest_completed_session(self):
            return {"date": "2026-08-28", "close": "2026-08-28T20:00:00+00:00"}

    class ObserveBroker:
        execution_mode = "OBSERVE"
        mutations_allowed = False

        def get_execution_quote(self, symbol):
            raise TimeoutError("no current streaming tick")

        def get_historical_fallback(self, symbol):
            return {"symbol": symbol, "bar_date": "2026-08-28", "close": 769.33, "source": "IBKR historical"}

    quote = IBKRExecutionQuoteProvider(ObserveBroker(), market_clock=ClosedClock()).get_quote("SPY")

    assert quote.price == pytest.approx(769.33)
    assert quote.timestamp == "2026-08-28T20:00:00+00:00"
    assert quote.market_status == "CLOSED"
    assert quote.data_type == "FROZEN"
    assert quote.source == "IBKR historical"


@pytest.mark.parametrize("market_open,mutations_allowed", [(True, False), (False, True)])
def test_ibkr_quote_provider_never_uses_historical_for_open_or_mutating_execution(market_open, mutations_allowed):
    class Clock:
        def is_open(self):
            return market_open

    class Broker:
        execution_mode = "PAPER" if mutations_allowed else "OBSERVE"
        historical_calls = 0

        def get_execution_quote(self, symbol):
            raise TimeoutError("no executable quote")

        def get_historical_fallback(self, symbol):
            self.historical_calls += 1
            return {"symbol": symbol, "bar_date": "2026-08-28", "close": 769.33}

    broker = Broker()
    broker.mutations_allowed = mutations_allowed

    with pytest.raises(TimeoutError, match="executable quote"):
        IBKRExecutionQuoteProvider(broker, market_clock=Clock()).get_quote("SPY")
    assert broker.historical_calls == 0


def test_ibkr_paper_closed_market_reconciliation_uses_historical_without_execution_quote(tmp_path):
    from src.data_provider import MockDataProvider
    from src.runner import StrategyRunner
    from src.service import BackendService

    class ClosedClock:
        def is_open(self): return False
        def latest_completed_session(self):
            return {"date": "2026-08-31", "close": "2026-08-31T20:00:00+00:00"}

    class PaperBroker:
        mode = "IBKR_PAPER"
        broker_source = "IBKR_PAPER"
        execution_mode = "PAPER"
        mutations_allowed = True
        connected = True
        broker_state_known = False
        execution_quote_calls = 0
        historical_calls = 0

        def position_symbols(self, timeout_seconds=None): return ["MPC"]
        def get_execution_quote(self, symbol):
            self.execution_quote_calls += 1
            raise TimeoutError("no current streaming tick")
        def get_historical_fallback(self, symbol):
            self.historical_calls += 1
            return {"symbol": symbol, "bar_date": "2026-08-31", "close": 374.15, "source": "IBKR historical"}
        def reconcile(self, prices=None, timeout_seconds=None):
            self.broker_state_known = True
            price = prices["MPC"]
            return BrokerSnapshot(
                equity=1000,
                cash=500,
                positions=[PositionSnapshot(symbol="MPC", quantity=1, market_price=price, market_value=price, average_cost=370)],
                source="IBKR_PAPER",
            )

    broker = PaperBroker()
    clock = ClosedClock()
    with SQLiteStore(tmp_path / "closed-reconciliation.sqlite3") as store:
        runner = StrategyRunner(
            {"portfolio": {"max_positions": 1}, "agent": {"decision_horizon_days": 20}},
            MockDataProvider(),
            executor=broker,
            store=store,
            market_clock=clock,
            quote_provider=IBKRExecutionQuoteProvider(broker, market_clock=clock),
        )

        service = BackendService({}, runner, store)
        portfolio = service.run_reconciliation_now()

        assert portfolio["current_symbol"] == "MPC"
        assert portfolio["current_quantity"] == 1
        assert broker.execution_quote_calls == 0
        assert broker.historical_calls == 1
        assert store.get_runtime("market_data_status") == "MARKET_CLOSED"
        assert store.get_runtime("market_data_source") == "IBKR historical"
        assert store.get_runtime("safe_mode") is False
        assert store.get_runtime("trading_enabled") is False

        with pytest.raises(TimeoutError, match="streaming tick"):
            runner._execution_quote("MPC")
        assert broker.execution_quote_calls == 1


def test_ibkr_paper_permission_keeps_mutating_executor_separate(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main, "IBKRPaperExecutor", _ReadOnlyBrokerStub)
    with SQLiteStore(":memory:") as store:
        broker = build_executor({"execution": {"mode": "OBSERVE"}}, store, broker_source="IBKR_PAPER", execution_mode="PAPER")

    assert isinstance(broker, _ReadOnlyBrokerStub)
    assert not isinstance(broker, ReadOnlyBroker)
    assert broker.execution_mode == "PAPER"
    assert broker.mutations_allowed is True


def test_ibkr_observe_runner_records_would_buy_without_mutating_broker(tmp_path):
    from src.data_provider import MockDataProvider
    from src.runner import StrategyRunner

    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    cfg = {
        "portfolio": {"starting_equity": 1000.0, "max_positions": 1},
        "agent": {"decision_horizon_days": 20},
        "execution": {"minimum_order_notional": 1.0, "buy_cash_safety_factor": 0.98},
        "risk": {
            "hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0,
            "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_decision_age_minutes_for_execution": 60,
            "max_gap_pct": 0.08, "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000,
            "drawdown_tiers": [], "event_risk": {},
        },
    }
    raw = _ReadOnlyBrokerStub()
    broker = ReadOnlyBroker(raw)
    intent = TradeIntent(
        action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20,
        expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01, thesis=["observe"],
        risk_factors=["observe"], invalidation_conditions=["observe"], evidence_used=["observe"],
        decision_id="ibkr-observe-decision", timestamp=datetime.now(timezone.utc).isoformat(),
    )

    with SQLiteStore(tmp_path / "ibkr-observe.sqlite3") as store:
        runner = StrategyRunner(cfg, data, executor=broker, store=store, quote_provider=IBKRExecutionQuoteProvider(broker))
        result = runner.run(intent=intent)

        assert result["state"] == "OBSERVED"
        assert "WOULD_BUY" in store.recent("executions", 1)[0]["execution_json"]
        assert raw.place_calls == 0
        assert raw.cancel_calls == 0
