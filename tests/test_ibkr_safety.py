import threading
import time
from types import SimpleNamespace

import pytest

from src.execution.ibkr import IBKRApp, IBKRConfig, IBKRPaperExecutor
from src.models import OrderRequest
from src.storage import SQLiteStore


class FakeIBKRApp:
    def __init__(self, ready=False):
        self.next_order_id = 10
        self.next_valid_id_event = threading.Event()
        self.managed_accounts_event = threading.Event()
        self.managed_accounts = ["DU123456"]
        self.account_summary_end_event = threading.Event()
        self.positions_end_event = threading.Event()
        self.open_orders_end_event = threading.Event()
        self.executions_end_event = threading.Event()
        self.account_values = {}
        self.position_values = {}
        self.order_values = {}
        self.execution_values = []
        self.request_errors = {}
        self.request_warnings = {}
        self.error_records = []
        self.state_callback = None
        self.contract_detail_values = {}
        self.contract_detail_events = {}
        self.disconnect_calls = 0
        if ready:
            self.next_valid_id_event.set()
            self.managed_accounts_event.set()

    def connect(self, host, port, client_id):
        return None

    def run(self):
        return None

    def disconnect(self):
        self.disconnect_calls += 1

    def reqManagedAccts(self):
        return None

    def reqAllOpenOrders(self):
        return self.reqOpenOrders()

    def cancelAccountSummary(self, *_):
        return None

    def cancelPositions(self):
        return None

    def reqContractDetails(self, request_id, query_contract):
        symbol = query_contract if isinstance(query_contract, str) else query_contract.symbol
        contract = SimpleNamespace(
            symbol=symbol, localSymbol=symbol, conId=1000 + request_id,
            primaryExchange="NASDAQ", currency="USD", secType="STK",
            exchange="SMART", validExchanges="SMART,NASDAQ",
        )
        self.contract_detail_values[request_id] = [SimpleNamespace(contract=contract)]
        self.contract_detail_events[request_id].set()

    def cancelContractDetails(self, request_id):
        return None

    def error(self, request_id, error_code, error_string, advanced_order_reject_json=""):
        record = {
            "request_id": request_id, "error_code": error_code,
            "error_string": error_string,
            "advanced_order_reject_json": advanced_order_reject_json,
        }
        self.request_errors[request_id] = record
        self.error_records.append(record)
        if request_id in getattr(self, "quote_events", {}):
            self.quote_events[request_id].set()
        if request_id in getattr(self, "contract_detail_events", {}):
            self.contract_detail_events[request_id].set()
        if request_id in self.order_values:
            self.order_values[request_id].update({
                "status": "Inactive" if error_code == 201 else "Error",
                "error_code": error_code,
                "error_string": error_string,
                "advanced_reject_reason": advanced_order_reject_json,
            })
        if self.state_callback:
            self.state_callback(record)


def test_ibkr_readiness_timeout_fails_closed():
    executor = IBKRPaperExecutor(
        IBKRConfig(port=7497, readiness_timeout_seconds=0.01),
        app_factory=lambda: FakeIBKRApp(ready=False),
    )

    with pytest.raises(TimeoutError, match="ready"):
        executor.connect()
    assert not executor.connected


def test_connect_closes_stale_api_session_before_reusing_client_id():
    stale = FakeIBKRApp(ready=False)
    replacement = FakeIBKRApp(ready=True)
    executor = IBKRPaperExecutor(
        IBKRConfig(port=7497, readiness_timeout_seconds=0.01),
        app_factory=lambda: replacement,
    )
    executor.app = stale
    executor.connected = False

    executor.connect()

    assert stale.disconnect_calls == 1
    assert executor.app is replacement
    assert executor.connected is True


def test_unknown_ibkr_account_state_blocks_order():
    executor = IBKRPaperExecutor(
        IBKRConfig(port=7497, readiness_timeout_seconds=0.01),
        app_factory=lambda: FakeIBKRApp(ready=True),
    )
    executor.connect()

    report = executor.submit_order(OrderRequest(decision_id="d1", symbol="NVDA", action="BUY", quantity=1, reference_price=100))

    assert report.status == "ERROR"
    assert "reconcil" in report.message.lower()


def test_ibkr_live_account_identifier_is_blocked():
    app = FakeIBKRApp(ready=True)
    app.managed_accounts = ["U123456"]
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app)

    with pytest.raises(RuntimeError, match="paper account"):
        executor.connect()
    assert not executor.connected


def test_ibkr_config_rejects_invalid_connection_values():
    with pytest.raises(ValueError, match="port"):
        IBKRConfig(port=0).validate()
    with pytest.raises(ValueError, match="client_id"):
        IBKRConfig(client_id=-1).validate()


def test_ibkr_negative_position_fails_reconciliation_closed():
    class ShortPositionApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)

        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self):
            self.position_values = {"NVDA": {"symbol": "NVDA", "quantity": -1.0, "average_cost": 100.0}}
            self.positions_end_event.set()

        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=ShortPositionApp,
    )
    executor.connect()

    with pytest.raises(RuntimeError, match="short positions"):
        executor.position_symbols()


def test_ibkr_option_position_fails_reconciliation_closed():
    class OptionPositionApp(FakeIBKRApp):
        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self):
            self.position_values = {
                "AAPL-option": {
                    "symbol": "AAPL", "sec_type": "OPT", "currency": "USD",
                    "exchange": "SMART", "quantity": 1.0, "average_cost": 100.0,
                }
            }
            self.positions_end_event.set()

        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=lambda: OptionPositionApp(ready=True),
    )
    executor.connect()

    with pytest.raises(RuntimeError, match="unsupported security"):
        executor.position_symbols()


def reconciled_executor_with_positions(position_values):
    class PositionApp(FakeIBKRApp):
        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self):
            self.position_values = position_values
            self.positions_end_event.set()

        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=lambda: PositionApp(ready=True),
    )
    executor.connect()
    return executor


def valid_stock(symbol="NVDA", currency="USD"):
    return {
        "symbol": symbol, "sec_type": "STK", "currency": currency,
        "exchange": "SMART", "quantity": 1.0, "average_cost": 100.0,
    }


def test_ibkr_future_position_fails_reconciliation_closed():
    future = {**valid_stock("ES"), "sec_type": "FUT"}
    executor = reconciled_executor_with_positions({"ES-future": future})

    with pytest.raises(RuntimeError, match="unsupported security"):
        executor.position_symbols()


def test_ibkr_non_usd_position_fails_reconciliation_closed():
    executor = reconciled_executor_with_positions({"SAP": valid_stock("SAP", "EUR")})

    with pytest.raises(RuntimeError, match="unsupported currency"):
        executor.position_symbols()


def test_ibkr_multiple_long_positions_fail_reconciliation_closed():
    executor = reconciled_executor_with_positions({"NVDA": valid_stock("NVDA"), "META": valid_stock("META")})

    with pytest.raises(RuntimeError, match="multiple long positions"):
        executor.position_symbols()


def test_ibkr_unknown_contract_fails_reconciliation_closed():
    executor = reconciled_executor_with_positions({"unknown": {"symbol": "NVDA", "quantity": 1.0, "average_cost": 100.0}})

    with pytest.raises(RuntimeError, match="unknown contract"):
        executor.position_symbols()


def test_ibkr_reconciliation_waits_for_complete_broker_snapshot():
    class ReconciledApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)

        def reqAccountSummary(self, req_id, group, tags):
            self.account_values = {"TotalCashValue": "500", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self):
            self.position_values = {
                "NVDA": {
                    "symbol": "NVDA", "sec_type": "STK", "currency": "USD",
                    "exchange": "SMART", "quantity": 5.0, "average_cost": 90.0,
                }
            }
            self.positions_end_event.set()

        def reqOpenOrders(self):
            self.order_values = {}
            self.open_orders_end_event.set()

        def reqExecutions(self, req_id, execution_filter):
            self.execution_values = []
            self.executions_end_event.set()

    executor = IBKRPaperExecutor(
        IBKRConfig(port=7497, readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=ReconciledApp,
    )
    executor.connect()

    snapshot = executor.reconcile({"NVDA": 100.0})

    assert executor.broker_state_known
    assert snapshot.equity == pytest.approx(1000.0)
    assert snapshot.cash == pytest.approx(500.0)
    assert snapshot.positions[0].symbol == "NVDA"
    assert snapshot.positions[0].market_value == pytest.approx(500.0)


def test_ibkr_reconciliation_timeout_after_readiness_fails_closed():
    class NeverCompletesApp(FakeIBKRApp):
        def reqAccountSummary(self, *_): pass
        def reqPositions(self): pass
        def reqOpenOrders(self): pass
        def reqExecutions(self, *_): pass

    executor = IBKRPaperExecutor(
        IBKRConfig(port=7497, readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.01),
        app_factory=lambda: NeverCompletesApp(ready=True),
    )
    executor.connect()

    with pytest.raises(TimeoutError, match="account summary"):
        executor.position_symbols()
    assert not executor.broker_state_known


def test_ibkr_recovers_exact_client_order_and_perm_id_from_execution(tmp_path):
    client_order_id = "client-order-exact"

    class RecoveredApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)

        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self): self.positions_end_event.set()
        def reqOpenOrders(self): self.open_orders_end_event.set()

        def reqExecutions(self, *_):
            self.execution_values = [{"order_id": 77, "perm_id": 9001, "client_order_id": client_order_id, "symbol": "NVDA", "shares": 2.0, "price": 100.0}]
            self.executions_end_event.set()

    with SQLiteStore(tmp_path / "ibkr.sqlite3") as store:
        request = OrderRequest(decision_id="decision-exact", symbol="NVDA", action="BUY", quantity=2, reference_price=100, client_order_id=client_order_id)
        store.save_order(request)
        executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=RecoveredApp, store=store)
        executor.connect()
        assert executor.position_symbols() == []
        executor.reconcile({})

        report = executor.order_status(client_order_id)
        store.save_execution(report)
        stored = store.order_by_client_id(client_order_id)

        assert report.decision_id == "decision-exact"
        assert report.status == "FILLED"
        assert report.broker_order_id == 77
        assert report.perm_id == 9001
        assert stored["broker_order_id"] == 77
        assert stored["perm_id"] == 9001


def test_ibkr_allows_two_exact_order_ids_for_switch_decision(monkeypatch):
    class SubmitApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.placed = []

        def placeOrder(self, order_id, contract, order):
            self.placed.append((order_id, contract, order))

    app = SubmitApp()
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app)
    executor.connect()
    executor.broker_state_known = True
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)
    monkeypatch.setattr("src.execution.ibkr.market_order", lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity, orderRef=None))

    sell = executor.submit_order(OrderRequest(decision_id="switch", symbol="NVDA", action="SELL", quantity=1, reference_price=100, client_order_id="switch-sell"))
    buy = executor.submit_order(OrderRequest(decision_id="switch", symbol="META", action="BUY", quantity=1, reference_price=100, client_order_id="switch-buy"))

    assert sell.status == buy.status == "PENDING"
    assert [item[2].orderRef for item in app.placed] == ["switch-sell", "switch-buy"]
    assert [item[0] for item in app.placed] == [10, 11]


def test_req_all_open_orders_detects_foreign_api_client_order():
    class ForeignOrderApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.all_open_order_requests = 0

        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def reqPositions(self): self.positions_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()
        def reqOpenOrders(self): pytest.fail("client-only reqOpenOrders must not be used")

        def reqAllOpenOrders(self):
            self.all_open_order_requests += 1
            self.order_values = {
                99: {"order_id": 99, "client_order_id": "foreign-order", "client_id": 7, "symbol": "AAPL", "action": "BUY", "status": "Submitted"}
            }
            self.open_orders_end_event.set()

    app = ForeignOrderApp()
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=lambda: app,
    )
    executor.connect()

    with pytest.raises(RuntimeError, match="UNKNOWN BROKER OPEN ORDER"):
        executor.position_symbols()
    assert app.all_open_order_requests == 1


def test_repeated_reconciliation_does_not_leak_subscriptions():
    class SubscriptionApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.account_starts = self.account_cancels = 0
            self.position_starts = self.position_cancels = 0

        def reqAccountSummary(self, *_):
            self.account_starts += 1
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()

        def cancelAccountSummary(self, *_): self.account_cancels += 1
        def reqPositions(self):
            self.position_starts += 1
            self.positions_end_event.set()
        def cancelPositions(self): self.position_cancels += 1
        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    app = SubscriptionApp()
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=lambda: app,
    )
    executor.connect()

    for _ in range(1000):
        assert executor.position_symbols() == []
        assert executor.reconcile({}).positions == []

    assert (app.account_starts, app.account_cancels) == (1000, 1000)
    assert (app.position_starts, app.position_cancels) == (1000, 1000)
    assert app.order_values == {}
    assert app.execution_values == []


def test_reconnect_restores_subscription_and_broker_state():
    class ReconnectApp(FakeIBKRApp):
        def __init__(self, cash):
            super().__init__(ready=True)
            self.cash = cash
            self.position_requests = 0

        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": str(self.cash), "NetLiquidation": str(self.cash)}
            self.account_summary_end_event.set()
        def reqPositions(self):
            self.position_requests += 1
            self.positions_end_event.set()
        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    apps = iter([ReconnectApp(1000), ReconnectApp(900)])
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, reconciliation_timeout_seconds=0.05),
        app_factory=lambda: next(apps),
    )
    executor.connect()
    first = executor.reconcile({})
    executor.reconnect()
    assert executor.broker_state_known is False

    second = executor.reconcile({})

    assert first.cash == 1000
    assert second.cash == 900
    assert executor.app.position_requests == 1


def test_ibkr_commission_callback_is_attached_to_execution(tmp_path):
    client_order_id = "commission-client-order"

    class CommissionApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.commission_values = {"exec-1": {"commission": 1.23, "currency": "USD"}}

        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()
        def reqPositions(self): self.positions_end_event.set()
        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_):
            self.commission_values = {"exec-1": {"commission": 1.23, "currency": "USD"}}
            self.execution_values = [{
                "order_id": 77, "perm_id": 9001, "client_order_id": client_order_id,
                "symbol": "NVDA", "exec_id": "exec-1", "shares": 2.0, "price": 100.0,
            }]
            self.executions_end_event.set()

    with SQLiteStore(tmp_path / "ibkr-commission.sqlite3") as store:
        request = OrderRequest(decision_id="commission-decision", symbol="NVDA", action="BUY", quantity=2, reference_price=100, client_order_id=client_order_id)
        store.save_order(request)
        executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=CommissionApp, store=store)
        executor.connect()
        executor.reconcile({})

        report = executor.order_status(client_order_id)

        assert report.status == "FILLED"
        assert report.commission == pytest.approx(1.23)


class QuoteApp(FakeIBKRApp):
    def __init__(self, quote):
        super().__init__(ready=True)
        self.quote = quote
        self.quote_values = {}
        self.quote_events = {}
        self.cancelled_quote_ids = []
        self.market_data_requests = []

    def reqMarketDataType(self, market_data_type):
        self.requested_market_data_type = market_data_type

    def reqMktData(self, request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options):
        self.market_data_requests.append((request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options))
        self.quote_values[request_id] = dict(self.quote)
        values = self.quote_values[request_id]
        if ((values.get("bid") and values.get("ask")) or values.get("last")) and values.get("timestamp") and values.get("data_type"):
            self.quote_events[request_id].set()

    def cancelMktData(self, request_id):
        self.cancelled_quote_ids.append(request_id)


class MarketDataProbeApp(QuoteApp):
    def __init__(self, quotes_by_type):
        super().__init__({})
        self.quotes_by_type = quotes_by_type
        self.probe_market_data_type = None

    def reqMarketDataType(self, market_data_type):
        self.probe_market_data_type = market_data_type

    def reqMktData(self, request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options):
        self.market_data_requests.append((request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options))
        self.quote_values[request_id] = {}
        IBKRApp.marketDataType(self, request_id, self.probe_market_data_type)
        quote = self.quotes_by_type.get(self.probe_market_data_type, {})
        for tick_type, value in quote.get("prices", {}).items():
            IBKRApp.tickPrice(self, request_id, tick_type, value, None)
        for tick_type, value in quote.get("sizes", {}).items():
            IBKRApp.tickSize(self, request_id, tick_type, value)
        if quote.get("timestamp"):
            IBKRApp.tickString(self, request_id, 45, str(quote["timestamp"]))
        if quote.get("error"):
            error_code, error_string = quote["error"]
            self.error(request_id, error_code, error_string)


def probe_executor(monkeypatch, quotes_by_type):
    app = MarketDataProbeApp(quotes_by_type)
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.02),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)
    return executor, app


def quote_executor(monkeypatch, quote, *, allow_delayed=False):
    app = QuoteApp(quote)
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.02, allow_delayed_quotes=allow_delayed),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)
    return executor, app


def test_ibkr_execution_quote_uses_streaming_generic_ticks_and_completes_immediately(monkeypatch):
    executor, app = quote_executor(monkeypatch, {
        "bid": 99.0, "ask": 101.0, "last": 100.5,
        "timestamp": "2026-08-28T14:00:00+00:00", "data_type": "REALTIME",
    })

    started = time.monotonic()
    quote = executor.get_execution_quote("nvda")

    assert quote["symbol"] == "NVDA"
    assert quote["price"] == pytest.approx(100.0)
    assert quote["data_type"] == "REALTIME"
    assert time.monotonic() - started < 0.02
    assert app.market_data_requests[0][2:] == ("165,233", False, False, [])
    assert app.cancelled_quote_ids == [10000]


def test_ibkr_market_data_probe_uses_all_types_and_accepts_frozen_price(monkeypatch):
    executor, app = probe_executor(monkeypatch, {
        1: {"prices": {1: -1.0, 2: -1.0, 4: -1.0}, "timestamp": "1777557600"},
        2: {
            "prices": {1: 769.28, 2: 769.42, 4: 769.33, 9: 769.33},
            "sizes": {0: 100, 3: 120, 5: 80},
            "timestamp": "1777557600",
        },
        3: {},
        4: {},
    })

    result = executor.probe_market_data("SPY", timeout_seconds=0.02)

    assert [item[0] for item in app.market_data_requests] == [10000, 10001, 10002, 10003]
    assert [item[2:] for item in app.market_data_requests] == [
        ("", False, False, []),
        ("", False, False, []),
        ("", False, False, []),
        ("", False, False, []),
    ]
    assert [item["requested_type"] for item in result["probes"]] == [1, 2, 3, 4]
    assert result["probes"][0]["result"] == "NO_CURRENT_TICK"
    frozen = result["probes"][1]
    assert frozen["result"] == "PASS"
    assert frozen["callback_market_data_type"] == 2
    assert frozen["bid"] == pytest.approx(769.28)
    assert frozen["ask"] == pytest.approx(769.42)
    assert frozen["last"] == pytest.approx(769.33)
    assert frozen["close"] == pytest.approx(769.33)
    assert frozen["bid_size"] == pytest.approx(100)
    assert frozen["ask_size"] == pytest.approx(120)
    assert frozen["last_size"] == pytest.approx(80)
    assert frozen["received_tick"] is True
    assert result["selected"]["requested_type"] == 2
    assert result["final_state"] == "MARKET_CLOSED / FROZEN"
    assert app.cancelled_quote_ids == [10000, 10001, 10002, 10003]


def test_ibkr_market_data_probe_maps_delayed_ticks_and_preserves_error(monkeypatch):
    executor, app = probe_executor(monkeypatch, {
        1: {"error": (354, "Requested market data is not subscribed")},
        2: {"error": (414, "Snapshot market data subscription is not applicable to generic ticks")},
        3: {
            "prices": {66: 769.28, 67: 769.42, 68: 769.33, 75: 769.33},
            "sizes": {69: 100, 70: 120, 71: 80},
            "timestamp": "1777557600",
        },
        4: {},
    })

    result = executor.probe_market_data("SPY", timeout_seconds=0.02)

    delayed = result["probes"][2]
    assert delayed["result"] == "PASS"
    assert delayed["data_type"] == "DELAYED"
    assert delayed["bid"] == pytest.approx(769.28)
    assert delayed["close"] == pytest.approx(769.33)
    assert delayed["bid_size"] == pytest.approx(100)
    assert result["probes"][0]["error_code"] == 354
    assert result["probes"][0]["error_message"] == "Requested market data is not subscribed"
    assert result["probes"][1]["error_code"] == 414
    assert result["probes"][1]["error_message"] == "Snapshot market data subscription is not applicable to generic ticks"
    assert result["selected"]["requested_type"] == 3
    assert app.cancelled_quote_ids == [10000, 10001, 10002, 10003]


def test_ibkr_market_data_probe_close_only_is_valid_for_observe(monkeypatch):
    executor, _ = probe_executor(monkeypatch, {
        1: {},
        2: {"prices": {9: 769.33}, "timestamp": "1777557600"},
        3: {},
        4: {},
    })

    result = executor.probe_market_data("SPY", timeout_seconds=0.02)

    assert result["selected"]["close"] == pytest.approx(769.33)
    assert result["selected"]["result"] == "PASS"


def test_ibkr_market_data_probe_price_does_not_require_generic_tick_timestamp(monkeypatch):
    executor, app = probe_executor(monkeypatch, {
        1: {"prices": {4: 100.0}},
        2: {},
        3: {},
        4: {},
    })

    result = executor.probe_market_data("SPY", timeout_seconds=0.02)

    assert result["probes"][0]["elapsed_ms"] < 20
    assert result["probes"][0]["result"] == "PASS"
    assert result["probes"][0]["last"] == pytest.approx(100.0)
    assert app.cancelled_quote_ids == [10000, 10001, 10002, 10003]


@pytest.mark.parametrize("warning_code", [2186, 2187])
def test_ibkr_probe_warning_does_not_end_request_before_delayed_price(monkeypatch, warning_code):
    class WarningThenPriceApp(MarketDataProbeApp):
        def reqMktData(self, request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options):
            self.market_data_requests.append((request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options))
            self.quote_values[request_id] = {}
            IBKRApp.marketDataType(self, request_id, 3)
            IBKRApp.error(self, request_id, warning_code, "Delayed market data warning")

            def deliver_price():
                IBKRApp.tickPrice(self, request_id, 68, 769.33, None)
                IBKRApp.tickString(self, request_id, 88, "1777557600")

            threading.Timer(0.01, deliver_price).start()

    app = WarningThenPriceApp({})
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.05),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    started = time.monotonic()
    result = executor.probe_market_data("SPY", market_data_types=(3,), timeout_seconds=0.05)

    delayed = result["probes"][0]
    assert time.monotonic() - started >= 0.005
    assert delayed["result"] == "PASS"
    assert delayed["last"] == pytest.approx(769.33)
    assert delayed["error_code"] is None
    assert delayed["warning_code"] == warning_code
    assert app.market_data_requests[0][2] == ""
    assert app.cancelled_quote_ids == [10000]


def test_ibkr_probe_warning_without_price_waits_then_fails(monkeypatch):
    class WarningOnlyApp(MarketDataProbeApp):
        def reqMktData(self, request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options):
            self.market_data_requests.append((request_id, contract, generic_ticks, snapshot, regulatory_snapshot, options))
            self.quote_values[request_id] = {}
            IBKRApp.marketDataType(self, request_id, 3)
            IBKRApp.error(self, request_id, 2187, "Delayed generic ticks are unavailable")

    app = WarningOnlyApp({})
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.02),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    started = time.monotonic()
    result = executor.probe_market_data("SPY", market_data_types=(3,), timeout_seconds=0.02)

    delayed = result["probes"][0]
    assert time.monotonic() - started >= 0.01
    assert delayed["result"] == "FAIL"
    assert delayed["error_code"] == 2187
    assert delayed["warning_code"] == 2187
    assert app.cancelled_quote_ids == [10000]


def test_ibkr_historical_fallback_uses_completed_snapshot_and_cleans_request(monkeypatch):
    class HistoricalApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.historical_requests = []
            self.cancelled_historical_ids = []

        def reqHistoricalData(self, request_id, contract, end_time, duration, bar_size, what_to_show, use_rth, format_date, keep_up_to_date, options):
            self.historical_requests.append((request_id, contract, end_time, duration, bar_size, what_to_show, use_rth, format_date, keep_up_to_date, options))
            IBKRApp.historicalData(self, request_id, SimpleNamespace(date="20260828", open=768.0, high=771.0, low=767.0, close=769.33, volume=1000000))
            IBKRApp.historicalDataEnd(self, request_id, "20260824", "20260828")

        def cancelHistoricalData(self, request_id):
            self.cancelled_historical_ids.append(request_id)

    app = HistoricalApp()
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.02),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    result = executor.get_historical_fallback("SPY", timeout_seconds=0.02)

    assert result["close"] == pytest.approx(769.33)
    assert result["bar_date"] == "2026-08-28"
    assert result["source"] == "IBKR historical"
    request = app.historical_requests[0]
    assert request[3:9] == ("5 D", "1 day", "TRADES", 1, 2, False)
    assert app.cancelled_historical_ids == []
    assert app.historical_events == {}
    assert app.historical_values == {}


def test_ibkr_historical_warning_waits_for_completed_daily_bar(monkeypatch):
    class HistoricalWarningApp(FakeIBKRApp):
        def __init__(self):
            super().__init__(ready=True)
            self.cancelled_historical_ids = []

        def reqHistoricalData(self, request_id, *args):
            IBKRApp.error(self, request_id, 2188, "Up-to-the-second historical data requires additional subscription")

            def complete():
                IBKRApp.historicalData(self, request_id, SimpleNamespace(date="20260828", open=768.0, high=771.0, low=767.0, close=769.33, volume=1000000))
                IBKRApp.historicalDataEnd(self, request_id, "20260824", "20260828")

            threading.Timer(0.01, complete).start()

        def cancelHistoricalData(self, request_id):
            self.cancelled_historical_ids.append(request_id)

    app = HistoricalWarningApp()
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.05),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    started = time.monotonic()
    result = executor.get_historical_fallback("SPY", timeout_seconds=0.05)

    assert time.monotonic() - started >= 0.005
    assert result["close"] == pytest.approx(769.33)
    assert result["warning_code"] == 2188
    assert app.cancelled_historical_ids == []


def test_ibkr_delayed_quote_fails_closed_unless_enabled(monkeypatch):
    delayed = {
        "last": 100.0, "timestamp": "2026-08-28T14:00:00+00:00",
        "data_type": "DELAYED",
    }
    executor, app = quote_executor(monkeypatch, delayed)

    with pytest.raises(RuntimeError, match="delayed quote"):
        executor.get_execution_quote("NVDA")
    assert app.cancelled_quote_ids == [10000]

    enabled, _ = quote_executor(monkeypatch, delayed, allow_delayed=True)
    assert enabled.get_execution_quote("NVDA")["data_type"] == "DELAYED"


def test_ibkr_frozen_quote_is_observation_only_and_execution_fails_closed(monkeypatch):
    executor, app = quote_executor(monkeypatch, {
        "bid": 99.0, "ask": 101.0, "last": 100.0,
        "timestamp": "2026-08-28T14:00:00+00:00", "data_type": "FROZEN",
    })

    with pytest.raises(RuntimeError, match="observation only"):
        executor.get_execution_quote("NVDA")

    assert app.cancelled_quote_ids == [10000]


def test_explicit_delayed_quote_request_changes_requested_market_data_type(monkeypatch):
    delayed = {
        "last": 100.0, "timestamp": "2026-08-28T14:00:00+00:00",
        "data_type": "DELAYED",
    }
    executor, app = quote_executor(monkeypatch, delayed, allow_delayed=False)

    executor.get_execution_quote("NVDA", allow_delayed=True)

    assert app.requested_market_data_type == 3


def test_ibkr_market_data_type_preserves_frozen_states():
    from src.execution import ibkr

    if not getattr(ibkr, "_IBAPI_AVAILABLE", False):
        pytest.skip("official IBKR API is not installed in this environment")
    app = ibkr.IBKRApp()
    app.quote_values[10000] = {}

    app.marketDataType(10000, 2)
    assert app.quote_values[10000]["data_type"] == "FROZEN"
    app.marketDataType(10000, 4)
    assert app.quote_values[10000]["data_type"] == "DELAYED_FROZEN"


@pytest.mark.parametrize("quote", [
    {"last": 100.0, "data_type": "REALTIME"},
    {"timestamp": "2026-08-28T14:00:00+00:00", "data_type": "REALTIME"},
])
def test_incomplete_ibkr_quote_times_out_and_cancels(monkeypatch, quote):
    executor, app = quote_executor(monkeypatch, quote)

    with pytest.raises(TimeoutError, match="timed out"):
        executor.get_execution_quote("NVDA")

    assert app.cancelled_quote_ids == [10000]


@pytest.mark.parametrize("error_code,error_string", [
    (414, "Snapshot market data subscription is not applicable to generic ticks"),
    (354, "Requested market data is not subscribed"),
])
def test_ibkr_market_data_error_wakes_quote_waiter_and_fails(monkeypatch, error_code, error_string):
    class ErrorQuoteApp(QuoteApp):
        def reqMktData(self, request_id, *args):
            self.market_data_requests.append((request_id, *args))
            self.error(request_id, error_code, error_string)

    app = ErrorQuoteApp({})
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=1),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match=str(error_code)):
        executor.get_execution_quote("NVDA")

    assert time.monotonic() - started < 0.1
    assert app.cancelled_quote_ids == [10000]


def test_ibkr_execution_quote_entitlement_warning_fails_immediately_without_order_mutation(monkeypatch):
    class EntitlementWarningApp(QuoteApp):
        def __init__(self):
            super().__init__({})
            self.place_order_calls = 0
            self.cancel_order_calls = 0

        def reqMktData(self, request_id, *args):
            self.market_data_requests.append((request_id, *args))
            IBKRApp.error(
                self,
                request_id,
                2186,
                "API real-time market data requires a separate subscription. Delayed market data is available.",
            )

        def placeOrder(self, *args):
            self.place_order_calls += 1

        def cancelOrder(self, *args):
            self.cancel_order_calls += 1

    app = EntitlementWarningApp()
    executor = IBKRPaperExecutor(
        IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.05),
        app_factory=lambda: app,
    )
    executor.connect()
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"realtime market data subscription unavailable.*2186.*delayed data cannot be used"):
        executor.get_execution_quote("MSFT")

    assert time.monotonic() - started < 0.1
    assert app.place_order_calls == 0
    assert app.cancel_order_calls == 0
    assert app.cancelled_quote_ids == []


def test_ibkr_order_reject_error_is_linked_to_execution_report(monkeypatch):
    class RejectingApp(FakeIBKRApp):
        def placeOrder(self, order_id, contract, order):
            self.error(order_id, 201, "Order rejected", '{"reason":"paper reject"}')

    app = RejectingApp(ready=True)
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app)
    executor.connect()
    executor.broker_state_known = True
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: symbol)
    monkeypatch.setattr("src.execution.ibkr.market_order", lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity, orderRef=None))

    request = OrderRequest(decision_id="reject-decision", symbol="NVDA", action="BUY", quantity=1, reference_price=100, client_order_id="reject-order")
    executor.submit_order(request)
    report = executor.order_status(request.client_order_id)

    assert report.status == "REJECTED"
    assert report.error_code == 201
    assert report.error_string == "Order rejected"
    assert report.advanced_reject_reason == '{"reason":"paper reject"}'


def test_connectivity_loss_invalidates_broker_readiness_and_blocks_order(tmp_path):
    app = FakeIBKRApp(ready=True)
    with SQLiteStore(tmp_path / "connection-loss.sqlite3") as store:
        executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app, store=store)
        executor.connect()
        executor.broker_state_known = True

        app.error(-1, 1100, "Connectivity between IB and Trader Workstation has been lost")
        report = executor.submit_order(OrderRequest(decision_id="blocked", symbol="NVDA", action="BUY", quantity=1, reference_price=100))

        assert executor.connected is False
        assert executor.broker_state_known is False
        assert report.status == "ERROR"
        assert store.get_runtime("service_status") == "SAFE_MODE"
        assert store.get_runtime("trading_enabled") is False


def test_ibkr_informational_warning_does_not_enter_safe_mode(tmp_path):
    app = FakeIBKRApp(ready=True)
    with SQLiteStore(tmp_path / "warning.sqlite3") as store:
        executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app, store=store)
        executor.connect()

        app.error(-1, 2104, "Market data farm connection is OK")

        assert executor.connected is True
        assert store.get_runtime("safe_mode", False) is False


@pytest.mark.parametrize("error_code", (2103, 2105))
def test_market_data_farm_warning_does_not_invalidate_reconciled_broker(tmp_path, error_code):
    app = FakeIBKRApp(ready=True)
    with SQLiteStore(tmp_path / f"farm-warning-{error_code}.sqlite3") as store:
        executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app, store=store)
        executor.connect()
        executor.broker_state_known = True

        app.error(-1, error_code, "Market data farm connection is broken")

        assert executor.connected is True
        assert executor.broker_state_known is True
        assert store.get_runtime("safe_mode", False) is False


def test_yahoo_brk_b_resolves_to_unique_verified_ibkr_contract(monkeypatch):
    app = FakeIBKRApp(ready=True)
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app)
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: SimpleNamespace(symbol=symbol))
    executor.connect()

    instrument = executor.resolve_instrument("BRK-B")

    assert instrument.canonical_symbol == "BRK.B"
    assert instrument.yahoo_symbol == "BRK-B"
    assert instrument.contract.symbol == "BRK B"
    assert instrument.con_id > 0
    assert instrument.local_symbol == "BRK B"
    assert instrument.primary_exchange == "NASDAQ"


def test_ambiguous_ibkr_contract_resolution_fails_closed(monkeypatch):
    class AmbiguousContractApp(FakeIBKRApp):
        def reqContractDetails(self, request_id, query_contract):
            self.contract_detail_values[request_id] = [
                SimpleNamespace(contract=SimpleNamespace(symbol="NVDA", localSymbol="NVDA", conId=1, primaryExchange="NASDAQ", currency="USD", secType="STK", exchange="SMART", validExchanges="SMART,NASDAQ")),
                SimpleNamespace(contract=SimpleNamespace(symbol="NVDA", localSymbol="NVDA", conId=2, primaryExchange="NYSE", currency="USD", secType="STK", exchange="SMART", validExchanges="SMART,NYSE")),
            ]
            self.contract_detail_events[request_id].set()

    app = AmbiguousContractApp(ready=True)
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: app)
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: SimpleNamespace(symbol=symbol))
    executor.connect()

    with pytest.raises(RuntimeError, match="unique"):
        executor.resolve_instrument("NVDA")


def test_no_security_definition_error_wakes_contract_resolution(monkeypatch):
    class MissingContractApp(FakeIBKRApp):
        def reqContractDetails(self, request_id, query_contract):
            self.error(request_id, 200, "No security definition has been found")

    app = MissingContractApp(ready=True)
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=1), app_factory=lambda: app)
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: SimpleNamespace(symbol=symbol))
    executor.connect()

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="200"):
        executor.resolve_instrument("MISSING")

    assert time.monotonic() - started < 0.1


def test_unresolved_contract_blocks_quote_and_order(monkeypatch):
    class UnresolvedContractApp(QuoteApp):
        def __init__(self):
            super().__init__({"last": 100, "timestamp": "2026-08-28T14:00:00+00:00", "data_type": "REALTIME"})
            self.placed = []

        def reqContractDetails(self, request_id, query_contract):
            self.contract_detail_values[request_id] = []
            self.contract_detail_events[request_id].set()

        def placeOrder(self, *args):
            self.placed.append(args)

    app = UnresolvedContractApp()
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01, quote_timeout_seconds=0.02), app_factory=lambda: app)
    monkeypatch.setattr("src.execution.ibkr.stock_contract", lambda symbol: SimpleNamespace(symbol=symbol))
    monkeypatch.setattr("src.execution.ibkr.market_order", lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity, orderRef=None))
    executor.connect()
    executor.broker_state_known = True

    with pytest.raises(RuntimeError, match="resolve"):
        executor.get_execution_quote("UNKNOWN")
    report = executor.submit_order(OrderRequest(decision_id="unresolved", symbol="UNKNOWN", action="BUY", quantity=1, reference_price=100))

    assert report.status == "ERROR"
    assert "resolve" in report.message
    assert app.market_data_requests == []
    assert app.placed == []


@pytest.mark.parametrize("reconnect_code", [1101, 1102])
def test_ibkr_reconnect_status_requires_full_reconciliation_before_quote_or_order(monkeypatch, reconnect_code):
    class ReconnectReadyApp(FakeIBKRApp):
        def reqAccountSummary(self, *_):
            self.account_values = {"TotalCashValue": "1000", "NetLiquidation": "1000"}
            self.account_summary_end_event.set()
        def reqPositions(self): self.positions_end_event.set()
        def reqOpenOrders(self): self.open_orders_end_event.set()
        def reqExecutions(self, *_): self.executions_end_event.set()

    first = ReconnectReadyApp(ready=True)
    second = ReconnectReadyApp(ready=True)
    apps = iter([first, second])
    executor = IBKRPaperExecutor(IBKRConfig(readiness_timeout_seconds=0.01), app_factory=lambda: next(apps))
    monkeypatch.setattr("src.execution.ibkr.market_order", lambda action, quantity: SimpleNamespace(action=action, totalQuantity=quantity, orderRef=None))
    executor.connect()
    executor.reconcile({})

    first.error(-1, reconnect_code, "Connectivity restored; broker state must be refreshed")
    executor.reconnect()

    with pytest.raises(RuntimeError, match="reconciliation required"):
        executor.get_execution_quote("NVDA")
    blocked = executor.submit_order(OrderRequest(decision_id="before-reconcile", symbol="NVDA", action="BUY", quantity=1, reference_price=100))
    assert blocked.status == "ERROR"

    executor.reconcile({})

    assert executor.broker_state_known is True
