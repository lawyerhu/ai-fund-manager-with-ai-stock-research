"""Explicit IBKR Paper Trading adapter.

The adapter never selects an account or submits on import. It rejects live
accounts and the usual live TWS/Gateway ports; V1 has no live-trading path.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable

from .base import Executor
from ..config import env
from ..instruments import InstrumentResolver, instrument_identity
from ..models import ExecutionReport, OrderRequest
from ..models import BrokerSnapshot, PositionSnapshot


_INFO_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}
_MARKET_DATA_FARM_WARNING_CODES = {2103, 2105}
_MARKET_DATA_ERROR_CODES = {200, 354, 414, 10090, 10167, 10168, 10186}
_NON_FATAL_MARKET_DATA_WARNING_CODES = {2186, 2187}
_NON_FATAL_HISTORICAL_WARNING_CODES = {2188}
_TERMINAL_MARKET_DATA_ERROR_CODES = {2186}
_ORDER_ERROR_CODES = {201, 202, 10147, 10148}
_CONNECTIVITY_ERROR_CODES = {1100, 1101, 1102, 1300, 2110}
_FATAL_BROKER_ERROR_CODES = {326, 502, 504}
_MARKET_DATA_TYPE_NAMES = {
    1: "REALTIME",
    2: "FROZEN",
    3: "DELAYED",
    4: "DELAYED_FROZEN",
}
_MARKET_DATA_PRICE_TICKS = {
    1: "bid",
    2: "ask",
    4: "last",
    9: "close",
    66: "bid",
    67: "ask",
    68: "last",
    72: "high",
    73: "low",
    75: "close",
    76: "open",
}
_MARKET_DATA_SIZE_TICKS = {
    0: "bid_size",
    3: "ask_size",
    5: "last_size",
    69: "bid_size",
    70: "ask_size",
    71: "last_size",
    74: "volume",
}


def _error_record(request_id: int, error_time: int | None, error_code: int, error_string: str, advanced_reject: str = "") -> dict[str, Any]:
    if error_code in _INFO_ERROR_CODES:
        category, severity = "INFO", "INFO"
    elif error_code in _MARKET_DATA_FARM_WARNING_CODES:
        # These describe an IBKR market-data farm reconnect, not loss of the
        # TWS/API socket or account state.  Existing executable quotes still
        # decide whether an order may proceed.
        category, severity = "MARKET_DATA", "WARNING"
    elif error_code in _MARKET_DATA_ERROR_CODES:
        category, severity = "MARKET_DATA", "ERROR"
    elif error_code in _ORDER_ERROR_CODES:
        category, severity = "ORDER", "ERROR"
    elif error_code in _CONNECTIVITY_ERROR_CODES:
        category, severity = "CONNECTIVITY", "ERROR" if error_code in {1100, 1300, 2110} else "WARNING"
    elif error_code in _FATAL_BROKER_ERROR_CODES:
        category, severity = "FATAL_BROKER", "ERROR"
    else:
        category, severity = "BROKER", "WARNING"
    return {
        "request_id": int(request_id),
        "error_time": error_time,
        "error_code": int(error_code),
        "error_string": str(error_string),
        "advanced_order_reject_json": advanced_reject or "",
        "local_received_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "severity": severity,
    }


def _initialize_ibkr_app_state(app):
    app.next_order_id = None
    app.account_values = {}
    app.position_values = {}
    app.order_values = {}
    app.execution_values = []
    app.commission_values = {}
    app.quote_values = {}
    app.quote_events = {}
    app.probe_quote_requests = set()
    app.historical_values = {}
    app.historical_events = {}
    app.historical_completed = set()
    app.contract_detail_values = {}
    app.contract_detail_events = {}
    app.request_errors = {}
    app.request_warnings = {}
    app.error_records = []
    app.state_callback = None
    app.transport_connected = False
    app.next_valid_id_event = threading.Event()
    app.managed_accounts_event = threading.Event()
    app.account_summary_end_event = threading.Event()
    app.positions_end_event = threading.Event()
    app.open_orders_end_event = threading.Event()
    app.executions_end_event = threading.Event()
    app.managed_accounts = []


def _quote_values(app, req_id: int) -> dict[str, Any]:
    return app.quote_values.setdefault(req_id, {})


def _positive_quote_value(values: dict[str, Any], field: str) -> bool:
    try:
        return float(values.get(field, 0)) > 0
    except (TypeError, ValueError):
        return False


def _quote_has_price(values: dict[str, Any]) -> bool:
    return any(_positive_quote_value(values, field) for field in ("bid", "ask", "last", "close"))


def _quote_is_complete_values(values: dict[str, Any]) -> bool:
    return (
        _quote_has_price(values)
        and bool(values.get("timestamp"))
        and values.get("data_type") in _MARKET_DATA_TYPE_NAMES.values()
    )


def _remember_tick(values: dict[str, Any], tick_type: int) -> None:
    values["received_tick"] = True
    tick_types = values.setdefault("received_tick_types", [])
    if tick_type not in tick_types:
        tick_types.append(tick_type)


def _record_tick_price(app, req_id: int, tick_type: int, price: float) -> None:
    values = _quote_values(app, req_id)
    _remember_tick(values, int(tick_type))
    field = _MARKET_DATA_PRICE_TICKS.get(int(tick_type))
    if field:
        try:
            values[field] = float(price)
        except (TypeError, ValueError):
            values[field] = None
    _signal_quote_if_complete(app, req_id)


def _record_tick_size(app, req_id: int, tick_type: int, size: float) -> None:
    values = _quote_values(app, req_id)
    _remember_tick(values, int(tick_type))
    field = _MARKET_DATA_SIZE_TICKS.get(int(tick_type))
    if field:
        try:
            values[field] = float(size)
        except (TypeError, ValueError):
            values[field] = None
    _signal_quote_if_complete(app, req_id)


def _record_tick_string(app, req_id: int, tick_type: int, value: str) -> None:
    values = _quote_values(app, req_id)
    _remember_tick(values, int(tick_type))
    if value and int(tick_type) in {45, 88}:
        try:
            values["timestamp"] = datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    elif value and int(tick_type) == 48:
        fields = value.split(";")
        if len(fields) >= 3 and fields[2]:
            try:
                values["timestamp"] = datetime.fromtimestamp(float(fields[2]) / 1000, timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError):
                pass
    _signal_quote_if_complete(app, req_id)


def _record_market_data_type(app, req_id: int, market_data_type: int) -> None:
    values = _quote_values(app, req_id)
    callback_type = int(market_data_type)
    values["callback_market_data_type"] = callback_type
    values["data_type"] = _MARKET_DATA_TYPE_NAMES.get(callback_type, "UNKNOWN")
    _signal_quote_if_complete(app, req_id)


def _normalize_historical_bar_date(value: Any) -> str:
    text = str(value).strip()
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(float(text), timezone.utc).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return text


def _record_historical_bar(app, req_id: int, bar) -> None:
    app.historical_values.setdefault(req_id, []).append({
        "bar_date": _normalize_historical_bar_date(getattr(bar, "date", "")),
        "open": float(getattr(bar, "open", 0) or 0),
        "high": float(getattr(bar, "high", 0) or 0),
        "low": float(getattr(bar, "low", 0) or 0),
        "close": float(getattr(bar, "close", 0) or 0),
        "volume": float(getattr(bar, "volume", 0) or 0),
    })


def _complete_historical_request(app, req_id: int) -> None:
    app.historical_completed.add(req_id)
    event = app.historical_events.get(req_id)
    if event is not None:
        event.set()


def _signal_quote_if_complete(app, req_id: int) -> None:
    event = app.quote_events.get(req_id)
    values = app.quote_values.get(req_id, {})
    probe_complete = (
        req_id in getattr(app, "probe_quote_requests", set())
        and _quote_has_price(values)
        and values.get("data_type") in _MARKET_DATA_TYPE_NAMES.values()
    )
    if event is not None and (probe_complete or _quote_is_complete_values(values)):
        event.set()

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.order import Order
    try:
        from ibapi.order_cancel import OrderCancel
    except ImportError:  # Older official clients do not expose the current cancel object.
        OrderCancel = None
    from ibapi.execution import ExecutionFilter
    from ibapi.wrapper import EWrapper
    _IBAPI_AVAILABLE = True
except ImportError:  # Optional until the user configures IBKR.
    EClient = EWrapper = Contract = Order = ExecutionFilter = OrderCancel = None
    _IBAPI_AVAILABLE = False


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7947
    client_id: int = 41
    paper_only: bool = True
    readiness_timeout_seconds: float = 10.0
    reconciliation_timeout_seconds: float = 15.0
    order_timeout_seconds: float = 30.0
    quote_timeout_seconds: float = 10.0
    allow_delayed_quotes: bool = False

    @classmethod
    def from_env(cls):
        return cls(host=env("IBKR_HOST", cls.host), port=int(env("IBKR_PORT", cls.port)), client_id=int(env("IBKR_CLIENT_ID", cls.client_id)))

    def validate(self):
        if not self.paper_only:
            raise ValueError("Live trading is not supported")
        if self.port in {7496, 7946, 4001}:
            raise ValueError("Live IBKR ports are blocked by the paper-only safety boundary")
        if not 1 <= self.port <= 65535:
            raise ValueError("IBKR port must be between 1 and 65535")
        if self.client_id < 0:
            raise ValueError("IBKR client_id must be non-negative")


if _IBAPI_AVAILABLE:
    class IBKRApp(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            _initialize_ibkr_app_state(self)

        def run(self):
            """Contain the official client's reset/reader race during reconnect."""
            try:
                # The official reader can outlive disconnect() after it resets
                # serverVersion to None.  Avoid entering its parser in that
                # state; the reader is already disconnected and must fail
                # closed instead of leaking an exception from its thread.
                if self.serverVersion() is None:
                    self.transport_connected = False
                    record = _error_record(-1, None, 1100, "IBKR API reader stopped during connection reset")
                    self.error_records.append(record)
                    if self.state_callback:
                        self.state_callback(record)
                    return
                return EClient.run(self)
            except TypeError:
                if self.serverVersion() is not None:
                    raise
                self.transport_connected = False
                record = _error_record(-1, None, 1100, "IBKR API reader stopped during connection reset")
                self.error_records.append(record)
                if self.state_callback:
                    self.state_callback(record)

        def nextValidId(self, orderId: int):
            self.transport_connected = True
            self.next_order_id = orderId
            self.next_valid_id_event.set()

        def connectAck(self):
            self.transport_connected = True

        def managedAccounts(self, accountsList: str):
            self.managed_accounts = [account for account in accountsList.split(",") if account]
            self.managed_accounts_event.set()

        def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
            self.account_values[tag] = value

        def accountSummaryEnd(self, reqId: int):
            self.account_summary_end_event.set()

        def position(self, account: str, contract: Contract, position: Decimal, avgCost: float):
            key = f"{contract.symbol}:{contract.secType}:{contract.currency}:{getattr(contract, 'conId', 0)}"
            self.position_values[key] = {
                "account": account,
                "symbol": contract.symbol,
                "sec_type": contract.secType,
                "currency": contract.currency,
                "exchange": contract.exchange,
                "quantity": float(position),
                "average_cost": avgCost,
            }

        def positionEnd(self):
            self.positions_end_event.set()

        def contractDetails(self, reqId: int, contractDetails):
            self.contract_detail_values.setdefault(reqId, []).append(contractDetails)

        def contractDetailsEnd(self, reqId: int):
            event = self.contract_detail_events.get(reqId)
            if event is not None:
                event.set()

        def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
            self.order_values[orderId] = {"order_id": orderId, "client_order_id": order.orderRef or None, "client_id": getattr(order, "clientId", None), "symbol": contract.symbol, "action": order.action, "quantity": float(order.totalQuantity), "status": orderState.status, "perm_id": getattr(order, "permId", None)}

        def openOrderEnd(self):
            self.open_orders_end_event.set()

        def orderStatus(self, orderId: int, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
            self.order_values.setdefault(orderId, {}).update({"order_id": orderId, "status": status, "filled": filled, "remaining": remaining, "avg_fill_price": avgFillPrice, "perm_id": permId})

        def execDetails(self, reqId: int, contract: Contract, execution):
            self.execution_values.append({"req_id": reqId, "order_id": execution.orderId, "perm_id": execution.permId, "client_order_id": getattr(execution, "orderRef", None) or None, "symbol": contract.symbol, "exec_id": execution.execId, "shares": float(execution.shares), "price": execution.price})

        def execDetailsEnd(self, reqId: int):
            self.executions_end_event.set()

        def historicalData(self, reqId: int, bar):
            _record_historical_bar(self, reqId, bar)

        def historicalDataEnd(self, reqId: int, start: str, end: str):
            _complete_historical_request(self, reqId)

        def commissionAndFeesReport(self, report):
            cost = float(getattr(report, "commissionAndFees", 0.0) or 0.0)
            self.commission_values[report.execId] = {
                "total_execution_cost": cost,
                "commission_and_fees": cost,
                "currency": report.currency,
                "realized_pnl": float(getattr(report, "realizedPNL", 0.0) or 0.0),
            }

        def commissionReport(self, report):
            # Compatibility for an older installed API; reconciliation uses the current callback above.
            cost = float(getattr(report, "commission", 0.0) or 0.0)
            self.commission_values[report.execId] = {
                "total_execution_cost": cost,
                "commission_and_fees": cost,
                "currency": report.currency,
                "realized_pnl": float(getattr(report, "realizedPNL", 0.0) or 0.0),
            }

        def error(self, reqId: int, errorTime: int, errorCode: int | str, errorString: str | None = None, advancedOrderRejectJson: str = ""):
            if isinstance(errorCode, str):
                advancedOrderRejectJson = errorString or ""
                errorString = errorCode
                errorCode = errorTime
                errorTime = None
            record = _error_record(reqId, errorTime, int(errorCode), errorString or "", advancedOrderRejectJson)
            if reqId in self.order_values and record["category"] not in {"CONNECTIVITY", "FATAL_BROKER"}:
                record.update(category="ORDER", severity="ERROR")
            elif (reqId in getattr(self, "quote_events", {}) or reqId in getattr(self, "contract_detail_events", {}) or reqId in getattr(self, "historical_events", {})) and record["category"] not in {"CONNECTIVITY", "FATAL_BROKER"}:
                record.update(category="MARKET_DATA", severity="ERROR")
            elif record["category"] == "BROKER" and reqId < 0:
                record.update(category="FATAL_BROKER", severity="ERROR")
            non_fatal_market_warning = (
                reqId in getattr(self, "quote_events", {})
                and int(errorCode) in _NON_FATAL_MARKET_DATA_WARNING_CODES
                and (int(errorCode) != 2186 or reqId in getattr(self, "probe_quote_requests", set()))
            ) or (
                reqId in getattr(self, "historical_events", {}) and int(errorCode) in _NON_FATAL_HISTORICAL_WARNING_CODES
            )
            if non_fatal_market_warning:
                record.update(category="MARKET_DATA", severity="WARNING")
                self.request_warnings.setdefault(reqId, []).append(record)
            else:
                self.request_errors[reqId] = record
            self.error_records.append(record)
            if int(errorCode) in {1100, 1300, 2110}:
                self.transport_connected = False
            elif int(errorCode) in {1101, 1102}:
                self.transport_connected = True
            if reqId in getattr(self, "quote_events", {}) and not non_fatal_market_warning:
                self.quote_events[reqId].set()
            if hasattr(self, "contract_detail_events") and reqId in self.contract_detail_events:
                self.contract_detail_events[reqId].set()
            if reqId in getattr(self, "historical_events", {}) and not non_fatal_market_warning:
                self.historical_events[reqId].set()
            if reqId in self.order_values:
                self.order_values[reqId].update({
                    "status": "Inactive" if errorCode == 201 else "Error",
                    "error_code": errorCode,
                    "error_string": errorString,
                    "advanced_reject_reason": advancedOrderRejectJson or None,
                })
            if self.state_callback:
                self.state_callback(record)

        def connectionClosed(self):
            self.transport_connected = False
            self.error(-1, 1100, "IBKR socket connection closed")

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
            _record_tick_price(self, reqId, tickType, price)

        def tickSize(self, reqId: int, tickType: int, size: int):
            _record_tick_size(self, reqId, tickType, size)

        def tickString(self, reqId: int, tickType: int, value: str):
            _record_tick_string(self, reqId, tickType, value)

        def marketDataType(self, reqId: int, marketDataType: int):
            _record_market_data_type(self, reqId, marketDataType)

        def tickSnapshotEnd(self, reqId: int):
            pass

        def _quote_is_complete(self, reqId: int) -> bool:
            return _quote_is_complete_values(self.quote_values.get(reqId, {}))

        def _signal_quote_if_complete(self, reqId: int):
            _signal_quote_if_complete(self, reqId)
else:
    class _IBKRCallbackCompatibility:
        def __init__(self):
            _initialize_ibkr_app_state(self)

        def commissionAndFeesReport(self, report):
            cost = float(getattr(report, "commissionAndFees", 0.0) or 0.0)
            self.commission_values[report.execId] = {
                "total_execution_cost": cost,
                "commission_and_fees": cost,
                "currency": report.currency,
                "realized_pnl": float(getattr(report, "realizedPNL", 0.0) or 0.0),
            }

        def commissionReport(self, report):
            self.commissionAndFeesReport(SimpleNamespace(
                execId=report.execId,
                commissionAndFees=getattr(report, "commission", 0.0),
                currency=report.currency,
                realizedPNL=getattr(report, "realizedPNL", 0.0),
            ))

        def historicalData(self, reqId: int, bar):
            _record_historical_bar(self, reqId, bar)

        def historicalDataEnd(self, reqId: int, start: str, end: str):
            _complete_historical_request(self, reqId)

        def error(self, reqId: int, errorTime: int, errorCode: int | str, errorString: str | None = None, advancedOrderRejectJson: str = ""):
            if isinstance(errorCode, str):
                advancedOrderRejectJson = errorString or ""
                errorString = errorCode
                errorCode = errorTime
                errorTime = None
            record = _error_record(reqId, errorTime, int(errorCode), errorString or "", advancedOrderRejectJson)
            if (reqId in getattr(self, "quote_events", {}) or reqId in getattr(self, "contract_detail_events", {}) or reqId in getattr(self, "historical_events", {})) and record["category"] not in {"CONNECTIVITY", "FATAL_BROKER"}:
                record.update(category="MARKET_DATA", severity="ERROR")
            non_fatal_market_warning = (
                reqId in getattr(self, "quote_events", {})
                and int(errorCode) in _NON_FATAL_MARKET_DATA_WARNING_CODES
                and (int(errorCode) != 2186 or reqId in getattr(self, "probe_quote_requests", set()))
            ) or (
                reqId in getattr(self, "historical_events", {}) and int(errorCode) in _NON_FATAL_HISTORICAL_WARNING_CODES
            )
            if non_fatal_market_warning:
                record.update(category="MARKET_DATA", severity="WARNING")
                self.request_warnings.setdefault(reqId, []).append(record)
            else:
                self.request_errors[reqId] = record
            self.error_records.append(record)
            if int(errorCode) in {1100, 1300, 2110}:
                self.transport_connected = False
            elif int(errorCode) in {1101, 1102}:
                self.transport_connected = True
            if reqId in getattr(self, "quote_events", {}) and not non_fatal_market_warning:
                self.quote_events[reqId].set()
            if reqId in self.contract_detail_events:
                self.contract_detail_events[reqId].set()
            if reqId in getattr(self, "historical_events", {}) and not non_fatal_market_warning:
                self.historical_events[reqId].set()
            if self.state_callback:
                self.state_callback(record)

        def connectionClosed(self):
            self.transport_connected = False
            self.error(-1, 1100, "IBKR socket connection closed")

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
            _record_tick_price(self, reqId, tickType, price)

        def tickSize(self, reqId: int, tickType: int, size: int):
            _record_tick_size(self, reqId, tickType, size)

        def tickString(self, reqId: int, tickType: int, value: str):
            _record_tick_string(self, reqId, tickType, value)

        def marketDataType(self, reqId: int, marketDataType: int):
            _record_market_data_type(self, reqId, marketDataType)

        def tickSnapshotEnd(self, reqId: int):
            pass

        def _quote_is_complete(self, reqId: int) -> bool:
            return _quote_is_complete_values(self.quote_values.get(reqId, {}))

        def _signal_quote_if_complete(self, reqId: int):
            _signal_quote_if_complete(self, reqId)

    class IBKRApp(_IBKRCallbackCompatibility):
        def __init__(self):
            super().__init__()

        def connect(self, *args, **kwargs):
            raise RuntimeError("OFFICIAL IBKR TWS API NOT INSTALLED OR INCOMPATIBLE; install the official TWS API Python client")


def stock_contract(symbol: str):
    if not _IBAPI_AVAILABLE:
        raise RuntimeError("ibapi is not installed")
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    if contract.symbol == "SPY":
        contract.primaryExchange = "ARCA"
    return contract


def market_order(action: str, quantity: Decimal):
    if not _IBAPI_AVAILABLE:
        raise RuntimeError("ibapi is not installed")
    if action not in {"BUY", "SELL"} or quantity <= 0:
        raise ValueError("market order requires BUY/SELL and positive quantity")
    order = Order()
    order.action = action
    order.orderType = "MKT"
    order.totalQuantity = quantity
    order.tif = "DAY"
    return order


def limit_order(action: str, quantity: Decimal, limit_price: Decimal):
    if not _IBAPI_AVAILABLE:
        raise RuntimeError("ibapi is not installed")
    if action not in {"BUY", "SELL"} or quantity <= 0 or limit_price <= 0:
        raise ValueError("limit order requires BUY/SELL, positive quantity, and positive limit price")
    order = Order()
    order.action = action
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = float(limit_price)
    order.tif = "DAY"
    return order


class IBKRPaperExecutor(Executor):
    """Paper-only broker adapter with callback-complete reconciliation."""

    mode = "IBKR_PAPER"
    broker_source = "IBKR_PAPER"
    execution_mode = "PAPER"
    mutations_allowed = True

    def __init__(self, config: IBKRConfig | None = None, app_factory: Callable[[], IBKRApp] = IBKRApp, store=None):
        self.config = config or IBKRConfig.from_env()
        self.app_factory = app_factory
        self.app: IBKRApp | None = None
        self.connected = False
        self._thread: threading.Thread | None = None
        self._client_orders: dict[str, int] = {}
        self._order_decisions: dict[str, str] = {}
        self._known_client_order_ids: set[str] = set()
        self.broker_state_known = False
        self._last_snapshot: BrokerSnapshot | None = None
        self._raw_state_ready = False
        self.store = store
        self._next_quote_request_id = 10000
        self.place_order_calls = 0
        self.cancel_order_calls = 0
        self._instrument_resolver: InstrumentResolver | None = None
        self._reconciliation_required = False
        self._account_summary_active = False
        self._positions_active = False

    def connect(self):
        self.config.validate()
        if self.connected and self._transport_ready():
            return {"connected": True, "already_connected": True}
        if self.app is not None:
            # A connectivity callback can invalidate executor state while the
            # old API socket is still alive. Close it before reusing client_id.
            self.disconnect()
        if self.app_factory is IBKRApp:
            from ..ibkr_diagnostics import OFFICIAL_API_ERROR, inspect_api_capabilities

            if not inspect_api_capabilities()["ok"]:
                raise RuntimeError(OFFICIAL_API_ERROR)
        self.app = self.app_factory()
        self.app.state_callback = self._handle_app_error
        self._instrument_resolver = None
        self.app.connect(self.config.host, self.config.port, self.config.client_id)
        self._thread = threading.Thread(target=self.app.run, name="ibkr-api", daemon=True)
        self._thread.start()
        timeout = self.config.readiness_timeout_seconds
        if not self.app.next_valid_id_event.wait(timeout):
            self.disconnect()
            raise TimeoutError("IBKR API did not become ready: nextValidId timeout")
        self.app.reqManagedAccts()
        if not self.app.managed_accounts_event.wait(timeout):
            self.disconnect()
            raise TimeoutError("IBKR API did not become ready: managedAccounts timeout")
        if len(self.app.managed_accounts) != 1 or not self.app.managed_accounts[0].upper().startswith("DU"):
            self.disconnect()
            raise RuntimeError("IBKR connection must expose exactly one verified DU paper account; live or ambiguous accounts are blocked")
        if not self._transport_ready():
            self.disconnect()
            raise RuntimeError("IBKR transport is not connected after readiness callbacks")
        self.connected = True
        return {"connected": True, "host": self.config.host, "port": self.config.port, "paper_only": self.config.paper_only}

    def disconnect(self):
        thread = self._thread
        if self.app is not None:
            if self._account_summary_active and hasattr(self.app, "cancelAccountSummary"):
                self.app.cancelAccountSummary(9001)
            if self._positions_active and hasattr(self.app, "cancelPositions"):
                self.app.cancelPositions()
            self.app.state_callback = None
            self.app.disconnect()
        if thread and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=min(2.0, self.config.readiness_timeout_seconds))
        self._thread = None
        self.connected = False
        self.broker_state_known = False
        self._raw_state_ready = False
        self._last_snapshot = None
        self._reconciliation_required = True
        self._instrument_resolver = None
        self._account_summary_active = False
        self._positions_active = False

    def _transport_ready(self) -> bool:
        return bool(self.app is not None and getattr(self.app, "transport_connected", True))

    def reconnect(self):
        self._reconciliation_required = True
        self.disconnect()
        return self.connect()

    def account_summary(self):
        return dict(self.app.account_values) if self._transport_ready() and self.broker_state_known else {}

    def cash_balance(self) -> float | None:
        value = self.account_summary().get("TotalCashValue")
        return float(value) if value is not None else None

    def current_positions(self):
        return [position.model_dump(mode="json") for position in self._last_snapshot.positions] if self._last_snapshot else []

    def positions(self):
        return self.current_positions()

    def current_orders(self):
        return list(self.app.order_values.values()) if self.app else []

    def resolve_instrument(self, symbol: str):
        if not self.connected or not self._transport_ready():
            raise RuntimeError("IBKR is disconnected; contract resolution unavailable")
        if self._instrument_resolver is None:
            self._instrument_resolver = InstrumentResolver(self.app, stock_contract, self.config.readiness_timeout_seconds)
        return self._instrument_resolver.resolve(symbol)

    def _handle_app_error(self, raw_record: dict[str, Any]):
        record = raw_record if raw_record.get("category") else _error_record(
            raw_record.get("request_id", -1), raw_record.get("error_time"),
            raw_record.get("error_code", -1), raw_record.get("error_string", "Unknown IBKR error"),
            raw_record.get("advanced_order_reject_json", ""),
        )
        order_id = record["request_id"]
        client_order_id = next((client_id for client_id, broker_id in self._client_orders.items() if broker_id == order_id), None)
        decision_id = self._order_decisions.get(client_order_id) if client_order_id else None
        if self.store:
            self.store.save_error(
                "ibkr", f"IBKR {record['error_code']}: {record['error_string']}", record,
                severity=record["severity"], component=record["category"].lower(),
                decision_id=decision_id, order_id=client_order_id,
            )
        if self.app and record["category"] == "MARKET_DATA" and record["severity"] != "WARNING":
            quote_event = getattr(self.app, "quote_events", {}).get(record["request_id"])
            if quote_event is not None:
                quote_event.set()
            contract_event = getattr(self.app, "contract_detail_events", {}).get(record["request_id"])
            if contract_event is not None:
                contract_event.set()
            historical_event = getattr(self.app, "historical_events", {}).get(record["request_id"])
            if historical_event is not None:
                historical_event.set()
        if self.app and client_order_id and record["category"] == "ORDER":
            order = self.app.order_values.get(order_id)
            if order is not None:
                order.update({
                    "status": "Inactive" if record["error_code"] == 201 else "Error",
                    "error_code": record["error_code"],
                    "error_string": record["error_string"],
                    "advanced_reject_reason": record["advanced_order_reject_json"] or None,
                })
        if record["category"] in {"CONNECTIVITY", "FATAL_BROKER"}:
            reconnecting = record["error_code"] in {1101, 1102}
            self.connected = False
            self.broker_state_known = False
            self._raw_state_ready = False
            self._last_snapshot = None
            self._reconciliation_required = True
            if self.app:
                self.app.next_order_id = None
            if self.store:
                self.store.set_runtime("service_status", "RECONNECTING" if reconnecting else "SAFE_MODE")
                self.store.set_runtime("safe_mode", False if reconnecting else True)
                self.store.set_runtime("trading_enabled", False)
                self.store.set_runtime("broker_status", "RECONCILIATION_REQUIRED" if reconnecting else "DISCONNECTED")
                self.store.set_runtime("last_error", f"IBKR {record['error_code']}: {record['error_string']}")

    def probe_market_data(
        self,
        symbol: str = "SPY",
        market_data_types: tuple[int, ...] = (1, 2, 3, 4),
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Probe market-data modes independently for read-only setup diagnostics.

        This deliberately does not replace ``get_execution_quote``. A probe may
        establish that research data is available while execution still fails
        closed on delayed, stale, or incomplete data.
        """
        if not self.connected or not self._transport_ready():
            raise RuntimeError("IBKR is disconnected; market data probe unavailable")
        instrument = self.resolve_instrument(symbol)
        if not hasattr(self.app, "probe_quote_requests"):
            self.app.probe_quote_requests = set()
        timeout = max(self.config.quote_timeout_seconds, 8.0) if timeout_seconds is None else float(timeout_seconds)
        probes: list[dict[str, Any]] = []

        for requested_type in market_data_types:
            request_id = self._next_quote_request_id
            self._next_quote_request_id += 1
            event = threading.Event()
            self.app.quote_values[request_id] = {"requested_type": int(requested_type)}
            self.app.quote_events[request_id] = event
            self.app.probe_quote_requests.add(request_id)
            started = time.monotonic()
            request_error: Exception | None = None
            cancel_error: Exception | None = None
            try:
                self.app.reqMarketDataType(int(requested_type))
                self.app.reqMktData(request_id, instrument.contract, "", False, False, [])
                event.wait(max(0.0, timeout))
            except Exception as exc:
                request_error = exc
            finally:
                try:
                    self.app.cancelMktData(request_id)
                except Exception as exc:
                    cancel_error = exc
                values = dict(self.app.quote_values.pop(request_id, {}))
                error = getattr(self.app, "request_errors", {}).pop(request_id, None)
                warnings = getattr(self.app, "request_warnings", {}).pop(request_id, [])
                self.app.quote_events.pop(request_id, None)
                self.app.probe_quote_requests.discard(request_id)

            if cancel_error is not None and request_error is None and error is None:
                request_error = cancel_error
            if request_error is not None and error is None:
                error = {
                    "error_code": None,
                    "error_string": str(request_error),
                    "advanced_order_reject_json": "",
                }

            has_price = _quote_has_price(values)
            complete = _quote_has_price(values) and values.get("data_type") in _MARKET_DATA_TYPE_NAMES.values()
            if error is not None:
                result = "ERROR"
            elif complete:
                result = "PASS"
            elif has_price and not values.get("data_type"):
                result = "NO_DATA_TYPE"
            elif warnings:
                result = "FAIL"
            else:
                result = "NO_CURRENT_TICK"

            warning = warnings[-1] if warnings else None
            if error is None and warning is not None and result != "PASS":
                error = warning

            callback_type = values.get("callback_market_data_type")
            probe = {
                "request_id": request_id,
                "requested_type": int(requested_type),
                "requested_type_name": _MARKET_DATA_TYPE_NAMES.get(int(requested_type), "UNKNOWN"),
                "callback_market_data_type": callback_type,
                "data_type": values.get("data_type", "UNKNOWN"),
                "bid": values.get("bid"),
                "ask": values.get("ask"),
                "last": values.get("last"),
                "close": values.get("close"),
                "open": values.get("open"),
                "high": values.get("high"),
                "low": values.get("low"),
                "volume": values.get("volume"),
                "bid_size": values.get("bid_size"),
                "ask_size": values.get("ask_size"),
                "last_size": values.get("last_size"),
                "timestamp": values.get("timestamp"),
                "broker_timestamp": values.get("timestamp"),
                "received_tick": bool(values.get("received_tick", False)),
                "received_tick_types": list(values.get("received_tick_types", [])),
                "error_code": error.get("error_code") if error else None,
                "error_message": (error.get("error_string") or error.get("error_message", "")) if error else "",
                "advanced_order_reject_json": error.get("advanced_order_reject_json", "") if error else "",
                "warning_code": warning.get("error_code") if warning else None,
                "warning_message": warning.get("error_string", "") if warning else "",
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                "result": result,
            }
            probes.append(probe)

        selected = next((probe for probe in probes if probe["result"] == "PASS"), None)
        selected_type = selected.get("data_type") if selected else None
        final_state = {
            "REALTIME": "LIVE",
            "FROZEN": "MARKET_CLOSED / FROZEN",
            "DELAYED": "DELAYED",
            "DELAYED_FROZEN": "MARKET_CLOSED / DELAYED_FROZEN",
        }.get(selected_type, "UNAVAILABLE")
        contract = instrument.contract
        return {
            "symbol": instrument.yahoo_symbol,
            "contract": {
                "symbol": getattr(contract, "symbol", ""),
                "sec_type": getattr(contract, "secType", ""),
                "exchange": getattr(contract, "exchange", ""),
                "primary_exchange": getattr(contract, "primaryExchange", ""),
                "currency": getattr(contract, "currency", ""),
                "con_id": getattr(contract, "conId", instrument.con_id),
                "local_symbol": getattr(contract, "localSymbol", instrument.local_symbol),
            },
            "probes": probes,
            "selected": selected,
            "final_state": final_state,
        }

    def get_execution_quote(self, symbol: str, allow_delayed: bool | None = None):
        if not self.connected or not self._transport_ready():
            raise RuntimeError("IBKR is disconnected; execution quote unavailable")
        if self._reconciliation_required and not self._raw_state_ready:
            raise RuntimeError("IBKR reconciliation required before execution quote")
        instrument = self.resolve_instrument(symbol)
        request_id = self._next_quote_request_id
        self._next_quote_request_id += 1
        event = threading.Event()
        self.app.quote_values[request_id] = {}
        self.app.quote_events[request_id] = event
        values = {}
        error = None
        delayed_allowed = self.config.allow_delayed_quotes if allow_delayed is None else bool(allow_delayed)
        try:
            self.app.reqMarketDataType(3 if delayed_allowed else 1)
            self.app.reqMktData(request_id, instrument.contract, "165,233", False, False, [])
            if not event.wait(self.config.quote_timeout_seconds):
                raise TimeoutError(f"IBKR execution quote timed out for {symbol}")
        finally:
            values = self.app.quote_values.pop(request_id, {})
            error = getattr(self.app, "request_errors", {}).pop(request_id, None)
            getattr(self.app, "request_warnings", {}).pop(request_id, None)
            if not error or int(error.get("error_code", -1)) not in _TERMINAL_MARKET_DATA_ERROR_CODES:
                self.app.cancelMktData(request_id)
            self.app.quote_events.pop(request_id, None)
        if error:
            if int(error["error_code"]) == 2186:
                raise RuntimeError(
                    f"IBKR API realtime market data subscription unavailable for {symbol.upper()} "
                    "(2186); delayed data cannot be used for PAPER execution"
                )
            raise RuntimeError(f"IBKR market data error {error['error_code']}: {error['error_string']}")
        data_type = values.get("data_type")
        if data_type in {"FROZEN", "DELAYED_FROZEN"}:
            raise RuntimeError("IBKR frozen quote is available for observation only; execution is fail-closed")
        if data_type in {"DELAYED", "DELAYED_FROZEN"} and not delayed_allowed:
            raise RuntimeError("IBKR delayed quote received but delayed quotes are not enabled")
        bid = values.get("bid") if _positive_quote_value(values, "bid") else None
        ask = values.get("ask") if _positive_quote_value(values, "ask") else None
        last = values.get("last") if _positive_quote_value(values, "last") else None
        mid = (bid + ask) / 2 if bid and ask else None
        price = mid or last
        if not price or not values.get("timestamp"):
            raise RuntimeError("IBKR execution quote is missing price or broker timestamp")
        market_status = "CLOSED" if data_type in {"FROZEN", "DELAYED_FROZEN"} else "OPEN"
        return {"symbol": instrument.yahoo_symbol, "price": price, "bid": bid, "ask": ask, "mid": mid, "timestamp": values["timestamp"], "market_status": market_status, "source": "IBKR", "data_type": data_type}

    def get_historical_fallback(self, symbol: str, timeout_seconds: float | None = None) -> dict[str, Any]:
        """Return the latest completed daily IBKR bar for closed-market observation."""
        if not self.connected or not self._transport_ready():
            raise RuntimeError("IBKR is disconnected; historical market data unavailable")
        instrument = self.resolve_instrument(symbol)
        request_id = self._next_quote_request_id
        self._next_quote_request_id += 1
        event = threading.Event()
        if not hasattr(self.app, "historical_values"):
            self.app.historical_values = {}
            self.app.historical_events = {}
            self.app.historical_completed = set()
        self.app.historical_values[request_id] = []
        self.app.historical_events[request_id] = event
        timeout = max(self.config.quote_timeout_seconds, 8.0) if timeout_seconds is None else float(timeout_seconds)
        submitted = False
        try:
            self.app.reqHistoricalData(
                request_id, instrument.contract, "", "5 D", "1 day", "TRADES", 1, 2, False, [],
            )
            submitted = True
            event.wait(max(0.0, timeout))
        finally:
            completed = request_id in self.app.historical_completed
            if submitted and not completed:
                self.app.cancelHistoricalData(request_id)
            bars = list(self.app.historical_values.pop(request_id, []))
            self.app.historical_events.pop(request_id, None)
            self.app.historical_completed.discard(request_id)
            error = getattr(self.app, "request_errors", {}).pop(request_id, None)
            warnings = getattr(self.app, "request_warnings", {}).pop(request_id, [])
        if error:
            raise RuntimeError(f"IBKR historical data error {error['error_code']}: {error['error_string']}")
        if not completed:
            if warnings:
                warning = warnings[-1]
                raise RuntimeError(
                    f"IBKR historical data warning {warning['error_code']}: {warning['error_string']}; no completed bar received"
                )
            raise TimeoutError(f"IBKR historical data timed out for {symbol}")
        valid = [bar for bar in bars if bar.get("bar_date") and float(bar.get("close", 0)) > 0]
        if not valid:
            raise RuntimeError(f"IBKR historical data returned no completed positive close for {symbol}")
        latest = valid[-1]
        warning = warnings[-1] if warnings else None
        return {
            "symbol": instrument.yahoo_symbol,
            **latest,
            "source": "IBKR historical",
            "warning_code": warning.get("error_code") if warning else None,
            "warning_message": warning.get("error_string", "") if warning else "",
        }

    def position_symbols(self, timeout_seconds: float | None = None) -> list[str]:
        self._refresh_broker_state(timeout_seconds)
        self._validate_position_mandate()
        return sorted(instrument_identity(value["symbol"])[1] for value in self.app.position_values.values() if float(value.get("quantity", 0)) > 0)

    def _validate_position_mandate(self):
        positions = [value for value in self.app.position_values.values() if abs(float(value.get("quantity", 0))) > 1e-9]
        if any(float(value.get("quantity", 0)) < 0 for value in positions):
            raise RuntimeError("IBKR short positions are unsupported; reconciliation failed closed")
        for value in positions:
            if not all(value.get(field) for field in ("symbol", "sec_type", "currency", "exchange")):
                raise RuntimeError("IBKR position has an unknown contract; reconciliation failed closed")
            if str(value["sec_type"]).upper() != "STK":
                raise RuntimeError("IBKR unsupported security; only STK positions are allowed")
            if str(value["currency"]).upper() != "USD":
                raise RuntimeError("IBKR unsupported currency; only USD positions are allowed")
        if len(positions) > 1:
            raise RuntimeError("IBKR multiple long positions violate max_positions=1")

    def _refresh_broker_state(self, timeout_seconds: float | None = None):
        if not self.connected or not self._transport_ready():
            raise RuntimeError("IBKR is disconnected")
        self.broker_state_known = False
        self._raw_state_ready = False
        timeout = self.config.reconciliation_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        for event in (self.app.account_summary_end_event, self.app.positions_end_event, self.app.open_orders_end_event, self.app.executions_end_event):
            event.clear()
        self.app.account_values.clear()
        self.app.position_values.clear()
        self.app.order_values.clear()
        self.app.execution_values.clear()
        if hasattr(self.app, "commission_values"):
            self.app.commission_values.clear()

        self._account_summary_active = True
        self._positions_active = True
        try:
            self.app.reqAccountSummary(9001, "All", "TotalCashValue,NetLiquidation,AvailableFunds")
            self.app.reqPositions()
            self.app.reqAllOpenOrders()
            execution_filter = ExecutionFilter() if _IBAPI_AVAILABLE else None
            self.app.reqExecutions(9002, execution_filter)

            deadline = time.monotonic() + timeout
            events = (
                (self.app.account_summary_end_event, "account summary"),
                (self.app.positions_end_event, "positions"),
                (self.app.open_orders_end_event, "open orders"),
                (self.app.executions_end_event, "executions"),
            )
            for event, label in events:
                remaining = max(0.0, deadline - time.monotonic())
                if not event.wait(remaining):
                    raise TimeoutError(f"IBKR reconciliation timed out waiting for {label}")
        finally:
            if self._account_summary_active and hasattr(self.app, "cancelAccountSummary"):
                self.app.cancelAccountSummary(9001)
            if self._positions_active and hasattr(self.app, "cancelPositions"):
                self.app.cancelPositions()
            self._account_summary_active = False
            self._positions_active = False

        if self.app.account_values.get("TotalCashValue") is None or self.app.account_values.get("NetLiquidation") is None:
            raise RuntimeError("IBKR reconciliation missing cash or net liquidation")
        self._rebuild_order_mappings()
        self._reject_unknown_open_orders()
        self._raw_state_ready = True

    def reconcile(self, prices: dict[str, float] | None = None, timeout_seconds: float | None = None) -> BrokerSnapshot:
        if not self._raw_state_ready:
            self._refresh_broker_state(timeout_seconds)

        cash_value = self.app.account_values.get("TotalCashValue")
        equity_value = self.app.account_values.get("NetLiquidation")

        price_map = {symbol.upper(): float(value) for symbol, value in (prices or {}).items()}
        positions = []
        self._validate_position_mandate()
        for raw in self.app.position_values.values():
            quantity = float(raw.get("quantity", 0))
            if quantity <= 0:
                continue
            symbol = instrument_identity(raw["symbol"])[1]
            price = price_map.get(symbol)
            if price is None or price <= 0:
                raise RuntimeError(f"IBKR position price is unknown for {symbol}")
            average_cost = float(raw.get("average_cost", 0.0) or 0.0)
            positions.append(PositionSnapshot(symbol=symbol, quantity=quantity, market_price=price, market_value=quantity * price, average_cost=average_cost, unrealized_pnl=(price - average_cost) * quantity if average_cost else 0.0))

        open_orders = []
        for order_id, order in self.app.order_values.items():
            client_order_id = order.get("client_order_id")
            if order.get("status") not in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
                open_orders.append(dict(order))

        snapshot = BrokerSnapshot(equity=float(equity_value), cash=float(cash_value), positions=positions, open_orders=open_orders, source=self.mode, invested_value=sum(position.market_value for position in positions), unrealized_pnl=sum(position.unrealized_pnl for position in positions))
        self._last_snapshot = snapshot
        self.broker_state_known = True
        self._reconciliation_required = False
        self._raw_state_ready = False
        return snapshot

    def _rebuild_order_mappings(self):
        self._client_orders.clear()
        for order_id, order in self.app.order_values.items():
            client_order_id = order.get("client_order_id")
            if client_order_id:
                self._client_orders[client_order_id] = int(order_id)
                if self.store:
                    stored = self.store.order_by_client_id(client_order_id)
                    if stored:
                        self._order_decisions[client_order_id] = stored["decision_id"]
        for execution in self.app.execution_values:
            client_order_id = execution.get("client_order_id")
            order_id = int(execution["order_id"])
            if not client_order_id and self.store:
                stored = self.store.order_by_broker_ids(order_id, execution.get("perm_id"))
                client_order_id = stored["client_order_id"] if stored else None
            if not client_order_id:
                continue
            self._client_orders[client_order_id] = order_id
            if self.store:
                stored = self.store.order_by_client_id(client_order_id)
                if stored:
                    self._order_decisions[client_order_id] = stored["decision_id"]
            order = self.app.order_values.setdefault(order_id, {})
            commission = getattr(self.app, "commission_values", {}).get(execution.get("exec_id"), {})
            execution_cost = float(commission.get("total_execution_cost", commission.get("commission", 0)) or 0)
            order.update({
                "order_id": order_id,
                "client_order_id": client_order_id,
                "perm_id": execution.get("perm_id"),
                "status": "Filled",
                "filled": float(order.get("filled", 0)) + float(execution.get("shares", 0)),
                "remaining": 0.0,
                "avg_fill_price": execution.get("price"),
                "commission": float(order.get("commission", 0)) + execution_cost,
                "total_execution_cost": float(order.get("total_execution_cost", 0)) + execution_cost,
            })

    def _reject_unknown_open_orders(self):
        terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
        for order in self.app.order_values.values():
            if order.get("status") in terminal:
                continue
            client_order_id = order.get("client_order_id")
            stored = self.store.order_by_client_id(client_order_id) if self.store and client_order_id else None
            if client_order_id not in self._known_client_order_ids and stored is None:
                raise RuntimeError("UNKNOWN BROKER OPEN ORDER; account-wide reconciliation failed closed")

    def _cancel_broker_order(self, order_id: int):
        if not self.app:
            raise RuntimeError("IBKR is disconnected; order cancellation unavailable")
        if OrderCancel is not None:
            cancel = OrderCancel()
            cancel.manualOrderIndicator = 0
            cancel.manualOrderCancelTime = ""
            try:
                self.app.cancelOrder(order_id, cancel)
                return
            except TypeError:
                pass
        self.app.cancelOrder(order_id)

    def cancel_order(self, client_order_id: str):
        if not self.app or client_order_id not in self._client_orders:
            return ExecutionReport(client_order_id=client_order_id, decision_id="unknown", status="REJECTED", message="Unknown order")
        self.cancel_order_calls += 1
        self._cancel_broker_order(self._client_orders[client_order_id])
        pending = ExecutionReport(client_order_id=client_order_id, decision_id=self._order_decisions.get(client_order_id, "unknown"), status="PENDING", broker_order_id=self._client_orders[client_order_id], message="Cancel requested")
        return self.await_order(client_order_id) or pending

    def order_status(self, client_order_id: str):
        if not self.app or client_order_id not in self._client_orders:
            return None
        order = self.app.order_values.get(self._client_orders[client_order_id])
        if not order:
            return None
        return self._status_report(client_order_id, order)

    def execution_report(self, client_order_id: str):
        return self.order_status(client_order_id)

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        request = request.model_copy(update={"symbol": request.symbol.upper()})
        if request.client_order_id in self._client_orders:
            return self.order_status(request.client_order_id) or ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="PENDING", broker_order_id=self._client_orders[request.client_order_id], message="Order already exists at broker")
        if not self.connected or not self._transport_ready() or not self.app or self.app.next_order_id is None:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message="IBKR is disconnected or has no nextValidId")
        if not self.broker_state_known:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message="IBKR account state is not reconciled; order blocked")
        try:
            instrument = self.resolve_instrument(request.symbol)
        except (RuntimeError, TimeoutError) as exc:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message=str(exc))
        order_id = self.app.next_order_id
        if request.order_type == "LMT":
            order = limit_order(request.action, Decimal(str(request.quantity)), Decimal(str(request.limit_price)))
        else:
            order = market_order(request.action, Decimal(str(request.quantity)))
        order.orderRef = request.client_order_id
        self._client_orders[request.client_order_id] = order_id
        self._known_client_order_ids.add(request.client_order_id)
        self._order_decisions[request.client_order_id] = request.decision_id
        self.app.order_values[order_id] = {"order_id": order_id, "client_order_id": request.client_order_id, "symbol": request.symbol, "action": request.action, "quantity": request.quantity, "order_type": request.order_type, "limit_price": request.limit_price, "status": "Submitted", "filled": 0.0, "remaining": request.quantity}
        self.place_order_calls += 1
        self.app.placeOrder(order_id, instrument.contract, order)
        self.app.next_order_id += 1
        rejected = self.order_status(request.client_order_id)
        if rejected and rejected.status in {"REJECTED", "ERROR"}:
            return rejected
        return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="PENDING", broker_order_id=order_id, remaining_quantity=request.quantity, message=f"Submitted paper order {order_id}")

    def await_order(self, client_order_id: str, timeout_seconds: float | None = None) -> ExecutionReport | None:
        timeout = self.config.order_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        deadline = time.monotonic() + timeout
        while True:
            report = self.order_status(client_order_id)
            if report is None or report.status in {"FILLED", "CANCELLED", "REJECTED", "ERROR"}:
                return report
            if time.monotonic() >= deadline:
                return report.model_copy(update={"status": "TIMEOUT", "message": "IBKR order status timeout"})
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _status_report(self, client_order_id: str, order: dict[str, Any]) -> ExecutionReport:
        status_map = {"ApiPending": "PENDING", "PendingSubmit": "PENDING", "PreSubmitted": "PENDING", "Submitted": "PENDING", "PendingCancel": "PENDING", "Filled": "FILLED", "PartiallyFilled": "PARTIALLY_FILLED", "Cancelled": "CANCELLED", "ApiCancelled": "CANCELLED", "Inactive": "REJECTED"}
        filled = float(order.get("filled", 0))
        remaining = float(order.get("remaining", 0))
        status = status_map.get(order.get("status"), "ERROR")
        if filled > 0 and remaining > 0 and status == "PENDING":
            status = "PARTIALLY_FILLED"
        error_string = order.get("error_string")
        message = f"IBKR {order['error_code']}: {error_string}" if order.get("error_code") else order.get("status", "")
        total_execution_cost = float(order.get("total_execution_cost", order.get("commission", 0)) or 0)
        return ExecutionReport(client_order_id=client_order_id, decision_id=self._order_decisions.get(client_order_id, "unknown"), status=status, filled_quantity=filled, average_price=order.get("avg_fill_price"), broker_order_id=order.get("order_id"), perm_id=order.get("perm_id"), remaining_quantity=remaining, commission=float(order.get("commission", 0)), fees=float(order.get("fees", 0)), slippage=float(order.get("slippage", 0)), total_execution_cost=total_execution_cost, message=message, error_code=order.get("error_code"), error_string=error_string, advanced_reject_reason=order.get("advanced_reject_reason"))
