from __future__ import annotations

from .base import Executor
from ..models import BrokerSnapshot, ExecutionReport, OrderRequest


class ObserveExecutor(Executor):
    """Read-only executor that records hypothetical fills without changing an account."""

    mode = "OBSERVE"
    broker_source = "LOCAL"
    execution_mode = "OBSERVE"
    mutations_allowed = False

    def __init__(self, starting_cash: float = 0.0):
        self.starting_cash = float(starting_cash)
        self._cash = self.starting_cash
        self.connected = False
        self.broker_state_known = False
        self._reports: dict[str, ExecutionReport] = {}
        self.place_order_calls = 0
        self.cancel_order_calls = 0

    def connect(self):
        self.connected = True
        self.broker_state_known = True
        return {"connected": True, "mode": self.mode}

    def disconnect(self):
        self.connected = False
        self.broker_state_known = False

    def reconnect(self):
        self.disconnect()
        return self.connect()

    def position_symbols(self, timeout_seconds: float | None = None) -> list[str]:
        if not self.connected:
            raise RuntimeError("Observe executor is disconnected")
        return []

    def reconcile(self, prices: dict[str, float] | None = None, timeout_seconds: float | None = None) -> BrokerSnapshot:
        if not self.connected:
            raise RuntimeError("Observe executor is disconnected")
        return BrokerSnapshot(equity=self._cash, cash=self._cash, positions=[], open_orders=[], source=self.mode)

    def account_summary(self):
        snapshot = self.reconcile()
        return {"cash": snapshot.cash, "equity": snapshot.equity, "positions": [], "mode": self.mode}

    def cash_balance(self):
        return self._cash

    def current_positions(self):
        return []

    def positions(self):
        return []

    def current_orders(self):
        return []

    def simulate_order(self, request: OrderRequest, label: str | None = None) -> ExecutionReport:
        if not self.connected:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message="Observe executor is disconnected")
        if request.client_order_id in self._reports:
            return self._reports[request.client_order_id]
        action_label = label or request.action
        report = ExecutionReport(
            client_order_id=request.client_order_id,
            decision_id=request.decision_id,
            status="SIMULATED",
            filled_quantity=request.quantity,
            average_price=request.reference_price,
            message=f"WOULD_{action_label}: OBSERVE mode; order was not sent",
        )
        self._reports[request.client_order_id] = report
        return report

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        return self.simulate_order(request)

    def await_order(self, client_order_id: str, timeout_seconds: float | None = None) -> ExecutionReport | None:
        return self.order_status(client_order_id)

    def simulate_cancel(self, client_order_id: str) -> ExecutionReport:
        report = self._reports.get(client_order_id)
        if report is None:
            return ExecutionReport(client_order_id=client_order_id, decision_id="unknown", status="REJECTED", message="Unknown observed order")
        return report.model_copy(update={"status": "CANCELLED", "message": "WOULD_CANCEL: OBSERVE mode; broker order was not cancelled"})

    def cancel_order(self, client_order_id: str) -> ExecutionReport:
        return self.simulate_cancel(client_order_id)

    def order_status(self, client_order_id: str) -> ExecutionReport | None:
        return self._reports.get(client_order_id)


class ReadOnlyBroker(Executor):
    """Read-only facade over a broker adapter for real Paper OBSERVE runs."""

    execution_mode = "OBSERVE"
    mutations_allowed = False

    def __init__(self, broker: Executor):
        self._broker = broker
        self.mode = getattr(broker, "mode", "IBKR_PAPER")
        self.broker_source = getattr(broker, "broker_source", "IBKR_PAPER")
        self._reports: dict[str, ExecutionReport] = {}
        self.place_order_calls = 0
        self.cancel_order_calls = 0

    @property
    def connected(self):
        return bool(getattr(self._broker, "connected", False))

    @connected.setter
    def connected(self, value):
        self._broker.connected = value

    @property
    def broker_state_known(self):
        return bool(getattr(self._broker, "broker_state_known", False))

    @broker_state_known.setter
    def broker_state_known(self, value):
        self._broker.broker_state_known = value

    def connect(self):
        return self._broker.connect()

    def disconnect(self):
        return self._broker.disconnect()

    def reconnect(self):
        return self._broker.reconnect()

    def position_symbols(self, timeout_seconds: float | None = None) -> list[str]:
        return self._broker.position_symbols(timeout_seconds)

    def reconcile(self, prices: dict[str, float] | None = None, timeout_seconds: float | None = None) -> BrokerSnapshot:
        return self._broker.reconcile(prices, timeout_seconds)

    def account_summary(self):
        return self._broker.account_summary()

    def cash_balance(self):
        return self._broker.cash_balance()

    def current_positions(self):
        return self._broker.current_positions()

    def positions(self):
        return self._broker.positions()

    def current_orders(self):
        return self._broker.current_orders()

    def get_execution_quote(self, symbol: str, allow_delayed: bool | None = None):
        if allow_delayed is None:
            return self._broker.get_execution_quote(symbol)
        return self._broker.get_execution_quote(symbol, allow_delayed=allow_delayed)

    def resolve_instrument(self, symbol: str):
        return self._broker.resolve_instrument(symbol)

    def probe_market_data(self, symbol: str = "SPY", market_data_types=(1, 2, 3, 4), timeout_seconds=None):
        """Expose the underlying read-only market-data diagnostic only."""
        return self._broker.probe_market_data(symbol, market_data_types, timeout_seconds)

    def simulate_order(self, request: OrderRequest, label: str | None = None) -> ExecutionReport:
        if not self.connected:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message="IBKR OBSERVE broker is disconnected")
        if request.client_order_id in self._reports:
            return self._reports[request.client_order_id]
        action_label = label or request.action
        report = ExecutionReport(
            client_order_id=request.client_order_id,
            decision_id=request.decision_id,
            status="SIMULATED",
            filled_quantity=request.quantity,
            average_price=request.reference_price,
            message=f"WOULD_{action_label}: IBKR OBSERVE; order was not sent",
        )
        self._reports[request.client_order_id] = report
        return report

    def simulate_cancel(self, client_order_id: str) -> ExecutionReport:
        report = self._reports.get(client_order_id)
        if report is None:
            existing = self._broker.order_status(client_order_id)
            if existing is None:
                return ExecutionReport(client_order_id=client_order_id, decision_id="unknown", status="REJECTED", message="Unknown observed order")
            return existing.model_copy(update={"status": "SIMULATED", "message": "WOULD_CANCEL: IBKR OBSERVE; broker order was not cancelled"})
        return report.model_copy(update={"status": "CANCELLED", "message": "WOULD_CANCEL: IBKR OBSERVE; broker order was not cancelled"})

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        return self.simulate_order(request)

    def await_order(self, client_order_id: str, timeout_seconds: float | None = None) -> ExecutionReport | None:
        return self.order_status(client_order_id)

    def cancel_order(self, client_order_id: str) -> ExecutionReport:
        return self.simulate_cancel(client_order_id)

    def order_status(self, client_order_id: str) -> ExecutionReport | None:
        if client_order_id in self._reports:
            return self._reports[client_order_id]
        return self._broker.order_status(client_order_id)

    def execution_report(self, client_order_id: str) -> ExecutionReport | None:
        return self.order_status(client_order_id)
