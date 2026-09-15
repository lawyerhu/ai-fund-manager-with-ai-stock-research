from __future__ import annotations

from .base import Executor
from ..models import BrokerSnapshot, ExecutionReport, OrderRequest, PositionSnapshot


class LocalPaperExecutor(Executor):
    """Deterministic in-process broker used for offline tests and dry runs."""

    mode = "LOCAL_PAPER"
    broker_source = "LOCAL"
    execution_mode = "PAPER"
    mutations_allowed = True

    def __init__(self, fractional_shares: bool = True, minimum_order_notional: float = 1.0, starting_cash: float = 0.0, store=None, commission_per_order: float = 0.0, commission_per_share: float = 0.0, minimum_commission: float = 0.0, slippage_bps: float = 0.0):
        self.fractional_shares = fractional_shares
        self.minimum_order_notional = minimum_order_notional
        self.connected = False
        self.broker_state_known = False
        self._orders: dict[str, OrderRequest] = {}
        self._reports: dict[str, ExecutionReport] = {}
        self._positions: dict[str, float] = {}
        self._average_costs: dict[str, float] = {}
        self._realized_pnl = 0.0
        self._cash = starting_cash
        self._prices: dict[str, float] = {}
        self.store = store
        self.commission_per_order = float(commission_per_order)
        self.commission_per_share = float(commission_per_share)
        self.minimum_commission = float(minimum_commission)
        self.slippage_bps = float(slippage_bps)
        self._restored = False

    def connect(self):
        self._restore_state()
        self.connected = True
        self.broker_state_known = True
        self._persist_state()
        return {"connected": True, "mode": "LOCAL_PAPER"}

    def disconnect(self):
        self.connected = False
        self.broker_state_known = False

    def reconnect(self):
        self.disconnect()
        return self.connect()

    def account_summary(self) -> dict:
        snapshot = self.reconcile()
        return {"cash": snapshot.cash, "equity": snapshot.equity, "positions": self.current_positions(), "mode": self.mode}

    def cash_balance(self) -> float:
        return self._cash

    def current_positions(self) -> list[dict]:
        positions = []
        for symbol, quantity in self._positions.items():
            if quantity <= 0:
                continue
            price = self._prices.get(symbol)
            average_cost = self._average_costs.get(symbol, 0.0)
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "market_price": price,
                "market_value": quantity * price if price is not None else None,
                "average_cost": average_cost,
                "unrealized_pnl": (price - average_cost) * quantity if price is not None else None,
                "realized_pnl": self._realized_pnl,
            })
        return positions

    def positions(self):
        return self.current_positions()

    def position_symbols(self, timeout_seconds: float | None = None) -> list[str]:
        if not self.connected:
            raise RuntimeError("Paper broker is disconnected")
        return sorted(symbol for symbol, quantity in self._positions.items() if quantity > 0)

    def current_orders(self) -> list[OrderRequest]:
        terminal = {"FILLED", "CANCELLED", "REJECTED", "ERROR"}
        pending = []
        for order_id, request in self._orders.items():
            status = self._reports.get(order_id)
            if status is None or status.status not in terminal:
                pending.append(request)
        return pending

    def cancel_order(self, client_order_id: str) -> ExecutionReport:
        request = self._orders.get(client_order_id)
        if request is None:
            return ExecutionReport(client_order_id=client_order_id, decision_id="unknown", status="REJECTED", message="Unknown order")
        existing = self._reports.get(client_order_id)
        if existing and existing.status in {"FILLED", "CANCELLED", "REJECTED", "ERROR"}:
            return existing
        report = ExecutionReport(client_order_id=client_order_id, decision_id=request.decision_id, status="CANCELLED", message="Cancelled by operator")
        self._reports[client_order_id] = report
        self._persist_state()
        return report

    def order_status(self, client_order_id: str) -> ExecutionReport | None:
        return self._reports.get(client_order_id)

    def execution_report(self, client_order_id: str) -> ExecutionReport | None:
        return self.order_status(client_order_id)

    def submit_order(self, request: OrderRequest, fill_ratio: float = 1.0, fill_price: float | None = None) -> ExecutionReport:
        request = request.model_copy(update={"symbol": request.symbol.upper()})
        existing_report = self._reports.get(request.client_order_id)
        if existing_report is not None:
            return existing_report
        if request.client_order_id in self._orders:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="PENDING", message="Order exists without a broker report")
        duplicate = next((item for item in self._orders.values() if item.decision_id == request.decision_id and item.symbol == request.symbol and item.action == request.action), None)
        if duplicate is not None:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="REJECTED", message="Duplicate decision/order submission")
        if not self.connected:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="ERROR", message="Paper broker is disconnected")

        def reject(message: str) -> ExecutionReport:
            report = ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="REJECTED", message=message)
            self._orders[request.client_order_id] = request
            self._reports[request.client_order_id] = report
            self._persist_state()
            return report

        if request.reference_price is not None and request.quantity * request.reference_price < self.minimum_order_notional:
            return reject("Order is below minimum notional")
        if not self.fractional_shares and request.quantity != int(request.quantity):
            return reject("Fractional shares are not enabled")
        if not 0 <= fill_ratio <= 1:
            raise ValueError("fill_ratio must be between 0 and 1")

        self._orders[request.client_order_id] = request
        filled = request.quantity * fill_ratio
        price = fill_price or self._slipped_price(request)
        commission = self._commission(filled) if filled else 0.0
        if filled and (price is None or price <= 0):
            return reject("Latest price is unavailable")
        if filled and price:
            self._prices[request.symbol] = float(price)
            signed_quantity = filled if request.action == "BUY" else -filled
            if request.action == "BUY" and signed_quantity * price + commission > self._cash + 1e-9:
                return reject("Insufficient cash; leverage is disabled")
            if request.action == "SELL" and filled > self._positions.get(request.symbol, 0.0) + 1e-9:
                return reject("Insufficient position; short selling is disabled")
            self._apply_fill(request, filled, price, commission)
        status = "FILLED" if fill_ratio >= 1 else "PENDING" if fill_ratio == 0 else "PARTIALLY_FILLED"
        slippage = abs(price - request.reference_price) * filled if filled and request.reference_price else 0.0
        report = ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status=status, filled_quantity=filled, average_price=price if filled else None, remaining_quantity=request.quantity - filled, commission=commission, slippage=slippage, total_execution_cost=commission, message="Local paper fill")
        self._reports[request.client_order_id] = report
        self._persist_state()
        return report

    def fill_order(self, client_order_id: str, fill_ratio: float = 1.0, fill_price: float | None = None) -> ExecutionReport:
        """Apply a later fill to an existing partial order without resubmitting it."""
        request = self._orders.get(client_order_id)
        previous = self._reports.get(client_order_id)
        if request is None or previous is None or previous.status not in {"PENDING", "PARTIALLY_FILLED"}:
            raise ValueError("Only an existing pending or partially filled order can receive a later fill")
        if not 0 <= fill_ratio <= 1:
            raise ValueError("fill_ratio must be between 0 and 1")
        remaining = request.quantity - previous.filled_quantity
        additional = remaining * fill_ratio
        price = fill_price or self._slipped_price(request) or previous.average_price
        if price is None or price <= 0:
            raise ValueError("Latest price is unavailable")
        signed_quantity = additional if request.action == "BUY" else -additional
        total_commission = self._commission(previous.filled_quantity + additional)
        incremental_commission = max(0.0, total_commission - previous.commission)
        if request.action == "BUY" and signed_quantity * price + incremental_commission > self._cash + 1e-9:
            raise ValueError("Insufficient cash; leverage is disabled")
        if request.action == "SELL" and additional > self._positions.get(request.symbol, 0.0) + 1e-9:
            raise ValueError("Insufficient position; short selling is disabled")
        self._apply_fill(request, additional, price, incremental_commission)
        filled = previous.filled_quantity + additional
        status = "FILLED" if filled >= request.quantity - 1e-9 else "PARTIALLY_FILLED"
        previous_price = previous.average_price or request.reference_price or price
        average_price = ((previous_price * previous.filled_quantity) + (price * additional)) / filled if filled else None
        slippage = abs(average_price - request.reference_price) * filled if request.reference_price else 0.0
        report = ExecutionReport(client_order_id=client_order_id, decision_id=request.decision_id, status=status, filled_quantity=filled, average_price=average_price, remaining_quantity=max(0.0, request.quantity - filled), commission=total_commission, slippage=slippage, total_execution_cost=total_commission, message="Local paper fill update")
        self._reports[client_order_id] = report
        self._prices[request.symbol] = float(price)
        self._persist_state()
        return report

    def reconcile(self, prices: dict[str, float] | None = None, timeout_seconds: float | None = None) -> BrokerSnapshot:
        if not self.connected:
            raise RuntimeError("Paper broker is disconnected")
        if prices:
            self._prices.update({symbol.upper(): float(price) for symbol, price in prices.items() if price is not None and float(price) > 0})
        positions: list[PositionSnapshot] = []
        for symbol, quantity in self._positions.items():
            if quantity <= 0:
                continue
            price = self._prices.get(symbol)
            if price is None or price <= 0:
                raise RuntimeError(f"Unknown market price for {symbol}; account state is not safe to trade")
            average_cost = self._average_costs.get(symbol, 0.0)
            positions.append(PositionSnapshot(symbol=symbol, quantity=quantity, market_price=price, market_value=quantity * price, average_cost=average_cost, unrealized_pnl=(price - average_cost) * quantity, realized_pnl=self._realized_pnl))
        equity = self._cash + sum(position.market_value for position in positions)
        self._persist_state()
        return BrokerSnapshot(equity=equity, cash=self._cash, positions=positions, open_orders=[request.model_dump(mode="json") for request in self.current_orders()], source=self.mode, invested_value=sum(position.market_value for position in positions), unrealized_pnl=sum(position.unrealized_pnl for position in positions), realized_pnl=self._realized_pnl)

    def await_order(self, client_order_id: str, timeout_seconds: float | None = None) -> ExecutionReport | None:
        return self.order_status(client_order_id)

    def _restore_state(self):
        if self._restored:
            return
        self._restored = True
        if self.store is None:
            return
        saved = self.store.load_local_broker_state()
        if not saved:
            return
        self._cash = saved["cash"]
        self._positions = {symbol: float(quantity) for symbol, quantity in saved["positions"].items()}
        self._prices = {symbol: float(price) for symbol, price in saved["prices"].items()}
        self._average_costs = {symbol: float(cost) for symbol, cost in saved.get("average_costs", {}).items()}
        self._realized_pnl = float(saved.get("realized_pnl", 0.0))
        self._orders = {order_id: OrderRequest.model_validate(payload) for order_id, payload in saved["orders"].items()}
        self._reports = {order_id: ExecutionReport.model_validate(payload) for order_id, payload in saved["reports"].items()}

    def _persist_state(self):
        if self.store is None:
            return
        self.store.save_local_broker_state(
            self._cash,
            self._positions,
            self._prices,
            {order_id: request.model_dump(mode="json") for order_id, request in self._orders.items()},
            {order_id: report.model_dump(mode="json") for order_id, report in self._reports.items()},
            self._average_costs,
            self._realized_pnl,
        )

    def _apply_fill(self, request: OrderRequest, quantity: float, price: float, commission: float = 0.0):
        symbol = request.symbol.upper()
        existing = self._positions.get(symbol, 0.0)
        if request.action == "BUY":
            new_quantity = existing + quantity
            previous_cost = self._average_costs.get(symbol, price)
            self._average_costs[symbol] = ((existing * previous_cost) + (quantity * price)) / new_quantity
            self._positions[symbol] = new_quantity
            self._cash -= quantity * price + commission
        else:
            self._realized_pnl += (price - self._average_costs.get(symbol, price)) * quantity
            self._positions[symbol] = max(0.0, existing - quantity)
            self._cash += quantity * price - commission
            if self._positions[symbol] <= 1e-9:
                self._positions[symbol] = 0.0
                self._average_costs.pop(symbol, None)

    def _commission(self, quantity: float) -> float:
        if quantity <= 0:
            return 0.0
        return max(self.minimum_commission, self.commission_per_order + self.commission_per_share * quantity)

    def _slipped_price(self, request: OrderRequest) -> float | None:
        if request.reference_price is None:
            return None
        direction = 1 if request.action == "BUY" else -1
        price = request.reference_price * (1 + direction * self.slippage_bps / 10_000)
        if request.order_type == "LMT" and request.limit_price is not None:
            return min(price, request.limit_price) if request.action == "BUY" else max(price, request.limit_price)
        return price
