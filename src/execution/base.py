from abc import ABC, abstractmethod

from ..models import BrokerSnapshot, ExecutionReport, OrderRequest


class Executor(ABC):
    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError

    @abstractmethod
    def position_symbols(self, timeout_seconds: float | None = None) -> list[str]:
        """Refresh broker state and return symbols that require market prices."""
        raise NotImplementedError

    @abstractmethod
    def reconcile(self, prices: dict[str, float] | None = None, timeout_seconds: float | None = None) -> BrokerSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        raise NotImplementedError

    @abstractmethod
    def await_order(self, client_order_id: str, timeout_seconds: float | None = None) -> ExecutionReport | None:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> ExecutionReport:
        raise NotImplementedError

    @abstractmethod
    def order_status(self, client_order_id: str) -> ExecutionReport | None:
        raise NotImplementedError
