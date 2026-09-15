from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ExecutionQuote


class ExecutionQuoteProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> ExecutionQuote | dict:
        raise NotImplementedError


class ResearchSnapshotQuoteProvider(ExecutionQuoteProvider):
    """Simulation-only adapter for OBSERVE and LOCAL_PAPER."""

    def __init__(self, research_data):
        self.research_data = research_data

    def get_quote(self, symbol: str) -> ExecutionQuote:
        snapshot = self.research_data.stock_snapshot(symbol)
        price = snapshot.get("price")
        return ExecutionQuote(
            symbol=symbol,
            price=price,
            bid=price,
            ask=price,
            mid=price,
            timestamp=snapshot.get("quote_as_of") or snapshot.get("as_of"),
            market_status="OPEN" if not snapshot.get("trading_halted") else "HALTED",
            source="RESEARCH_SIMULATION",
            data_type="SIMULATED",
        )


class IBKRExecutionQuoteProvider(ExecutionQuoteProvider):
    def __init__(self, executor, market_clock=None):
        self.executor = executor
        if market_clock is None:
            from .market_clock import USEquityMarketClock

            market_clock = USEquityMarketClock()
        self.market_clock = market_clock

    def get_quote(self, symbol: str) -> ExecutionQuote:
        try:
            return ExecutionQuote.model_validate(self.executor.get_execution_quote(symbol))
        except Exception:
            observe = (
                str(getattr(self.executor, "execution_mode", "PAPER")).upper() == "OBSERVE"
                or not bool(getattr(self.executor, "mutations_allowed", True))
            )
            if not observe or self.market_clock.is_open():
                raise
            return self._closed_market_quote(symbol)

    def get_reconciliation_quote(self, symbol: str) -> ExecutionQuote:
        """Value broker positions without weakening the order quote policy."""
        if self.market_clock.is_open():
            return self.get_quote(symbol)
        return self._closed_market_quote(symbol)

    def _closed_market_quote(self, symbol: str) -> ExecutionQuote:
        raw_executor = getattr(self.executor, "_broker", self.executor)
        bar = raw_executor.get_historical_fallback(symbol)
        completed_session = self.market_clock.latest_completed_session()
        if str(bar.get("bar_date", "")) != completed_session["date"] or float(bar.get("close", 0)) <= 0:
            raise RuntimeError("IBKR historical fallback is not the latest completed market session")
        close = float(bar["close"])
        return ExecutionQuote(
            symbol=str(bar.get("symbol", symbol)).upper(),
            price=close,
            timestamp=completed_session["close"],
            market_status="CLOSED",
            source=bar.get("source", "IBKR historical"),
            data_type="FROZEN",
        )
