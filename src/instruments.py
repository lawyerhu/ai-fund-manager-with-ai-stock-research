from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


_SHARE_CLASS_IDENTITIES = {
    "BRK.B": ("BRK.B", "BRK-B", "BRK B"),
    "BRK-B": ("BRK.B", "BRK-B", "BRK B"),
    "BRK B": ("BRK.B", "BRK-B", "BRK B"),
    "BF.B": ("BF.B", "BF-B", "BF B"),
    "BF-B": ("BF.B", "BF-B", "BF B"),
    "BF B": ("BF.B", "BF-B", "BF B"),
}
_US_PRIMARY_EXCHANGES = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "IEX", "ISLAND"}


def instrument_identity(symbol: str) -> tuple[str, str, str]:
    key = symbol.strip().upper()
    return _SHARE_CLASS_IDENTITIES.get(key, (key, key, key))


@dataclass(frozen=True)
class ResolvedInstrument:
    canonical_symbol: str
    yahoo_symbol: str
    contract: Any
    con_id: int
    local_symbol: str
    primary_exchange: str
    currency: str
    sec_type: str
    min_tick: float = 0.01


class InstrumentResolver:
    """Resolve a research symbol to one verified US IBKR stock contract."""

    def __init__(self, app, contract_factory: Callable[[str], Any], timeout_seconds: float = 10.0):
        self.app = app
        self.contract_factory = contract_factory
        self.timeout_seconds = timeout_seconds
        self._next_request_id = 20000
        self._cache: dict[str, ResolvedInstrument] = {}

    def resolve(self, symbol: str) -> ResolvedInstrument:
        canonical, yahoo_symbol, ibkr_symbol = instrument_identity(symbol)
        if canonical in self._cache:
            return self._cache[canonical]

        request_id = self._next_request_id
        self._next_request_id += 1
        event = threading.Event()
        self.app.contract_detail_values[request_id] = []
        self.app.contract_detail_events[request_id] = event
        self.app.reqContractDetails(request_id, self.contract_factory(ibkr_symbol))
        try:
            if not event.wait(self.timeout_seconds):
                if hasattr(self.app, "cancelContractDetails"):
                    self.app.cancelContractDetails(request_id)
                raise TimeoutError(f"IBKR contract resolution timed out for {symbol}")
        finally:
            self.app.contract_detail_events.pop(request_id, None)

        error = getattr(self.app, "request_errors", {}).pop(request_id, None)
        details = self.app.contract_detail_values.pop(request_id, [])
        if error:
            raise RuntimeError(f"IBKR contract resolution error {error['error_code']}: {error['error_string']}")
        candidates = [detail for detail in details if self._is_eligible(detail)]
        unique = {int(detail.contract.conId): detail for detail in candidates if int(getattr(detail.contract, "conId", 0) or 0) > 0}
        if len(unique) != 1:
            raise RuntimeError(f"IBKR could not uniquely resolve {symbol}; found {len(unique)} eligible contracts")

        contract_details = next(iter(unique.values()))
        contract = contract_details.contract
        resolved = ResolvedInstrument(
            canonical_symbol=canonical,
            yahoo_symbol=yahoo_symbol,
            contract=contract,
            con_id=int(contract.conId),
            local_symbol=str(contract.localSymbol),
            primary_exchange=str(contract.primaryExchange),
            currency=str(contract.currency),
            sec_type=str(contract.secType),
            min_tick=float(getattr(contract_details, "minTick", 0.01) or 0.01),
        )
        self._cache[canonical] = resolved
        return resolved

    @staticmethod
    def _is_eligible(details: Any) -> bool:
        contract = details.contract
        valid_exchanges = getattr(details, "validExchanges", None) or getattr(contract, "validExchanges", "")
        exchanges = {value.strip().upper() for value in str(valid_exchanges).split(",") if value.strip()}
        exchange = str(getattr(contract, "exchange", "")).upper()
        primary = str(getattr(contract, "primaryExchange", "")).upper()
        return (
            str(getattr(contract, "secType", "")).upper() == "STK"
            and str(getattr(contract, "currency", "")).upper() == "USD"
            and (exchange == "SMART" or "SMART" in exchanges)
            and primary in _US_PRIMARY_EXCHANGES
            and bool(getattr(contract, "localSymbol", ""))
        )
