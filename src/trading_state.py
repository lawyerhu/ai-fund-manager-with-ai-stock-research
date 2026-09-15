from __future__ import annotations

from enum import StrEnum


class TradingState(StrEnum):
    IDLE = "IDLE"
    RESEARCHING = "RESEARCHING"
    DECISION_READY = "DECISION_READY"
    WAITING_FOR_MARKET = "WAITING_FOR_MARKET"
    WAITING_FOR_BROKER = "WAITING_FOR_BROKER"
    PENDING_ENTRY = "PENDING_ENTRY"
    RISK_APPROVED = "RISK_APPROVED"
    ORDER_PENDING = "ORDER_PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    RISK_HALTED = "RISK_HALTED"
    ERROR = "ERROR"
    OBSERVED = "OBSERVED"


_TRANSITIONS = {
    TradingState.IDLE: {TradingState.RESEARCHING, TradingState.RISK_HALTED, TradingState.ERROR},
    TradingState.RESEARCHING: {TradingState.DECISION_READY, TradingState.ERROR},
    TradingState.DECISION_READY: {TradingState.WAITING_FOR_MARKET, TradingState.WAITING_FOR_BROKER, TradingState.PENDING_ENTRY, TradingState.RISK_APPROVED, TradingState.ORDER_PENDING, TradingState.RISK_HALTED, TradingState.REJECTED, TradingState.ERROR},
    TradingState.PENDING_ENTRY: {TradingState.PENDING_ENTRY, TradingState.RESEARCHING, TradingState.RISK_APPROVED, TradingState.ERROR},
    TradingState.WAITING_FOR_MARKET: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.ERROR},
    TradingState.WAITING_FOR_BROKER: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.ERROR},
    TradingState.RISK_APPROVED: {TradingState.ORDER_PENDING, TradingState.FILLED, TradingState.OBSERVED, TradingState.ERROR},
    TradingState.ORDER_PENDING: {TradingState.PARTIALLY_FILLED, TradingState.FILLED, TradingState.OBSERVED, TradingState.REJECTED, TradingState.RISK_HALTED, TradingState.ERROR},
    TradingState.PARTIALLY_FILLED: {TradingState.PARTIALLY_FILLED, TradingState.FILLED, TradingState.REJECTED, TradingState.ERROR},
    TradingState.FILLED: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.RISK_APPROVED, TradingState.RISK_HALTED, TradingState.ERROR},
    TradingState.REJECTED: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.ERROR},
    TradingState.RISK_HALTED: {TradingState.IDLE, TradingState.ERROR},
    TradingState.ERROR: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.RISK_HALTED},
    TradingState.OBSERVED: {TradingState.IDLE, TradingState.RESEARCHING, TradingState.RISK_APPROVED, TradingState.RISK_HALTED, TradingState.ERROR},
}


class TradingStateMachine:
    def __init__(self, initial: TradingState | str = TradingState.IDLE):
        self.state = TradingState(initial)

    def transition(self, target: TradingState | str) -> TradingState:
        target = TradingState(target)
        if target not in _TRANSITIONS[self.state]:
            raise ValueError(f"Invalid transition {self.state} -> {target}")
        self.state = target
        return self.state

    @classmethod
    def restore(cls, persisted_state: str | None):
        return cls(persisted_state or TradingState.IDLE)
