"""Persist the security retained by GPT-6's decision, never an account holding."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


def stock_symbol(value):
    if not isinstance(value, str) or not value.strip() or value.upper().strip() in {"CASH", "WAIT", "NONE", "UNKNOWN"}:
        raise ValueError("Active selection must identify a stock")
    return value.strip().upper()


class SelectionTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incoming_active_selection: str
    new_first_symbol: str
    rebalance_decision: Literal["KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"]
    outgoing_active_selection: str

    @field_validator("incoming_active_selection", "new_first_symbol", "outgoing_active_selection")
    @classmethod
    def normalize_symbol(cls, value):
        return stock_symbol(value)

    @model_validator(mode="after")
    def check_transition(self):
        expected = self.incoming_active_selection if self.rebalance_decision == "KEEP_PREVIOUS" else self.new_first_symbol
        if self.outgoing_active_selection != expected:
            raise ValueError("Outgoing active selection contradicts GPT-6's decision")
        if self.incoming_active_selection == self.new_first_symbol and self.rebalance_decision != "KEEP_PREVIOUS":
            raise ValueError("Identical selections must KEEP_PREVIOUS")
        return self


def transition_fields(incoming, new_first, decision):
    """Apply an already-made model decision; this function does not choose KEEP/SWITCH."""
    state = SelectionTransition(
        incoming_active_selection=incoming, new_first_symbol=new_first, rebalance_decision=decision,
        outgoing_active_selection=incoming if decision == "KEEP_PREVIOUS" else new_first,
    )
    return {**state.model_dump(), "active_selection": state.outgoing_active_selection}


def resolve_active_selection(raw):
    if raw.get("status") != "COMPLETE":
        raise ValueError("Previous research is not COMPLETE")
    nested = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    records = [raw, nested]
    known = []
    for record in records:
        comparison = record.get("rebalance_comparison") or record.get("comparison") or {}
        action = record.get("rebalance_decision") or comparison.get("rebalance_decision")
        if not action and any(key in record for key in ("incoming_active_selection", "outgoing_active_selection")):
            raise ValueError("Persisted transition is missing its final KEEP/SWITCH decision")
        if action:
            if record.get("rebalance_decision") and comparison.get("rebalance_decision") and record["rebalance_decision"] != comparison["rebalance_decision"]:
                raise ValueError("Saved comparison actions disagree")
            incoming = record.get("incoming_active_selection") or record.get("previous_first_symbol") or comparison.get("previous_first_symbol")
            new_first = record.get("new_first_symbol") or comparison.get("new_first_symbol") or record.get("selected_symbol")
            fields = transition_fields(incoming, new_first, action)
            if record.get("previous_first_symbol") and stock_symbol(record["previous_first_symbol"]) != fields["incoming_active_selection"]:
                raise ValueError("Legacy previous_first_symbol disagrees with incoming active selection")
            known.append(fields["outgoing_active_selection"])
            for key, expected in (("previous_first_symbol", fields["incoming_active_selection"]),
                                  ("new_first_symbol", fields["new_first_symbol"])):
                if comparison.get(key) and stock_symbol(comparison[key]) != expected:
                    raise ValueError("Saved comparison symbols disagree with selection state")
        for key in ("outgoing_active_selection", "active_selection"):
            if key in record:
                known.append(stock_symbol(record[key]))
    if not known:
        raise ValueError("Previous result has only a ranking; explicit active_selection initialization or a saved KEEP/SWITCH decision is required")
    if len(set(known)) != 1:
        raise ValueError("Saved active selection contradicts the final decision")
    return known[0]
