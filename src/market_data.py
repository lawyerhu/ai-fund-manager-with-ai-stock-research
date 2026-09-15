from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MarketDataType = Literal["LIVE", "FROZEN", "DELAYED", "DELAYED_FROZEN", "HISTORICAL"]
MarketDataPurpose = Literal["RESEARCH", "RISK", "EXECUTION"]


class MarketDataSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    price: float = Field(gt=0)
    bid: float | None = Field(default=None, gt=0)
    ask: float | None = Field(default=None, gt=0)
    last: float | None = Field(default=None, gt=0)
    source: str
    market_data_type: MarketDataType
    broker_timestamp: str
    received_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    age_seconds: float | None = Field(default=None, ge=0)
    market_session: Literal["OPEN", "CLOSED", "HALTED", "UNKNOWN"] = "UNKNOWN"

    def current_age_seconds(self, now: datetime | None = None) -> float:
        timestamp = datetime.fromisoformat(self.broker_timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, ((now or datetime.now(timezone.utc)) - timestamp.astimezone(timezone.utc)).total_seconds())


class MarketDataPolicy:
    """Purpose-specific data rules. It never upgrades delayed data to live."""

    @staticmethod
    def allows(snapshot: MarketDataSnapshot, *, purpose: MarketDataPurpose, execution_mode: str, max_age_seconds: float = 30.0) -> bool:
        purpose = purpose.upper()
        mode = execution_mode.upper()
        if purpose == "RESEARCH":
            return snapshot.market_data_type in {"LIVE", "FROZEN", "DELAYED", "DELAYED_FROZEN", "HISTORICAL"}
        if mode == "OBSERVE" and purpose in {"RISK", "EXECUTION"}:
            return snapshot.market_data_type in {"LIVE", "FROZEN", "DELAYED", "DELAYED_FROZEN", "HISTORICAL"}
        if purpose not in {"RISK", "EXECUTION"}:
            raise ValueError(f"Unknown market data purpose: {purpose}")
        return (
            snapshot.market_data_type == "LIVE"
            and snapshot.market_session == "OPEN"
            and snapshot.bid is not None
            and snapshot.ask is not None
            and snapshot.current_age_seconds() <= float(max_age_seconds)
        )
