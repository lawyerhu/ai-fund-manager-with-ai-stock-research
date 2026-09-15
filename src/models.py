from datetime import datetime, timezone
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

Action = Literal["BUY", "HOLD", "SWITCH", "CASH"]
RiskState = Literal["NORMAL", "REDUCED", "RISK_HALTED"]
OrderStatus = Literal["PENDING", "PARTIALLY_FILLED", "FILLED", "CANCELLED", "REJECTED", "TIMEOUT", "ERROR", "SIMULATED"]
EntryStatus = Literal[
    "PENDING_ENTRY",
    "ENTRY_REVALIDATING",
    "ENTRY_VALID",
    "ENTRY_REVIEW_REQUIRED",
    "ENTRY_CANCELLED",
    "ENTRY_SUBMITTED",
    "ENTRY_PARTIALLY_FILLED",
    "ENTRY_FILLED",
    "ENTRY_FAILED",
]
ThesisStatus = Literal["INTACT", "WEAKENING", "BROKEN", "IMPROVING", "EXPIRED_REVIEW_REQUIRED"]
PortfolioAction = Literal["HOLD", "ADD", "REDUCE", "SELL", "REPLACE", "EXIT_TO_CASH", "EXTEND_HOLD"]
PositionTriggerType = Literal[
    "PRICE_SHOCK", "RELATIVE_STRENGTH_BREAK", "VOLATILITY_SPIKE", "VOLUME_ANOMALY",
    "STOP_LOSS", "TAKE_PROFIT",
    "MAJOR_NEWS", "NEGATIVE_NEWS", "SEC_FILING", "GUIDANCE_CHANGE", "ANALYST_DOWNGRADE",
    "MANAGEMENT_CHANGE", "M_AND_A_EVENT", "LEGAL_EVENT", "REGULATORY_EVENT",
    "EARNINGS_APPROACHING", "EARNINGS_RESULT", "MARKET_REGIME_CHANGE", "SECTOR_REGIME_CHANGE",
    "THESIS_METRIC_BREAK",
]


class ExecutionQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    price: float = Field(gt=0)
    bid: float | None = Field(default=None, gt=0)
    ask: float | None = Field(default=None, gt=0)
    mid: float | None = Field(default=None, gt=0)
    timestamp: str
    market_status: Literal["OPEN", "CLOSED", "HALTED", "UNKNOWN"]
    source: str
    data_type: Literal["REALTIME", "FROZEN", "DELAYED", "DELAYED_FROZEN", "SIMULATED"]

    def to_market_data_snapshot(self):
        from .market_data import MarketDataSnapshot

        data_type = {"REALTIME": "LIVE", "SIMULATED": "HISTORICAL"}.get(self.data_type, self.data_type)
        now = datetime.now(timezone.utc)
        timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return MarketDataSnapshot(
            symbol=self.symbol,
            price=self.price,
            bid=self.bid,
            ask=self.ask,
            last=self.price if self.mid is None else None,
            source=self.source,
            market_data_type=data_type,
            broker_timestamp=self.timestamp,
            received_at=now.isoformat(),
            age_seconds=max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds()),
            market_session=self.market_status,
        )


def make_client_order_id(decision_id: str, leg: str, symbol: str, action: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"ai-fund-manager:{decision_id}:{leg}:{symbol}:{action}"))


class TradeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    symbol: str | None = Field(default=None, description="Target stock ticker; null for CASH")
    target_weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    holding_period_days: int = Field(ge=1, le=90)
    expected_alpha_vs_spy: float
    expected_alpha_vs_qqq: float
    thesis: list[str] = Field(min_length=1, max_length=6)
    risk_factors: list[str] = Field(min_length=1, max_length=6)
    invalidation_conditions: list[str] = Field(min_length=1, max_length=6)
    evidence_used: list[str] = Field(min_length=1, max_length=12)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str = "unknown"
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    current_symbol: str | None = None
    new_symbol: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action == "CASH":
            if self.symbol not in (None, "") or self.target_weight != 0:
                raise ValueError("CASH intents must have no symbol and zero target_weight")
        elif not self.symbol:
            raise ValueError("Non-CASH intents require symbol")
        if self.action == "SWITCH":
            if not self.current_symbol or not self.new_symbol:
                raise ValueError("SWITCH intents require current_symbol and new_symbol")
            if self.symbol.upper() != self.new_symbol.upper():
                raise ValueError("SWITCH symbol must match new_symbol")
            if self.current_symbol.upper() == self.new_symbol.upper():
                raise ValueError("SWITCH requires different current_symbol and new_symbol")
        return self


class PendingEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    run_id: str
    symbol: str
    decision_time: str
    decision_reference_price: float = Field(gt=0)
    sol_confidence: float = Field(ge=0.0, le=1.0)
    expected_alpha_vs_spy: float
    expected_alpha_vs_qqq: float
    thesis: list[str] = Field(min_length=1)
    positive_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    thesis_horizon_days: int = Field(ge=1)
    requested_weight: float = Field(ge=0.0, le=1.0)
    market_regime_at_decision: str
    status: EntryStatus = "PENDING_ENTRY"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_price: float | None = Field(default=None, gt=0)
    price_gap_pct: float | None = None
    spread_bps: float | None = Field(default=None, ge=0)
    approved_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    limit_price: float | None = Field(default=None, gt=0)
    client_order_id: str | None = None
    entry_price: float | None = Field(default=None, gt=0)
    fill_time: str | None = None
    filled_quantity: float = Field(default=0.0, ge=0)
    remaining_quantity: float = Field(default=0.0, ge=0)
    slippage_vs_decision: float | None = None
    slippage_vs_mid: float | None = None
    message: str = ""


class ManagedPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_id: str
    run_id: str
    decision_id: str
    symbol: str
    entry_time: str
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    current_weight: float = Field(ge=0.0, le=1.0)
    original_thesis_horizon_days: int = Field(ge=1)
    current_thesis_horizon_days: int = Field(ge=1)
    original_thesis: list[str] = Field(min_length=1)
    original_positive_factors: list[str] = Field(default_factory=list)
    original_risks: list[str] = Field(default_factory=list)
    latest_thesis: list[str] = Field(default_factory=list)
    current_thesis_status: ThesisStatus = "INTACT"
    monitoring_status: Literal["ACTIVE", "CLOSED", "SUSPENDED"] = "ACTIVE"
    last_sol_review_at: str | None = None
    next_scheduled_review_at: str | None = None
    last_trigger: str | None = None
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def days_held_at(self, at: datetime | None = None) -> int:
        current = at or datetime.now(timezone.utc)
        entered = datetime.fromisoformat(self.entry_time.replace("Z", "+00:00"))
        if entered.tzinfo is None:
            entered = entered.replace(tzinfo=timezone.utc)
        return max(0, (current.astimezone(timezone.utc).date() - entered.astimezone(timezone.utc).date()).days)

class SizingAudit(BaseModel):
    """Model-authored rationale, not a calibrated probability or risk approval."""
    model_config = ConfigDict(extra="forbid")

    new_negative_evidence: list[str] = Field(default_factory=list, max_length=6)
    unchanged_missing_data: list[str] = Field(default_factory=list, max_length=6)
    thesis_changes: list[str] = Field(default_factory=list, max_length=6)
    weight_basis: str = Field(min_length=1)
    incremental_reason: str = Field(min_length=1)
    risk_budget_basis: str = Field(min_length=1)
    downside_scenario: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1, max_length=12)
    estimate_type: Literal["SOL_MODEL_ESTIMATE"] = "SOL_MODEL_ESTIMATE"


class PositionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(default_factory=lambda: str(uuid4()))
    position_id: str
    run_id: str
    decision_id: str
    review_type: Literal["DAILY", "WEEKLY", "EMERGENCY", "HORIZON"]
    trigger: str | None = None
    current_holding: str
    action: PortfolioAction
    thesis_status: ThesisStatus
    current_holding_score: float = Field(ge=0, le=100)
    best_alternative: str | None = None
    best_alternative_score: float | None = Field(default=None, ge=0, le=100)
    replacement_gap: float = 0.0
    replacement_threshold: float = Field(default=10.0, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    days_held: int = Field(ge=0)
    original_horizon_days: int = Field(ge=1)
    new_horizon_days: int = Field(ge=1)
    reason: list[str] = Field(min_length=1, max_length=5)
    risk_level: Literal["NORMAL", "ELEVATED", "HIGH", "CRITICAL"] = "NORMAL"
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_full_research: bool = False
    sizing_audit: SizingAudit | None = None  # Older stored reviews remain readable.
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    comparison_context: dict[str, Any] = Field(default_factory=dict)
    reviewed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PositionReviewOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    position: ManagedPosition
    review: PositionReview
    trade_intent: "TradeIntent | None" = None


class PositionTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    position_id: str
    symbol: str
    event_type: PositionTriggerType
    source: str = "MONITOR"
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    occurred_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = Field(default_factory=dict)


class LLMDecision(BaseModel):
    """Strict model-authored fields; audit metadata is added by Python."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    symbol: str | None
    target_weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    holding_period_days: int = Field(ge=1, le=90)
    expected_alpha_vs_spy: float
    expected_alpha_vs_qqq: float
    thesis: list[str] = Field(min_length=1, max_length=6)
    risk_factors: list[str] = Field(min_length=1, max_length=6)
    invalidation_conditions: list[str] = Field(min_length=1, max_length=6)
    evidence_used: list[str] = Field(min_length=1, max_length=12)
    current_symbol: str | None
    new_symbol: str | None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action == "CASH":
            if self.symbol is not None or self.target_weight != 0:
                raise ValueError("CASH decisions must have null symbol and zero target_weight")
        elif not self.symbol:
            raise ValueError("Non-CASH decisions require symbol")
        if self.action == "SWITCH":
            if not self.current_symbol or not self.new_symbol:
                raise ValueError("SWITCH decisions require current_symbol and new_symbol")
            if self.symbol.upper() != self.new_symbol.upper():
                raise ValueError("SWITCH symbol must match new_symbol")
            if self.current_symbol.upper() == self.new_symbol.upper():
                raise ValueError("SWITCH requires different current_symbol and new_symbol")
        return self

    def to_trade_intent(self, model_name: str) -> TradeIntent:
        return TradeIntent(**self.model_dump(), model_name=model_name)


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equity: float
    peak_equity: float
    cash: float
    current_symbol: str | None = None
    current_weight: float = 0.0
    current_quantity: float = 0.0
    invested_value: float = Field(default=0.0, ge=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    historical_max_drawdown: float = Field(default=0.0, ge=0, le=1)
    risk_state: RiskState = "NORMAL"
    as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    requested_weight: float
    approved_weight: float
    reason: str
    forced_cash: bool = False
    risk_state: RiskState = "NORMAL"
    limit_reasons: list[str] = Field(default_factory=list)
    drawdown_weight_limit: float | None = None
    volatility_weight_limit: float | None = None
    event_weight_limit: float | None = None
    liquidity_min_dollar_volume: float | None = None
    confidence_minimum: float | None = None
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    symbol: str
    action: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    reference_price: float | None = Field(default=None, gt=0)
    order_type: Literal["MKT", "LMT"] = "MKT"
    limit_price: float | None = Field(default=None, gt=0)
    client_order_id: str = Field(default_factory=lambda: str(uuid4()))
    expected_position_quantity: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_limit_price(self):
        if self.order_type == "LMT" and self.limit_price is None:
            raise ValueError("LMT orders require limit_price")
        if self.order_type == "MKT" and self.limit_price is not None:
            raise ValueError("MKT orders cannot specify limit_price")
        return self


class ExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_order_id: str
    decision_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: float | None = None
    message: str = ""
    broker_order_id: int | None = None
    perm_id: int | None = None
    remaining_quantity: float = 0.0
    commission: float = Field(default=0.0, ge=0)
    fees: float = Field(default=0.0, ge=0)
    slippage: float = Field(default=0.0, ge=0)
    total_execution_cost: float = Field(default=0.0, ge=0)
    error_code: int | None = None
    error_string: str | None = None
    advanced_reject_reason: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def trading_costs(self) -> float:
        broker_cost = self.total_execution_cost if self.total_execution_cost > 0 else self.commission + self.fees
        return broker_cost + self.slippage


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: float = Field(gt=0)
    market_price: float = Field(gt=0)
    market_value: float = Field(ge=0)
    average_cost: float = Field(default=0.0, ge=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BrokerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equity: float = Field(ge=0)
    cash: float = Field(ge=0)
    positions: list[PositionSnapshot] = Field(default_factory=list)
    open_orders: list[dict] = Field(default_factory=list)
    as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str
    invested_value: float = Field(default=0.0, ge=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BenchmarkData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: int = Field(ge=1)
    dates: list[str]
    series: dict[str, list[float]]
    as_of: str
    source: str

    @model_validator(mode="after")
    def require_benchmarks(self):
        if set(self.series) != {"SPY", "QQQ"} or any(not values for values in self.series.values()):
            raise ValueError("BenchmarkData requires non-empty SPY and QQQ series")
        lengths = {len(self.dates), *(len(values) for values in self.series.values())}
        if len(lengths) != 1 or not self.dates:
            raise ValueError("BenchmarkData dates, SPY, and QQQ must be synchronized")
        return self
