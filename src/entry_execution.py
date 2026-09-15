from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import BrokerSnapshot, ExecutionQuote, ExecutionReport, OrderRequest, PendingEntry, RiskDecision, TradeIntent, make_client_order_id
from .storage import SQLiteStore


@dataclass(frozen=True)
class EntryRevalidationResult:
    entry: PendingEntry
    checks: dict[str, Any]
    reason: str


class EntryQuickReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["BUY_NOW", "WAIT", "CANCEL_ENTRY"]
    confidence: float = Field(ge=0.0, le=1.0)
    thesis_status: Literal["INTACT", "WEAKENED", "BROKEN"]
    max_acceptable_price_deviation: float = Field(ge=0.0)
    reasons: list[str] = Field(min_length=1, max_length=5)


class EntryExecutionEngine:
    """Owns durable entry intent state before any broker order may exist."""

    def __init__(self, cfg: dict[str, Any], *, store: SQLiteStore, market_clock):
        self.cfg = cfg
        self.store = store
        self.market_clock = market_clock

    def _execution_config(self) -> dict[str, Any]:
        return self.cfg.get("execution", {})

    def _revalidation_config(self) -> dict[str, Any]:
        return self._execution_config().get("entry_revalidation", {})

    def capture(
        self,
        intent: TradeIntent,
        *,
        run_id: str,
        decision_reference_price: float,
        market_regime_at_decision: str,
    ) -> PendingEntry:
        if intent.action != "BUY" or not intent.symbol:
            raise ValueError("EntryExecutionEngine.capture requires a BUY intent")
        existing = self.store.pending_entry(intent.decision_id)
        if existing is not None:
            return existing
        entry = PendingEntry(
            decision_id=intent.decision_id,
            run_id=run_id,
            symbol=intent.symbol.upper(),
            decision_time=intent.timestamp,
            decision_reference_price=decision_reference_price,
            sol_confidence=intent.confidence,
            expected_alpha_vs_spy=intent.expected_alpha_vs_spy,
            expected_alpha_vs_qqq=intent.expected_alpha_vs_qqq,
            thesis=intent.thesis,
            positive_factors=intent.thesis,
            risks=intent.risk_factors,
            thesis_horizon_days=intent.holding_period_days,
            requested_weight=intent.target_weight,
            market_regime_at_decision=market_regime_at_decision,
            message="Waiting for regular trading hours",
        )
        entry = self.store.save_pending_entry(entry)
        self.store.save_runtime_event(
            "PENDING_ENTRY_CREATED",
            "ENTRY_EXECUTION",
            f"{entry.symbol} BUY decision is waiting for entry revalidation",
            run_id=entry.run_id,
            decision_id=entry.decision_id,
            symbol=entry.symbol,
            metadata={"status": entry.status, "decision_reference_price": entry.decision_reference_price},
        )
        return entry

    def revalidate(
        self,
        entry: PendingEntry,
        quote: ExecutionQuote | dict[str, Any] | None,
        *,
        now: datetime | None = None,
        material_events: dict[str, Any] | None = None,
        opened_at: datetime | None = None,
    ) -> EntryRevalidationResult:
        """Revalidate an entry without submitting or cancelling a broker order."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        events = material_events or {}
        if not self.market_clock.is_open(now):
            return self._persist_result(entry, "PENDING_ENTRY", {"market_open": False}, "Market is closed")

        delay_seconds = float(self._execution_config().get("entry_delay_after_open_seconds", 120))
        if opened_at is not None:
            opened_at = opened_at if opened_at.tzinfo else opened_at.replace(tzinfo=timezone.utc)
            elapsed = (now - opened_at).total_seconds()
            if elapsed < delay_seconds:
                return self._persist_result(entry, "PENDING_ENTRY", {"market_open": True, "seconds_since_open": elapsed}, "Waiting for post-open stabilization")

        if quote is None:
            return self._persist_result(entry, "ENTRY_FAILED", {"quote_fresh": False}, "No executable quote")
        quote = quote if isinstance(quote, ExecutionQuote) else ExecutionQuote.model_validate(quote)
        age = self._quote_age(quote.timestamp, now)
        max_age = float(self._execution_config().get("entry_max_quote_age_seconds", self._execution_config().get("max_quote_age_seconds", 10)))
        # Allow at most one second of clock skew; retain negative ages for audit.
        quote_fresh = -1.0 <= age <= max_age and quote.market_status == "OPEN" and quote.data_type == "REALTIME"
        if not quote_fresh:
            return self._persist_result(entry, "ENTRY_FAILED", {"quote_fresh": False, "quote_age_seconds": age}, "Executable quote is stale, future-dated, or not realtime")

        mid = quote.mid
        if mid is None and quote.bid is not None and quote.ask is not None:
            mid = (quote.bid + quote.ask) / 2
        current_price = mid or quote.price
        price_gap = (current_price - entry.decision_reference_price) / entry.decision_reference_price
        spread_bps = None
        if quote.bid and quote.ask and mid:
            spread_bps = (quote.ask - quote.bid) / mid * 10000
        revalidation = self._revalidation_config()
        max_gap = float(revalidation.get("max_price_gap_pct_without_review", 0.03))
        max_spread = float(revalidation.get("max_spread_bps", 30))
        checks = {
            "market_open": True,
            "quote_fresh": True,
            "quote_age_seconds": age,
            "current_price": current_price,
            "price_gap_pct": price_gap,
            "spread_bps": spread_bps,
            "material_events": events,
        }
        if spread_bps is not None and spread_bps > max_spread:
            return self._persist_result(entry, "ENTRY_REVIEW_REQUIRED", checks, f"Spread {spread_bps:.1f} bps exceeds {max_spread:.1f} bps")
        event_keys = ("major_news", "sec_filing", "analyst_revision", "guidance_change", "market_regime_change", "sector_regime_change")
        if any(bool(events.get(key)) and bool(revalidation.get(f"review_on_{key}", True)) for key in event_keys):
            return self._persist_result(entry, "ENTRY_REVIEW_REQUIRED", checks, "Material information requires Sol quick review")
        if abs(price_gap) > max_gap:
            return self._persist_result(entry, "ENTRY_REVIEW_REQUIRED", checks, f"Price gap {price_gap:.2%} exceeds {max_gap:.2%}")
        return self._persist_result(entry, "ENTRY_VALID", checks, "Entry revalidation passed")

    def marketable_limit_price(self, quote: ExecutionQuote | dict[str, Any], *, action: Literal["BUY", "SELL"] = "BUY", tick_size: float = 0.01) -> float:
        """Calculate a bounded marketable limit from the executable quote side."""
        quote = quote if isinstance(quote, ExecutionQuote) else ExecutionQuote.model_validate(quote)
        quote_side = quote.ask if action == "BUY" else quote.bid
        side_name = "ask" if action == "BUY" else "bid"
        if quote.market_status != "OPEN" or quote.data_type != "REALTIME" or not quote_side or quote_side <= 0:
            raise ValueError(f"A fresh realtime {side_name} is required for a marketable limit order")
        bps = float(self._execution_config().get("limit_slippage_bps", 5))
        direction = Decimal("1") if action == "BUY" else Decimal("-1")
        raw = Decimal(str(quote_side)) * (Decimal("1") + direction * Decimal(str(bps)) / Decimal("10000"))
        tick = Decimal(str(tick_size))
        if tick <= 0:
            raise ValueError("tick_size must be positive")
        rounding = ROUND_CEILING if action == "BUY" else ROUND_FLOOR
        return float((raw / tick).to_integral_value(rounding=rounding) * tick)

    def apply_quick_review(self, entry: PendingEntry, review: EntryQuickReview | dict[str, Any]) -> PendingEntry:
        review = review if isinstance(review, EntryQuickReview) else EntryQuickReview.model_validate(review)
        status = {"BUY_NOW": "ENTRY_VALID", "WAIT": "ENTRY_REVIEW_REQUIRED", "CANCEL_ENTRY": "ENTRY_CANCELLED"}[review.action]
        updated = entry.model_copy(update={"status": status, "message": "; ".join(review.reasons)})
        self.store.update_pending_entry(updated)
        self.store.save_runtime_event("ENTRY_QUICK_REVIEW", "SOL", updated.message, run_id=entry.run_id, decision_id=entry.decision_id, symbol=entry.symbol, metadata={"action": review.action, "confidence": review.confidence, "thesis_status": review.thesis_status, "status": status})
        return updated

    def build_order(
        self,
        entry: PendingEntry,
        quote: ExecutionQuote | dict[str, Any],
        portfolio: BrokerSnapshot,
        risk: RiskDecision,
        *,
        broker_connected: bool,
        broker_reconciled: bool,
        execution_mode: str,
        tick_size: float = 0.01,
        fractional_shares: bool = True,
        requote_index: int = 0,
    ) -> OrderRequest:
        if execution_mode.upper() != "PAPER":
            raise RuntimeError("Entry orders are permitted only in PAPER execution mode")
        if entry.status != "ENTRY_VALID":
            raise RuntimeError("Entry must be ENTRY_VALID before order construction")
        if not broker_connected or not broker_reconciled:
            raise RuntimeError("Broker connection and reconciliation are required")
        if not risk.approved or risk.risk_state == "RISK_HALTED" or risk.approved_weight <= 0:
            raise RuntimeError("Risk Engine did not approve the entry")
        if risk.approved_weight > 1:
            raise RuntimeError("Gross exposure above 100% is blocked")
        quote = quote if isinstance(quote, ExecutionQuote) else ExecutionQuote.model_validate(quote)
        limit_price = self.marketable_limit_price(quote, tick_size=tick_size)
        initial_price = entry.current_price or quote.mid or quote.ask or quote.price
        max_chase_bps = float(self._execution_config().get("max_total_slippage_bps", 30))
        chase_bps = max(0.0, (limit_price - initial_price) / initial_price * 10000)
        if chase_bps > max_chase_bps:
            raise RuntimeError(f"Maximum entry slippage exceeded: {chase_bps:.1f} bps")
        target_notional = portfolio.equity * risk.approved_weight
        current_symbol_notional = sum(
            max(0.0, position.market_value)
            for position in portfolio.positions
            if position.symbol.upper() == entry.symbol.upper()
        )
        other_notional = sum(
            max(0.0, position.market_value)
            for position in portfolio.positions
            if position.symbol.upper() != entry.symbol.upper()
        )
        if other_notional + target_notional > portfolio.equity + 1e-6:
            raise RuntimeError("Gross exposure above 100% is blocked")
        delta_notional = max(0.0, target_notional - current_symbol_notional)
        cash_safety_factor = float(self._execution_config().get("buy_cash_safety_factor", 1.0))
        if not 0 < cash_safety_factor <= 1:
            raise ValueError("execution.buy_cash_safety_factor must be in (0, 1]")
        available_cash = min(portfolio.cash * cash_safety_factor, delta_notional)
        if available_cash <= 0:
            raise RuntimeError("Margin borrowing is blocked or the approved target has no positive buy delta")
        quantity = available_cash / limit_price
        if not fractional_shares:
            quantity = float(int(quantity))
        if quantity <= 0:
            raise RuntimeError("Available cash cannot fund a positive quantity")
        return OrderRequest(
            decision_id=entry.decision_id,
            symbol=entry.symbol,
            action="BUY",
            quantity=quantity,
            reference_price=quote.mid or quote.price,
            order_type="LMT",
            limit_price=limit_price,
            client_order_id=make_client_order_id(entry.decision_id, f"entry-{requote_index}", entry.symbol, "BUY"),
            expected_position_quantity=sum(
                position.quantity
                for position in portfolio.positions
                if position.symbol.upper() == entry.symbol.upper()
            ),
        )

    def apply_execution(self, entry: PendingEntry, report: ExecutionReport, *, quote_mid: float | None = None) -> PendingEntry:
        if report.filled_quantity > 0 and report.remaining_quantity <= 1e-9:
            status = "ENTRY_FILLED"
        elif report.filled_quantity > 0:
            status = "ENTRY_PARTIALLY_FILLED"
        else:
            status = {
                "PENDING": "ENTRY_SUBMITTED",
                "PARTIALLY_FILLED": "ENTRY_PARTIALLY_FILLED",
                "FILLED": "ENTRY_FILLED",
                "CANCELLED": "ENTRY_REVIEW_REQUIRED",
                "TIMEOUT": "ENTRY_REVIEW_REQUIRED",
                "REJECTED": "ENTRY_FAILED",
                "ERROR": "ENTRY_FAILED",
                "SIMULATED": "ENTRY_VALID",
            }.get(report.status, "ENTRY_FAILED")
        average = report.average_price
        updates = {
            "status": status,
            "client_order_id": report.client_order_id,
            "entry_price": average,
            "fill_time": report.timestamp if report.filled_quantity > 0 else None,
            "filled_quantity": report.filled_quantity,
            "remaining_quantity": report.remaining_quantity,
            "slippage_vs_decision": (average - entry.decision_reference_price) / entry.decision_reference_price if average else None,
            "slippage_vs_mid": (average - quote_mid) / quote_mid if average and quote_mid else None,
            "message": report.message,
        }
        updated = entry.model_copy(update=updates)
        self.store.update_pending_entry(updated)
        self.store.save_runtime_event(status, "ENTRY_EXECUTION", report.message, run_id=entry.run_id, decision_id=entry.decision_id, symbol=entry.symbol, metadata={"filled_quantity": report.filled_quantity, "remaining_quantity": report.remaining_quantity, "average_price": average})
        return updated

    def _persist_result(self, entry: PendingEntry, status: str, checks: dict[str, Any], reason: str) -> EntryRevalidationResult:
        current_price = checks.get("current_price")
        updates = {"status": status, "message": reason}
        if "price_gap_pct" in checks:
            updates.update({"price_gap_pct": checks["price_gap_pct"], "spread_bps": checks.get("spread_bps"), "current_price": current_price})
        updated = entry.model_copy(update=updates)
        if self.store.pending_entry(entry.decision_id) is not None:
            self.store.update_pending_entry(updated)
        self.store.save_runtime_event("ENTRY_REVALIDATION", "ENTRY_EXECUTION", reason, run_id=entry.run_id, decision_id=entry.decision_id, symbol=entry.symbol, metadata={"status": status, "checks": checks})
        return EntryRevalidationResult(updated, checks, reason)

    @staticmethod
    def _quote_age(timestamp: str, now: datetime) -> float:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (now - parsed.astimezone(timezone.utc)).total_seconds()
