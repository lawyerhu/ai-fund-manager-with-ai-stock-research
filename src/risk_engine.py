from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

from .models import PortfolioState, RiskDecision, TradeIntent


class RiskEngine:
    """Deterministic position sizing and trade veto layer.

    The model can request a weight, but it cannot relax any of these limits.
    """

    def __init__(self, risk_cfg: dict[str, Any]):
        self.cfg = risk_cfg
        self.tolerance = float(risk_cfg.get("comparison_tolerance", 1e-9))

    def evaluate(
        self,
        intent: TradeIntent,
        portfolio: PortfolioState,
        stock_ann_vol: float | None,
        data_as_of: str | None,
        stock_snapshot: dict[str, Any] | None = None,
        events: dict[str, Any] | None = None,
        market_open: bool = True,
    ) -> RiskDecision:
        requested = intent.target_weight
        drawdown = portfolio.drawdown
        hard_limit = float(self.cfg.get("hard_drawdown_limit", 0.25))
        recovery_limit = float(self.cfg.get("halt_recovery_drawdown", 0.20))

        if intent.action == "CASH":
            state = "RISK_HALTED" if self._at_or_above(drawdown, hard_limit) else portfolio.risk_state
            return RiskDecision(approved=True, requested_weight=requested, approved_weight=0.0, reason="Model chose cash", forced_cash=False, risk_state=state, drawdown_weight_limit=0.0, confidence_minimum=float(self.cfg.get("min_confidence_to_open", 0.55)))

        if intent.action == "HOLD":
            if not portfolio.current_symbol:
                return self._reject(requested, "HOLD requires an existing reconciled position", False, self._state_for_drawdown(drawdown))
            if portfolio.current_symbol.upper() != intent.symbol.upper():
                return self._reject(requested, "HOLD symbol does not match the reconciled position", False, self._state_for_drawdown(drawdown))

        if intent.action == "SWITCH":
            if not portfolio.current_symbol:
                return self._reject(requested, "SWITCH requires an existing position", False, self._state_for_drawdown(drawdown))
            if portfolio.current_symbol.upper() != intent.current_symbol.upper():
                return self._reject(requested, "SWITCH current_symbol does not match the reconciled position", False, self._state_for_drawdown(drawdown))
            if intent.symbol.upper() != intent.new_symbol.upper():
                return self._reject(requested, "SWITCH new_symbol does not match the target symbol", False, self._state_for_drawdown(drawdown))

        if portfolio.risk_state == "RISK_HALTED" and not self._at_or_below(drawdown, recovery_limit):
            return self._reject(requested, "Risk halt remains active until drawdown recovers to the configured recovery level", True, "RISK_HALTED")

        if self._at_or_above(drawdown, hard_limit):
            return self._reject(requested, "Hard drawdown limit reached; exit to cash", True, "RISK_HALTED")

        decision_age_limit = float(self.cfg.get("max_decision_age_minutes_for_execution", 60))
        if not self._fresh_with_limit(intent.timestamp, decision_age_limit):
            return self._reject(requested, "Trade decision is stale and must be refreshed", False, self._state_for_drawdown(drawdown))

        opens_target_exposure = (
            requested > self.tolerance
            and (not portfolio.current_symbol or portfolio.current_symbol.upper() != intent.symbol.upper() or portfolio.current_quantity <= self.tolerance)
        )
        if intent.confidence < float(self.cfg.get("min_confidence_to_open", 0.55)) and opens_target_exposure:
            return self._reject(requested, "Confidence below opening threshold", False, self._state_for_drawdown(drawdown))

        if not data_as_of or not self._fresh(data_as_of):
            return self._reject(requested, "Market data is missing or stale", False, self._state_for_drawdown(drawdown))

        snapshot = stock_snapshot or {}
        if stock_snapshot is not None and (snapshot.get("price_available") is False or snapshot.get("price") is None or float(snapshot["price"]) <= 0):
            return self._reject(requested, "Latest price is unavailable", False, self._state_for_drawdown(drawdown))
        if snapshot.get("trading_halted") is True:
            return self._reject(requested, "Trading is halted or unavailable", False, self._state_for_drawdown(drawdown))
        if not market_open:
            return self._reject(requested, "Market is not open for this order", False, self._state_for_drawdown(drawdown))

        if stock_ann_vol is None or not isfinite(stock_ann_vol) or stock_ann_vol <= 0:
            return self._reject(requested, "Missing/invalid volatility", False, self._state_for_drawdown(drawdown))
        if stock_ann_vol > float(self.cfg.get("max_annualized_volatility", 1.0)) + self.tolerance:
            return self._reject(requested, "Annualized volatility exceeds configured maximum", False, self._state_for_drawdown(drawdown))

        if "avg_dollar_volume" in snapshot and float(snapshot["avg_dollar_volume"]) < float(self.cfg.get("min_avg_dollar_volume", 0)):
            return self._reject(requested, "Average dollar volume is below liquidity minimum", False, self._state_for_drawdown(drawdown))

        if "gap_pct" in snapshot and abs(float(snapshot["gap_pct"])) > float(self.cfg.get("max_gap_pct", 1.0)) + self.tolerance:
            return self._reject(requested, "Abnormal opening gap exceeds configured maximum", False, self._state_for_drawdown(drawdown))

        volatility_limit = min(float(self.cfg.get("absolute_max_weight", 1.0)), float(self.cfg.get("target_annualized_vol", 0.25)) / stock_ann_vol)
        max_weight = volatility_limit
        drawdown_limit = 1.0
        event_limit = 1.0
        reasons: list[str] = [f"volatility cap={max_weight:.3f}"]
        state = self._state_for_drawdown(drawdown)

        for tier in sorted(self.cfg.get("drawdown_tiers", []), key=lambda x: float(x["drawdown"])):
            if self._at_or_above(drawdown, float(tier["drawdown"])):
                max_weight = min(max_weight, float(tier["max_weight"]))
                drawdown_limit = min(drawdown_limit, float(tier["max_weight"]))
                reasons.append(f"drawdown cap={float(tier['max_weight']):.3f}")
                if float(tier["max_weight"]) < 1.0:
                    state = "REDUCED"

        event_cfg = self.cfg.get("event_risk", {})
        event_data = events or {}
        if event_data.get("macro_event_supported") is False:
            unknown_policy = event_cfg.get("macro_unknown_policy", "reject_new")
            if unknown_policy == "reject_new" and intent.action in {"BUY", "SWITCH"}:
                return self._reject(requested, "Macro event data is unavailable; opening trade blocked by policy", False, state)
            if unknown_policy == "cap":
                max_weight = min(max_weight, float(event_cfg.get("macro_unknown_max_weight", 1.0)))
                reasons.append("macro data unknown cap")
                state = "REDUCED"
            elif unknown_policy not in {"allow", "reject_new"}:
                raise ValueError(f"Unsupported macro_unknown_policy: {unknown_policy}")
        earnings_days = event_data.get("days_to_earnings", snapshot.get("earnings_days"))
        if earnings_days is not None and float(earnings_days) <= float(event_cfg.get("earnings_blackout_days", 0)):
            if intent.action in {"BUY", "SWITCH"} and bool(event_cfg.get("reject_new_before_earnings", False)):
                return self._reject(requested, "Earnings are inside the configured opening blackout", False, "REDUCED")
            event_limit = min(event_limit, float(event_cfg.get("earnings_max_weight", 1.0)))
            max_weight = min(max_weight, event_limit)
            reasons.append("earnings event cap")
            state = "REDUCED"
        if event_data.get("macro_event"):
            event_limit = min(event_limit, float(event_cfg.get("macro_event_max_weight", 1.0)))
            max_weight = min(max_weight, event_limit)
            reasons.append("macro event cap")
            state = "REDUCED"

        approved_weight = min(requested, max_weight)
        if approved_weight <= self.tolerance:
            return self._reject(requested, "Risk cap reduced target to zero", True, state)

        return RiskDecision(
            approved=True,
            requested_weight=requested,
            approved_weight=approved_weight,
            reason="Approved with deterministic limits: " + "; ".join(reasons),
            risk_state=state,
            limit_reasons=reasons,
            drawdown_weight_limit=drawdown_limit,
            volatility_weight_limit=volatility_limit,
            event_weight_limit=event_limit,
            liquidity_min_dollar_volume=float(self.cfg.get("min_avg_dollar_volume", 0)),
            confidence_minimum=float(self.cfg.get("min_confidence_to_open", 0.55)),
        )

    def _reject(self, requested: float, reason: str, forced_cash: bool, state: str) -> RiskDecision:
        return RiskDecision(approved=False, requested_weight=requested, approved_weight=0.0, reason=reason, forced_cash=forced_cash, risk_state=state, limit_reasons=[reason])

    def _at_or_above(self, value: float, threshold: float) -> bool:
        return value >= threshold - self.tolerance

    def _at_or_below(self, value: float, threshold: float) -> bool:
        return value <= threshold + self.tolerance

    def _state_for_drawdown(self, drawdown: float) -> str:
        for tier in sorted(self.cfg.get("drawdown_tiers", []), key=lambda x: float(x["drawdown"])):
            if self._at_or_above(drawdown, float(tier["drawdown"])) and float(tier["max_weight"]) < 1.0:
                return "REDUCED"
        return "NORMAL"

    def _fresh(self, iso_time: str) -> bool:
        return self._fresh_with_limit(iso_time, float(self.cfg.get("max_data_age_minutes", 30)))

    def _fresh_with_limit(self, iso_time: str, max_age_minutes: float) -> bool:
        try:
            ts = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 60
        return -self.tolerance <= age_minutes <= max_age_minutes + self.tolerance
