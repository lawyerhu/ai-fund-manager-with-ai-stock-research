from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import ManagedPosition, PositionReview, PositionReviewOutcome, PositionTrigger, TradeIntent
from .storage import SQLiteStore


class PositionManager:
    """Applies deterministic portfolio policy to model-authored position reviews."""

    def __init__(self, cfg: dict[str, Any], store: SQLiteStore):
        self.cfg = cfg
        self.store = store
        review_cfg = cfg.get("portfolio_review", {})
        self.replacement_threshold = float(review_cfg.get("replacement_threshold", 10.0))
        self.normal_review_hours = float(review_cfg.get("normal_review_interval_hours", 24.0))
        self.review_cooldown_minutes = float(cfg.get("monitoring", {}).get("review_cooldown_minutes", 60.0))

    def save_position(self, position: ManagedPosition) -> ManagedPosition:
        return self.store.save_managed_position(position.model_copy(update={"symbol": position.symbol.upper()}))

    def ensure_entry_thesis_consistency(
        self,
        position: ManagedPosition,
        intent: TradeIntent | None,
    ) -> ManagedPosition:
        """Prevent an executed BUY from being managed with a cash thesis."""
        if intent is None or intent.action != "BUY" or not intent.symbol:
            return position
        if intent.symbol.upper() != position.symbol.upper():
            return position
        thesis_text = " ".join(position.original_thesis).lower()
        cash_markers = (
            "cash is valid",
            "hold cash",
            "holding cash",
            "remain in cash",
            "stay in cash",
            "does not justify concentration",
            "does not justify a position",
        )
        if not any(marker in thesis_text for marker in cash_markers):
            return position

        symbol = position.symbol.upper()
        supporting_evidence = [
            item
            for item in intent.evidence_used
            if symbol.lower() in item.lower()
            and any(term in item.lower() for term in ("rank", "score", "momentum", "growth", "strength", "alpha"))
        ][:4]
        repaired_thesis = [
            f"{symbol} entered under the configured alpha-validation selection policy after ranking first among the finalists.",
            *supporting_evidence,
            "The original committee rationale favored cash, so a fresh Sol review is required before this thesis can be treated as intact.",
        ]
        repaired_thesis = list(dict.fromkeys(repaired_thesis))[:6]
        positive_factors = supporting_evidence or [repaired_thesis[0]]
        return position.model_copy(update={
            "original_thesis": repaired_thesis,
            "original_positive_factors": positive_factors,
            "latest_thesis": repaired_thesis,
            "current_thesis_status": "EXPIRED_REVIEW_REQUIRED",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    @staticmethod
    def review_due(position: ManagedPosition, at: datetime | None = None) -> str | None:
        current = at or datetime.now(timezone.utc)
        if position.days_held_at(current) >= position.current_thesis_horizon_days:
            return "HORIZON"
        if position.next_scheduled_review_at:
            scheduled = datetime.fromisoformat(position.next_scheduled_review_at.replace("Z", "+00:00"))
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            if current.astimezone(timezone.utc) >= scheduled.astimezone(timezone.utc):
                return "DAILY"
        return None

    def register_trigger(
        self,
        position: ManagedPosition,
        event_id: str,
        event_type: str,
        *,
        occurred_at: datetime | None = None,
        source: str = "MONITOR",
        severity: str = "MEDIUM",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if self.store.position_trigger(event_id) is not None:
            return False
        current = occurred_at or datetime.now(timezone.utc)
        for prior in self.store.position_triggers(position.position_id):
            if prior.event_type != event_type:
                continue
            prior_at = datetime.fromisoformat(prior.occurred_at.replace("Z", "+00:00"))
            if prior_at.tzinfo is None:
                prior_at = prior_at.replace(tzinfo=timezone.utc)
            if 0 <= (current.astimezone(timezone.utc) - prior_at.astimezone(timezone.utc)).total_seconds() < self.review_cooldown_minutes * 60:
                return False
            break
        trigger = PositionTrigger(
            event_id=event_id,
            position_id=position.position_id,
            symbol=position.symbol,
            event_type=event_type,
            source=source,
            severity=severity,
            occurred_at=current.isoformat(),
            metadata=metadata or {},
        )
        self.store.save_position_trigger(trigger)
        self.store.save_managed_position(position.model_copy(update={"last_trigger": event_type, "updated_at": current.isoformat()}))
        self.store.save_runtime_event(
            event_type,
            "POSITION_MONITOR",
            f"{position.symbol}: {event_type}",
            run_id=position.run_id,
            decision_id=position.decision_id,
            symbol=position.symbol,
            metadata={"event_id": event_id, "severity": severity, **(metadata or {})},
            timestamp=current.isoformat(),
        )
        return True

    def apply_review(self, position: ManagedPosition, review: PositionReview) -> PositionReviewOutcome:
        if review.position_id != position.position_id or review.current_holding.upper() != position.symbol.upper():
            raise ValueError("Position review does not match the managed position")
        gap = (
            float(review.best_alternative_score) - float(review.current_holding_score)
            if review.best_alternative_score is not None
            else 0.0
        )
        threshold = self.replacement_threshold
        action = review.action
        reasons = list(review.reason)
        if review.thesis_status == "BROKEN" and action not in {"SELL", "EXIT_TO_CASH", "REPLACE"}:
            action = "EXIT_TO_CASH"
            reasons.append("Broken thesis cannot remain invested")
        if action == "REPLACE" and gap < threshold:
            if review.thesis_status == "BROKEN":
                action = "EXIT_TO_CASH"
                reasons.append(f"Replacement gap {gap:.1f} is below threshold; broken thesis exits to cash")
            else:
                action = "HOLD"
                reasons.append(f"Replacement gap {gap:.1f} is below threshold {threshold:.1f}")
        if action == "EXTEND_HOLD":
            if review.review_type != "HORIZON" or review.days_held < position.current_thesis_horizon_days:
                raise ValueError("EXTEND_HOLD is only valid at the thesis horizon")
            if review.new_horizon_days <= position.current_thesis_horizon_days:
                raise ValueError("EXTEND_HOLD requires a longer thesis horizon")
        reviewed_at = datetime.fromisoformat(review.reviewed_at.replace("Z", "+00:00"))
        if reviewed_at.tzinfo is None:
            reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
        normalized = review.model_copy(update={
            "action": action,
            "replacement_gap": gap,
            "replacement_threshold": threshold,
            "reason": reasons[:5],
        })
        updated = position.model_copy(update={
            "latest_thesis": reasons[:5],
            "current_thesis_status": normalized.thesis_status,
            "last_sol_review_at": normalized.reviewed_at,
            "next_scheduled_review_at": (reviewed_at + timedelta(hours=self.normal_review_hours)).isoformat(),
            "last_trigger": normalized.trigger,
            "current_thesis_horizon_days": normalized.new_horizon_days if normalized.action == "EXTEND_HOLD" else position.current_thesis_horizon_days,
            "updated_at": normalized.reviewed_at,
        })
        self.store.save_managed_position(updated)
        self.store.save_position_review(normalized)
        self.store.save_runtime_event(
            "POSITION_REVIEW_COMPLETED",
            "POSITION_MANAGER",
            f"{updated.symbol}: {normalized.action} / {normalized.thesis_status}",
            run_id=normalized.run_id,
            decision_id=normalized.decision_id,
            symbol=updated.symbol,
            metadata=normalized.model_dump(mode="json"),
        )
        return PositionReviewOutcome(position=updated, review=normalized, trade_intent=self._trade_intent(updated, normalized))

    def review_context(self, position, event_context=None, committee_decision=None):
        previous = self.store.position_reviews(position.position_id, 1)
        return {
            **(event_context or {}),
            "previous_review": previous[0].model_dump(mode="json", exclude={"comparison_context"}) if previous else None,
            "committee_decision": committee_decision,
            "configured_risk_budget": self.cfg.get("risk", {}),
            "comparison_note": "Compare evidence, not numerical scores from different model evaluations.",
        }

    def record_reduction_shadow(self, position, review, equity):
        if review.action not in {"REDUCE", "SELL", "EXIT_TO_CASH"} or equity <= 0:
            return
        target = 0.0 if review.action in {"SELL", "EXIT_TO_CASH"} else review.target_weight
        if target >= position.current_weight:
            return
        costs = self.cfg.get("execution", {})
        self.store.save_reduction_shadow(review.review_id, {
            "created_at": review.reviewed_at, "decision_id": review.decision_id,
            "symbol": position.symbol, "old_weight": position.current_weight,
            "target_weight": target, "equity": equity, "horizon_days": review.new_horizon_days,
            "cost_assumptions": {key: float(costs.get(key, 0)) for key in
                                 ("commission_per_order", "commission_per_share", "minimum_commission", "slippage_bps")},
            "method": "SIMULATION: next observed completed close after decision; frozen stock/cash sleeve; calendar-day horizon; no rebalancing; zero cash interest; price returns exclude dividends; shared LLM costs excluded; not actual fills or full portfolio returns",
        })

    @staticmethod
    def _trade_intent(position: ManagedPosition, review: PositionReview) -> TradeIntent | None:
        if review.action in {"HOLD", "EXTEND_HOLD"}:
            return None
        if review.action == "REPLACE":
            if not review.best_alternative or review.best_alternative.upper() == position.symbol.upper():
                raise ValueError("REPLACE requires a different best alternative")
            action = "SWITCH"
            symbol = review.best_alternative.upper()
            current_symbol = position.symbol.upper()
            new_symbol = symbol
            target_weight = review.target_weight
        elif review.action in {"SELL", "EXIT_TO_CASH"}:
            action = "CASH"
            symbol = None
            current_symbol = None
            new_symbol = None
            target_weight = 0.0
        elif review.action == "ADD":
            action = "BUY"
            symbol = position.symbol.upper()
            current_symbol = None
            new_symbol = None
            target_weight = review.target_weight
        elif review.action == "REDUCE":
            action = "HOLD"
            symbol = position.symbol.upper()
            current_symbol = None
            new_symbol = None
            target_weight = review.target_weight
        else:
            raise ValueError(f"Unsupported portfolio action: {review.action}")
        return TradeIntent(
            action=action,
            symbol=symbol,
            current_symbol=current_symbol,
            new_symbol=new_symbol,
            target_weight=target_weight,
            confidence=review.confidence,
            holding_period_days=review.new_horizon_days,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=review.reason,
            risk_factors=[f"Position review risk level: {review.risk_level}"],
            invalidation_conditions=["Replacement thesis no longer remains valid"],
            evidence_used=[f"position_review:{review.review_id}"],
            timestamp=review.reviewed_at,
            model_name="sol-position-review",
            decision_id=review.decision_id,
        )
