from __future__ import annotations

from datetime import datetime, timezone

from .models import TradeIntent


class SwitchHysteresisPolicy:
    """Require economic advantage plus a confidence or research-quality gain."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def allows(self, proposed: TradeIntent, current: TradeIntent | None) -> bool:
        if proposed.action != "SWITCH" or current is None:
            return True
        improvement = (
            (proposed.expected_alpha_vs_spy - current.expected_alpha_vs_spy)
            + (proposed.expected_alpha_vs_qqq - current.expected_alpha_vs_qqq)
        ) / 2
        minimum = float(self.config.get("min_expected_excess_improvement", 0.01))
        switching_cost = float(self.config.get("switching_cost_bps", 10)) / 10_000
        confidence_gain = proposed.confidence - current.confidence
        quality_gain = len(proposed.thesis) > len(current.thesis) and len(proposed.evidence_used) > len(current.evidence_used)
        return (
            improvement >= max(minimum, switching_cost)
            and (confidence_gain >= float(self.config.get("min_confidence_improvement", 0.05)) or quality_gain)
        )


class SwitchFrequencyGate:
    """Audit strategic switches without constraining risk reductions."""

    def __init__(self, store, config: dict | None = None, now=None):
        self.store = store
        self.config = config or {}
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _key(self) -> str:
        year, week, _ = self._now().isocalendar()
        return f"strategic_switches.{year}-W{week:02d}"

    def allow(self, action: str, *, category: str) -> bool:
        if category == "RISK_REDUCTION" or action.upper() in {"REDUCE", "EXIT", "CASH", "EMERGENCY_LIQUIDATION"}:
            return True
        if action.upper() != "SWITCH":
            return True
        maximum = int(self.config.get("max_strategic_switches_per_week", 1))
        return int(self.store.get_runtime(self._key(), 0)) < maximum

    def record(self, action: str, *, category: str, decision_id: str | None = None) -> None:
        if category != "STRATEGIC_REBALANCE" or action.upper() != "SWITCH":
            return
        key = self._key()
        count = int(self.store.get_runtime(key, 0)) + 1
        self.store.set_runtime(key, count)
        self.store.save_runtime_event(
            "STRATEGIC_REBALANCE",
            "PORTFOLIO",
            f"Strategic switch recorded ({count})",
            decision_id=decision_id,
            metadata={"category": category, "weekly_count": count},
        )
