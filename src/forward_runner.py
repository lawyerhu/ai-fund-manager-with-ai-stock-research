from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import PortfolioState
from .runner import StrategyRunner


class ForwardPaperRunner:
    """Small scheduling boundary for an immutable forward paper experiment.

    A supervisor may call ``run_weekly_if_due`` once per day. This class does not
    create a background process, which keeps scheduling and broker permissions
    outside the strategy code.
    """

    def __init__(self, strategy_runner: StrategyRunner, interval_days: int = 7):
        self.strategy_runner = strategy_runner
        self.interval_days = interval_days

    def run_weekly_if_due(self, last_decision_at: str | None = None, now: datetime | None = None, use_llm: bool = True) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        if not self.is_due(last_decision_at, now):
            return {"status": "SKIPPED", "reason": "weekly decision interval has not elapsed"}
        return {"status": "RAN", "result": self.strategy_runner.run(use_llm=use_llm)}

    def risk_monitor(self, portfolio: PortfolioState | None = None) -> dict[str, Any]:
        return self.strategy_runner.run_daily_risk_check()

    def is_due(self, last_decision_at: str | None, now: datetime | None = None) -> bool:
        if not last_decision_at:
            return True
        now = now or datetime.now(timezone.utc)
        previous = datetime.fromisoformat(last_decision_at.replace("Z", "+00:00"))
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc) - previous.astimezone(timezone.utc) >= timedelta(days=self.interval_days)
