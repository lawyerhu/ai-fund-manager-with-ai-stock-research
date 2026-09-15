from __future__ import annotations

from math import sqrt
from statistics import mean, stdev
from typing import Iterable


def _returns(values: list[float]) -> list[float]:
    return [values[index] / values[index - 1] - 1 for index in range(1, len(values)) if values[index - 1] != 0]


def compute_metrics(equity: Iterable[float], periods_per_year: int = 252, turnover: float = 0.0) -> dict[str, float | None]:
    values = [float(value) for value in equity]
    if len(values) < 2 or values[0] <= 0:
        return {key: None for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe", "sortino", "calmar", "win_rate", "turnover")}
    returns = _returns(values)
    peak = values[0]
    drawdowns = []
    for value in values:
        peak = max(peak, value)
        drawdowns.append(1 - value / peak)
    total_return = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (periods_per_year / max(1, len(values) - 1)) - 1
    volatility = stdev(returns) * sqrt(periods_per_year) if len(returns) > 1 else 0.0
    downside = [min(0.0, value) for value in returns]
    downside_deviation = sqrt(sum(value * value for value in downside) / len(downside)) * sqrt(periods_per_year) if downside else 0.0
    sharpe = mean(returns) / stdev(returns) * sqrt(periods_per_year) if len(returns) > 1 and stdev(returns) else None
    sortino = mean(returns) / downside_deviation * sqrt(periods_per_year) if downside_deviation else None
    max_drawdown = max(drawdowns)
    calmar = cagr / max_drawdown if max_drawdown else None
    return {"total_return": total_return, "cagr": cagr, "max_drawdown": max_drawdown, "volatility": volatility, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "win_rate": sum(value > 0 for value in returns) / len(returns) if returns else None, "turnover": turnover}


def compare_to_benchmarks(equity: Iterable[float], spy: Iterable[float], qqq: Iterable[float]) -> dict[str, float | None]:
    strategy = compute_metrics(equity)
    spy_metrics = compute_metrics(spy)
    qqq_metrics = compute_metrics(qqq)
    strategy_return = strategy["total_return"]
    spy_return = spy_metrics["total_return"]
    qqq_return = qqq_metrics["total_return"]
    return {
        **strategy,
        "excess_return_vs_spy": strategy_return - spy_return if strategy_return is not None and spy_return is not None else None,
        "excess_return_vs_qqq": strategy_return - qqq_return if strategy_return is not None and qqq_return is not None else None,
    }
