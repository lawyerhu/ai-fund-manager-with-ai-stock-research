"""Read-only evidence comparison and non-executable reduction counterfactuals."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite

from .market_clock import USEquityMarketClock


def evidence_changes(previous, current):
    # Retrieval timestamps alone are not new investment evidence. No semantic
    # inference: a changed source is NOT automatically negative information.
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()
                    if k not in {"as_of", "quote_as_of", "last_bar_at", "recorded_at"}}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    def missing_fields(value, prefix=""):
        if isinstance(value, dict):
            return {path for k, v in value.items() for path in missing_fields(v, f"{prefix}.{k}".strip("."))}
        if value is None or (isinstance(value, str) and value.upper() in {"UNKNOWN", "DATA_UNAVAILABLE", "UNAVAILABLE"}):
            return {prefix}
        return set()
    old_missing, new_missing = missing_fields(previous), missing_fields(current)
    return {
        "baseline_available": bool(previous),
        "changed_sources": [k for k in current if k in previous and normalize(current[k]) != normalize(previous[k])],
        "unchanged_sources": [k for k in current if k in previous and normalize(current[k]) == normalize(previous[k])],
        "new_sources": [k for k in current if k not in previous],
        "unchanged_missing_fields": sorted(old_missing & new_missing),
        "currently_missing_fields": sorted(new_missing),
    }


def shadow_summary(proposal, observations):
    """Frozen stock/cash sleeve; price returns, not actual portfolio performance."""
    if not observations:
        return {"status": "WAITING_FOR_BASELINE"}
    base = observations[0]
    costs = proposal["cost_assumptions"]
    delta = proposal["old_weight"] - proposal["target_weight"]
    quantity = delta * proposal["equity"] / base["price"]
    commission = max(costs["minimum_commission"], costs["commission_per_order"] + quantity * costs["commission_per_share"])
    cost_fraction = commission / proposal["equity"] + delta * costs["slippage_bps"] / 10000
    hold = [1 + proposal["old_weight"] * (o["price"] / base["price"] - 1) for o in observations]
    reduce = [1 - cost_fraction + proposal["target_weight"] * (o["price"] / base["price"] - 1) for o in observations]
    def drawdown(values):
        peak, worst = 1.0, 0.0
        for value in values:
            peak = max(peak, value)
            worst = max(worst, 1 - value / peak)
        return worst
    spy = observations[-1]["spy"] / base["spy"] - 1
    qqq = observations[-1]["qqq"] / base["qqq"] - 1
    return {
        "status": "OBSERVING", "start_date": base["date"], "last_date": observations[-1]["date"],
        "observation_count": len(observations), "simulation_cost_fraction": cost_fraction,
        "hold_return": hold[-1] - 1, "reduce_net_return": reduce[-1] - 1,
        "reduction_benefit": reduce[-1] - hold[-1],
        "hold_observed_max_drawdown": drawdown(hold), "reduce_observed_max_drawdown": drawdown(reduce),
        "hold_excess_vs_spy": hold[-1] - 1 - spy, "reduce_net_excess_vs_spy": reduce[-1] - 1 - spy,
        "hold_excess_vs_qqq": hold[-1] - 1 - qqq, "reduce_net_excess_vs_qqq": reduce[-1] - 1 - qqq,
    }


def record_shadow_observations(store, data, benchmark, at=None):
    """Backend-only. Never use a query date or intraday bar as an official close."""
    now = at or datetime.now(timezone.utc)
    session = USEquityMarketClock().latest_completed_session(now)
    dates = benchmark.get("dates", [])
    series = benchmark.get("series", {})
    if not dates or dates[-1] != session["date"]:
        return
    if any(len(series.get(s, [])) != len(dates) for s in ("SPY", "QQQ")):
        return
    for row in store.reduction_shadows():
        try:
            proposal = row["proposal"]
            if row["status"] == "COMPLETE":
                continue
            created = datetime.fromisoformat(proposal["created_at"])
            if datetime.fromisoformat(session["close"]) <= created:
                continue  # Never simulate a fill at a close preceding the decision.
            observations = row["observations"]
            if observations and observations[-1]["date"] >= session["date"]:
                continue
            if observations and datetime.fromisoformat(session["date"]).date() > (
                datetime.fromisoformat(observations[0]["date"]).date() + timedelta(days=proposal["horizon_days"])
            ):
                store.complete_reduction_shadow(row["review_id"])
                continue
            history = data.price_history(proposal["symbol"], 20)
            if str(history.get("last_bar_at", ""))[:10] != session["date"]:
                continue
            values = [history.get("price"), series["SPY"][-1], series["QQQ"][-1]]
            if not all(isinstance(v, (int, float)) and isfinite(v) and v > 0 for v in values):
                continue
            store.save_shadow_observation(row["review_id"], {"date": session["date"], "price": values[0], "spy": values[1], "qqq": values[2]})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            store.save_error("shadow-observation", str(exc))
