from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.market_data import MarketDataPolicy, MarketDataSnapshot
from src.data_provider import MockDataProvider
from src.execution.observe import ObserveExecutor
from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest, PortfolioState, TradeIntent
from src.operations import SwitchFrequencyGate, SwitchHysteresisPolicy
from src.scheduler import PaperScheduler, SchedulerConfig
from src.service import BackendService
from src.storage import SQLiteStore
from src.runner import StrategyRunner


class SessionClock:
    def is_open(self, current=None):
        return current is not None and current.weekday() < 5 and (9, 30) <= (current.hour, current.minute) < (16, 0)

    def is_session_day(self, current=None):
        return current is not None and current.weekday() < 5


def test_weekly_daily_and_risk_schedules_enqueue_distinct_work():
    calls = []
    scheduler = PaperScheduler(
        SchedulerConfig(
            full_research_day="saturday",
            full_research_time="10:00",
            pre_execution_time="09:00",
            daily_review_time="16:05",
            timezone="America/New_York",
            risk_monitor_interval_minutes=15,
        ),
        weekly_callback=lambda: calls.append("FULL_RESEARCH"),
        pre_execution_callback=lambda: calls.append("PRE_EXECUTION"),
        daily_callback=lambda: calls.append("DAILY_REVIEW"),
        risk_callback=lambda: calls.append("FULL_RISK"),
        market_clock=SessionClock(),
    )

    scheduler.run_once(datetime(2026, 8, 29, 10, 0, tzinfo=ZoneInfo("America/New_York")))
    scheduler.run_once(datetime(2026, 8, 31, 9, 0, tzinfo=ZoneInfo("America/New_York")))
    scheduler.run_once(datetime(2026, 8, 31, 9, 30, tzinfo=ZoneInfo("America/New_York")))
    scheduler.run_once(datetime(2026, 8, 31, 16, 5, tzinfo=ZoneInfo("America/New_York")))

    assert calls == ["FULL_RESEARCH", "PRE_EXECUTION", "FULL_RISK", "DAILY_REVIEW"]


def test_market_data_policy_separates_research_risk_and_execution_levels():
    old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    delayed = MarketDataSnapshot(symbol="SPY", price=700, source="IBKR", market_data_type="DELAYED", broker_timestamp=old, received_at=old, market_session="CLOSED")
    historical = delayed.model_copy(update={"market_data_type": "HISTORICAL", "source": "IBKR historical"})
    stale_live = delayed.model_copy(update={"market_data_type": "LIVE", "market_session": "OPEN", "bid": 699.9, "ask": 700.1})

    assert MarketDataPolicy.allows(delayed, purpose="RESEARCH", execution_mode="OBSERVE")
    assert MarketDataPolicy.allows(delayed, purpose="RISK", execution_mode="OBSERVE")
    assert MarketDataPolicy.allows(historical, purpose="RESEARCH", execution_mode="OBSERVE")
    assert not MarketDataPolicy.allows(delayed, purpose="EXECUTION", execution_mode="PAPER")
    assert not MarketDataPolicy.allows(stale_live, purpose="EXECUTION", execution_mode="PAPER", max_age_seconds=30)


def test_service_risk_event_persists_and_enqueues_immediate_check(tmp_path):
    with SQLiteStore(tmp_path / "events.sqlite3") as store:
        service = BackendService({}, SimpleNamespace(executor=SimpleNamespace(connected=True, broker_state_known=True)), store)
        event_id = service.trigger_risk_event("PRICE_SHOCK", source="MARKET_DATA", symbol="NVDA", severity="HIGH", metadata={"move": -0.08})

        event = next(row for row in store.runtime_events() if row["id"] == event_id)
        command = store.pending_commands()[0]
        assert event["event_type"] == "PRICE_SHOCK"
        assert command["command"] == "RUN_IMMEDIATE_RISK_CHECK"
        assert command["source"] == "EVENT"
        assert command["trigger_reason"] == "PRICE_SHOCK"


def test_daily_and_event_reviews_do_not_require_full_luna_screening(tmp_path):
    class ReviewRunner:
        executor = SimpleNamespace(connected=True, broker_state_known=True)

        def run_position_review(self, event_context=None):
            return {"review_action": "HOLD", "event_context": event_context, "luna_calls": 0}

    with SQLiteStore(tmp_path / "reviews.sqlite3") as store:
        service = BackendService({}, ReviewRunner(), store, use_llm=True)
        daily = service._handle_command("RUN_DAILY_POSITION_REVIEW", {})
        event = service._handle_command("RUN_EVENT_SOL_REVIEW", {"event_type": "MATERIAL_NEWS"})

        assert daily["luna_calls"] == 0
        assert event["luna_calls"] == 0
        assert event["event_context"]["event_type"] == "MATERIAL_NEWS"


def test_strategic_switch_limit_does_not_block_risk_exit(tmp_path):
    cfg = {"max_strategic_switches_per_week": 1}
    with SQLiteStore(tmp_path / "switches.sqlite3") as store:
        gate = SwitchFrequencyGate(store, cfg, now=lambda: datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert gate.allow("SWITCH", category="STRATEGIC_REBALANCE")
        gate.record("SWITCH", category="STRATEGIC_REBALANCE", decision_id="first")
        assert not gate.allow("SWITCH", category="STRATEGIC_REBALANCE")
        assert gate.allow("CASH", category="RISK_REDUCTION")


def test_switch_hysteresis_requires_material_advantage_beyond_cost_and_one_quality_gain():
    policy = SwitchHysteresisPolicy({"min_expected_excess_improvement": 0.01, "min_confidence_improvement": 0.05, "switching_cost_bps": 10})
    current = TradeIntent(action="HOLD", symbol="NVDA", target_weight=0.5, confidence=0.75, holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02, thesis=["durable"], risk_factors=["risk"], invalidation_conditions=["invalid"], evidence_used=["fundamentals"], model_name="sol")
    marginal = TradeIntent(action="SWITCH", symbol="META", current_symbol="NVDA", new_symbol="META", target_weight=0.5, confidence=0.76, holding_period_days=20, expected_alpha_vs_spy=0.035, expected_alpha_vs_qqq=0.025, thesis=["similar"], risk_factors=["risk"], invalidation_conditions=["invalid"], evidence_used=["fundamentals"], model_name="sol")
    material = marginal.model_copy(update={"confidence": 0.84, "expected_alpha_vs_spy": 0.06, "expected_alpha_vs_qqq": 0.05, "thesis": ["stronger", "catalyst"], "evidence_used": ["fundamentals", "earnings", "news"]})

    assert not policy.allows(marginal, current)
    assert policy.allows(material, current)


def test_broker_monitor_is_independent_from_stopped_scheduler(tmp_path):
    runner = SimpleNamespace(executor=SimpleNamespace(connected=True, broker_state_known=True))
    with SQLiteStore(tmp_path / "broker-monitor.sqlite3") as store:
        store.set_runtime("scheduler_status", "STOPPED")
        service = BackendService({}, runner, store)

        service.monitor_broker_state()

        assert store.get_runtime("broker_monitor_status") == "RUNNING"
        assert store.get_runtime("broker_status") == "CONNECTED"
        assert store.get_runtime("scheduler_status") == "STOPPED"


def test_real_position_review_calls_sol_without_full_universe_screening(tmp_path):
    class PositionOnlyData(MockDataProvider):
        def universe_snapshot(self):
            raise AssertionError("daily review must not screen the full universe")

    class Sol:
        last_tool_calls = [{"name": "get_news"}]

        def decide(self, portfolio, candidates, horizon_days=20):
            assert candidates == ["NVDA"]
            return TradeIntent(action="HOLD", symbol="NVDA", target_weight=portfolio.current_weight, confidence=0.8, holding_period_days=1, expected_alpha_vs_spy=0, expected_alpha_vs_qqq=0, thesis=["intact"], risk_factors=["volatility"], invalidation_conditions=["guidance cut"], evidence_used=["news"], model_name="sol")

    data = PositionOnlyData()
    with SQLiteStore(tmp_path / "position-review.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=1, reference_price=180))
        agent = SimpleNamespace(sol=Sol(), set_event_sink=lambda sink: None)
        result = StrategyRunner({"portfolio": {"starting_equity": 1000, "max_positions": 1}}, data, broker, store, agent=agent).run_position_review()

        assert result["review_action"] == "HOLD"
        assert result["luna_calls"] == 0
        assert result["sol_tool_calls"] == 1


def test_market_closed_observe_revalidation_is_non_executable_but_research_valid(tmp_path):
    class ClosedClock:
        def is_open(self): return False

    cfg = {"portfolio": {"starting_equity": 1000, "max_positions": 1}, "execution": {"max_quote_age_seconds": 30}, "risk": {"max_decision_age_minutes_for_execution": 60}}
    with SQLiteStore(tmp_path / "observe-revalidation.sqlite3") as store:
        broker = ObserveExecutor(starting_cash=1000)
        runner = StrategyRunner(cfg, MockDataProvider(), broker, store, market_clock=ClosedClock())
        intent = TradeIntent(action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02, thesis=["test"], risk_factors=["test"], invalidation_conditions=["test"], evidence_used=["test"])

        result = runner.pre_execution_revalidate(intent, {})

        assert result["valid"] is True
        assert result["executable"] is False
        assert result["market_data"]["market_data_type"] == "HISTORICAL"


def test_delayed_execution_quote_blocks_paper_new_entry(tmp_path):
    class OpenClock:
        def is_open(self): return True

    class DelayedQuote:
        def get_quote(self, symbol):
            now = datetime.now(timezone.utc).isoformat()
            return {"symbol": symbol, "price": 180, "bid": 179.9, "ask": 180.1, "mid": 180, "timestamp": now, "market_status": "OPEN", "source": "IBKR", "data_type": "DELAYED"}

    cfg = {"portfolio": {"starting_equity": 1000, "max_positions": 1}, "agent": {"decision_horizon_days": 20}, "execution": {"minimum_order_notional": 1, "max_quote_age_seconds": 30}, "risk": {"hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1, "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_decision_age_minutes_for_execution": 60, "max_annualized_volatility": 1, "min_avg_dollar_volume": 0, "drawdown_tiers": [], "event_risk": {}}}
    with SQLiteStore(tmp_path / "delayed-paper.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        intent = TradeIntent(action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02, thesis=["test"], risk_factors=["test"], invalidation_conditions=["test"], evidence_used=["test"])

        result = StrategyRunner(cfg, MockDataProvider(), broker, store, market_clock=OpenClock(), quote_provider=DelayedQuote()).run(intent=intent)

        assert result["risk"].approved is False
        assert "DELAYED" in result["risk"].reason
        assert broker.current_orders() == []
