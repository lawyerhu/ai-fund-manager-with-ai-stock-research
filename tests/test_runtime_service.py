from datetime import datetime, timedelta, timezone
import threading
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.data_provider import MockDataProvider
from src.execution.observe import ObserveExecutor
from src.first_run import FirstRunResult, HealthCheck
from src.market_clock import USEquityMarketClock
from src.models import OrderRequest, PortfolioState, TradeIntent
from src.runner import StrategyRunner
from src.scheduler import PaperScheduler, SchedulerConfig
from src.service import BackendService
from src.storage import SQLiteStore


def test_market_clock_weekend_uses_last_completed_nyse_session():
    weekend = datetime(2026, 8, 30, 14, 0, tzinfo=timezone.utc)
    clock = USEquityMarketClock(now=lambda: weekend)

    assert clock.is_open() is False
    assert clock.latest_completed_session()["date"] == "2026-08-28"


def test_expired_pending_entry_queues_one_fresh_ai_research_command(tmp_path):
    runner = SimpleNamespace(
        process_pending_entries=lambda: [{
            "decision_id": "expired",
            "status": "DECISION_EXPIRED",
            "reason": "fresh AI research is required",
        }]
    )
    with SQLiteStore(tmp_path / "expired-entry-refresh.sqlite3") as store:
        service = BackendService({}, runner, store)

        service.run_pending_entries()
        service.run_pending_entries()

        research = [
            row for row in store.recent("control_commands", 10)
            if row["command"] == "RUN_FULL_AI_RESEARCH"
        ]
        assert len(research) == 1
        assert research[0]["status"] == "QUEUED"
        assert research[0]["trigger_reason"] == "PENDING_DECISION_EXPIRED"


def test_after_close_reduce_is_persisted_and_retried_in_next_open_session(tmp_path):
    intent = TradeIntent(
        action="HOLD", symbol="MPC", target_weight=0.30, confidence=0.78,
        holding_period_days=10, expected_alpha_vs_spy=0.0, expected_alpha_vs_qqq=0.0,
        thesis=["Reduce concentration"], risk_factors=["Cyclical exposure"],
        invalidation_conditions=["Thesis improves"], evidence_used=["position review"],
        decision_id="reduce-after-close",
    )
    completed = {
        "intent": intent,
        "portfolio": PortfolioState(equity=1000, peak_equity=1000, cash=500, current_symbol="MPC", current_quantity=5, current_weight=0.5),
        "risk": None,
        "state": "FILLED",
    }

    class Runner:
        market_clock = SimpleNamespace(is_open=lambda: True)

        def __init__(self):
            self.calls = []

        def run_position_review(self, event_context=None):
            return {
                "review_action": "REDUCE",
                "symbol": "MPC",
                "decision": intent.model_dump(mode="json"),
                "execution": {"state": "REJECTED", "risk": {"reason": "Market data is missing or stale"}},
            }

        def process_pending_entries(self):
            return []

        def run(self, *, intent, use_llm=False):
            self.calls.append(intent)
            return completed

    with SQLiteStore(tmp_path / "pending-adjustment.sqlite3") as store:
        runner = Runner()
        service = BackendService({}, runner, store)

        service.run_position_review()
        pending = store.get_runtime("pending_execution_intent")
        assert pending["intent"]["decision_id"] == "reduce-after-close"
        assert pending["source"] == "DAILY_POSITION_REVIEW"

        results = service.run_pending_entries()

        assert results[-1]["status"] == "FILLED"
        assert runner.calls[0].decision_id == "reduce-after-close"
        assert store.get_runtime("pending_execution_intent") is None


@pytest.mark.parametrize(
    ("action", "symbol", "target_weight"),
    [("HOLD", "MPC", 0.3), ("SWITCH", "MSFT", 0.5), ("CASH", None, 0.0)],
)
def test_weekly_non_buy_adjustment_waiting_for_market_is_durable(tmp_path, action, symbol, target_weight):
    intent = TradeIntent(
        action=action, symbol=symbol, current_symbol="MPC" if action == "SWITCH" else None,
        new_symbol=symbol if action == "SWITCH" else None, target_weight=target_weight,
        confidence=0.8, holding_period_days=10, expected_alpha_vs_spy=0.0,
        expected_alpha_vs_qqq=0.0, thesis=["weekly"], risk_factors=["risk"],
        invalidation_conditions=["change"], evidence_used=["weekly"],
    )
    runner = SimpleNamespace(run=lambda **kwargs: {
        "intent": intent,
        "portfolio": PortfolioState(equity=1000, peak_equity=1000, cash=500, current_symbol="MPC", current_quantity=5, current_weight=0.5),
        "risk": SimpleNamespace(reason="Market is not open"),
        "state": "WAITING_FOR_MARKET",
    })
    with SQLiteStore(tmp_path / f"pending-{action}.sqlite3") as store:
        service = BackendService({}, runner, store)

        service.run_weekly_cycle(use_llm=False)

        pending = store.get_runtime("pending_execution_intent")
        assert pending["intent"]["action"] == action
        assert pending["source"] == "WEEKLY_FULL_RESEARCH"


def test_risk_cycle_uses_expired_pending_entry_refresh_path(tmp_path):
    runner = SimpleNamespace(
        run_daily_risk_check=lambda: {
            "portfolio": PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            "risk": None,
        },
        process_pending_entries=lambda: [{
            "decision_id": "expired-from-risk-cycle",
            "status": "DECISION_EXPIRED",
            "reason": "fresh AI research is required",
        }],
    )
    with SQLiteStore(tmp_path / "risk-cycle-expired-entry.sqlite3") as store:
        service = BackendService({}, runner, store)

        service.run_risk_check_now()

        research = [
            row for row in store.recent("control_commands", 10)
            if row["command"] == "RUN_FULL_AI_RESEARCH"
        ]
        assert len(research) == 1
        assert research[0]["trigger_reason"] == "PENDING_DECISION_EXPIRED"


def test_sol_resume_command_passes_source_run_to_production_runner(tmp_path):
    intent = TradeIntent(
        action="CASH", symbol=None, target_weight=0.0, confidence=0.7, holding_period_days=20,
        expected_alpha_vs_spy=0.0, expected_alpha_vs_qqq=0.0, thesis=["cash"],
        risk_factors=["risk"], invalidation_conditions=["change"], evidence_used=["persisted evidence"],
    )
    calls = []
    runner = SimpleNamespace(
        run=lambda **kwargs: calls.append(kwargs) or {
            "intent": intent,
            "portfolio": PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            "risk": None,
            "state": "COMPLETED",
        },
    )
    with SQLiteStore(tmp_path / "sol-resume-command.sqlite3") as store:
        service = BackendService({}, runner, store, use_llm=True)

        service._handle_command("RUN_SOL_DECISION_RESUME", {"source_run_id": "failed-run-1"})

    assert calls == [{"use_llm": True, "resume_source_run_id": "failed-run-1"}]


def test_runner_loads_sol_resume_context_from_one_failed_runtime_run(tmp_path):
    symbols = ["NVDA", "META", "AVGO", "MSFT", "MPC"]
    with SQLiteStore(tmp_path / "sol-resume-context.sqlite3") as store:
        store.save_runtime_event("SOL_RESEARCH_STARTED", "SOL", "started", run_id="failed-run", metadata={"candidate_symbols": symbols})
        store.save_runtime_event(
            "LUNA_FINAL_SCREENING_COMPLETED", "LUNA", "completed", run_id="failed-run",
            metadata={"final_screening_stage": "GLOBAL_TIE_BREAK", "candidate_symbols": symbols},
        )
        for symbol in symbols:
            for tool in ("get_price_history", "get_fundamentals", "get_analyst_revisions", "get_upcoming_events"):
                store.save_runtime_event(
                    "SOL_TOOL_RESULT", "SOL", "succeeded", run_id="failed-run", symbol=symbol,
                    metadata={"tool": tool, "status": "SUCCESS", "result": {"symbol": symbol, "tool": tool}},
                )
        store.save_runtime_event("LLM_ERROR", "LLM", "502", run_id="failed-run", metadata={"stage": "SOL_RESEARCH"})
        store.save_runtime_event("AI_RUN_FAILED", "PIPELINE", "failed", run_id="failed-run")
        runner = StrategyRunner({}, MockDataProvider(), store=store)

        candidates, evidence = runner._persisted_sol_resume_context("failed-run")

    assert candidates == symbols
    assert len(evidence) == 20
    assert {item["tool"] for item in evidence} == {
        "get_price_history", "get_fundamentals", "get_analyst_revisions", "get_upcoming_events",
    }


def test_observe_executor_never_changes_account_or_sends_order():
    executor = ObserveExecutor(starting_cash=1000.0)
    executor.connect()
    before = executor.reconcile()

    report = executor.submit_order(OrderRequest(decision_id="observe", symbol="NVDA", action="BUY", quantity=1, reference_price=100))
    after = executor.reconcile()

    assert report.status == "SIMULATED"
    assert "not sent" in report.message.lower()
    assert after.cash == before.cash == 1000.0
    assert after.positions == []
    assert executor.current_orders() == []


def test_real_backend_progress_reaches_100_only_after_success_and_stops_below_100_on_failure(tmp_path):
    from test_safety_reconciliation import config

    intent = TradeIntent(action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.01, thesis=["test"], risk_factors=["test"], invalidation_conditions=["test"], evidence_used=["test"])
    with SQLiteStore(tmp_path / "progress-success.sqlite3") as store:
        executor = ObserveExecutor(starting_cash=1000)
        executor.connect()
        runner = StrategyRunner(config(store.path), MockDataProvider(), executor=executor, store=store)
        runner.run(intent=intent)
        progress = store.get_runtime("ai_research_progress")
        assert progress["stage"] == "COMPLETE"
        assert progress["progress_percent"] == 100

    class FailingAgent:
        def set_event_sink(self, sink): self.sink = sink
        def decide(self, *args): raise RuntimeError("luna failed")

    with SQLiteStore(tmp_path / "progress-failed.sqlite3") as store:
        executor = ObserveExecutor(starting_cash=1000)
        executor.connect()
        runner = StrategyRunner(config(store.path), MockDataProvider(), executor=executor, store=store, agent=FailingAgent())
        with pytest.raises(RuntimeError, match="luna failed"):
            runner.run(use_llm=True)
        progress = store.get_runtime("ai_research_progress")
        assert progress["status"] == "FAILED"
        assert progress["progress_percent"] < 100


def test_scheduler_runs_configured_daily_and_weekly_cycles_once_per_period():
    calls = []
    scheduler = PaperScheduler(
        SchedulerConfig(weekly_day="monday", weekly_time="09:30", daily_time="16:00", timezone="Asia/Shanghai"),
        weekly_callback=lambda: calls.append("weekly"),
        daily_callback=lambda: calls.append("daily"),
    )

    monday = datetime(2026, 8, 31, 16, 1)
    scheduler.run_once(monday)
    scheduler.run_once(monday)

    assert calls == ["weekly", "daily"]
    assert scheduler.status()["last_weekly_run"] == "2026-08-31"
    assert scheduler.status()["last_daily_run"] == "2026-08-31"


def test_scheduler_persists_current_market_session_for_dashboard():
    class Clock:
        open = True

        def is_open(self, _at=None):
            return self.open

        def is_session_day(self, _at=None):
            return True

    with SQLiteStore(":memory:") as store:
        clock = Clock()
        scheduler = PaperScheduler(
            SchedulerConfig(timezone="America/New_York"),
            store=store,
            market_clock=clock,
        )
        scheduler.run_once(datetime(2026, 9, 1, 10, 0, tzinfo=ZoneInfo("America/New_York")))
        assert store.get_runtime("market_session") == "OPEN"

        clock.open = False
        scheduler.run_once(datetime(2026, 9, 1, 17, 0, tzinfo=ZoneInfo("America/New_York")))
        assert store.get_runtime("market_session") == "CLOSED"


def test_scheduler_waits_for_enqueued_weekly_command_before_marking_complete(tmp_path):
    with SQLiteStore(tmp_path / "scheduler-command.sqlite3") as store:
        command_id = store.enqueue_command("RUN_FULL_AI_RESEARCH", source="SCHEDULER")
        calls = []

        def enqueue_once():
            calls.append("weekly")
            return command_id

        scheduler = PaperScheduler(
            SchedulerConfig(weekly_day="monday", weekly_time="09:45", timezone="America/New_York"),
            weekly_callback=enqueue_once,
            store=store,
        )

        scheduler.run_once(datetime(2026, 8, 31, 9, 45))
        assert calls == ["weekly"]
        assert scheduler.status()["weekly_status"] == "QUEUED"
        assert scheduler.status()["last_weekly_run"] is None

        store.complete_command(command_id, result={"state": "FILLED"})
        scheduler.run_once(datetime(2026, 8, 31, 10, 0))

        assert scheduler.status()["weekly_status"] == "COMPLETED"
        assert scheduler.status()["last_weekly_run"] == "2026-08-31"


def test_failed_weekly_research_requeues_at_next_scheduled_week_without_same_day_loop(tmp_path):
    with SQLiteStore(tmp_path / "scheduler-failed-recovery.sqlite3") as store:
        command_ids = []

        def enqueue_weekly():
            command_id = store.enqueue_command("RUN_FULL_AI_RESEARCH", source="SCHEDULER")
            command_ids.append(command_id)
            return command_id

        scheduler = PaperScheduler(
            SchedulerConfig(weekly_day="monday", weekly_time="10:00", timezone="America/New_York"),
            weekly_callback=enqueue_weekly,
            store=store,
        )

        scheduler.run_once(datetime(2026, 8, 31, 10, 0))
        store.complete_command(command_ids[0], error="provider unavailable")
        scheduler.run_once(datetime(2026, 8, 31, 10, 5))
        scheduler.run_once(datetime(2026, 8, 31, 11, 0))

        assert scheduler.status()["weekly_status"] == "FAILED"
        assert len(command_ids) == 1

        scheduler.run_once(datetime(2026, 9, 7, 10, 0))

        assert len(command_ids) == 2
        assert scheduler.status()["weekly_status"] == "QUEUED"


def test_full_research_command_uses_weekly_state_machine(tmp_path):
    class WeeklyRunner:
        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "state": "FILLED",
                "intent": SimpleNamespace(timestamp=datetime.now(timezone.utc).isoformat()),
                "portfolio": SimpleNamespace(risk_state="NORMAL"),
            }

    with SQLiteStore(tmp_path / "full-research-command.sqlite3") as store:
        runner = WeeklyRunner()
        service = BackendService({}, runner, store, use_llm=False)

        service._handle_command("RUN_FULL_AI_RESEARCH", {})

        assert runner.calls == [{"use_llm": False}]


def test_scheduler_can_be_disabled_by_configuration():
    scheduler = PaperScheduler(SchedulerConfig(enabled=False), weekly_callback=lambda: pytest.fail("weekly callback"), daily_callback=lambda: pytest.fail("daily callback"))
    assert scheduler.config.enabled is False


def test_scheduler_defaults_to_new_york_and_shanghai_monday_does_not_trigger_weekly():
    calls = []
    config = SchedulerConfig()
    scheduler = PaperScheduler(config, weekly_callback=lambda: calls.append("weekly"))

    scheduler.run_once(datetime(2026, 8, 31, 9, 35, tzinfo=ZoneInfo("Asia/Shanghai")))

    assert config.timezone == "America/New_York"
    assert config.weekly_time == "09:45"
    assert config.risk_monitor_interval_minutes == 15
    assert calls == []


def test_waiting_for_market_does_not_consume_weekly_execution():
    class OpenClock:
        def is_open(self, current=None): return True

    outcomes = iter([{"state": "WAITING_FOR_MARKET"}, {"state": "FILLED"}])
    scheduler = PaperScheduler(
        SchedulerConfig(weekly_day="monday", weekly_time="09:45", timezone="America/New_York"),
        weekly_callback=lambda: next(outcomes),
        market_clock=OpenClock(),
    )

    scheduler.run_once(datetime(2026, 8, 31, 9, 45))
    waiting = scheduler.status()
    scheduler.run_once(datetime(2026, 8, 31, 10, 0))

    assert waiting["weekly_status"] == "WAITING_FOR_MARKET"
    assert waiting["last_weekly_run"] is None
    assert scheduler.status()["weekly_status"] == "COMPLETED"
    assert scheduler.status()["last_weekly_run"] == "2026-08-31"


def test_waiting_weekly_execution_retries_on_next_open_trading_day():
    class SessionClock:
        def is_open(self, current=None):
            return current is not None and current.date().isoformat() == "2026-09-01"

    outcomes = iter([{"state": "WAITING_FOR_MARKET"}, {"state": "FILLED"}])
    scheduler = PaperScheduler(
        SchedulerConfig(weekly_day="monday", weekly_time="09:45", timezone="America/New_York"),
        weekly_callback=lambda: next(outcomes),
        market_clock=SessionClock(),
    )

    scheduler.run_once(datetime(2026, 8, 31, 9, 45))
    scheduler.run_once(datetime(2026, 8, 31, 10, 0))
    assert scheduler.status()["weekly_status"] == "WAITING_FOR_MARKET"

    scheduler.run_once(datetime(2026, 9, 1, 9, 45))

    assert scheduler.status()["weekly_status"] == "COMPLETED"
    assert scheduler.status()["last_weekly_run"] == "2026-09-01"


def test_fifteen_minute_risk_cycle_runs_only_during_valid_session():
    class SessionClock:
        def is_open(self, current=None):
            return current is not None and (9, 30) <= (current.hour, current.minute) < (16, 0)

    calls = []
    scheduler = PaperScheduler(
        SchedulerConfig(timezone="America/New_York", risk_monitor_interval_minutes=15),
        risk_callback=lambda: calls.append("risk"),
        market_clock=SessionClock(),
    )

    for current in (
        datetime(2026, 9, 1, 9, 15),
        datetime(2026, 9, 1, 9, 30),
        datetime(2026, 9, 1, 9, 44),
        datetime(2026, 9, 1, 9, 45),
        datetime(2026, 9, 1, 16, 0),
    ):
        scheduler.run_once(current)

    assert calls == ["risk", "risk"]


def test_thirty_minute_position_monitor_runs_only_during_valid_session():
    class SessionClock:
        def is_open(self, current=None):
            return current is not None and (9, 30) <= (current.hour, current.minute) < (16, 0)

    calls = []
    scheduler = PaperScheduler(
        SchedulerConfig(timezone="America/New_York", position_monitor_interval_minutes=30),
        position_callback=lambda: calls.append("position"),
        market_clock=SessionClock(),
    )

    for current in (
        datetime(2026, 9, 1, 9, 15),
        datetime(2026, 9, 1, 9, 30),
        datetime(2026, 9, 1, 9, 59),
        datetime(2026, 9, 1, 10, 0),
        datetime(2026, 9, 1, 16, 0),
    ):
        scheduler.run_once(current)

    assert calls == ["position", "position"]


def test_expired_waiting_decision_requires_fresh_ai_decision(tmp_path):
    stale_intent = SimpleNamespace(timestamp=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    fresh_intent = SimpleNamespace(timestamp=datetime.now(timezone.utc).isoformat())

    class WaitingRunner:
        def __init__(self):
            self.calls = []

        def run(self, intent=None, use_llm=False):
            self.calls.append({"intent": intent, "use_llm": use_llm})
            selected = stale_intent if len(self.calls) == 1 else fresh_intent
            state = "WAITING_FOR_MARKET" if len(self.calls) == 1 else "FILLED"
            return {"state": state, "intent": selected, "portfolio": SimpleNamespace(risk_state="NORMAL")}

    with SQLiteStore(tmp_path / "waiting.sqlite3") as store:
        runner = WaitingRunner()
        service = BackendService(
            {"risk": {"max_decision_age_minutes_for_execution": 30}}, runner, store,
        )

        first = service.run_weekly_cycle(use_llm=True)
        second = service.run_weekly_cycle(use_llm=True)

        assert first["state"] == "WAITING_FOR_MARKET"
        assert second["state"] == "FILLED"
        assert runner.calls == [
            {"intent": None, "use_llm": True},
            {"intent": None, "use_llm": True},
        ]


def test_fresh_waiting_decision_is_reused_in_next_market_window(tmp_path):
    pending_intent = SimpleNamespace(timestamp=datetime.now(timezone.utc).isoformat())

    class WaitingRunner:
        def __init__(self): self.calls = []
        def run(self, intent=None, use_llm=False):
            self.calls.append({"intent": intent, "use_llm": use_llm})
            return {
                "state": "WAITING_FOR_MARKET" if len(self.calls) == 1 else "FILLED",
                "intent": pending_intent,
                "portfolio": SimpleNamespace(risk_state="NORMAL"),
            }

    with SQLiteStore(tmp_path / "fresh-waiting.sqlite3") as store:
        runner = WaitingRunner()
        service = BackendService({"risk": {"max_decision_age_minutes_for_execution": 30}}, runner, store)

        service.run_weekly_cycle(use_llm=True)
        service.run_weekly_cycle(use_llm=True)

        assert runner.calls[1] == {"intent": pending_intent, "use_llm": False}


def test_broker_quote_loss_after_llm_waits_without_failing_the_research_run(tmp_path):
    class Agent:
        model = "gpt-5.6-sol"
        prompt_version = "test"
        last_pipeline_metadata = {}
        last_usage = {}
        last_research_evidence = []

        def decide(self, portfolio, horizon_days):
            return TradeIntent(
                action="BUY",
                symbol="NVDA",
                target_weight=0.5,
                confidence=0.8,
                holding_period_days=horizon_days,
                expected_alpha_vs_spy=0.03,
                expected_alpha_vs_qqq=0.02,
                thesis=["test thesis"],
                risk_factors=["test risk"],
                invalidation_conditions=["test invalidation"],
                evidence_used=["test evidence"],
                model_name=self.model,
                decision_id="broker-wait-decision",
            )

    class DisconnectedQuoteProvider:
        def get_quote(self, symbol):
            raise RuntimeError("IBKR is disconnected; execution quote unavailable")

    config = {
        "portfolio": {"starting_equity": 1000.0},
        "agent": {"decision_horizon_days": 20},
        "execution": {"entry_engine_enabled": False},
    }
    with SQLiteStore(tmp_path / "broker-wait.sqlite3") as store:
        executor = ObserveExecutor(starting_cash=1000.0)
        runner = StrategyRunner(
            config,
            MockDataProvider(),
            executor=executor,
            store=store,
            agent=Agent(),
            quote_provider=DisconnectedQuoteProvider(),
        )

        result = runner.run(use_llm=True)

        assert result["state"] == "WAITING_FOR_BROKER"
        assert store.get_runtime("ai_cycle")["status"] == "WAITING_FOR_BROKER"
        events = store.runtime_events(run_id=runner._active_run_id)
        assert any(row["event_type"] == "EXECUTION_WAITING_FOR_BROKER" for row in events)
        assert not any(row["event_type"] == "AI_RUN_FAILED" for row in events)
        assert store.recent("llm_decisions", 1)[0]["decision_id"] == "broker-wait-decision"


def test_broker_waiting_execution_is_durable_and_reused_without_llm(tmp_path):
    intent = TradeIntent(
        action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8,
        holding_period_days=20, expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02,
        thesis=["test thesis"], risk_factors=["test risk"],
        invalidation_conditions=["test invalidation"], evidence_used=["test evidence"],
        decision_id="durable-broker-wait",
    )
    portfolio = PortfolioState(equity=1000.0, peak_equity=1000.0, cash=1000.0)

    class Runner:
        market_clock = SimpleNamespace(is_open=lambda: True)

        def __init__(self):
            self.calls = []

        def run(self, intent=None, use_llm=False):
            self.calls.append((intent, use_llm))
            if intent is None:
                return {
                    "state": "WAITING_FOR_BROKER",
                    "intent": intent_value,
                    "portfolio": portfolio,
                    "risk": None,
                    "broker_waiting_reason": "IBKR is disconnected; execution quote unavailable",
                }
            return {"state": "COMPLETED", "intent": intent, "portfolio": portfolio, "risk": None}

    intent_value = intent
    with SQLiteStore(tmp_path / "durable-broker-wait.sqlite3") as store:
        runner = Runner()
        service = BackendService({}, runner, store)

        first = service.run_weekly_cycle(use_llm=False)
        pending = store.get_runtime("pending_execution_intent")

        assert first["state"] == "WAITING_FOR_BROKER"
        assert pending["intent"]["decision_id"] == intent.decision_id
        assert store.get_runtime("service_status") == "SAFE_MODE"
        assert store.get_runtime("trading_enabled") is False

        resumed = service.run_pending_entries()

        assert resumed[-1]["status"] == "COMPLETED"
        assert store.get_runtime("pending_execution_intent") is None
        assert runner.calls[1] == (intent, False)


def test_broker_recovery_queues_deferred_execution_resume_once(tmp_path):
    intent = TradeIntent(
        action="HOLD", symbol="MPC", target_weight=0.25, confidence=0.7,
        holding_period_days=20, expected_alpha_vs_spy=0.0, expected_alpha_vs_qqq=0.0,
        thesis=["test thesis"], risk_factors=["test risk"],
        invalidation_conditions=["test invalidation"], evidence_used=["test evidence"],
        decision_id="recovery-resume",
    )

    class Executor:
        mode = "IBKR_PAPER"
        broker_source = "IBKR_PAPER"
        execution_mode = "OBSERVE"
        connected = False
        broker_state_known = False

        def reconnect(self):
            self.connected = True
            self.broker_state_known = True

    class Runner:
        def __init__(self):
            self.executor = Executor()

        def reconcile_portfolio(self):
            return PortfolioState(equity=1000.0, peak_equity=1000.0, cash=750.0, current_symbol="MPC", current_quantity=1.0, current_weight=0.25)

    scheduler = PaperScheduler(SchedulerConfig(enabled=False))
    with SQLiteStore(tmp_path / "broker-recovery-resume.sqlite3") as store:
        store.set_runtime("pending_execution_intent", {"intent": intent.model_dump(mode="json"), "source": "WEEKLY_FULL_RESEARCH"})
        store.set_runtime("safe_mode", True)
        store.set_runtime("last_error", "IBKR is disconnected; execution quote unavailable")
        store.set_runtime("broker_monitor_connected", False)
        store.set_runtime("broker_reconciliation_ready", False)
        service = BackendService(
            {"execution": {"broker_reconnect_interval_seconds": 0}},
            Runner(),
            store,
            scheduler=scheduler,
        )

        service.monitor_broker_state()
        service.monitor_broker_state()

        resumes = [row for row in store.recent("control_commands", 10) if row["command"] == "RUN_PENDING_ENTRIES"]
        assert len(resumes) == 1
        assert resumes[0]["source"] == "BROKER_RECOVERY"


def test_backend_enters_safe_mode_when_startup_reconciliation_fails(tmp_path):
    class FailingRunner:
        def reconcile_portfolio(self):
            raise RuntimeError("broker callback timeout")

    with SQLiteStore(tmp_path / "runtime.sqlite3") as store:
        service = BackendService(
            cfg={"execution": {"mode": "local_paper"}},
            runner=FailingRunner(),
            store=store,
            scheduler=None,
        )

        result = service.start()

        assert result["service_status"] == "SAFE_MODE"
        assert result["scheduler_status"] == "STOPPED"
        assert result["trading_enabled"] is False
        assert store.get_runtime("safe_mode") is True
        service.stop()


def test_successful_reconciliation_clears_stale_backend_error(tmp_path):
    class Portfolio:
        risk_state = "NORMAL"

        def model_dump(self, mode="json"):
            return {"risk_state": self.risk_state}

    class HealthyRunner:
        def reconcile_portfolio(self):
            return Portfolio()

    with SQLiteStore(tmp_path / "reconciliation-clears-error.sqlite3") as store:
        store.set_runtime("last_error", "IBKR API did not become ready: nextValidId timeout")
        service = BackendService({}, HealthyRunner(), store)

        service.run_reconciliation_now()

        assert store.get_runtime("last_error") is None
        assert store.get_runtime("safe_mode") is False


def test_backend_restart_clears_stale_llm_failure_after_verified_runtime_probe(tmp_path):
    signature = {
        "luna_model": "gpt-5.6-luna",
        "sol_model": "gpt-5.6-sol",
        "api_protocol": "CHAT_COMPLETIONS",
        "protocol_fallback": True,
        "timeout_seconds": 300.0,
    }

    class HealthyRunner:
        executor = SimpleNamespace(
            mode="OBSERVE",
            broker_source="LOCAL",
            execution_mode="OBSERVE",
            connected=True,
            broker_state_known=True,
        )
        agent = SimpleNamespace(
            runtime=SimpleNamespace(
                luna_model="gpt-5.6-luna",
                sol_model="gpt-5.6-sol",
                api_protocol="CHAT_COMPLETIONS",
                protocol_fallback=True,
                timeout_seconds=300.0,
            )
        )

        def reconcile_portfolio(self):
            return SimpleNamespace(
                risk_state="NORMAL",
                as_of=datetime.now(timezone.utc).isoformat(),
            )

    with SQLiteStore(tmp_path / "stale-llm-state.sqlite3") as store:
        store.set_runtime("llm_status", "ERROR")
        store.set_runtime("llm_gateway_status", "OFFLINE")
        store.set_runtime("llm_luna_status", "ERROR")
        store.set_runtime("llm_sol_status", "ONLINE")
        store.set_runtime("llm_capability_status", "PASS")
        store.set_runtime("last_error", "Request timed out.")
        store.set_runtime(
            "llm_runtime_exact_probe",
            {"status": "PASS", "ok": True, "signature": signature},
        )

        service = BackendService(
            {"execution": {"mode": "OBSERVE"}},
            HealthyRunner(),
            store,
            use_llm=True,
        )
        try:
            service.start()

            assert store.get_runtime("llm_status") == "ONLINE"
            assert store.get_runtime("llm_gateway_status") == "ONLINE"
            assert store.get_runtime("llm_luna_status") == "ONLINE"
            assert store.get_runtime("llm_sol_status") == "ONLINE"
        finally:
            service.stop()


def test_runtime_probe_cache_is_invalidated_when_timeout_changes(tmp_path):
    class Runner:
        agent = SimpleNamespace(
            runtime=SimpleNamespace(
                luna_model="gpt-5.6-luna",
                sol_model="gpt-5.6-sol",
                api_protocol="CHAT_COMPLETIONS",
                protocol_fallback=True,
                timeout_seconds=300.0,
            )
        )

        def __init__(self):
            self.probe_calls = 0

        def probe_runtime_llm_exact_path(self):
            self.probe_calls += 1
            return {"status": "PASS", "ok": True}

    with SQLiteStore(tmp_path / "probe-cache-invalidation.sqlite3") as store:
        store.set_runtime(
            "llm_runtime_exact_probe",
            {
                "status": "PASS",
                "ok": True,
                "signature": {
                    "luna_model": "gpt-5.6-luna",
                    "sol_model": "gpt-5.6-sol",
                    "api_protocol": "CHAT_COMPLETIONS",
                    "protocol_fallback": True,
                },
            },
        )
        runner = Runner()
        service = BackendService({}, runner, store, use_llm=True)

        service._ensure_runtime_llm_probe()

        assert runner.probe_calls == 1
        assert store.get_runtime("llm_runtime_exact_probe")["signature"]["timeout_seconds"] == 300.0


def test_unknown_broker_open_order_puts_backend_in_safe_mode(tmp_path):
    class UnknownOrderRunner:
        def reconcile_portfolio(self):
            raise RuntimeError("UNKNOWN BROKER OPEN ORDER")

    with SQLiteStore(tmp_path / "unknown-order.sqlite3") as store:
        service = BackendService({}, UnknownOrderRunner(), store)

        result = service.start()

        assert result["service_status"] == "SAFE_MODE"
        assert result["safe_mode"] is True
        assert result["trading_enabled"] is False
        assert "UNKNOWN BROKER OPEN ORDER" in result["last_error"]
        service.stop()


def test_risk_monitor_exception_enters_safe_mode_and_blocks_ai_execution(tmp_path):
    class FailingRiskRunner:
        def run_daily_risk_check(self, record_performance=True):
            raise RuntimeError("quote feed down")

        def run(self, **kwargs):
            pytest.fail("AI execution must not run after risk-monitor failure")

    class RunningScheduler:
        def __init__(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "risk-monitor-safe-mode.sqlite3") as store:
        scheduler = RunningScheduler()
        service = BackendService({}, FailingRiskRunner(), store, scheduler=scheduler)

        with pytest.raises(RuntimeError, match="quote feed down"):
            service.run_risk_monitor_cycle()

        status = service.status()
        assert status["service_status"] == "SAFE_MODE"
        assert status["safe_mode"] is True
        assert status["trading_enabled"] is False
        assert status["scheduler_status"] == "STOPPED"
        assert scheduler.started is False
        with pytest.raises(RuntimeError, match="SAFE_MODE"):
            service.run_weekly_cycle(use_llm=True)


def test_sol_pipeline_failure_blocks_ai_but_keeps_risk_monitor_running(tmp_path):
    from src.llm_agent import PipelineError

    class FailingLLMRunner:
        def run(self, **kwargs):
            raise PipelineError("Sol research/CIO failed: gateway offline")

        def run_daily_risk_check(self, record_performance=True):
            return {"portfolio": SimpleNamespace(risk_state="NORMAL"), "state": "IDLE", "risk": None, "intent": None}

    class RunningScheduler:
        def __init__(self):
            self.started = True

        def stop(self):
            self.started = False

        def status(self):
            return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "llm-failure.sqlite3") as store:
        scheduler = RunningScheduler()
        service = BackendService({}, FailingLLMRunner(), store, scheduler=scheduler, use_llm=True)

        with pytest.raises(PipelineError, match="Sol research"):
            service.run_weekly_cycle(use_llm=True)

        status = service.status()
        assert status["service_status"] == "ONLINE"
        assert status["safe_mode"] is False
        assert status["trading_enabled"] is False
        assert status["llm_status"] == "ERROR"
        assert scheduler.started is True
        risk_result = service.run_risk_monitor_cycle()
        assert risk_result["portfolio"].risk_state == "NORMAL"
        assert service.status()["trading_enabled"] is False


def test_backend_starts_scheduler_only_after_reconciliation(tmp_path):
    class HealthyRunner:
        def reconcile_portfolio(self):
            return type("Portfolio", (), {"risk_state": "NORMAL"})()

    class FakeScheduler:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def status(self):
            return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "runtime.sqlite3") as store:
        scheduler = FakeScheduler()
        service = BackendService(
            cfg={"execution": {"mode": "local_paper"}},
            runner=HealthyRunner(),
            store=store,
            scheduler=scheduler,
        )

        result = service.start()

        assert scheduler.started
        assert result["service_status"] == "ONLINE"
        assert result["trading_enabled"] is True
        service.stop()
        assert not scheduler.started


def test_backend_clears_position_before_starting_after_hard_drawdown(tmp_path):
    from src.data_provider import MockDataProvider
    from src.execution.paper import LocalPaperExecutor
    from src.runner import StrategyRunner
    from test_safety_reconciliation import config

    class FakeScheduler:
        def __init__(self): self.started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    database = tmp_path / "startup-risk.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed-startup", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        runner = StrategyRunner(config(database), data, executor=broker, store=store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 50.0
        scheduler = FakeScheduler()
        service = BackendService(config(database), runner, store, scheduler=scheduler)

        result = service.start()

        assert broker.reconcile({"NVDA": 50.0}).positions == []
        assert result["risk_state"] == "RISK_HALTED"
        assert result["trading_enabled"] is False
        assert scheduler.started is True
        service.stop()


def test_singleton_lock_rejects_second_backend(tmp_path):
    from src.process_lock import LockAlreadyHeld, SingletonLock

    first = SingletonLock(tmp_path / "backend.lock")
    second = SingletonLock(tmp_path / "backend.lock")
    first.acquire()
    try:
        with pytest.raises(LockAlreadyHeld):
            second.acquire()
    finally:
        first.release()
        second.release()


def test_emergency_stop_liquidates_and_requires_explicit_resume(tmp_path):
    from src.data_provider import MockDataProvider
    from src.execution.paper import LocalPaperExecutor
    from src.models import OrderRequest
    from src.runner import StrategyRunner
    from test_safety_reconciliation import config

    class FakeScheduler:
        def __init__(self):
            self.started = False

        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    database = tmp_path / "manual-halt.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 100.0
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
        scheduler = FakeScheduler()
        service = BackendService(config(database), StrategyRunner(config(database), data, broker, store), store, scheduler=scheduler)
        service.start()

        result = service.emergency_stop()

        assert result["portfolio"]["current_symbol"] is None
        assert broker.reconcile({"NVDA": 100}).positions == []
        assert store.get_runtime("manual_halt") is True
        assert service.status()["service_status"] == "MANUAL_HALT"
        assert not scheduler.started
        service.resume_manual_halt()
        assert store.get_runtime("manual_halt") is False
        assert scheduler.started
        service.stop()


def test_backend_processes_dashboard_control_commands(tmp_path):
    class HealthyRunner:
        def reconcile_portfolio(self):
            return type("Portfolio", (), {"risk_state": "NORMAL", "model_dump": lambda self, **_: {"risk_state": "NORMAL"}})()

    class FakeScheduler:
        def __init__(self): self.started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "commands.sqlite3") as store:
        scheduler = FakeScheduler()
        service = BackendService({}, HealthyRunner(), store, scheduler=scheduler)
        service.start()
        try:
            command_id = store.enqueue_command("STOP_SCHEDULER")
            service.process_commands()

            command = store.connection.execute("SELECT status FROM control_commands WHERE id=?", (command_id,)).fetchone()
            assert command["status"] == "SUCCEEDED"
            assert scheduler.started is False
        finally:
            service.stop()


def test_command_worker_consumes_manual_command_while_scheduler_is_stopped(tmp_path):
    class HealthyRunner:
        def reconcile_portfolio(self):
            return type("Portfolio", (), {"risk_state": "NORMAL", "model_dump": lambda self, **_: {"risk_state": "NORMAL"}})()

    class StoppedScheduler:
        def start(self): pass
        def stop(self): pass
        def status(self): return {"status": "STOPPED"}

    with SQLiteStore(tmp_path / "worker.sqlite3") as store:
        service = BackendService({}, HealthyRunner(), store, scheduler=StoppedScheduler())
        service.start()
        command_id = store.enqueue_command("RUN_RECONCILIATION")
        deadline = datetime.now(timezone.utc) + timedelta(seconds=3)
        command = None
        while datetime.now(timezone.utc) < deadline:
            command = store.connection.execute("SELECT * FROM control_commands WHERE id=?", (command_id,)).fetchone()
            if command["status"] == "SUCCEEDED":
                break
            service._stop_event.wait(0.05)

        assert command["status"] == "SUCCEEDED"
        assert command["worker_id"]
        assert command["claimed_at"]
        assert command["started_at"]
        assert command["completed_at"]
        assert store.get_runtime("command_worker_status") == "RUNNING"
        assert store.get_runtime("scheduler_status") == "STOPPED"
        service.stop()


def test_command_worker_heartbeat_stays_fresh_during_long_command(tmp_path):
    class BlockingRunner:
        def __init__(self):
            self.calls = 0
            self.entered = threading.Event()
            self.release = threading.Event()

        def reconcile_portfolio(self):
            self.calls += 1
            if self.calls > 1:
                self.entered.set()
                self.release.wait(5)
            return SimpleNamespace(risk_state="NORMAL", as_of=datetime.now(timezone.utc).isoformat())

    class StoppedScheduler:
        def start(self): pass
        def stop(self): pass
        def status(self): return {"status": "STOPPED"}

    with SQLiteStore(tmp_path / "long-command-heartbeat.sqlite3") as store:
        runner = BlockingRunner()
        service = BackendService({}, runner, store, scheduler=StoppedScheduler())
        service.start()
        try:
            store.enqueue_command("RUN_RECONCILIATION")
            assert runner.entered.wait(2)
            before = store.get_runtime("command_worker_heartbeat")
            time.sleep(1.5)
            after = store.get_runtime("command_worker_heartbeat")
            assert after != before
        finally:
            runner.release.set()
            service.stop()


def test_atomic_command_claim_and_safe_stale_recovery(tmp_path):
    database = tmp_path / "claims.sqlite3"
    with SQLiteStore(database) as first, SQLiteStore(database) as second:
        command_id = first.enqueue_command("RUN_AI_RESEARCH")

        claimed = first.claim_next_command("worker-a")
        assert claimed["id"] == command_id
        assert second.claim_next_command("worker-b") is None
        assert first.mark_command_running(command_id, "worker-a") is True
        assert first.recover_stale_claims("9999-01-01T00:00:00+00:00") == 0

        first.complete_command(command_id, result={"ok": True}, worker_id="worker-a")
        row = second.connection.execute("SELECT * FROM control_commands WHERE id=?", (command_id,)).fetchone()
        assert row["status"] == "SUCCEEDED"
        assert row["completed_at"]


def test_stale_claim_that_never_started_is_requeued(tmp_path):
    with SQLiteStore(tmp_path / "stale.sqlite3") as store:
        command_id = store.enqueue_command("RUN_RISK_CHECK")
        store.claim_next_command("dead-worker")

        assert store.recover_stale_claims("9999-01-01T00:00:00+00:00") == 1
        row = store.connection.execute("SELECT * FROM control_commands WHERE id=?", (command_id,)).fetchone()
        assert row["status"] == "QUEUED"
        assert row["worker_id"] is None


def test_abandoned_running_command_is_failed_without_replay(tmp_path):
    class HealthyRunner:
        def reconcile_portfolio(self):
            return type("Portfolio", (), {"risk_state": "NORMAL", "model_dump": lambda self, **_: {"risk_state": "NORMAL"}})()

    with SQLiteStore(tmp_path / "abandoned-running.sqlite3") as store:
        command_id = store.enqueue_command("RUN_AI_RESEARCH")
        store.claim_next_command("dead-worker")
        store.mark_command_running(command_id, "dead-worker")

        service = BackendService({}, HealthyRunner(), store)
        service.start()
        try:
            row = store.command(command_id)
            assert row["status"] == "FAILED"
            assert row["worker_id"] == "dead-worker"
            assert row["error_stage"] == "WORKER_RECOVERY"
            assert "not replayed" in row["error_message"]
            assert any(event["event_type"] == "COMMAND_WORKER_RECOVERY" for event in store.runtime_events())
        finally:
            service.stop()


def test_first_run_observe_stays_ready_without_starting_scheduler(tmp_path, monkeypatch):
    class HealthyRunner:
        executor = SimpleNamespace(mode="IBKR_PAPER", broker_source="IBKR_PAPER", execution_mode="OBSERVE")

    class FakeScheduler:
        def __init__(self): self.started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    result = FirstRunResult(
        "READY_FOR_OBSERVE",
        [HealthCheck("TWS", "PASS"), HealthCheck("Broker Mutation", "PASS")],
        "READY FOR OBSERVE",
        {},
    )
    monkeypatch.setattr("src.service.run_first_run_observe", lambda *args, **kwargs: result)

    with SQLiteStore(tmp_path / "first-run.sqlite3") as store:
        scheduler = FakeScheduler()
        service = BackendService({}, HealthyRunner(), store, scheduler=scheduler, first_run_observe=True)

        status = service.start()

        assert status["service_status"] == "READY_FOR_OBSERVE"
        assert status["first_run_ready"] is True
        assert status["trading_enabled"] is False
        assert scheduler.started is False
        service.stop()


def test_startup_socket_failure_shows_waiting_for_tws_login(tmp_path):
    class WaitingRunner:
        def reconcile_portfolio(self):
            raise TimeoutError("IBKR API did not become ready: nextValidId timeout")

    with SQLiteStore(tmp_path / "waiting-tws.sqlite3") as store:
        service = BackendService({}, WaitingRunner(), store)

        status = service.start()

        assert status["service_status"] == "WAITING_FOR_TWS_PAPER_LOGIN"
        assert status["trading_enabled"] is False
        assert status["safe_mode"] is False
        service.stop()


def test_manual_tws_login_is_followed_by_automatic_reconnect_reconcile_and_scheduler_recovery(tmp_path):
    class RecoveringExecutor:
        mode = "IBKR_PAPER"
        broker_source = "IBKR_PAPER"
        connected = False
        broker_state_known = False

        def reconnect(self):
            self.connected = True
            return {"connected": True}

    class RecoveringRunner:
        def __init__(self):
            self.executor = RecoveringExecutor()
            self.reconcile_calls = 0

        def reconcile_portfolio(self):
            self.reconcile_calls += 1
            self.executor.broker_state_known = True
            return SimpleNamespace(
                risk_state="NORMAL",
                as_of=datetime.now(timezone.utc).isoformat(),
                model_dump=lambda **_: {"risk_state": "NORMAL"},
            )

    class FakeScheduler:
        def __init__(self): self.started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "automatic-tws-recovery.sqlite3") as store:
        runner = RecoveringRunner()
        scheduler = FakeScheduler()
        service = BackendService(
            {"execution": {"broker_reconnect_interval_seconds": 0}},
            runner,
            store,
            scheduler=scheduler,
        )
        store.set_runtime("service_status", "WAITING_FOR_TWS_PAPER_LOGIN")
        store.set_runtime("safe_mode", False)
        store.set_runtime("trading_enabled", False)

        result = service.monitor_broker_state()

        assert result == {"connected": True, "reconciled": True}
        assert runner.reconcile_calls == 1
        assert scheduler.started is True
        assert store.get_runtime("service_status") == "ONLINE"
        assert store.get_runtime("broker_status") == "CONNECTED"
        assert store.get_runtime("trading_enabled") is True


def test_automatic_broker_recovery_does_not_clear_manual_or_risk_halt(tmp_path):
    class Executor:
        mode = "IBKR_PAPER"
        broker_source = "IBKR_PAPER"
        connected = False
        broker_state_known = False

        def reconnect(self):
            self.connected = True

    class Runner:
        def __init__(self): self.executor = Executor()
        def reconcile_portfolio(self):
            self.executor.broker_state_known = True
            return SimpleNamespace(risk_state="RISK_HALTED", as_of=datetime.now(timezone.utc).isoformat())

    class Scheduler:
        started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "halted-tws-recovery.sqlite3") as store:
        scheduler = Scheduler()
        service = BackendService({"execution": {"broker_reconnect_interval_seconds": 0}}, Runner(), store, scheduler=scheduler)
        store.set_runtime("manual_halt", True)

        service.monitor_broker_state()

        assert scheduler.started is False
        assert store.get_runtime("service_status") == "MANUAL_HALT"
        assert store.get_runtime("trading_enabled") is False


def test_transient_quote_safe_mode_is_retried_and_cleared_after_full_reconciliation(tmp_path):
    class Executor:
        mode = "IBKR_PAPER"
        broker_source = "IBKR_PAPER"
        connected = True
        broker_state_known = True

    class Runner:
        def __init__(self):
            self.executor = Executor()
            self.calls = 0

        def reconcile_portfolio(self):
            self.calls += 1
            return SimpleNamespace(risk_state="NORMAL", as_of=datetime.now(timezone.utc).isoformat())

    class Scheduler:
        def __init__(self): self.started = False
        def start(self): self.started = True
        def stop(self): self.started = False
        def status(self): return {"status": "RUNNING" if self.started else "STOPPED"}

    with SQLiteStore(tmp_path / "transient-quote-recovery.sqlite3") as store:
        runner = Runner()
        scheduler = Scheduler()
        service = BackendService({"execution": {"broker_reconnect_interval_seconds": 0}}, runner, store, scheduler=scheduler)
        store.set_runtime("safe_mode", True)
        store.set_runtime("service_status", "SAFE_MODE")
        store.set_runtime("last_error", "IBKR execution quote timed out for MPC")

        result = service.monitor_broker_state()

        assert result == {"connected": True, "reconciled": True}
        assert runner.calls == 1
        assert store.get_runtime("safe_mode") is False
        assert store.get_runtime("service_status") == "ONLINE"
        assert scheduler.started is True
