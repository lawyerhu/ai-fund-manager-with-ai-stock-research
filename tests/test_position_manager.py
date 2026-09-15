from datetime import datetime, timedelta, timezone

from src.models import ManagedPosition, PositionReview
from src.position_manager import PositionManager
from src.storage import SQLiteStore
import pytest
import json
from types import SimpleNamespace

from src.data_provider import MockDataProvider
from src.llm_agent import LLMProvider, SolResearchCIOAgent
from src.models import PortfolioState, TradeIntent
from src.service import BackendService
from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest
from src.runner import StrategyRunner


def managed_position(**updates):
    values = {
        "position_id": "position-mpc",
        "run_id": "run-entry",
        "decision_id": "decision-entry",
        "symbol": "MPC",
        "entry_time": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
        "entry_price": 180.50,
        "current_price": 190.20,
        "current_weight": 0.98,
        "original_thesis_horizon_days": 20,
        "current_thesis_horizon_days": 20,
        "original_thesis": ["Refining margins remain strong"],
        "original_positive_factors": ["Capacity discipline"],
        "original_risks": ["Crack spread compression"],
    }
    values.update(updates)
    return ManagedPosition(**values)


def test_policy_override_buy_cannot_keep_cash_thesis_intact(tmp_path):
    with SQLiteStore(tmp_path / "policy-thesis.sqlite3") as store:
        manager = PositionManager({}, store)
        position = managed_position(
            original_thesis=["Cash is valid and the evidence does not justify concentration."],
            original_positive_factors=["Cash is valid and the evidence does not justify concentration."],
            latest_thesis=["Cash is valid and the evidence does not justify concentration."],
        )
        intent = TradeIntent(
            action="BUY",
            symbol="MPC",
            target_weight=1.0,
            confidence=0.88,
            holding_period_days=20,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=position.original_thesis,
            risk_factors=["Refining-cycle normalization"],
            invalidation_conditions=["Crack spreads normalize"],
            evidence_used=[
                "Stage A ranked MPC first with an ordinal score of 86.4.",
                "MPC showed strong cross-sectional momentum but causal data remained incomplete.",
            ],
            decision_id=position.decision_id,
        )

        repaired = manager.ensure_entry_thesis_consistency(position, intent)

        assert repaired.current_thesis_status == "EXPIRED_REVIEW_REQUIRED"
        assert repaired.original_thesis[0].startswith("MPC entered under the configured alpha-validation selection policy")
        assert any("ranked MPC first" in reason for reason in repaired.original_positive_factors)
        assert not any("Cash is valid" in reason for reason in repaired.original_thesis)


def weekly_review(**updates):
    values = {
        "position_id": "position-mpc",
        "run_id": "run-review",
        "decision_id": "decision-review",
        "review_type": "WEEKLY",
        "current_holding": "MPC",
        "action": "REPLACE",
        "thesis_status": "INTACT",
        "current_holding_score": 80,
        "best_alternative": "NVDA",
        "best_alternative_score": 85,
        "replacement_gap": 5,
        "replacement_threshold": 10,
        "confidence": 0.80,
        "days_held": 7,
        "original_horizon_days": 20,
        "new_horizon_days": 20,
        "reason": ["NVDA is only marginally stronger"],
        "target_weight": 0.98,
    }
    values.update(updates)
    return PositionReview(**values)


def test_marginal_replacement_is_held_and_audited(tmp_path):
    with SQLiteStore(tmp_path / "position-manager.sqlite3") as store:
        manager = PositionManager({"portfolio_review": {"replacement_threshold": 10}}, store)
        position = manager.save_position(managed_position())

        outcome = manager.apply_review(position, weekly_review())

        assert outcome.review.action == "HOLD"
        assert outcome.trade_intent is None
        assert store.managed_position(position.position_id).current_thesis_status == "INTACT"
        assert store.position_reviews(position.position_id)[0].action == "HOLD"


def test_material_replacement_becomes_risk_gated_switch_intent(tmp_path):
    with SQLiteStore(tmp_path / "material-replacement.sqlite3") as store:
        manager = PositionManager({"portfolio_review": {"replacement_threshold": 10}}, store)
        position = manager.save_position(managed_position())

        outcome = manager.apply_review(
            position,
            weekly_review(best_alternative_score=95, replacement_gap=15),
        )

        assert outcome.review.action == "REPLACE"
        assert outcome.review.replacement_gap == 15
        assert outcome.trade_intent.action == "SWITCH"
        assert outcome.trade_intent.current_symbol == "MPC"
        assert outcome.trade_intent.new_symbol == "NVDA"
        assert outcome.trade_intent.target_weight == 0.98


@pytest.mark.parametrize(
    ("action", "target_weight", "expected_intent_action"),
    [
        ("ADD", 1.0, "BUY"),
        ("REDUCE", 0.50, "HOLD"),
        ("SELL", 0.0, "CASH"),
        ("EXIT_TO_CASH", 0.0, "CASH"),
    ],
)
def test_portfolio_actions_reuse_existing_trade_intents(tmp_path, action, target_weight, expected_intent_action):
    with SQLiteStore(tmp_path / f"{action}.sqlite3") as store:
        manager = PositionManager({}, store)
        outcome = manager.apply_review(
            manager.save_position(managed_position()),
            weekly_review(action=action, target_weight=target_weight, best_alternative=None, best_alternative_score=None),
        )

        assert outcome.trade_intent.action == expected_intent_action
        assert outcome.trade_intent.target_weight == target_weight


def test_broken_thesis_cannot_remain_hold(tmp_path):
    with SQLiteStore(tmp_path / "broken.sqlite3") as store:
        manager = PositionManager({}, store)
        outcome = manager.apply_review(
            manager.save_position(managed_position()),
            weekly_review(action="HOLD", thesis_status="BROKEN", best_alternative=None, best_alternative_score=None, target_weight=0),
        )

        assert outcome.review.action == "EXIT_TO_CASH"
        assert outcome.trade_intent.action == "CASH"
        assert outcome.position.current_thesis_status == "BROKEN"


def test_expired_intact_thesis_can_extend_horizon(tmp_path):
    position = managed_position(entry_time=(datetime.now(timezone.utc) - timedelta(days=20)).isoformat())
    with SQLiteStore(tmp_path / "extend.sqlite3") as store:
        manager = PositionManager({}, store)
        assert manager.review_due(position, datetime.now(timezone.utc)) == "HORIZON"

        outcome = manager.apply_review(
            manager.save_position(position),
            weekly_review(
                review_type="HORIZON",
                action="EXTEND_HOLD",
                days_held=20,
                new_horizon_days=35,
                best_alternative=None,
                best_alternative_score=None,
            ),
        )

        assert outcome.trade_intent is None
        assert outcome.position.current_thesis_horizon_days == 35


def test_extend_hold_before_horizon_is_rejected(tmp_path):
    with SQLiteStore(tmp_path / "early-extend.sqlite3") as store:
        manager = PositionManager({}, store)
        with pytest.raises(ValueError, match="horizon"):
            manager.apply_review(
                manager.save_position(managed_position()),
                weekly_review(action="EXTEND_HOLD", best_alternative=None, best_alternative_score=None),
            )


@pytest.mark.parametrize("event_type", ["MAJOR_NEWS", "SEC_FILING", "MARKET_REGIME_CHANGE"])
def test_material_events_are_persisted_for_emergency_review(tmp_path, event_type):
    now = datetime.now(timezone.utc)
    with SQLiteStore(tmp_path / f"{event_type}.sqlite3") as store:
        manager = PositionManager({"monitoring": {"review_cooldown_minutes": 60}}, store)
        position = manager.save_position(managed_position())

        accepted = manager.register_trigger(position, f"event-{event_type}", event_type, occurred_at=now)

        assert accepted is True
        assert store.position_triggers(position.position_id)[0].event_type == event_type


def test_duplicate_event_and_review_cooldown_suppress_repeated_sol_calls(tmp_path):
    now = datetime.now(timezone.utc)
    with SQLiteStore(tmp_path / "dedup.sqlite3") as store:
        manager = PositionManager({"monitoring": {"review_cooldown_minutes": 60}}, store)
        position = manager.save_position(managed_position())

        assert manager.register_trigger(position, "news-1", "NEGATIVE_NEWS", occurred_at=now)
        assert not manager.register_trigger(position, "news-1", "NEGATIVE_NEWS", occurred_at=now + timedelta(minutes=1))
        assert not manager.register_trigger(position, "news-2", "NEGATIVE_NEWS", occurred_at=now + timedelta(minutes=30))
        assert manager.register_trigger(position, "news-3", "NEGATIVE_NEWS", occurred_at=now + timedelta(minutes=61))

        assert len(store.position_triggers(position.position_id)) == 2


def test_sol_position_review_uses_dedicated_structured_schema_without_full_universe():
    class PositionData(MockDataProvider):
        def universe_snapshot(self):
            raise AssertionError("position review must not run Luna or load the full universe")

    class Provider(LLMProvider):
        def __init__(self):
            self.calls = []

        def create_response(self, **kwargs):
            self.calls.append(kwargs)
            payload = {
                "action": "REPLACE",
                "thesis_status": "WEAKENING",
                "current_holding_score": 72,
                "best_alternative": "META",
                "best_alternative_score": 91,
                "replacement_gap": 19,
                "confidence": 0.84,
                "new_horizon_days": 20,
                "reason": ["META has materially stronger comparative alpha"],
                "risk_level": "ELEVATED",
                "target_weight": 0.80,
                "requires_full_research": False,
                "sizing_audit": {
                    "weight_basis": "Test weight rationale", "incremental_reason": "New comparative evidence",
                    "risk_budget_basis": "UNKNOWN", "downside_scenario": "Qualitative downside",
                    "evidence_refs": ["holding_fundamentals"],
                },
            }
            return SimpleNamespace(id="review-1", output=[], output_text=json.dumps(payload), usage=None)

    provider = Provider()
    sol = SolResearchCIOAgent("gpt-5.6-sol", PositionData(), provider=provider)
    position = managed_position(symbol="NVDA")
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=20, current_symbol="NVDA", current_weight=0.98, current_quantity=5)

    review = sol.review_position(
        position,
        portfolio,
        candidate_symbols=["META"],
        review_type="WEEKLY",
        event_context={},
        replacement_threshold=10,
    )

    assert review.action == "REPLACE"
    assert review.position_id == position.position_id
    assert review.replacement_gap == 19
    assert provider.calls[0]["text"]["format"]["name"] == "sol_position_review"
    assert "Current holding, supplied alternatives, and CASH" in provider.calls[0]["input"]


def test_backend_material_event_enqueues_one_deduplicated_emergency_review(tmp_path):
    with SQLiteStore(tmp_path / "service-trigger.sqlite3") as store:
        manager = PositionManager({"monitoring": {"review_cooldown_minutes": 60}}, store)
        position = manager.save_position(managed_position())
        runner = SimpleNamespace(position_manager=manager, executor=SimpleNamespace(connected=True, broker_state_known=True))
        service = BackendService({}, runner, store)

        assert service.trigger_position_event("filing-1", "SEC_FILING", symbol=position.symbol)
        assert not service.trigger_position_event("filing-1", "SEC_FILING", symbol=position.symbol)

        commands = store.pending_commands()
        assert len(commands) == 1
        assert commands[0]["command"] == "RUN_EVENT_SOL_REVIEW"


def position_runner_config(min_confidence=0.55):
    return {
        "portfolio": {"starting_equity": 1000, "max_positions": 1},
        "agent": {"decision_horizon_days": 20},
        "portfolio_review": {"replacement_threshold": 10},
        "strategy": {"max_strategic_switches_per_week": 1, "switch_hysteresis": {}},
        "execution": {"minimum_order_notional": 1, "buy_cash_safety_factor": 0.98},
        "risk": {
            "hard_drawdown_limit": 0.25,
            "halt_recovery_drawdown": 0.20,
            "target_annualized_vol": 1.0,
            "absolute_max_weight": 1.0,
            "min_confidence_to_open": min_confidence,
            "max_data_age_minutes": 30,
            "max_decision_age_minutes_for_execution": 60,
            "max_annualized_volatility": 1.0,
            "min_avg_dollar_volume": 0,
            "max_gap_pct": 0.20,
            "drawdown_tiers": [],
            "event_risk": {},
        },
    }


class ReplacementSol:
    last_tool_calls = [{"name": "get_fundamentals"}]

    def review_position(self, position, portfolio, **kwargs):
        return weekly_review(
            position_id=position.position_id,
            current_holding=position.symbol,
            action="REPLACE",
            best_alternative="META",
            best_alternative_score=95,
            current_holding_score=80,
            replacement_gap=15,
            target_weight=0.80,
        )


def seeded_position_runner(tmp_path, min_confidence=0.55):
    store = SQLiteStore(tmp_path / f"runner-{min_confidence}.sqlite3")
    broker = LocalPaperExecutor(starting_cash=1000, store=store)
    broker.connect()
    broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=1, reference_price=180))
    agent = SimpleNamespace(sol=ReplacementSol(), set_event_sink=lambda sink: None)
    return store, broker, StrategyRunner(position_runner_config(min_confidence), MockDataProvider(), broker, store, agent=agent)


def test_risk_approved_replace_sells_old_before_buying_new(tmp_path):
    store, broker, runner = seeded_position_runner(tmp_path)
    with store:
        result = runner.run_position_review({"candidate_symbols": ["META"], "event_type": "WEEKLY_DISCOVERY"})

        assert result["review_action"] == "REPLACE"
        assert result["execution"]["risk"].approved is True
        positions = broker.current_positions()
        assert [position["symbol"] for position in positions] == ["META"]
        actions = [json.loads(row["order_json"])["action"] for row in reversed(store.recent("order_records", 2))]
        assert actions == ["SELL", "BUY"]


def test_risk_rejected_replace_keeps_current_holding(tmp_path):
    store, broker, runner = seeded_position_runner(tmp_path, min_confidence=0.90)
    with store:
        result = runner.run_position_review({"candidate_symbols": ["META"], "event_type": "WEEKLY_DISCOVERY"})

        assert result["execution"]["risk"].approved is False
        assert [position["symbol"] for position in broker.current_positions()] == ["NVDA"]


def test_position_monitor_stop_loss_is_risk_gated_and_idempotent(tmp_path):
    cfg = position_runner_config()
    cfg["monitoring"] = {
        "review_drawdown_pct": 0.10,
        "hard_stop_loss_pct": 0.15,
        "take_profit_pct": None,
    }
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 180
    with SQLiteStore(tmp_path / "position-stop-loss.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=1, reference_price=180))
        runner = StrategyRunner(cfg, data, broker, store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 150

        result = runner.run_position_monitor()
        repeated = runner.run_position_monitor()

        assert result["monitor_action"] == "STOP_LOSS"
        assert result["execution"]["risk"].approved is True
        assert broker.current_positions() == []
        assert repeated["monitor_action"] == "NO_POSITION"
        sell_orders = [row for row in store.recent("order_records", 10) if json.loads(row["order_json"])["action"] == "SELL"]
        assert len(sell_orders) == 1


def test_position_monitor_warning_drawdown_requests_sol_review_without_selling(tmp_path):
    cfg = position_runner_config()
    cfg["monitoring"] = {
        "review_drawdown_pct": 0.10,
        "hard_stop_loss_pct": 0.15,
        "take_profit_pct": None,
    }
    data = MockDataProvider()
    data.rows["NVDA"]["price"] = 180
    with SQLiteStore(tmp_path / "position-review-drawdown.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=1, reference_price=180))
        runner = StrategyRunner(cfg, data, broker, store)
        runner.reconcile_portfolio()
        data.rows["NVDA"]["price"] = 160

        result = BackendService(cfg, runner, store).run_position_monitor()

        assert result["monitor_action"] == "REVIEW_REQUIRED"
        assert result["event_type"] == "PRICE_SHOCK"
        assert result["review_queued"] is True
        assert [position["symbol"] for position in broker.current_positions()] == ["NVDA"]
        sell_orders = [row for row in store.recent("order_records", 10) if json.loads(row["order_json"])["action"] == "SELL"]
        assert sell_orders == []
        commands = store.recent("control_commands", 5)
        assert commands[0]["command"] == "RUN_EVENT_SOL_REVIEW"


@pytest.mark.parametrize(
    ("scenario", "position_updates", "review_updates", "expected_action"),
    [
        ("A", {}, {"action": "REPLACE", "best_alternative_score": 84, "current_holding_score": 80}, "HOLD"),
        ("B", {}, {"action": "REPLACE", "best_alternative_score": 95, "current_holding_score": 80}, "REPLACE"),
        ("C", {"current_price": 166.14}, {"action": "HOLD", "thesis_status": "INTACT", "best_alternative": None, "best_alternative_score": None}, "HOLD"),
        ("D", {"current_price": 166.14}, {"action": "HOLD", "thesis_status": "BROKEN", "best_alternative": None, "best_alternative_score": None, "target_weight": 0}, "EXIT_TO_CASH"),
        ("E", {"entry_time": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()}, {"review_type": "HORIZON", "action": "EXTEND_HOLD", "days_held": 20, "new_horizon_days": 35, "best_alternative": None, "best_alternative_score": None}, "EXTEND_HOLD"),
        ("F", {"entry_time": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()}, {"review_type": "HORIZON", "action": "EXIT_TO_CASH", "days_held": 20, "best_alternative": None, "best_alternative_score": None, "target_weight": 0}, "EXIT_TO_CASH"),
    ],
)
def test_mpc_six_scenario_dry_run(tmp_path, scenario, position_updates, review_updates, expected_action):
    with SQLiteStore(tmp_path / f"mpc-{scenario}.sqlite3") as store:
        manager = PositionManager({"portfolio_review": {"replacement_threshold": 10}}, store)
        position = manager.save_position(managed_position(position_id=f"position-mpc-{scenario}", **position_updates))
        review = weekly_review(position_id=position.position_id, **review_updates)

        outcome = manager.apply_review(position, review)

        assert outcome.review.action == expected_action


def test_weekly_llm_discovery_automatically_enters_position_manager_before_execution(tmp_path):
    store, broker, runner = seeded_position_runner(tmp_path)

    class WeeklyAgent:
        sol = ReplacementSol()
        model = "gpt-5.6-sol"
        prompt_version = "test"
        last_usage = {}
        last_research_evidence = []
        last_pipeline_metadata = {"candidate_symbols": ["META"], "pipeline": "LUNA_SOL"}

        def set_event_sink(self, sink):
            return None

        def decide(self, portfolio, horizon_days):
            return TradeIntent(
                action="HOLD", symbol="NVDA", target_weight=portfolio.current_weight,
                confidence=0.8, holding_period_days=20, expected_alpha_vs_spy=0,
                expected_alpha_vs_qqq=0, thesis=["Discovery completed"], risk_factors=["risk"],
                invalidation_conditions=["invalid"], evidence_used=["weekly_discovery"], model_name="sol",
            )

    runner.agent = WeeklyAgent()
    with store:
        result = runner.run(use_llm=True)

        assert result["intent"].action == "SWITCH"
        assert result["intent"].new_symbol == "META"
        assert [position["symbol"] for position in broker.current_positions()] == ["META"]
