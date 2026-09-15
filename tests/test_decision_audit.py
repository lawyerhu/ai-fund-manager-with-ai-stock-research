from datetime import datetime, timezone
from types import SimpleNamespace
import json

import pytest

from src.decision_audit import evidence_changes, record_shadow_observations, shadow_summary
from src.models import SizingAudit
from src.storage import SQLiteStore
from src.position_manager import PositionManager
from src.llm_agent import SolResearchCIOAgent
from src.data_provider import MockDataProvider
from src.models import PortfolioState
from tests.test_position_manager import managed_position, weekly_review


def test_timestamps_are_not_new_negative_evidence():
    old = {"fundamentals": {"eps": None, "as_of": "yesterday"}}
    current = {"fundamentals": {"eps": None, "as_of": "today"}}
    assert evidence_changes(old, current)["unchanged_sources"] == ["fundamentals"]
    assert evidence_changes(old, current)["unchanged_missing_fields"] == ["fundamentals.eps"]
    assert evidence_changes({}, current)["baseline_available"] is False
    current["fundamentals"]["eps"] = 2
    assert evidence_changes(old, current)["changed_sources"] == ["fundamentals"]


def test_context_includes_prior_and_committee_without_recursive_history(tmp_path):
    with SQLiteStore(tmp_path / "audit.db") as store:
        review = weekly_review(comparison_context={"previous_review": "must not recurse"},
                               evidence_snapshot={"fundamentals": {"eps": None}})
        store.save_position_review(review)
        context = PositionManager({}, store).review_context(managed_position(), committee_decision={"target_weight": .3})
        assert context["previous_review"]["review_id"] == review.review_id
        assert "comparison_context" not in context["previous_review"]
        assert context["committee_decision"]["target_weight"] == .3
        assert context["previous_review"]["evidence_snapshot"]["fundamentals"]["eps"] is None


def test_missing_sizing_explanation_is_invalid():
    with pytest.raises(ValueError):
        SizingAudit.model_validate({"weight_basis": ""})


def test_real_review_path_passes_history_and_validates_evidence_refs():
    calls = []
    audit = dict(weight_basis="Correct documented concentration", incremental_reason="New measured volatility",
                 risk_budget_basis="UNKNOWN", downside_scenario="Qualitative", evidence_refs=["holding_fundamentals"])
    def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(dict(
            action="REDUCE", thesis_status="WEAKENING", current_holding_score=75,
            confidence=.7, new_horizon_days=20, reason=["New evidence"], target_weight=.25,
            sizing_audit=audit)), usage=None)
    sol = SolResearchCIOAgent("test-sol", MockDataProvider(), provider=SimpleNamespace(create_response=respond))
    portfolio = PortfolioState(equity=1000, peak_equity=1000, cash=500, current_symbol="NVDA", current_weight=.5)
    context = {"previous_review": {"review_id": "previous", "target_weight": .3,
                                   "evidence_snapshot": {"holding_fundamentals": {"eps": None}}},
               "committee_decision": {"target_weight": .3}}
    review = sol.review_position(managed_position(symbol="NVDA", current_weight=.5), portfolio,
                                candidate_symbols=[], review_type="WEEKLY", event_context=context, replacement_threshold=10)
    assert review.sizing_audit.weight_basis == audit["weight_basis"]
    assert review.comparison_context["previous_review_id"] == "previous"
    assert "holding_fundamentals" in review.evidence_snapshot
    assert '"target_weight": 0.3' in calls[0]["input"]
    assert "Do not apply another haircut" in calls[0]["input"]
    audit["evidence_refs"] = ["invented_source"]
    with pytest.raises(ValueError, match="invalid position review"):
        sol.review_position(managed_position(symbol="NVDA"), portfolio, candidate_symbols=[],
                            review_type="DAILY", event_context=context, replacement_threshold=10)


def proposal():
    return dict(created_at="2026-09-08T15:00:00+00:00", symbol="MPC", old_weight=.5, target_weight=.25,
                equity=10000, horizon_days=20, cost_assumptions=dict(
                    minimum_commission=1, commission_per_order=0, commission_per_share=.005, slippage_bps=5))


def test_shadow_reduction_costs_and_benchmark_alignment():
    rows = [dict(date="2026-09-08", price=100, spy=100, qqq=100),
            dict(date="2026-09-09", price=90, spy=101, qqq=102)]
    result = shadow_summary(proposal(), rows)
    assert result["hold_return"] == pytest.approx(-.05)
    assert result["reduce_net_return"] == pytest.approx(-.025225)
    assert result["reduction_benefit"] == pytest.approx(.024775)
    assert result["reduce_net_excess_vs_qqq"] == pytest.approx(-.045225)
    assert result["hold_observed_max_drawdown"] == pytest.approx(.05)
    rows[-1]["price"] = 110
    assert shadow_summary(proposal(), rows)["reduction_benefit"] < 0  # Counts missed upside too.


def test_shadow_is_idempotent_and_never_mutates_orders(tmp_path):
    with SQLiteStore(tmp_path / "shadow.db") as store:
        store.save_reduction_shadow("r", proposal())
        store.save_reduction_shadow("r", {**proposal(), "target_weight": 0})
        data = SimpleNamespace(price_history=lambda *args: {"last_bar_at": "2026-09-08T00:00:00-04:00", "price": 100})
        benchmark = dict(dates=["2026-09-08"], series={"SPY": [100], "QQQ": [100]})
        for _ in range(2):
            record_shadow_observations(store, data, benchmark, datetime(2026, 9, 8, 21, tzinfo=timezone.utc))
        row = store.reduction_shadows()[0]
        assert len(row["observations"]) == 1
        assert row["proposal"]["target_weight"] == .25
        assert store.recent("order_records") == []
        assert store.pending_commands() == []
        assert store.recent("llm_decisions") == []


@pytest.mark.parametrize("bar,at,dates", [
    ("2026-09-08", "2026-09-08T19:00:00+00:00", ["2026-09-08"]),  # Incomplete session
    ("2026-09-10", "2026-09-11T21:00:00+00:00", ["2026-09-11"]),  # Unsynchronized symbol
    ("2026-09-12", "2026-09-12T21:00:00+00:00", ["2026-09-12"]),  # Fake Saturday
    ("2026-09-04", "2026-09-07T21:00:00+00:00", ["2026-09-04"]),  # Before decision
])
def test_invalid_shadow_dates_do_not_create_observations(tmp_path, bar, at, dates):
    with SQLiteStore(tmp_path / "dates.db") as store:
        store.save_reduction_shadow("r", proposal())
        record_shadow_observations(store, SimpleNamespace(price_history=lambda *a: dict(last_bar_at=bar, price=100)),
                                   dict(dates=dates, series={"SPY": [100], "QQQ": [100]}), datetime.fromisoformat(at))
        assert store.reduction_shadows()[0]["observations"] == []


def test_reduction_proposal_freezes_costs_and_does_not_execute(tmp_path):
    with SQLiteStore(tmp_path / "proposal.db") as store:
        manager = PositionManager({"execution": {"slippage_bps": 5}}, store)
        review = weekly_review(action="REDUCE", target_weight=.25)
        manager.record_reduction_shadow(managed_position(current_weight=.5), review, 10000)
        manager.record_reduction_shadow(managed_position(current_weight=.4), review, 20000)
        assert len(store.reduction_shadows()) == 1
        assert store.reduction_shadows()[0]["proposal"]["old_weight"] == .5
        assert store.recent("order_records") == []


def test_shadow_fixed_horizon_stops_collecting_and_can_be_read_only(tmp_path):
    path = tmp_path / "horizon.db"
    with SQLiteStore(path) as store:
        store.save_reduction_shadow("r", {**proposal(), "horizon_days": 1})
        store.save_shadow_observation("r", dict(date="2026-09-08", price=100, spy=100, qqq=100))
        data = SimpleNamespace(price_history=lambda *a: pytest.fail("expired experiment must not fetch prices"))
        record_shadow_observations(store, data, dict(dates=["2026-09-10"], series={"SPY": [100], "QQQ": [100]}),
                                   datetime(2026, 9, 10, 21, tzinfo=timezone.utc))
        assert store.reduction_shadows()[0]["status"] == "COMPLETE"
    with SQLiteStore(path, read_only=True) as store:
        assert len(store.reduction_shadows()[0]["observations"]) == 1
