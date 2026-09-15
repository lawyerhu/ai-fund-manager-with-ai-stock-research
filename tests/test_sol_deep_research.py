import json
from types import SimpleNamespace

import pytest

from src.data_provider import MockDataProvider
from src.llm_agent import InvestmentCommitteeDecision, LLMProvider, LLMRuntimeConfig, LunaSolPipeline, SolResearchCIOAgent
from src.models import PortfolioState


class ScriptedProvider(LLMProvider):
    gateway = "ccswitch"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            output=[],
            output_text=json.dumps(self.payloads.pop(0)),
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


def ranking_payload():
    symbols = ["NVDA", "META", "AVGO", "MSFT", "MPC"]
    return {
        "ranking": [
            {
                "rank": rank,
                "symbol": symbol,
                "preliminary_alpha_score": 96 - rank * 4,
                "confidence": 0.85 - rank * 0.04,
                "key_strengths": [f"{symbol} strength"],
                "key_weaknesses": [f"{symbol} risk"],
                "research_priority": "HIGH" if rank <= 3 else "MEDIUM",
                "unknown_fields": ["commodity differentials"] if symbol == "MPC" else [],
            }
            for rank, symbol in enumerate(symbols, start=1)
        ],
        "comparison_basis": ["momentum", "fundamentals", "valuation", "revisions", "risk"],
        "missing_data_sources": ["Crack spread data unavailable"],
    }


def deep_payload(symbol):
    return {
        "symbol": symbol,
        "business_driver_summary": f"Structured causal research for {symbol}",
        "causal_drivers": [{
            "driver": "earnings revisions",
            "category": "COMPANY",
            "direction": "POSITIVE",
            "estimated_importance": "HIGH",
            "evidence_quality": "HIGH",
            "persistence": "MEDIUM",
            "priced_in": "PARTIAL",
            "mechanism": "Higher estimates support earnings and valuation expectations",
            "earnings_impact": "POSITIVE",
            "valuation_impact": "POSITIVE",
            "price_impact": "POSITIVE",
            "reversal_conditions": ["Estimate revisions turn negative"],
            "evidence": ["analyst revisions"],
        }],
        "return_attribution": {
            "company_specific_alpha": "MEDIUM",
            "industry_beta": "MEDIUM",
            "energy_beta": "UNKNOWN",
            "broad_market_beta": "LOW",
            "evidence_quality": "MEDIUM",
            "explanation": "Relative performance supports partial company attribution",
        },
        "why_now": ["Recent estimate revisions create a near-term expectation gap"],
        "market_expectations": ["Consensus expects continued growth"],
        "pricing_assessment": "FAIRLY_PRICED",
        "priced_in_drivers": [{
            "driver": "earnings revisions",
            "priced_in_level": "MEDIUM",
            "remaining_alpha_potential": "MEDIUM",
        }],
        "alternative_hypotheses": [
            {"hypothesis": "company execution", "supporting_evidence": ["earnings"], "opposing_evidence": ["valuation"], "evidence_quality": "HIGH"},
            {"hypothesis": "sector beta", "supporting_evidence": ["peer move"], "opposing_evidence": ["relative strength"], "evidence_quality": "MEDIUM"},
            {"hypothesis": "broad market beta", "supporting_evidence": ["SPY trend"], "opposing_evidence": ["excess return"], "evidence_quality": "MEDIUM"},
        ],
        "primary_hypothesis": "company execution",
        "near_term_catalyst": "Estimate revisions",
        "geopolitical_contribution": "UNKNOWN",
        "geopolitical_mechanism": "DATA_UNAVAILABLE",
        "twenty_day_persistence": "MEDIUM",
        "bull_case": {"scenario": "Drivers persist", "key_assumptions": ["Revisions continue"], "expected_direction": "STRONG_UPSIDE", "probability_confidence": "MEDIUM"},
        "base_case": {"scenario": "Trend moderates", "key_assumptions": ["Fundamentals hold"], "expected_direction": "MODERATE_UPSIDE", "probability_confidence": "MEDIUM"},
        "bear_case": {"scenario": "Expectations reverse", "key_assumptions": ["Revisions weaken"], "expected_direction": "MODERATE_DOWNSIDE", "probability_confidence": "MEDIUM"},
        "key_risks": ["Expectations reverse"],
        "thesis_invalidation_conditions": ["Relative strength and revisions turn negative"],
        "evidence": [{"claim": "Estimates improved", "source": "analyst revisions", "evidence_quality": "MEDIUM"}],
        "missing_data": ["Crack spread data unavailable"] if symbol == "MPC" else [],
        "required_data": ["Reliable current crack spread series"] if symbol == "MPC" else [],
        "mpc_specific_answers": {"war_primary_driver": "UNKNOWN"} if symbol == "MPC" else {},
    }


def bear_payload():
    return {
        "reviews": [{
            "symbol": symbol,
            "strongest_bear_argument": f"{symbol} expectations may already be priced in",
            "failure_modes": ["momentum exhaustion", "earnings reversal"],
            "what_would_make_me_change_my_mind": ["Negative revisions"],
            "better_alternative_exists": symbol != "NVDA",
            "evidence_quality": "MEDIUM",
        } for symbol in ("NVDA", "META", "AVGO")]
    }


def final_payload():
    return {
        "action": "BUY",
        "selected_symbol": "NVDA",
        "rank": 1,
        "runner_up": "META",
        "alpha_score": 92,
        "runner_up_score": 88,
        "alpha_gap": 4,
        "confidence": 0.74,
        "confidence_basis": ["Multiple independent evidence sets agree"],
        "confidence_reducers": ["Recent appreciation"],
        "expected_alpha_vs_spy": 0.04,
        "expected_alpha_vs_qqq": None,
        "expected_alpha_basis": ["Relative strength and revisions"],
        "estimate_type": "SOL_MODEL_ESTIMATE",
        "thesis_horizon_days": 20,
        "target_weight": 0.50,
        "primary_alpha_drivers": ["Earnings revisions"],
        "sizing_audit": {
            "weight_basis": "Test allocation", "incremental_reason": "New evidence",
            "risk_budget_basis": "UNKNOWN", "downside_scenario": "Qualitative downside",
            "evidence_refs": ["STAGE_B"],
        },
        "company_specific_drivers": ["Execution"],
        "industry_drivers": ["AI spending"],
        "macro_drivers": ["Constructive market regime"],
        "geopolitical_drivers": ["UNKNOWN"],
        "priced_in_assessment": "FAIRLY_PRICED",
        "bull_case": {"scenario": "Upside persists", "key_assumptions": ["Revisions continue"], "expected_direction": "STRONG_UPSIDE", "probability_confidence": "MEDIUM"},
        "base_case": {"scenario": "Trend moderates", "key_assumptions": ["Growth holds"], "expected_direction": "MODERATE_UPSIDE", "probability_confidence": "MEDIUM"},
        "bear_case": {"scenario": "Expectations reverse", "key_assumptions": ["Revisions weaken"], "expected_direction": "MODERATE_DOWNSIDE", "probability_confidence": "MEDIUM"},
        "thesis_invalidation_conditions": ["Revisions and relative strength turn negative"],
        "why_selected": ["Best risk-adjusted causal evidence"],
        "why_not_runner_up": ["META catalyst is less clear"],
        "why_not_rank3": ["AVGO downside asymmetry is weaker"],
        "pairwise_comparisons": [
            {"selected_symbol": "NVDA", "alternative_symbol": "META", "selected_advantages": ["revisions"], "alternative_advantages": ["valuation"], "higher_alpha_potential": "NVDA", "lower_downside_risk": "META", "clearer_catalyst": "NVDA", "more_priced_in": "NVDA", "decision_reason": "Higher causal confidence"},
            {"selected_symbol": "NVDA", "alternative_symbol": "AVGO", "selected_advantages": ["liquidity"], "alternative_advantages": ["valuation"], "higher_alpha_potential": "NVDA", "lower_downside_risk": "AVGO", "clearer_catalyst": "NVDA", "more_priced_in": "NVDA", "decision_reason": "Stronger revisions"},
        ],
        "strongest_bear_argument": "The market has already priced in revisions",
        "what_would_make_me_change_my_mind": ["Negative revisions"],
        "missing_data": ["Crack spread data unavailable"],
        "tool_coverage": ["price", "fundamentals", "news", "SEC", "analyst revisions"],
        "evidence_used": ["stage_a", "stage_b", "stage_c"],
        "risk_factors": ["Valuation compression"],
    }


def test_deep_research_runs_rank_top3_bear_and_ic_as_separate_bounded_stages():
    final = final_payload()
    final["risk_factors"] = [f"risk-{index}" for index in range(7)]
    final["thesis_invalidation_conditions"] = [f"condition-{index}" for index in range(7)]
    provider = ScriptedProvider([
        ranking_payload(),
        deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
        bear_payload(), final,
    ])
    events = []
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol",
        MockDataProvider(),
        provider=provider,
        deep_research_enabled=True,
        event_sink=events.append,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "BUY"
    assert intent.symbol == "NVDA"
    assert intent.expected_alpha_vs_qqq == 0.0
    assert len(intent.risk_factors) == 6
    assert len(intent.invalidation_conditions) == 6
    assert len(provider.calls) == 6
    assert [item.symbol for item in agent.last_deep_research.top_three] == ["NVDA", "META", "AVGO"]
    assert agent.last_deep_research.final_decision.expected_alpha_vs_qqq is None
    assert all(item.bull_case.scenario for item in agent.last_deep_research.top_three)
    assert all(item.base_case.scenario for item in agent.last_deep_research.top_three)
    assert all(item.bear_case.scenario for item in agent.last_deep_research.top_three)
    assert agent.last_deep_research.tool_coverage.missing_sources
    assert "Reliable current crack spread series" not in agent.last_deep_research.tool_coverage.missing_sources
    event_types = [event["event_type"] for event in events]
    assert event_types.count("SOL_DEEP_DIVE_COMPLETED") == 3
    assert event_types[-1] == "SOL_IC_DECISION_COMPLETED"
    assert "chain-of-thought" not in str(events).lower()


def test_committee_reference_schema_is_enumerated_without_restricting_position_reviews():
    from src.models import SizingAudit
    schema = InvestmentCommitteeDecision.model_json_schema()
    ref = schema["properties"]["sizing_audit"]["$ref"].split("/")[-1]
    items = schema["$defs"][ref]["properties"]["evidence_refs"]["items"]
    assert set(items["enum"]) == {"STAGE_A", "STAGE_B", "STAGE_C", "TOOL_COVERAGE"}
    audit = final_payload()["sizing_audit"]
    audit["evidence_refs"] = ["holding_fundamentals"]
    assert SizingAudit.model_validate(audit).evidence_refs == ["holding_fundamentals"]


@pytest.mark.parametrize("repaired", [True, False])
def test_committee_bad_reference_repair_has_original_evidence_and_fails_closed(repaired):
    bad = final_payload()
    bad["sizing_audit"]["evidence_refs"] = ["STAGE_B_JSON.NVDA"]
    provider = ScriptedProvider([
        ranking_payload(), deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
        bear_payload(), bad, final_payload() if repaired else bad,
    ])
    agent = SolResearchCIOAgent("gpt-6-astra", MockDataProvider(), provider=provider,
                                deep_research_enabled=True, max_repair_retries=1)
    def run():
        return agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000),
                            ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"])
    if repaired:
        assert run().symbol == "NVDA"
        assert agent.last_deep_research.final_decision.sizing_audit.evidence_refs == ["STAGE_B"]
    else:
        with pytest.raises(ValueError, match="invalid structured output"):
            run()
        assert agent.last_deep_research is None
    assert len(provider.calls) == 7
    repair = provider.calls[-1]["input"]
    assert provider.calls[-2]["input"] in repair
    assert "STAGE_B_JSON.NVDA" in repair
    assert "Structured causal research for NVDA" in repair


def _persisted_ranking_evidence(symbols):
    return [
        {
            "symbol": symbol,
            "tool": tool,
            "arguments": {"symbol": symbol},
            "result": {"symbol": symbol, "source": tool, "value": "persisted"},
        }
        for symbol in symbols
        for tool in ("get_price_history", "get_fundamentals", "get_analyst_revisions", "get_upcoming_events")
    ]


def test_sol_resume_reuses_complete_ranking_evidence_and_continues_deep_research():
    symbols = ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"]
    provider = ScriptedProvider([
        ranking_payload(),
        deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
        bear_payload(), final_payload(),
    ])
    events = []
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(), provider=provider,
        deep_research_enabled=True, event_sink=events.append,
    )

    intent = agent.decide_from_persisted_ranking_evidence(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        symbols,
        _persisted_ranking_evidence(symbols),
        source_run_id="failed-run-1",
    )

    assert intent.symbol == "NVDA"
    assert len(provider.calls) == 6
    assert sum(event["event_type"] == "SOL_TOOL_RESULT_REUSED" for event in events) == 24
    assert any(item.get("reused_from_run_id") == "failed-run-1" for item in agent.last_research_evidence)


def test_sol_resume_fails_closed_when_persisted_ranking_evidence_is_incomplete():
    symbols = ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"]
    evidence = _persisted_ranking_evidence(symbols)[:-1]
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(), provider=ScriptedProvider([]), deep_research_enabled=True,
    )

    with pytest.raises(ValueError, match="incomplete"):
        agent.decide_from_persisted_ranking_evidence(
            PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            symbols,
            evidence,
            source_run_id="failed-run-1",
        )


def test_pipeline_enables_deep_research_and_exposes_structured_stage_metadata(monkeypatch):
    class SixStockData(MockDataProvider):
        def universe_snapshot(self):
            return [{"symbol": symbol} for symbol in ("NVDA", "META", "AVGO", "MSFT", "MPC", "VLO")]

    provider = ScriptedProvider([
        ranking_payload(),
        deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
        bear_payload(), final_payload(),
    ])
    monkeypatch.setenv("LLM_PIPELINE", "SOL_ONLY")
    monkeypatch.setenv("SOL_DEEP_RESEARCH_ENABLED", "true")
    runtime = LLMRuntimeConfig.from_mapping({
        "llm": {
            "pipeline": "SOL_ONLY",
            "deep_research_enabled": True,
            "deep_research_top_five": 5,
            "deep_research_top_three": 3,
        }
    })
    pipeline = LunaSolPipeline(SixStockData(), provider, runtime)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    metadata = pipeline.last_pipeline_metadata
    assert pipeline.sol.deep_research_enabled is True
    assert [item["symbol"] for item in metadata["sol_stage_a"]["ranking"]] == ["NVDA", "META", "AVGO", "MSFT", "MPC"]
    assert [item["symbol"] for item in metadata["sol_top_three"]] == ["NVDA", "META", "AVGO"]
    assert metadata["sol_final_decision"]["expected_alpha_vs_qqq"] is None
    assert metadata["missing_data_sources"]
    assert metadata["total_sol_tool_calls"] == len(pipeline.sol.last_tool_calls)
    assert [item["stage"] for item in metadata["sol_stage_metrics"]] == [
        "SOL_STAGE_A_RANKING", "SOL_DEEP_DIVE", "SOL_DEEP_DIVE", "SOL_DEEP_DIVE",
        "SOL_STAGE_C_ADVERSARIAL", "SOL_STAGE_D_IC_DECISION",
    ]
    assert all(item["latency_ms"] >= 0 for item in metadata["sol_stage_metrics"])


def test_deep_research_fails_closed_when_stage_a_returns_unknown_symbol():
    ranking = ranking_payload()
    ranking["ranking"][0]["symbol"] = "UNKNOWN_TICKER"
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol",
        MockDataProvider(),
        provider=ScriptedProvider([ranking]),
        deep_research_enabled=True,
        max_repair_retries=0,
    )

    with pytest.raises(ValueError, match="outside Final Candidates"):
        agent.decide(
            PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
        )


def test_investment_committee_cannot_select_outside_top_three():
    final = final_payload()
    final["selected_symbol"] = "MSFT"
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol",
        MockDataProvider(),
        provider=ScriptedProvider([
            ranking_payload(),
            deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
            bear_payload(), final,
        ]),
        deep_research_enabled=True,
        max_repair_retries=0,
    )

    with pytest.raises(ValueError, match="outside Top3"):
        agent.decide(
            PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
        )


def test_mpc_missing_specialist_data_remains_explicitly_unavailable():
    ranking = ranking_payload()
    ranking["ranking"][0]["symbol"] = "MPC"
    ranking["ranking"][4]["symbol"] = "NVDA"
    final = final_payload()
    final["selected_symbol"] = "MPC"
    for comparison in final["pairwise_comparisons"]:
        comparison["selected_symbol"] = "MPC"
    provider = ScriptedProvider([
        ranking,
        deep_payload("MPC"), deep_payload("META"), deep_payload("AVGO"),
        {
            "reviews": [
                {**item, "symbol": symbol}
                for item, symbol in zip(bear_payload()["reviews"], ("MPC", "META", "AVGO"))
            ]
        },
        final,
    ])
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol",
        MockDataProvider(),
        provider=provider,
        deep_research_enabled=True,
        max_repair_retries=0,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    mpc = agent.last_deep_research.top_three[0]
    assert intent.symbol == "MPC"
    assert mpc.mpc_specific_answers.oil_vs_crack_spread == "DATA_UNAVAILABLE"
    assert "Crack spread data unavailable" in mpc.missing_data
    assert "Reliable current crack spread series" in agent.last_deep_research.tool_coverage.missing_sources


def test_investment_committee_may_hold_cash_when_evidence_is_insufficient():
    final = final_payload()
    final.update({
        "action": "HOLD_CASH",
        "selected_symbol": None,
        "rank": None,
        "alpha_score": None,
        "runner_up_score": None,
        "alpha_gap": None,
        "target_weight": 0.0,
        "expected_alpha_vs_spy": None,
        "expected_alpha_vs_qqq": None,
    })
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol",
        MockDataProvider(),
        provider=ScriptedProvider([
            ranking_payload(),
            deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
            bear_payload(), final,
        ]),
        deep_research_enabled=True,
        max_repair_retries=0,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "CASH"
    assert intent.symbol is None
    assert intent.target_weight == 0
    assert agent.last_deep_research.final_decision.expected_alpha_vs_spy is None


def test_force_stock_selection_projects_cash_ic_to_top1_buy_without_bypassing_risk():
    final = final_payload()
    final.update({
        "action": "HOLD_CASH", "selected_symbol": None, "rank": None,
        "alpha_score": None, "runner_up_score": None, "alpha_gap": None,
        "target_weight": 0.0, "expected_alpha_vs_spy": None, "expected_alpha_vs_qqq": None,
    })
    provider = ScriptedProvider([
        ranking_payload(),
        deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
        bear_payload(), final,
    ])
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(), provider=provider,
        deep_research_enabled=True, force_stock_selection=True,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "BUY"
    assert intent.symbol == "NVDA"
    assert intent.target_weight == 1.0
    assert agent.last_deep_research.final_decision.action == "BUY"
    assert agent.last_deep_research.final_decision.expected_alpha_vs_qqq is None


def test_data_limited_top1_gets_aggressive_seventy_percent_request():
    ranking = ranking_payload()
    ranking["ranking"][0]["symbol"] = "MPC"
    ranking["ranking"][4]["symbol"] = "NVDA"
    final = final_payload()
    final.update({
        "action": "HOLD_CASH", "selected_symbol": None, "rank": None,
        "alpha_score": None, "runner_up_score": None, "alpha_gap": None,
        "target_weight": 0.0, "expected_alpha_vs_spy": None, "expected_alpha_vs_qqq": None,
    })
    events = []
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(),
        provider=ScriptedProvider([
            ranking,
            deep_payload("MPC"), deep_payload("META"), deep_payload("AVGO"),
            {
                "reviews": [
                    {**item, "symbol": symbol}
                    for item, symbol in zip(bear_payload()["reviews"], ("MPC", "META", "AVGO"))
                ]
            },
            final,
        ]),
        deep_research_enabled=True,
        force_stock_selection=True,
        event_sink=events.append,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "BUY"
    assert intent.symbol == "MPC"
    assert intent.target_weight == 0.70
    assert intent.confidence == ranking["ranking"][0]["confidence"]
    policy_event = next(event for event in events if event["event_type"] == "SOL_FORCED_STOCK_SELECTION")
    assert policy_event["metadata"]["selection_tier"] == "DATA_LIMITED"
    assert policy_event["metadata"]["original_action"] == "HOLD_CASH"


def test_force_stock_selection_preserves_cash_for_explicit_negative_base_case():
    final = final_payload()
    final.update({
        "action": "HOLD_CASH", "selected_symbol": None, "rank": None,
        "alpha_score": None, "runner_up_score": None, "alpha_gap": None,
        "target_weight": 0.0, "expected_alpha_vs_spy": -0.03, "expected_alpha_vs_qqq": -0.04,
        "base_case": {
            "scenario": "Expected alpha and price direction are negative",
            "key_assumptions": ["Earnings revisions weaken"],
            "expected_direction": "MODERATE_DOWNSIDE",
            "probability_confidence": "MEDIUM",
        },
    })
    events = []
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(),
        provider=ScriptedProvider([
            ranking_payload(),
            deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
            bear_payload(), final,
        ]),
        deep_research_enabled=True,
        force_stock_selection=True,
        event_sink=events.append,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "CASH"
    assert intent.symbol is None
    assert intent.target_weight == 0
    assert not any(event["event_type"] == "SOL_FORCED_STOCK_SELECTION" for event in events)


def test_weak_nonnegative_top1_gets_thirty_five_percent_exploration_request():
    ranking = ranking_payload()
    ranking["ranking"][0]["preliminary_alpha_score"] = 70
    ranking["ranking"][0]["confidence"] = 0.60
    final = final_payload()
    final.update({
        "action": "HOLD_CASH", "selected_symbol": None, "rank": None,
        "alpha_score": None, "runner_up_score": None, "alpha_gap": None,
        "target_weight": 0.0, "expected_alpha_vs_spy": None, "expected_alpha_vs_qqq": None,
    })
    events = []
    agent = SolResearchCIOAgent(
        "gpt-5.6-sol", MockDataProvider(),
        provider=ScriptedProvider([
            ranking,
            deep_payload("NVDA"), deep_payload("META"), deep_payload("AVGO"),
            bear_payload(), final,
        ]),
        deep_research_enabled=True,
        force_stock_selection=True,
        event_sink=events.append,
    )

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
    )

    assert intent.action == "BUY"
    assert intent.symbol == "NVDA"
    assert intent.target_weight == 0.35
    assert intent.confidence == 0.60
    policy_event = next(event for event in events if event["event_type"] == "SOL_FORCED_STOCK_SELECTION")
    assert policy_event["metadata"]["selection_tier"] == "EXPLORATORY"
