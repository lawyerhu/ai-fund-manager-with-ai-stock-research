import json
from types import SimpleNamespace

import pytest

from src.data_provider import MockDataProvider
from src.llm_agent import (
    CCSwitchError,
    CCSwitchProvider,
    LLMProvider,
    LLMRuntimeConfig,
    LunaScreeningAgent,
    LunaSolPipeline,
    PipelineError,
    ScreeningCandidate,
    SolResearchCIOAgent,
)
from src.models import PortfolioState
from src.storage import SQLiteStore


def _decision(action="BUY", symbol="META", weight=0.5):
    return json.dumps({
        "action": action,
        "symbol": symbol if action != "CASH" else None,
        "target_weight": weight if action != "CASH" else 0.0,
        "confidence": 0.82,
        "holding_period_days": 20,
        "expected_excess_vs_spy": 0.03,
        "expected_excess_vs_qqq": 0.01,
        "thesis": ["Independent research supports the decision"],
        "risk_factors": ["Model uncertainty"],
        "invalidation_conditions": ["Research evidence changes"],
        "evidence_used": ["get_stock_snapshot"],
        "current_symbol": None,
        "new_symbol": None,
    })


def _comparative_decision(symbol="MPC", runner_up="ANET", ranking_symbols=None):
    symbols = list(ranking_symbols or [symbol, runner_up, "VEEV", "CRM", "ABNB"])
    symbol, runner_up = symbols[:2]
    ranking = []
    for rank, candidate in enumerate(symbols, start=1):
        ranking.append({
            "rank": rank,
            "symbol": candidate,
            "alpha_score": 90 - rank * 5,
            "confidence": 0.72 - rank * 0.03,
            "expected_alpha": 0.08 - rank * 0.01,
            "key_advantage": f"{candidate} advantage",
            "key_weakness": f"{candidate} weakness",
            "why_not_selected": [] if rank == 1 else [f"Lower risk-adjusted alpha than {symbol}"],
            "comparison": {
                "momentum_relative_strength": "POSITIVE",
                "earnings_trend": "POSITIVE",
                "revenue_eps_quality": "POSITIVE",
                "valuation": "ATTRACTIVE" if candidate == symbol else "MIXED",
                "analyst_revisions": "POSITIVE",
                "catalyst": "KNOWN",
                "event_risk": "LOW",
                "volatility": "MODERATE",
                "liquidity": "HIGH",
                "market_sector_fit": "POSITIVE",
                "downside_risk": "MODERATE",
                "expected_alpha": "POSITIVE",
            },
        })
    return json.dumps({
        "action": "BUY", "symbol": symbol, "target_weight": 0.5, "confidence": 0.72,
        "holding_period_days": 20, "expected_excess_vs_spy": 0.06, "expected_excess_vs_qqq": 0.08,
        "thesis": ["Best risk-adjusted candidate in the common Sol evaluation"],
        "risk_factors": ["Cyclical downside risk"],
        "thesis_invalidation_conditions": ["Relative strength breaks"],
        "evidence_used": ["get_price_history", "get_fundamentals"],
        "current_symbol": None, "new_symbol": None,
        "top_five": ranking,
        "selected_symbol": symbol, "runner_up_symbol": runner_up,
        "selected_alpha_score": 85, "runner_up_alpha_score": 80, "alpha_gap": 5,
        "why_selected": ["Superior valuation-adjusted momentum"],
        "why_selected_over_runner_up": [f"{symbol} has lower valuation risk than {runner_up}"],
        "bull_case": {"scenario": "Margins expand", "expected_return_or_direction": "Upside", "key_assumptions": ["Demand holds"]},
        "base_case": {"scenario": "Trend persists", "expected_return_or_direction": "Moderate upside", "key_assumptions": ["No earnings shock"]},
        "bear_case": {"scenario": "Cycle reverses", "expected_return_or_direction": "Downside", "key_assumptions": ["Margins compress"]},
        "expected_alpha_basis": ["momentum", "earnings", "valuation", "risk discount"],
        "estimate_type": "SOL_MODEL_ESTIMATE",
        "confidence_basis": ["Multiple independent evidence sources agree"],
        "confidence_reducers": ["Cyclical uncertainty prevents higher confidence"],
    })


class ScriptedProvider(LLMProvider):
    gateway = "ccswitch"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("unexpected gateway call")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    def estimate_cost(self, input_tokens, output_tokens, model=None):
        return round((input_tokens or 0) * 0.001 + (output_tokens or 0) * 0.002, 6)


def _response(text="", calls=None, response_id="response-1", usage=None):
    return SimpleNamespace(
        id=response_id,
        output=calls or [],
        output_text=text,
        usage=usage,
    )


def _screening(symbols=("NVDA", "META")):
    return json.dumps({
        "candidates": [
            {"symbol": symbol, "rationale": f"Screened {symbol}", "positive_signals": ["trend"], "risks": ["volatility"]}
            for symbol in symbols
        ],
        "screening_rationale": "Compact universe screen only",
    })


def test_ccswitch_provider_uses_custom_gateway_url_and_selected_model():
    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _response(_screening())

    client = SimpleNamespace(responses=Responses())
    provider = CCSwitchProvider(
        base_url="http://127.0.0.1:8787/v1",
        api_key="local-token",
        luna_model="gateway/luna",
        sol_model="gateway/sol",
        client=client,
    )

    provider.create_response(model="gateway/luna", input="screen")

    assert provider.base_url == "http://127.0.0.1:8787/v1"
    assert client.responses.calls[0]["model"] == "gateway/luna"
    assert "api.openai.com" not in provider.base_url


def test_ccswitch_provider_applies_per_model_reasoning_effort():
    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _response(_screening())

    client = SimpleNamespace(responses=Responses())
    runtime = LLMRuntimeConfig(
        provider="ccswitch",
        base_url="http://127.0.0.1:8787/v1",
        api_key="local-token",
        luna_model="gateway/luna",
        sol_model="gateway/sol",
        luna_reasoning_effort="max",
        sol_reasoning_effort="medium",
    )
    provider = CCSwitchProvider(runtime=runtime, client=client)

    provider.create_response(model="gateway/luna", input="screen")
    provider.create_response(model="gateway/sol", input="research")

    assert client.responses.calls[0]["reasoning"] == {"effort": "max"}
    assert client.responses.calls[1]["reasoning"] == {"effort": "medium"}


def test_luna_screening_returns_candidates_only_and_cannot_emit_trade_intent():
    provider = ScriptedProvider([_response(_screening(("NVDA",)))])
    agent = LunaScreeningAgent("gateway/luna", MockDataProvider(), provider=provider)

    result = agent.screen(MockDataProvider().universe_snapshot())

    assert result.candidate_symbols == ["NVDA"]
    assert not hasattr(result, "action")
    assert "action" not in agent.output_schema()["schema"]["properties"]


def test_sol_receives_candidates_and_can_reject_luna_first_candidate():
    tool_call = SimpleNamespace(type="function_call", name="get_stock_snapshot", arguments='{"symbol":"META"}', call_id="call-1")
    provider = ScriptedProvider([
        _response(calls=[tool_call], response_id="sol-1"),
        _response(_decision(symbol="META"), response_id="sol-2"),
    ])
    agent = SolResearchCIOAgent("gateway/sol", MockDataProvider(), provider=provider)

    result = agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000), ["NVDA", "META"])

    assert result.symbol == "META"
    assert provider.calls[0]["model"] == "gateway/sol"
    assert "NVDA, META" in provider.calls[0]["input"]
    assert agent.last_tool_calls[0]["name"] == "get_stock_snapshot"


def test_sol_comparative_decision_ranks_top_five_in_one_evaluation_without_changing_trade_intent():
    provider = ScriptedProvider([_response(_comparative_decision())])
    agent = SolResearchCIOAgent("gateway/sol", MockDataProvider(), provider=provider)

    intent = agent.decide(
        PortfolioState(equity=1000, peak_equity=1000, cash=1000),
        ["MPC", "ANET", "VEEV", "CRM", "ABNB", "PANW"],
    )

    analysis = agent.last_decision_analysis
    assert intent.symbol == "MPC"
    assert intent.expected_alpha_vs_spy == 0.06
    assert [item.symbol for item in analysis.top_five] == ["MPC", "ANET", "VEEV", "CRM", "ABNB"]
    assert analysis.runner_up_symbol == "ANET"
    assert analysis.alpha_gap == 5
    assert analysis.estimate_type == "SOL_MODEL_ESTIMATE"
    assert analysis.thesis_invalidation_conditions == ["Relative strength breaks"]


def test_sol_comparative_decision_rejects_incomplete_top_five_for_large_candidate_set():
    provider = ScriptedProvider([_response(_decision(symbol="MPC"))])
    agent = SolResearchCIOAgent("gateway/sol", MockDataProvider(), provider=provider, max_repair_retries=0)

    with pytest.raises(ValueError, match="comparative"):
        agent.decide(
            PortfolioState(equity=1000, peak_equity=1000, cash=1000),
            ["MPC", "ANET", "VEEV", "CRM", "ABNB", "PANW"],
        )


def test_pipeline_persists_sol_comparative_analysis_separately_from_trade_intent(tmp_path):
    symbols = ("MPC", "ANET", "VEEV", "CRM", "ABNB", "PANW")
    class FinalistProvider(MockDataProvider):
        def universe_snapshot(self):
            return [{"symbol": symbol, "price": 100 + index, "ret_20d": index / 100} for index, symbol in enumerate(symbols)]

    provider = ScriptedProvider([
        _response(_screening(symbols)),
        _response(_comparative_decision()),
    ])
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local",
        luna_model="luna-id", sol_model="sol-id", pipeline="LUNA_SOL",
    )
    pipeline = LunaSolPipeline(FinalistProvider(), provider, runtime)

    intent = pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))
    with SQLiteStore(tmp_path / "comparative.sqlite3") as store:
        store.save_decision(intent, pipeline_metadata=pipeline.last_pipeline_metadata)
        saved = json.loads(store.pipeline_history(1)[0]["final_decision_json"])

    assert saved["selected_symbol"] == "MPC"
    assert saved["runner_up_symbol"] == "ANET"
    assert len(saved["top_five"]) == 5
    assert saved["bull_case"]["scenario"] == "Margins expand"
    assert "top_five" not in intent.model_dump()


def test_sol_tool_calls_emit_only_structured_runtime_events():
    tool_call = SimpleNamespace(type="function_call", name="get_fundamentals", arguments='{"symbol":"NVDA"}', call_id="call-1")
    provider = ScriptedProvider([
        _response(calls=[tool_call], response_id="sol-tool-1"),
        _response(_decision(symbol="NVDA"), response_id="sol-tool-2"),
    ])
    events = []
    agent = SolResearchCIOAgent("gateway/sol", MockDataProvider(), provider=provider, event_sink=events.append)

    agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000), ["NVDA"])

    tool_events = [event for event in events if event["event_type"] in {"SOL_TOOL_CALL", "SOL_TOOL_RESULT"}]
    assert [event["event_type"] for event in tool_events] == ["SOL_TOOL_CALL", "SOL_TOOL_RESULT"]
    assert tool_events[0]["metadata"] == {"tool": "get_fundamentals", "arguments": {"symbol": "NVDA"}}
    assert tool_events[1]["metadata"]["status"] == "SUCCESS"
    assert tool_events[1]["metadata"]["result"]["symbol"] == "NVDA"
    assert "prompt" not in str(tool_events)


def test_sol_can_choose_cash_without_a_luna_trade_intent():
    provider = ScriptedProvider([_response(_decision(action="CASH"))])
    agent = SolResearchCIOAgent("gateway/sol", MockDataProvider(), provider=provider)

    result = agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000), ["NVDA"])

    assert result.action == "CASH"
    assert result.symbol is None
    assert result.target_weight == 0


def test_luna_sol_pipeline_records_gateway_models_and_stage_outputs():
    provider = ScriptedProvider([_response(_screening(("NVDA",))), _response(_decision(symbol="NVDA"))])
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id", pipeline="LUNA_SOL"
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    intent = pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    assert intent.model_name == "sol-id"
    assert pipeline.last_pipeline_metadata["pipeline"] == "LUNA_SOL"
    assert pipeline.last_pipeline_metadata["gateway"] == "ccswitch"
    assert pipeline.last_pipeline_metadata["luna_candidates"] == ["NVDA"]
    assert pipeline.last_pipeline_metadata["sol_model"] == "sol-id"


def test_luna_sol_pipeline_emits_safe_runtime_lifecycle_events():
    provider = ScriptedProvider([
        _response(_screening(("NVDA",))),
        _response(_decision(symbol="NVDA")),
    ])
    events = []
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id", pipeline="LUNA_SOL"
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime, event_sink=events.append)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    event_types = [event["event_type"] for event in events]
    assert "LUNA_SCREEN_STARTED" in event_types
    assert "LUNA_SCREEN_COMPLETED" in event_types
    assert "SOL_RESEARCH_STARTED" in event_types
    assert "SOL_DECISION_COMPLETED" in event_types
    assert events[0]["metadata"]["universe_size"] == 4
    assert next(event for event in events if event["event_type"] == "SOL_DECISION_COMPLETED")["metadata"]["action"] == "BUY"
    assert "chain" not in str(events)


def _large_universe(count):
    return [{"symbol": f"S{index:03d}", "price": 100 + index, "ret_20d": index / 1000} for index in range(count)]


class LargeUniverseProvider(MockDataProvider):
    def __init__(self, count):
        super().__init__()
        self.large_rows = _large_universe(count)

    def universe_snapshot(self):
        return self.large_rows


def test_luna_batches_cover_full_universe_and_final_screen_receives_merged_candidates():
    provider = ScriptedProvider([
        _response(_screening(("S000",))),
        _response(_screening(("S060",))),
        _response(_screening(("S000", "S060"))),
        _response(_decision(symbol="S000")),
    ])
    events = []
    runtime = LLMRuntimeConfig(luna_model="luna", sol_model="sol", luna_screen_batch_size=60, luna_screen_batch_top_k=6, luna_min_batch_size=15, luna_final_top_k=20)
    pipeline = LunaSolPipeline(LargeUniverseProvider(120), provider, runtime, event_sink=events.append)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    completed = [event for event in events if event["event_type"] == "LUNA_BATCH_COMPLETED"]
    covered = {symbol for event in completed for symbol in event["metadata"]["symbols"]}
    assert len(completed) == 2
    assert covered == {f"S{index:03d}" for index in range(120)}
    assert pipeline.last_pipeline_metadata["luna_candidates"] == ["S000", "S060"]
    assert len(provider.calls) == 4
    progress = [event["metadata"] for event in events if event["event_type"] == "AI_PROGRESS"]
    assert progress[-3]["universe_processed"] == 120
    assert progress[-3]["missing"] == 0


def test_luna_adaptive_split_60_to_30_preserves_coverage_and_progress():
    provider = ScriptedProvider([
        CCSwitchError("upstream overload", status_code=502, category="upstream"),
        _response(_screening(("S000",))),
        _response(_screening(("S030",))),
        _response(_screening(("S000", "S030"))),
        _response(_decision(symbol="S000")),
    ])
    events = []
    runtime = LLMRuntimeConfig(luna_model="luna", sol_model="sol", luna_screen_batch_size=60, luna_screen_batch_top_k=6, luna_min_batch_size=15, luna_final_top_k=20)
    pipeline = LunaSolPipeline(LargeUniverseProvider(60), provider, runtime, event_sink=events.append)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    split = next(event for event in events if event["event_type"] == "LUNA_BATCH_SPLIT")
    assert split["metadata"]["original_batch_size"] == 60
    assert split["metadata"]["split_sizes"] == [30, 30]
    progress = [event["metadata"]["progress_percent"] for event in events if event["event_type"] == "AI_PROGRESS"]
    assert progress == sorted(progress)
    assert {symbol for row in pipeline.last_luna_batches for symbol in row["symbols"]} == {f"S{index:03d}" for index in range(60)}


def test_luna_adaptive_split_30_to_15_and_minimum_failure_fails_closed():
    provider = ScriptedProvider([
        CCSwitchError("request too large", status_code=502),
        CCSwitchError("request too large", status_code=502),
        CCSwitchError("still unavailable", status_code=502),
    ])
    runtime = LLMRuntimeConfig(luna_model="luna", sol_model="sol", luna_screen_batch_size=60, luna_screen_batch_top_k=6, luna_min_batch_size=15, luna_final_top_k=20)
    events = []
    pipeline = LunaSolPipeline(LargeUniverseProvider(60), provider, runtime, event_sink=events.append)

    with pytest.raises(PipelineError, match="screening incomplete"):
        pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    splits = [event["metadata"]["split_sizes"] for event in events if event["event_type"] == "LUNA_BATCH_SPLIT"]
    assert splits == [[30, 30], [15, 15]]
    assert not any(event["event_type"] == "SOL_RESEARCH_STARTED" for event in events)
    assert len(provider.calls) == 3
    progress = [event["metadata"] for event in events if event["event_type"] == "AI_PROGRESS"][-1]
    assert progress["luna_batch_calls"] == 3
    assert progress["luna_retry_calls"] == 2
    assert progress["universe_processed"] == 0
    assert progress["error_stage"] == "LUNA_BATCH_SCREENING"


def test_luna_rate_limit_does_not_split_batch():
    provider = ScriptedProvider([
        CCSwitchError("rate limited", status_code=429, category="rate_limit"),
    ])
    runtime = LLMRuntimeConfig(
        luna_model="luna",
        sol_model="sol",
        luna_screen_batch_size=60,
        luna_screen_batch_top_k=6,
        luna_min_batch_size=15,
        luna_final_top_k=20,
    )
    events = []
    pipeline = LunaSolPipeline(LargeUniverseProvider(60), provider, runtime, event_sink=events.append)

    with pytest.raises(PipelineError, match="screening incomplete"):
        pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    assert len(provider.calls) == 1
    assert not any(event["event_type"] == "LUNA_BATCH_SPLIT" for event in events)


def test_luna_batch_usage_and_cost_are_aggregated_across_batch_and_final_calls():
    provider = ScriptedProvider([
        _response(_screening(("S000",)), usage=SimpleNamespace(input_tokens=10, output_tokens=2)),
        _response(_screening(("S060",)), usage=SimpleNamespace(input_tokens=20, output_tokens=3)),
        _response(_screening(("S000", "S060")), usage=SimpleNamespace(input_tokens=5, output_tokens=1)),
        _response(_decision(symbol="S000"), usage=SimpleNamespace(input_tokens=30, output_tokens=6)),
    ])
    runtime = LLMRuntimeConfig(luna_model="luna", sol_model="sol", luna_screen_batch_size=60, luna_screen_batch_top_k=6, luna_min_batch_size=15, luna_final_top_k=20)
    pipeline = LunaSolPipeline(LargeUniverseProvider(120), provider, runtime)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    usage = pipeline.last_pipeline_metadata["luna_usage"]
    assert usage["call_count"] == 3
    assert usage["input_tokens"] == 35
    assert usage["output_tokens"] == 6
    assert usage["estimated_cost"] == pytest.approx(0.047)


def test_luna_final_screening_stages_53_compact_candidates_before_global_tie_break():
    batch_symbols = [
        tuple(f"S{batch * 60 + offset:03d}" for offset in range(6 if batch < 8 else 5))
        for batch in range(9)
    ]
    merged = [symbol for symbols in batch_symbols for symbol in symbols]
    partitions = [merged[index::3] for index in range(3)]
    staged = [tuple(partition[:8]) for partition in partitions]
    provider = ScriptedProvider([
        *[_response(_screening(symbols)) for symbols in batch_symbols],
        *[_response(_screening(symbols)) for symbols in staged],
        _response(_screening(tuple(symbol for symbols in staged for symbol in symbols)[:20])),
        _response(_comparative_decision(
            symbol=staged[0][0],
            runner_up=staged[1][0],
            ranking_symbols=tuple(symbol for symbols in staged for symbol in symbols)[:5],
        )),
    ])
    runtime = LLMRuntimeConfig(
        luna_model="luna",
        sol_model="sol",
        luna_screen_batch_size=60,
        luna_screen_batch_top_k=6,
        luna_min_batch_size=15,
        luna_final_top_k=20,
        luna_final_max_candidates=24,
    )
    events = []
    pipeline = LunaSolPipeline(LargeUniverseProvider(518), provider, runtime, event_sink=events.append)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    final_calls = provider.calls[9:13]
    assert [call["input"].count('"symbol":') for call in final_calls[:3]] == [18, 18, 17]
    assert final_calls[3]["input"].count('"symbol":') == 24
    assert all('"price":' not in call["input"] for call in final_calls)
    assert all('"rationale":' in call["input"] for call in final_calls)
    final_events = [event for event in events if event["event_type"] == "LUNA_FINAL_SCREENING_STARTED"]
    assert [event["metadata"]["input_candidate_count"] for event in final_events] == [18, 18, 17, 24]
    assert pipeline.last_luna_final_screening == {
        "input_candidate_count": 53,
        "partition_sizes": [18, 18, 17],
        "partition_keep_counts": [8, 8, 8],
        "global_input_candidate_count": 24,
        "call_count": 4,
    }


def test_luna_final_screening_failure_does_not_call_sol():
    candidates = [
        ScreeningCandidate(symbol=f"S{index:03d}", rationale="batch", positive_signals=[], risks=[])
        for index in range(25)
    ]
    provider = ScriptedProvider([CCSwitchError("Request timed out", category="timeout")])
    runtime = LLMRuntimeConfig(luna_model="luna", sol_model="sol", luna_final_max_candidates=24)
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    with pytest.raises(CCSwitchError, match="timed out"):
        pipeline._screen_final_candidates(candidates, [])

    assert len(provider.calls) == 1


def test_luna_final_max_candidates_configuration(monkeypatch):
    monkeypatch.setenv("LUNA_FINAL_MAX_CANDIDATES", "24")

    runtime = LLMRuntimeConfig.from_mapping({})

    assert runtime.luna_final_max_candidates == 24


def test_aggressive_paper_selection_weights_are_configurable():
    runtime = LLMRuntimeConfig.from_mapping({
        "llm": {
            "forced_selection_high_weight": 1.0,
            "forced_selection_data_limited_weight": 0.70,
            "forced_selection_exploratory_weight": 0.35,
        }
    })

    assert runtime.forced_selection_high_weight == 1.0
    assert runtime.forced_selection_data_limited_weight == 0.70
    assert runtime.forced_selection_exploratory_weight == 0.35


def test_sol_only_does_not_call_luna():
    provider = ScriptedProvider([_response(_decision(symbol="META"))])
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id", pipeline="SOL_ONLY"
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "sol-id"
    assert pipeline.last_pipeline_metadata["pipeline"] == "SOL_ONLY"


def test_luna_failure_falls_back_to_sol_only_with_explicit_event():
    provider = ScriptedProvider([ValueError("luna offline"), _response(_decision(symbol="META"))])
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id", pipeline="LUNA_SOL", fallback_to_sol_only=True
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    events = pipeline.last_pipeline_metadata["fallback_events"]
    assert events[0]["event"] == "PIPELINE_FALLBACK"
    assert events[0]["reason"] == "luna offline"
    assert pipeline.last_pipeline_metadata["pipeline"] == "SOL_ONLY"


def test_sol_failure_fails_closed_without_a_new_trade_intent():
    provider = ScriptedProvider([_response(_screening(("NVDA",))), RuntimeError("sol offline")])
    runtime = LLMRuntimeConfig(provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id")
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    with pytest.raises(PipelineError, match="Sol"):
        pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))


def test_pipeline_usage_is_persisted_separately_for_luna_and_sol(tmp_path):
    provider = ScriptedProvider([
        _response(_screening(("NVDA",)), usage=SimpleNamespace(input_tokens=10, output_tokens=2)),
        _response(_decision(symbol="NVDA"), usage=SimpleNamespace(input_tokens=30, output_tokens=6)),
    ])
    runtime = LLMRuntimeConfig(provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id")
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)
    intent = pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    with SQLiteStore(tmp_path / "pipeline.sqlite3") as store:
        store.save_decision(intent, pipeline_metadata=pipeline.last_pipeline_metadata)
        usage = store.connection.execute("SELECT stage,model_name,input_tokens,output_tokens FROM model_usage ORDER BY id").fetchall()
        history = store.pipeline_history(1)

    assert [(row["stage"], row["model_name"], row["input_tokens"]) for row in usage] == [("LUNA", "luna-id", 10), ("SOL", "sol-id", 30)]
    assert history[0]["pipeline"] == "LUNA_SOL"


def test_sol_only_usage_does_not_create_a_fake_luna_stage(tmp_path):
    provider = ScriptedProvider([_response(_decision(symbol="META"), usage=SimpleNamespace(input_tokens=30, output_tokens=6))])
    runtime = LLMRuntimeConfig(
        provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id", pipeline="SOL_ONLY"
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)
    intent = pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    with SQLiteStore(tmp_path / "sol-only.sqlite3") as store:
        store.save_decision(intent, pipeline_metadata=pipeline.last_pipeline_metadata)
        usage = store.connection.execute("SELECT stage,model_name FROM model_usage ORDER BY id").fetchall()

    assert [(row["stage"], row["model_name"]) for row in usage] == [("SOL", "sol-id")]


def test_runner_records_model_change_for_a_new_pipeline_signature(tmp_path):
    from src.runner import StrategyRunner
    from test_safety_reconciliation import config

    class Agent:
        model = "sol-new"
        prompt_version = "luna-sol-v1"
        last_usage = {}
        last_research_evidence = []
        last_pipeline_metadata = {
            "pipeline": "LUNA_SOL",
            "gateway": "ccswitch",
            "luna_model": "luna-new",
            "sol_model": "sol-new",
            "luna_candidates": [],
            "candidate_symbols": [],
            "sol_tool_calls": [],
            "sol_research_evidence": [],
            "fallback_events": [],
            "prompt_versions": {"luna": "luna-screening-v1", "sol": "sol-research-cio-v1"},
            "luna_usage": {},
            "sol_usage": {},
            "usage": {},
            "final_decision": {},
        }

        def decide(self, portfolio, horizon_days=20):
            return _decision_intent("change-decision")

    with SQLiteStore(tmp_path / "model-change.sqlite3") as store:
        store.set_runtime("llm.model_signature", {"gateway": "ccswitch", "luna_model": "luna-old", "sol_model": "sol-old"})
        runner = StrategyRunner(config(tmp_path / "model-change.sqlite3"), MockDataProvider(), store=store, agent=Agent())
        runner.run(use_llm=True)
        events = store.recent("state_events", 20)

    assert any(row["state"] == "MODEL_CHANGE" for row in events)


def _decision_intent(decision_id):
    from src.models import TradeIntent

    return TradeIntent(
        action="CASH", symbol=None, target_weight=0.0, confidence=0.8, holding_period_days=20,
        expected_alpha_vs_spy=0.0, expected_alpha_vs_qqq=0.0, thesis=["cash"], risk_factors=["risk"],
        invalidation_conditions=["change"], evidence_used=["test"], model_name="sol-new", decision_id=decision_id,
    )


def test_first_run_dashboard_capability_can_be_injected(tmp_path):
    from src.first_run import run_first_run_observe
    from test_first_run_setup import _FixedMarketClock, _ObserveBroker

    result = run_first_run_observe(
        _ObserveBroker(),
        llm_check=lambda: True,
        dashboard_checker=lambda: True,
        database_path=tmp_path / "first-run.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.check("Dashboard") == "PASS"


@pytest.mark.parametrize(
    ("error", "status", "category"),
    [(Exception("unauthorized"), 401, "authentication"), (Exception("rate limited"), 429, "rate_limit"), (TimeoutError("gateway timeout"), None, "timeout")],
)
def test_ccswitch_provider_normalizes_gateway_failures_and_redacts_token(error, status, category):
    if status is not None:
        error.status_code = status

    class Responses:
        def create(self, **kwargs):
            raise error

    provider = CCSwitchProvider(
        base_url="http://127.0.0.1:8787/v1",
        api_key="local-secret",
        client=SimpleNamespace(responses=Responses()),
    )

    with pytest.raises(CCSwitchError) as raised:
        provider.create_response(model="sol")

    assert raised.value.category == category
    assert raised.value.status_code == status
    assert "local-secret" not in str(raised.value)


def test_ccswitch_health_check_rejects_invalid_structured_output():
    class Responses:
        def __init__(self):
            self.responses = iter([
                _response("not-json"),
                _response('{"ok": true}'),
                _response(calls=[SimpleNamespace(type="function_call", name="probe", arguments='{}', call_id="probe-1")]),
                _response("done"),
            ])

        def create(self, **kwargs):
            return next(self.responses)

    result = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        luna_model="luna",
        sol_model="sol",
        client=SimpleNamespace(responses=Responses()),
    ).health_check()

    assert result["ok"] is False
    assert result["structured_output"] == "FAIL"


def test_pipeline_records_structured_output_fallback_after_limited_repair():
    provider = ScriptedProvider([
        _response("not-json"),
        _response(_screening(("NVDA",))),
        _response(_decision(symbol="NVDA")),
    ])
    runtime = LLMRuntimeConfig(provider="ccswitch", base_url="http://gateway/v1", api_key="local", luna_model="luna-id", sol_model="sol-id")
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    pipeline.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    assert pipeline.last_pipeline_metadata["structured_output_status"] == "FALLBACK"
