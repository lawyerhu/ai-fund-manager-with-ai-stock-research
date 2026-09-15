from types import SimpleNamespace

import pytest

from src.data_provider import MockDataProvider
from src.llm_agent import LLMPortfolioManager, LLMProvider
from src.models import PortfolioState


class FakeProvider(LLMProvider):
    def __init__(self, output_text):
        self.output_text = output_text

    def create_response(self, **kwargs):
        return SimpleNamespace(id="response-1", output=[], output_text=self.output_text)


def test_all_required_tools_are_exposed():
    agent = LLMPortfolioManager("test-model", MockDataProvider(), provider=FakeProvider("{}"))
    names = {tool["name"] for tool in agent.tools}
    assert names == {"get_market_regime", "get_universe", "get_stock_snapshot", "get_price_history", "get_fundamentals", "get_earnings", "get_analyst_revisions", "get_news", "get_sec_filings", "get_portfolio", "get_current_positions", "get_benchmark_data", "get_upcoming_events"}


def test_invalid_json_cannot_become_a_decision():
    agent = LLMPortfolioManager("test-model", MockDataProvider(), provider=FakeProvider("not-json"))
    with pytest.raises(ValueError, match="invalid TradeIntent"):
        agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))


def test_structured_output_schema_requires_all_llm_fields_and_excludes_metadata():
    agent = LLMPortfolioManager("test-model", MockDataProvider(), provider=FakeProvider("{}"))
    schema = agent._trade_schema()["schema"]

    assert set(schema["required"]) == set(schema["properties"])
    assert {"timestamp", "decision_id", "model_name"}.isdisjoint(schema["properties"])
    assert {item.get("type") for item in schema["properties"]["symbol"]["anyOf"]} == {"string", "null"}


def test_continuations_preserve_instructions_and_accumulate_usage():
    decision_json = """{
      "action":"CASH","symbol":null,"target_weight":0.0,"confidence":0.8,
      "holding_period_days":20,"expected_alpha_vs_spy":0.0,"expected_alpha_vs_qqq":0.0,
      "thesis":["No edge"],"risk_factors":["Uncertainty"],
      "invalidation_conditions":["New evidence"],"evidence_used":["get_market_regime"],
      "current_symbol":null,"new_symbol":null
    }"""

    class ToolProvider(LLMProvider):
        def __init__(self):
            self.calls = []

        def create_response(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                tool_call = SimpleNamespace(type="function_call", name="get_market_regime", arguments="{}", call_id="call-1")
                usage = SimpleNamespace(input_tokens=10, output_tokens=2, input_tokens_details=SimpleNamespace(cached_tokens=3), output_tokens_details=SimpleNamespace(reasoning_tokens=1))
                return SimpleNamespace(id="response-1", output=[tool_call], output_text="", usage=usage)
            usage = SimpleNamespace(input_tokens=20, output_tokens=4, input_tokens_details=SimpleNamespace(cached_tokens=5), output_tokens_details=SimpleNamespace(reasoning_tokens=2))
            return SimpleNamespace(id="response-2", output=[], output_text=decision_json, usage=usage)

    provider = ToolProvider()
    agent = LLMPortfolioManager("test-model", MockDataProvider(), provider=provider)
    result = agent.decide(PortfolioState(equity=1000, peak_equity=1000, cash=1000))

    assert result.model_name == "test-model"
    assert result.decision_id
    assert all(call["instructions"] == agent.instructions for call in provider.calls)
    assert agent.last_usage["input_tokens"] == 30
    assert agent.last_usage["output_tokens"] == 6
    assert agent.last_usage["cached_tokens"] == 8
    assert agent.last_usage["reasoning_tokens"] == 3
