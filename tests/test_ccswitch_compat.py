import json
from types import SimpleNamespace

import pytest

from src.data_provider import MockDataProvider
from src.llm_agent import (
    CCSwitchError,
    CCSwitchProvider,
    LLMProvider,
    LunaScreeningAgent,
)


def _response(text="", calls=None, response_id="response-1", usage=None):
    return SimpleNamespace(id=response_id, output=calls or [], output_text=text, usage=usage)


def _screening(symbols=("NVDA",)):
    return {
        "candidates": [
            {
                "symbol": symbol,
                "rationale": f"Screened {symbol}",
                "positive_signals": ["trend"],
                "risks": ["volatility"],
            }
            for symbol in symbols
        ],
        "screening_rationale": "Compact universe screen only",
    }


def _decision_cash():
    return {
        "action": "CASH",
        "symbol": None,
        "target_weight": 0.0,
        "confidence": 0.8,
        "holding_period_days": 20,
        "expected_excess_vs_spy": 0.0,
        "expected_excess_vs_qqq": 0.0,
        "thesis": ["No edge"],
        "risk_factors": ["Uncertainty"],
        "invalidation_conditions": ["New evidence"],
        "evidence_used": ["probe"],
        "current_symbol": None,
        "new_symbol": None,
    }


class _ScriptedProvider(LLMProvider):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def test_luna_missing_positive_signals_is_repaired_with_validation_context():
    incomplete = {"candidates": [{"symbol": "NVDA", "rationale": "Trend", "risks": []}], "screening_rationale": "Screen"}
    provider = _ScriptedProvider([_response(json.dumps(incomplete)), _response(json.dumps(_screening()))])
    agent = LunaScreeningAgent("luna", MockDataProvider(), provider=provider)

    result = agent.screen(MockDataProvider().universe_snapshot())

    assert result.candidate_symbols == ["NVDA"]
    assert agent.structured_output_status == "FALLBACK"
    assert len(provider.calls) == 2
    repair_prompt = provider.calls[1]["input"]
    assert "positive_signals" in repair_prompt
    assert "VALIDATION_ERROR" in repair_prompt


def test_luna_repair_still_invalid_fails_closed():
    incomplete = {"candidates": [{"symbol": "NVDA", "rationale": "Trend", "risks": []}], "screening_rationale": "Screen"}
    provider = _ScriptedProvider([_response(json.dumps(incomplete)), _response(json.dumps(incomplete))])
    agent = LunaScreeningAgent("luna", MockDataProvider(), provider=provider)

    with pytest.raises(ValueError, match="execution is blocked"):
        agent.screen(MockDataProvider().universe_snapshot())

    assert agent.structured_output_status == "FAIL"


def test_luna_schema_requires_nested_candidate_properties_for_strict_gateways():
    schema = LunaScreeningAgent.output_schema()["schema"]
    candidate_schema = schema["$defs"]["ScreeningCandidate"]

    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])


class _502Error(RuntimeError):
    status_code = 502


def test_ccswitch_retries_upstream_502_then_succeeds():
    class Responses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _502Error("upstream unavailable")
            return _response("ok")

    delays = []
    responses = Responses()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=responses),
        sleep_fn=delays.append,
    )

    result = provider.create_response(model="sol", input="ping")

    assert result.output_text == "ok"
    assert responses.calls == 2
    assert delays == [2.0]


def test_ccswitch_retries_timeout_then_succeeds():
    class Responses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("read timed out")
            return _response("ok")

    delays = []
    responses = Responses()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=responses),
        retry_delays=(2, 8),
        sleep_fn=delays.append,
    )

    assert provider.create_response(model="luna", input="batch").output_text == "ok"
    assert responses.calls == 2
    assert delays == [2.0]


def test_ccswitch_429_honors_numeric_retry_after():
    class RateLimitError(RuntimeError):
        status_code = 429

        def __init__(self):
            super().__init__("rate limited")
            self.response = SimpleNamespace(headers={"retry-after": "5"})

    class Responses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError()
            return _response("ok")

    delays = []
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=Responses()),
        retry_delays=(2, 8),
        sleep_fn=delays.append,
    )

    assert provider.create_response(model="luna", input="batch").output_text == "ok"
    assert delays == [5.0]


def test_ccswitch_repeated_502_is_reported_without_infinite_retry():
    class Responses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise _502Error("upstream unavailable")

    responses = Responses()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=responses),
        sleep_fn=lambda _delay: None,
    )

    with pytest.raises(CCSwitchError) as raised:
        provider.create_response(model="sol", input="ping")

    assert raised.value.status_code == 502
    assert raised.value.category == "upstream"
    assert responses.calls == 3


def test_responses_502_falls_back_to_chat_completions_and_emits_event():
    class Responses:
        def create(self, **kwargs):
            raise _502Error("responses upstream unavailable")

    class ChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="chat-1",
                choices=[SimpleNamespace(message=SimpleNamespace(content="chat ok", tool_calls=None))],
                usage=None,
            )

    events = []
    chat = ChatCompletions()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=Responses(), chat=SimpleNamespace(completions=chat)),
        sleep_fn=lambda _delay: None,
        event_sink=events.append,
    )

    result = provider.create_response(model="sol", input="ping")

    assert result.output_text == "chat ok"
    assert chat.calls[0]["messages"][-1] == {"role": "user", "content": "ping"}
    assert provider.protocol == "CHAT_COMPLETIONS"
    assert provider.protocol_fallback is True
    assert any(event["event_type"] == "API_PROTOCOL_FALLBACK" for event in events)


def test_created_provider_uses_v1_chat_route_after_responses_fallback():
    class Responses:
        def create(self, **kwargs):
            raise _502Error("responses unavailable")

    class ChatCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                id="chat-v1",
                choices=[SimpleNamespace(message=SimpleNamespace(content="123", tool_calls=None))],
                usage=None,
            )

    created = []

    def factory(*, base_url, api_key, timeout, max_retries):
        created.append((base_url, max_retries))
        if base_url.endswith("/v1"):
            return SimpleNamespace(chat=SimpleNamespace(completions=ChatCompletions()))
        return SimpleNamespace(responses=Responses())

    provider = CCSwitchProvider(
        base_url="http://gateway",
        api_key="local",
        client_factory=factory,
        sleep_fn=lambda _delay: None,
    )

    response = provider.create_response(model="sol", input='Return exactly "123".')

    assert response.output_text == "123"
    assert created == [("http://gateway", 0), ("http://gateway/v1", 0)]


def test_chat_timeout_does_not_fall_back_to_unversioned_html_route():
    class TimedOutChat:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise TimeoutError("read timed out")

    class UnversionedChat:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return "<html>Sub2API dashboard</html>"

    timed_out = TimedOutChat()
    unversioned = UnversionedChat()

    def factory(*, base_url, api_key, timeout, max_retries):
        if base_url.endswith("/v1"):
            return SimpleNamespace(chat=SimpleNamespace(completions=timed_out))
        return SimpleNamespace(chat=SimpleNamespace(completions=unversioned))

    provider = CCSwitchProvider(
        base_url="http://gateway",
        api_key="local",
        client_factory=factory,
        api_protocol="CHAT_COMPLETIONS",
        retry_delays=(),
    )

    with pytest.raises(CCSwitchError) as raised:
        provider.create_response(model="luna", input="small batch")

    assert raised.value.category == "timeout"
    assert timed_out.calls == 1
    assert unversioned.calls == 0


def test_chat_tool_calling_supports_multi_turn_continuation():
    class Responses:
        def create(self, **kwargs):
            raise _502Error("responses unavailable")

    class ChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(name="probe", arguments='{"value":"ok"}'),
                        )
                    ],
                )
            else:
                message = SimpleNamespace(content="done", tool_calls=None)
            return SimpleNamespace(id=f"chat-{len(self.calls)}", choices=[SimpleNamespace(message=message)], usage=None)

    chat = ChatCompletions()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        client=SimpleNamespace(responses=Responses(), chat=SimpleNamespace(completions=chat)),
        sleep_fn=lambda _delay: None,
    )
    tools = [{"type": "function", "name": "probe", "description": "Probe", "parameters": {"type": "object"}}]

    first = provider.create_response(model="sol", input="Call probe", tools=tools)
    second = provider.create_response(
        model="sol",
        previous_response_id=first.id,
        input=[{"type": "function_call_output", "call_id": "call-1", "output": '{"value":"ok"}'}],
        tools=tools,
    )

    assert first.output[0].type == "function_call"
    assert first.output[0].name == "probe"
    assert second.output_text == "done"
    assert any(message.get("role") == "tool" for message in chat.calls[1]["messages"])


def test_chat_tool_calling_preserves_outputs_across_multiple_rounds():
    class ChatCompletions:
        def __init__(self): self.calls = []
        def create(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index < 3:
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id=f"call-{index}", type="function",
                        function=SimpleNamespace(name="probe", arguments='{"value":"ok"}'),
                    )],
                )
            else:
                message = SimpleNamespace(content="done", tool_calls=None)
            return SimpleNamespace(id=f"chat-{index}", choices=[SimpleNamespace(message=message)], usage=None)

    chat = ChatCompletions()
    provider = CCSwitchProvider(
        base_url="http://gateway/v1", api_key="local",
        client=SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        api_protocol="CHAT_COMPLETIONS", sleep_fn=lambda _delay: None,
    )
    tools = [{"type": "function", "name": "probe", "description": "Probe", "parameters": {"type": "object"}}]

    first = provider.create_response(model="sol", input="Call probe", tools=tools)
    second = provider.create_response(
        model="sol", previous_response_id=first.id,
        input=[{"type": "function_call_output", "call_id": "call-1", "output": '{"value":"one"}'}], tools=tools,
    )
    third = provider.create_response(
        model="sol", previous_response_id=second.id,
        input=[{"type": "function_call_output", "call_id": "call-2", "output": '{"value":"two"}'}], tools=tools,
    )

    tool_ids = [message.get("tool_call_id") for message in chat.calls[2]["messages"] if message.get("role") == "tool"]
    assert tool_ids == ["call-1", "call-2"]
    assert third.output_text == "done"


def test_health_check_keeps_gateway_pass_when_capabilities_fail():
    class Models:
        def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="gpt-5.6-luna"), SimpleNamespace(id="gpt-5.6-sol")])

    class Responses:
        def create(self, **kwargs):
            return _response("not valid for the requested capability")

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        luna_model="gpt-5.6-luna",
        sol_model="gpt-5.6-sol",
        client=SimpleNamespace(models=Models(), responses=Responses()),
    )

    result = provider.health_check()

    assert result["endpoint"] == "PASS"
    assert result["gateway"] == "PASS"
    assert result["model_discovery"] == "PASS"
    assert result["ok"] is False


def test_health_check_prefers_role_specific_ids_discovered_from_models():
    class Models:
        def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="provider-luna"), SimpleNamespace(id="provider-sol")])

    tool_call = SimpleNamespace(type="function_call", name="gateway_health_probe", arguments='{"value":"ok"}', call_id="probe-1")

    class Responses:
        def __init__(self):
            self.outputs = iter([
                _response("123"),
                _response(json.dumps(_screening())),
                _response("123"),
                _response('{"ok":true}'),
                _response(calls=[tool_call], response_id="tool-1"),
                _response("done", response_id="tool-2"),
                _response(json.dumps(_decision_cash()), response_id="decision-1"),
            ])

        def create(self, **kwargs):
            return next(self.outputs)

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        luna_model="gpt-5.6-luna",
        sol_model="gpt-5.6-sol",
        client=SimpleNamespace(models=Models(), responses=Responses()),
    )

    result = provider.health_check()

    assert result["model_discovery"] == "PASS"
    assert result["luna_model"] == "provider-luna"
    assert result["sol_model"] == "provider-sol"
    assert result["ok"] is True


def test_health_check_treats_missing_reasoning_and_usage_as_nonfatal():
    tool_call = SimpleNamespace(type="function_call", name="gateway_health_probe", arguments='{"value":"ok"}', call_id="probe-1")

    class Models:
        def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="gpt-5.6-luna"), SimpleNamespace(id="gpt-5.6-sol")])

    class Responses:
        def __init__(self):
            self.outputs = iter([
                _response("123"),
                _response(json.dumps(_screening())),
                _response("123"),
                _response('{"ok":true}'),
                _response(calls=[tool_call], response_id="tool-1"),
                _response("done", response_id="tool-2"),
                _response(json.dumps(_decision_cash()), response_id="decision-1"),
            ])

        def create(self, **kwargs):
            return next(self.outputs)

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="local",
        luna_model="gpt-5.6-luna",
        sol_model="gpt-5.6-sol",
        client=SimpleNamespace(models=Models(), responses=Responses()),
    )

    result = provider.health_check()

    assert result["ok"] is True
    assert result["reasoning_metadata"] == "UNAVAILABLE"
    assert result["token_usage"] == "UNAVAILABLE"
    assert result["luna_structured_output"] == "PASS"
    assert result["sol_tool_calling"] == "PASS"
    assert result["sol_decision"] == "PASS"
