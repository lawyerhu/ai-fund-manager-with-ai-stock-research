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
    _compact_universe,
)
from src.main import build_llm_agent
from src.runner import StrategyRunner
from src.service import BackendService
from src.setup_wizard import persist_detected_llm_config
from src.storage import SQLiteStore


def _response(text="", response_id="response-1"):
    return SimpleNamespace(id=response_id, output=[], output_text=text, usage=None)


def _screening(symbol="NVDA"):
    return {
        "candidates": [{
            "symbol": symbol,
            "rationale": "Compact probe candidate",
            "positive_signals": ["trend"],
            "risks": ["volatility"],
        }],
        "screening_rationale": "Runtime path probe",
    }


class _Provider(LLMProvider):
    gateway = "ccswitch"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class _HTTPError(RuntimeError):
    status_code = 403

    def __init__(self, message="Forbidden"):
        super().__init__(message)
        self.body = {
            "error": {
                "type": "permission_error",
                "code": "model_not_allowed",
                "message": "model is not permitted",
            }
        }


def test_runtime_config_reads_setup_protocol_and_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ccswitch")
    monkeypatch.setenv("LLM_BASE_URL", "http://gateway")
    monkeypatch.setenv("LLM_API_KEY", "runtime-secret")
    monkeypatch.setenv("LLM_API_PROTOCOL", "CHAT_COMPLETIONS")
    monkeypatch.setenv("LLM_PROTOCOL_FALLBACK", "YES")

    runtime = LLMRuntimeConfig.from_mapping({"llm": {}})
    provider = CCSwitchProvider(runtime=runtime, client=SimpleNamespace())

    assert runtime.api_protocol == "CHAT_COMPLETIONS"
    assert runtime.protocol_fallback is True
    assert provider.protocol == "CHAT_COMPLETIONS"
    assert provider.protocol_fallback is True


def test_chat_runtime_request_metadata_uses_v1_adapter_path():
    runtime = LLMRuntimeConfig(
        base_url="http://gateway",
        api_key="runtime-secret",
        api_protocol="CHAT_COMPLETIONS",
        protocol_fallback=True,
    )
    provider = CCSwitchProvider(runtime=runtime, client=SimpleNamespace())

    metadata = provider.request_metadata(model="detected-luna", request_id="request-1")

    assert metadata["endpoint_path"] == "/v1/chat/completions"
    assert "runtime-secret" not in str(metadata)


def test_setup_persists_detected_protocol_without_replacing_secret(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_BASE_URL=http://gateway\n"
        "LLM_API_KEY=existing-secret\n"
        "LLM_LUNA_MODEL=old-luna\n"
        "LLM_SOL_MODEL=old-sol\n",
        encoding="utf-8",
    )

    persist_detected_llm_config(
        env_path,
        luna_model="detected-luna",
        sol_model="detected-sol",
        api_protocol="CHAT_COMPLETIONS",
        protocol_fallback=True,
    )

    content = env_path.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=ccswitch" in content
    assert "LLM_LUNA_MODEL=detected-luna" in content
    assert "LLM_SOL_MODEL=detected-sol" in content
    assert "LLM_PIPELINE=LUNA_SOL" in content
    assert "LLM_API_PROTOCOL=CHAT_COMPLETIONS" in content
    assert "LLM_PROTOCOL_FALLBACK=YES" in content
    assert "LLM_API_KEY=existing-secret" in content


def test_runtime_builder_uses_detected_models_and_protocol(monkeypatch):
    import src.llm_agent as llm_module

    monkeypatch.setenv("LLM_PROVIDER", "ccswitch")
    monkeypatch.setenv("LLM_BASE_URL", "http://gateway")
    monkeypatch.setenv("LLM_API_KEY", "runtime-secret")
    monkeypatch.setenv("LLM_LUNA_MODEL", "detected-luna")
    monkeypatch.setenv("LLM_SOL_MODEL", "detected-sol")
    monkeypatch.setenv("LLM_API_PROTOCOL", "CHAT_COMPLETIONS")
    monkeypatch.setenv("LLM_PROTOCOL_FALLBACK", "YES")

    created = []

    class SpyProvider:
        gateway = "ccswitch"

        def __init__(self, *, runtime):
            created.append(runtime)

    monkeypatch.setattr(llm_module, "CCSwitchProvider", SpyProvider)
    pipeline = build_llm_agent({"llm": {}, "agent": {}}, MockDataProvider())

    assert pipeline.runtime.luna_model == "detected-luna"
    assert pipeline.runtime.sol_model == "detected-sol"
    assert pipeline.runtime.api_protocol == "CHAT_COMPLETIONS"
    assert created[0] is pipeline.runtime


def test_responses_403_uses_one_verified_chat_adapter_when_available():
    class Responses:
        def create(self, **kwargs):
            raise _HTTPError()

    class ChatCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                id="chat-1",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                usage=None,
            )

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="runtime-secret",
        client=SimpleNamespace(
            responses=Responses(),
            chat=SimpleNamespace(completions=ChatCompletions()),
        ),
        sleep_fn=lambda _delay: None,
    )

    response = provider.create_response(model="luna", input="ping")

    assert response.output_text == "ok"
    assert provider.protocol == "CHAT_COMPLETIONS"
    assert provider.protocol_fallback is True


def test_403_error_preserves_safe_provider_diagnostics_without_secret():
    class Responses:
        def create(self, **kwargs):
            raise _HTTPError()

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="runtime-secret",
        client=SimpleNamespace(responses=Responses()),
    )

    with pytest.raises(CCSwitchError) as raised:
        provider.create_response(model="detected-luna", input="ping")

    error = raised.value
    assert error.status_code == 403
    assert error.category == "authentication"
    assert error.provider_error_type == "permission_error"
    assert error.provider_error_code == "model_not_allowed"
    assert error.safe_response_message == "model is not permitted"
    assert error.retryable is False
    assert "runtime-secret" not in str(error)


def test_luna_request_events_identify_runtime_stage_and_redact_auth():
    events = []

    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise _HTTPError()

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="runtime-secret",
        luna_model="detected-luna",
        sol_model="detected-sol",
        client=SimpleNamespace(responses=Responses()),
    )
    agent = LunaScreeningAgent(
        "detected-luna",
        MockDataProvider(),
        provider=provider,
        event_sink=events.append,
    )

    with pytest.raises(CCSwitchError):
        agent.screen([{"symbol": "NVDA"}])

    started = next(event for event in events if event["event_type"] == "LUNA_SCREEN_REQUEST_STARTED")
    failed = next(event for event in events if event["event_type"] == "LUNA_SCREEN_REQUEST_FAILED")
    metadata = started["metadata"]
    assert metadata["provider"] == "ccswitch"
    assert metadata["endpoint_host"] == "gateway"
    assert metadata["endpoint_path"] == "/v1/responses"
    assert metadata["protocol"] == "RESPONSES"
    assert metadata["model_id"] == "detected-luna"
    assert metadata["auth_present"] is True
    assert metadata["ticker_count"] == 1
    assert metadata["batch_size"] == 1
    assert metadata["payload_bytes"] > 0
    assert metadata["estimated_input_tokens"] > 0
    assert metadata["reasoning_effort"] == "max"
    assert metadata["stream_enabled"] is False
    assert metadata["time_to_first_upstream_byte_ms"] is None
    assert "runtime-secret" not in str(events)
    assert failed["metadata"]["http_status"] == 403
    assert failed["metadata"]["provider_error_code"] == "model_not_allowed"
    assert failed["metadata"]["response_completed"] is False
    assert failed["metadata"]["total_latency_ms"] >= 0


def test_sol_runtime_403_fails_closed_with_stage_metadata():
    events = []

    class Responses:
        def create(self, **kwargs):
            raise _HTTPError()

    provider = CCSwitchProvider(
        base_url="http://gateway/v1",
        api_key="runtime-secret",
        luna_model="detected-luna",
        sol_model="detected-sol",
        client=SimpleNamespace(responses=Responses()),
    )
    from src.llm_agent import SolResearchCIOAgent

    agent = SolResearchCIOAgent(
        "detected-sol",
        MockDataProvider(),
        provider=provider,
        event_sink=events.append,
    )

    with pytest.raises(CCSwitchError):
        agent.decide(
            SimpleNamespace(
                equity=1000,
                peak_equity=1000,
                cash=1000,
                model_dump_json=lambda: "{}",
            ),
            ["NVDA"],
        )

    failed = next(event for event in events if event["event_type"] == "SOL_RESEARCH_REQUEST_FAILED")
    assert failed["metadata"]["http_status"] == 403
    assert failed["metadata"]["model_id"] == "detected-sol"
    assert "runtime-secret" not in str(events)


def test_runtime_exact_probe_uses_production_luna_agent_path():
    provider = _Provider(_response(json.dumps(_screening())))
    runtime = LLMRuntimeConfig(
        provider="ccswitch",
        base_url="http://gateway/v1",
        api_key="runtime-secret",
        luna_model="detected-luna",
        sol_model="detected-sol",
    )
    pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)

    result = pipeline.probe_runtime_llm_exact_path([{"symbol": "NVDA"}])

    assert result["status"] == "PASS"
    assert result["ok"] is True
    assert result["model"] == "detected-luna"
    assert provider.calls[0]["model"] == "detected-luna"
    assert provider.calls[0]["text"]["format"]["name"] == "luna_screening"


def test_compact_universe_omits_nulls_and_limits_float_precision():
    rows = _compact_universe([{"symbol": "NVDA", "price": 123.123456789, "forward_pe": None}])

    assert rows == [{"symbol": "NVDA", "price": 123.123457}]


def test_strategy_runner_exposes_runtime_exact_probe_through_production_agent(tmp_path):
    class Agent:
        def __init__(self): self.rows = None
        def probe_runtime_llm_exact_path(self, rows):
            self.rows = rows
            return {"status": "PASS", "ok": True}

    agent = Agent()
    with SQLiteStore(tmp_path / "probe.sqlite3") as store:
        runner = StrategyRunner({}, MockDataProvider(), store=store, agent=agent)
        result = runner.probe_runtime_llm_exact_path()

    assert result == {"status": "PASS", "ok": True}
    assert agent.rows == [{"symbol": "SPY"}]


def test_runtime_exact_probe_failure_is_not_allowed_to_run_ai(tmp_path):
    class Runner:
        def __init__(self):
            self.probe_calls = 0
            self.run_calls = 0

        def probe_runtime_llm_exact_path(self):
            self.probe_calls += 1
            return {"status": "FAIL", "ok": False, "error": "HTTP 403"}

        def run(self, **kwargs):
            self.run_calls += 1
            raise AssertionError("production AI run must be gated")

    runner = Runner()
    with SQLiteStore(tmp_path / "runtime-gate.sqlite3") as store:
        service = BackendService({}, runner, store, use_llm=True)

        with pytest.raises(PipelineError, match="runtime exact probe"):
            service.run_ai_research_now()

        assert runner.probe_calls == 1
        assert runner.run_calls == 0
        assert store.get_runtime("llm_status") == "ERROR"
        assert store.get_runtime("trading_enabled") is False
        event = store.runtime_events(1)[0]
        assert event["event_type"] == "LLM_ERROR"
