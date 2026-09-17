from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import env
from .data_provider import DataProvider
from .models import LLMDecision, ManagedPosition, PortfolioState, PositionReview, SizingAudit, TradeIntent
from .decision_audit import evidence_changes

ROOT = Path(__file__).resolve().parents[1]


class LLMProvider(ABC):
    """External LLM gateway boundary."""

    @abstractmethod
    def create_response(self, **kwargs):
        raise NotImplementedError


class CCSwitchError(RuntimeError):
    """Normalized error from the configured CC Switch gateway."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        category: str = "gateway",
        cause: Exception | None = None,
        provider_error_type: str | None = None,
        provider_error_code: str | int | None = None,
        safe_response_message: str | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.cause = cause
        self.provider_error_type = provider_error_type
        self.provider_error_code = provider_error_code
        self.safe_response_message = safe_response_message or str(message)
        self.retryable = bool(retryable) if retryable is not None else status_code in {429, 500, 502, 503, 504}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "http_status": self.status_code,
            "status_code": self.status_code,
            "provider_error_type": self.provider_error_type,
            "provider_error_code": self.provider_error_code,
            "safe_response_message": self.safe_response_message,
            "retryable": self.retryable,
            "category": self.category,
        }


LLMGatewayError = CCSwitchError


class ScreeningIncompleteError(RuntimeError):
    """Raised when an adaptive Luna batch cannot complete at minimum size."""


def _env_or(mapping: dict[str, Any], env_name: str, key: str, default: Any = None, legacy_name: str | None = None) -> Any:
    value = env(env_name)
    if value is not None and value != "":
        return value
    value = mapping.get(key, default)
    if value not in (None, ""):
        return value
    if legacy_name:
        return env(legacy_name, default)
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _fraction(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1") from exc
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return result


_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "ultra", "max"})
_LLM_PROTOCOLS = frozenset({"RESPONSES", "CHAT_COMPLETIONS"})


def _select_discovered_model(configured: str, models: list[str], role: str) -> str:
    if configured in models:
        return configured
    casefolded = {model.casefold(): model for model in models}
    if configured.casefold() in casefolded:
        return casefolded[configured.casefold()]
    role_matches = [model for model in models if role in model.casefold()]
    return role_matches[0] if len(role_matches) == 1 else configured


def _reasoning_effort(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"{name} must be one of: {allowed}")
    return normalized


def _api_protocol(value: Any, name: str = "LLM_API_PROTOCOL") -> str:
    normalized = str(value or "RESPONSES").strip().upper().replace("-", "_").replace(" ", "_")
    if normalized not in _LLM_PROTOCOLS:
        allowed = ", ".join(sorted(_LLM_PROTOCOLS))
        raise ValueError(f"{name} must be one of: {allowed}")
    return normalized


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str = "ccswitch"
    base_url: str = ""
    api_key: str = ""
    luna_model: str = "gpt-5.6-luna"
    sol_model: str = "gpt-5.6-sol"
    luna_reasoning_effort: str = "max"
    sol_reasoning_effort: str = "medium"
    pipeline: str = "LUNA_SOL"
    timeout_seconds: float = 300.0
    fallback_to_sol_only: bool = False
    api_protocol: str = "RESPONSES"
    protocol_fallback: bool = False
    luna_screen_batch_size: int = 60
    luna_screen_batch_top_k: int = 6
    luna_min_batch_size: int = 15
    luna_final_top_k: int = 20
    luna_final_max_candidates: int = 24
    deep_research_enabled: bool = False
    deep_research_top_five: int = 5
    deep_research_top_three: int = 3
    force_stock_selection: bool = False
    forced_selection_high_weight: float = 1.0
    forced_selection_data_limited_weight: float = 0.70
    forced_selection_exploratory_weight: float = 0.35

    @classmethod
    def from_env(cls, cfg: dict[str, Any] | None = None) -> "LLMRuntimeConfig":
        return cls.from_mapping(cfg or {})

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None = None) -> "LLMRuntimeConfig":
        mapping = (cfg or {}).get("llm", {}) or {}
        provider = str(_env_or(mapping, "LLM_PROVIDER", "provider", "ccswitch")).strip().lower().replace("_", "")
        if provider != "ccswitch":
            raise ValueError("LLM_PROVIDER must be ccswitch")
        pipeline = str(_env_or(mapping, "LLM_PIPELINE", "pipeline", "LUNA_SOL")).strip().upper()
        if pipeline not in {"LUNA_SOL", "SOL_ONLY"}:
            raise ValueError("LLM_PIPELINE must be LUNA_SOL or SOL_ONLY")
        try:
            timeout = float(_env_or(mapping, "LLM_TIMEOUT_SECONDS", "timeout_seconds", 300.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number")
        return cls(
            provider="ccswitch",
            base_url=str(_env_or(mapping, "LLM_BASE_URL", "base_url", "") or "").strip(),
            api_key=str(_env_or(mapping, "LLM_API_KEY", "api_key", "", "OPENAI_API_KEY") or ""),
            luna_model=str(_env_or(mapping, "LLM_LUNA_MODEL", "luna_model", "gpt-5.6-luna", "OPENAI_MODEL") or "gpt-5.6-luna"),
            sol_model=str(_env_or(mapping, "LLM_SOL_MODEL", "sol_model", "gpt-5.6-sol", "OPENAI_MODEL") or "gpt-5.6-sol"),
            luna_reasoning_effort=_reasoning_effort(
                _env_or(mapping, "LLM_LUNA_REASONING_EFFORT", "luna_reasoning_effort", "max"),
                "LLM_LUNA_REASONING_EFFORT",
            ),
            sol_reasoning_effort=_reasoning_effort(
                _env_or(mapping, "LLM_SOL_REASONING_EFFORT", "sol_reasoning_effort", "medium"),
                "LLM_SOL_REASONING_EFFORT",
            ),
            pipeline=pipeline,
            timeout_seconds=timeout,
            fallback_to_sol_only=_as_bool(
                _env_or(mapping, "LLM_FALLBACK_TO_SOL_ONLY", "fallback_to_sol_only", False),
            ),
            api_protocol=_api_protocol(_env_or(mapping, "LLM_API_PROTOCOL", "api_protocol", "RESPONSES")),
            protocol_fallback=_as_bool(
                _env_or(mapping, "LLM_PROTOCOL_FALLBACK", "protocol_fallback", False),
            ),
            luna_screen_batch_size=int(_env_or(mapping, "LUNA_SCREEN_BATCH_SIZE", "screen_batch_size", 60)),
            luna_screen_batch_top_k=int(_env_or(mapping, "LUNA_SCREEN_BATCH_TOP_K", "screen_batch_top_k", 6)),
            luna_min_batch_size=int(_env_or(mapping, "LUNA_MIN_BATCH_SIZE", "min_batch_size", 15)),
            luna_final_top_k=int(_env_or(mapping, "LUNA_FINAL_TOP_K", "final_top_k", 20)),
            luna_final_max_candidates=int(
                _env_or(mapping, "LUNA_FINAL_MAX_CANDIDATES", "final_max_candidates", 24)
            ),
            deep_research_enabled=_as_bool(
                _env_or(mapping, "SOL_DEEP_RESEARCH_ENABLED", "deep_research_enabled", False)
            ),
            deep_research_top_five=int(
                _env_or(mapping, "SOL_DEEP_RESEARCH_TOP_FIVE", "deep_research_top_five", 5)
            ),
            deep_research_top_three=int(
                _env_or(mapping, "SOL_DEEP_RESEARCH_TOP_THREE", "deep_research_top_three", 3)
            ),
            force_stock_selection=_as_bool(
                _env_or(mapping, "SOL_FORCE_STOCK_SELECTION", "force_stock_selection", False)
            ),
            forced_selection_high_weight=_fraction(
                _env_or(mapping, "SOL_FORCED_SELECTION_HIGH_WEIGHT", "forced_selection_high_weight", 1.0),
                "SOL_FORCED_SELECTION_HIGH_WEIGHT",
            ),
            forced_selection_data_limited_weight=_fraction(
                _env_or(mapping, "SOL_FORCED_SELECTION_DATA_LIMITED_WEIGHT", "forced_selection_data_limited_weight", 0.70),
                "SOL_FORCED_SELECTION_DATA_LIMITED_WEIGHT",
            ),
            forced_selection_exploratory_weight=_fraction(
                _env_or(mapping, "SOL_FORCED_SELECTION_EXPLORATORY_WEIGHT", "forced_selection_exploratory_weight", 0.35),
                "SOL_FORCED_SELECTION_EXPLORATORY_WEIGHT",
            ),
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _emit_runtime_event(
    event_sink: Callable[[dict[str, Any]], None] | None,
    event_type: str,
    component: str,
    message: str,
    *,
    symbol: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    if event_sink is not None:
        event_sink({
            "event_type": event_type,
            "component": component,
            "message": message,
            "symbol": symbol,
            "metadata": metadata or {},
        })


def _provider_request_metadata(provider: Any, model: str, request_id: str) -> dict[str, Any] | None:
    metadata = getattr(provider, "request_metadata", None)
    if not callable(metadata):
        return None
    value = metadata(model=model, request_id=request_id)
    return value if isinstance(value, dict) else None


def _provider_error_metadata(provider: Any, exc: Exception) -> dict[str, Any]:
    diagnostics = getattr(provider, "error_diagnostics", None)
    if callable(diagnostics):
        value = diagnostics(exc)
        if isinstance(value, dict):
            return value
    return {
        "http_status": _status_code(exc),
        "status_code": _status_code(exc),
        "provider_error_type": getattr(exc, "provider_error_type", None),
        "provider_error_code": getattr(exc, "provider_error_code", None),
        "safe_response_message": str(exc) or exc.__class__.__name__,
        "retryable": bool(getattr(exc, "retryable", False)),
    }


def _request_with_runtime_events(
    provider: Any,
    model: str,
    event_sink: Callable[[dict[str, Any]], None] | None,
    *,
    component: str,
    started_event: str,
    failed_event: str,
    diagnostic_metadata: dict[str, Any] | None = None,
    **kwargs,
):
    request_id = str(uuid4())
    request_metadata = _provider_request_metadata(provider, model, request_id)
    if request_metadata is None:
        return provider.create_response(model=model, **kwargs)
    payload_bytes = len(json.dumps(kwargs, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8"))
    reasoning = kwargs.get("reasoning") or {}
    reasoning_effort = _field(reasoning, "effort")
    if reasoning_effort is None:
        reasoning_effort = getattr(provider, "luna_reasoning_effort" if component == "LUNA" else "sol_reasoning_effort", None)
    request_metadata = {
        **request_metadata,
        **(diagnostic_metadata or {}),
        "payload_bytes": payload_bytes,
        "estimated_input_tokens": (payload_bytes + 3) // 4,
        "reasoning_effort": reasoning_effort,
        "stream_enabled": bool(kwargs.get("stream", False)),
        "request_start_time": datetime.now(timezone.utc).isoformat(),
        "time_to_first_upstream_byte_ms": None,
        "time_to_first_semantic_output_ms": None,
        "timing_telemetry": "UNAVAILABLE",
    }
    started = time.perf_counter()
    _emit_runtime_event(
        event_sink,
        started_event,
        component,
        f"{component} LLM request started",
        metadata=request_metadata,
    )
    try:
        response = provider.create_response(model=model, **kwargs)
        _emit_runtime_event(
            event_sink,
            started_event.replace("_STARTED", "_COMPLETED"),
            component,
            f"{component} LLM request completed",
            metadata={
                **request_metadata,
                "total_latency_ms": (time.perf_counter() - started) * 1000,
                "response_completed": True,
            },
        )
        return response
    except Exception as exc:
        failure_metadata = {**request_metadata, **_provider_error_metadata(provider, exc)}
        failure_metadata.update({
            "total_latency_ms": (time.perf_counter() - started) * 1000,
            "response_completed": False,
            "upstream_error_type": failure_metadata.get("category") or failure_metadata.get("provider_error_type"),
            "upstream_error_message": failure_metadata.get("safe_response_message"),
        })
        _emit_runtime_event(
            event_sink,
            failed_event,
            component,
            f"{component} LLM request failed",
            metadata=failure_metadata,
        )
        raise


def _response_text(response: Any) -> str:
    text = _field(response, "output_text", "")
    if text:
        return str(text)
    parts: list[str] = []
    for item in _field(response, "output", []) or []:
        if _field(item, "type") not in {"message", "output_text", "text"}:
            continue
        content = _field(item, "content", []) or []
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content:
            value = _field(part, "text")
            if value:
                parts.append(str(value))
    return "".join(parts)


def _function_calls(response: Any) -> list[Any]:
    return [item for item in (_field(response, "output", []) or []) if _field(item, "type") == "function_call"]


def _usage_value(usage: Any, name: str, default: int = 0) -> int:
    value = _field(usage, name, default)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _accumulate_usage(totals: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    totals["input_tokens"] += _usage_value(usage, "input_tokens")
    totals["output_tokens"] += _usage_value(usage, "output_tokens")
    input_details = _field(usage, "input_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    totals["cached_tokens"] += _usage_value(input_details, "cached_tokens")
    totals["reasoning_tokens"] += _usage_value(output_details, "reasoning_tokens")


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make Pydantic's nested schema acceptable to strict OpenAI-compatible gateways."""
    result = deepcopy(schema)

    def visit(node: Any):
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(exc, "status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class CCSwitchProvider(LLMProvider):
    """OpenAI-compatible CC Switch client with bounded protocol compatibility."""

    gateway = "ccswitch"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        luna_model: str | None = None,
        sol_model: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        runtime: LLMRuntimeConfig | None = None,
        client: Any | None = None,
        client_factory: Any | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        retry_delays: Iterable[float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        api_protocol: str | None = None,
        protocol_fallback: bool | str | None = None,
    ):
        runtime_was_explicit = runtime is not None
        runtime = runtime or LLMRuntimeConfig.from_env()
        self.base_url = str(base_url if base_url is not None else runtime.base_url).strip()
        if not self.base_url:
            raise ValueError("LLM_BASE_URL must point to the CC Switch gateway")
        self.api_key = api_key if api_key is not None else runtime.api_key
        self.luna_model = luna_model or runtime.luna_model
        self.sol_model = sol_model or runtime.sol_model or model
        self.luna_reasoning_effort = runtime.luna_reasoning_effort
        self.sol_reasoning_effort = runtime.sol_reasoning_effort
        self.model = model or self.sol_model
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else runtime.timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive")
        self.event_sink = event_sink
        self._retry_delays = tuple(
            max(0.0, float(delay))
            for delay in (retry_delays if retry_delays is not None else (2.0, 8.0))
        )
        self._sleep = sleep_fn or time.sleep
        configured_protocol = api_protocol if api_protocol is not None else runtime.api_protocol
        configured_fallback = protocol_fallback if protocol_fallback is not None else runtime.protocol_fallback
        # Injected clients are isolated test doubles unless their runtime is explicit.
        if client is not None and not runtime_was_explicit and api_protocol is None:
            configured_protocol = "RESPONSES"
            configured_fallback = False
        self.protocol = _api_protocol(configured_protocol)
        self.protocol_fallback = (
            _as_bool(configured_fallback)
        )
        self._protocol_fallback_attempted = self.protocol == "CHAT_COMPLETIONS" and self.protocol_fallback
        self.protocol_events: list[dict[str, Any]] = []
        self.reasoning_parameter_status = "REQUESTED"
        self.reasoning_metadata_status = "UNAVAILABLE"
        self.token_usage_status = "UNAVAILABLE"
        self._response_conversations: dict[str, list[dict[str, Any]]] = {}
        self._chat_conversations: dict[str, list[dict[str, Any]]] = {}
        self._client_was_injected = client is not None
        self._client_factory = None
        self._chat_client = None
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the openai compatibility client for CC Switch") from exc
            factory = client_factory or OpenAI
            self._client_factory = factory
            self.client = factory(
                base_url=self.base_url,
                api_key=self.api_key or "ccswitch-local",
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        self.input_cost_per_1k = self._rate("LLM_INPUT_COST_PER_1K")
        self.output_cost_per_1k = self._rate("LLM_OUTPUT_COST_PER_1K")

    @staticmethod
    def _rate(name: str) -> float:
        try:
            return float(os.getenv(name, "0") or 0)
        except ValueError:
            return 0.0

    def estimate_cost(self, input_tokens: int | None, output_tokens: int | None, model: str | None = None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        return (float(input_tokens or 0) / 1000 * self.input_cost_per_1k) + (float(output_tokens or 0) / 1000 * self.output_cost_per_1k)

    def set_event_sink(self, event_sink: Callable[[dict[str, Any]], None] | None):
        self.event_sink = event_sink

    def create_response(self, model: str | None = None, **kwargs):
        selected_model = model or kwargs.pop("model", None) or self.model
        if not selected_model:
            raise ValueError("An LLM model must be configured")
        if "reasoning" not in kwargs:
            effort = self.luna_reasoning_effort if selected_model == self.luna_model else self.sol_reasoning_effort
            kwargs["reasoning"] = {"effort": effort}
        if self.protocol == "CHAT_COMPLETIONS":
            return self._create_chat_response(selected_model, kwargs)
        try:
            return self._create_responses_response(selected_model, kwargs)
        except CCSwitchError as exc:
            if (
                not self._protocol_fallback_attempted
                and self._can_fallback_to_chat(exc)
                and self._has_chat_completions()
            ):
                self._protocol_fallback_attempted = True
                self.protocol = "CHAT_COMPLETIONS"
                self.protocol_fallback = True
                self._record_protocol_fallback(exc)
                return self._create_chat_response(selected_model, kwargs)
            raise

    def _create_responses_response(self, model: str, kwargs: dict[str, Any]):
        responses_api = getattr(self.client, "responses", None)
        create = getattr(responses_api, "create", None)
        if not callable(create):
            raise CCSwitchError("Responses API is unavailable", status_code=404, category="protocol")
        request = dict(kwargs)
        try:
            response = self._call_with_retries(lambda: create(model=model, **request), "Responses")
        except CCSwitchError as exc:
            if "reasoning" in request and self._reasoning_unsupported(exc):
                request.pop("reasoning", None)
                self.reasoning_parameter_status = "UNAVAILABLE"
                response = self._call_with_retries(lambda: create(model=model, **request), "Responses")
            else:
                raise
        self.reasoning_parameter_status = "ACCEPTED" if "reasoning" in request else self.reasoning_parameter_status
        self._record_telemetry(response)
        self._remember_response_context(response, request, self._response_conversations)
        return response

    def _create_chat_response(self, model: str, kwargs: dict[str, Any]):
        chat_apis = list(self._chat_api_candidates())
        if not chat_apis:
            raise CCSwitchError("Chat Completions API is unavailable", status_code=404, category="protocol")

        request = dict(kwargs)
        previous_response_id = request.pop("previous_response_id", None)
        instructions = request.pop("instructions", None)
        input_value = request.pop("input", None)
        response_text = request.pop("text", None)
        reasoning = request.pop("reasoning", None)
        if "messages" not in request:
            contexts = {**self._response_conversations, **self._chat_conversations}
            request["messages"] = self._chat_messages(instructions, input_value, previous_response_id, contexts)
        if "tools" in request:
            request["tools"] = self._chat_tools(request["tools"])
        if response_text is not None:
            request["response_format"] = self._chat_response_format(response_text)
        if reasoning:
            effort = _field(reasoning, "effort")
            if effort:
                request["reasoning_effort"] = effort

        last_error = None
        for chat_api in chat_apis:
            create = getattr(chat_api, "create", None)
            if not callable(create):
                continue
            try:
                try:
                    completion = self._call_with_retries(lambda: create(model=model, **request), "Chat Completions")
                except CCSwitchError as exc:
                    if "reasoning_effort" in request and self._reasoning_unsupported(exc):
                        request.pop("reasoning_effort", None)
                        self.reasoning_parameter_status = "UNAVAILABLE"
                        completion = self._call_with_retries(lambda: create(model=model, **request), "Chat Completions")
                    else:
                        raise
                if isinstance(completion, str):
                    raise CCSwitchError("Chat Completions returned a non-API response", status_code=502, category="protocol")
                if "reasoning_effort" in request:
                    self.reasoning_parameter_status = "ACCEPTED"
                response = self._normalise_chat_completion(completion)
                self._record_telemetry(response)
                self._remember_response_context(response, {"previous_response_id": previous_response_id, "messages": request["messages"]}, self._chat_conversations)
                return response
            except CCSwitchError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise CCSwitchError("Chat Completions API is unavailable", status_code=404, category="protocol")

    def _call_with_retries(self, operation: Callable[[], Any], operation_name: str):
        for retry_number in range(len(self._retry_delays) + 1):
            try:
                return operation()
            except Exception as raw_exc:
                exc = self._normalise_exception(raw_exc)
                transient = exc.status_code in {429, 500, 502, 503, 504} or exc.category == "timeout"
                if transient and retry_number < len(self._retry_delays):
                    delay = self._retry_delay(raw_exc, retry_number)
                    _emit_runtime_event(
                        self.event_sink,
                        "LLM_UPSTREAM_RETRY",
                        "LLM_GATEWAY",
                        f"{operation_name} retry scheduled",
                        metadata={
                            "operation": operation_name,
                            "retry_attempt": retry_number + 1,
                            "next_attempt": retry_number + 2,
                            "status_code": exc.status_code,
                            "category": exc.category,
                            "backoff_seconds": delay,
                        },
                    )
                    self._sleep(delay)
                    continue
                if transient:
                    raise CCSwitchError(
                        f"{operation_name} upstream unavailable after {retry_number + 1} attempt(s): {self._safe_error(exc)}",
                        status_code=exc.status_code,
                        category=exc.category,
                        cause=exc,
                        provider_error_type=exc.provider_error_type,
                        provider_error_code=exc.provider_error_code,
                        safe_response_message=exc.safe_response_message,
                        retryable=False,
                    ) from exc
                raise exc
        raise CCSwitchError(f"{operation_name} failed", category="gateway")

    def _retry_delay(self, raw_exc: Exception, retry_number: int) -> float:
        if _status_code(raw_exc) == 429:
            response = getattr(raw_exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                value = headers.get("retry-after") or headers.get("Retry-After")
                try:
                    return min(300.0, max(0.0, float(value)))
                except (TypeError, ValueError):
                    pass
        return self._retry_delays[retry_number]

    def _normalise_exception(self, exc: Exception) -> CCSwitchError:
        if isinstance(exc, CCSwitchError):
            return exc
        code = _status_code(exc)
        message = str(exc).lower()
        is_timeout = isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message
        category = (
            "authentication" if code in {401, 403}
            else "rate_limit" if code == 429
            else "timeout" if is_timeout
            else "upstream" if code in {502, 503, 504}
            else "gateway"
        )
        provider_error_type, provider_error_code, safe_response_message = self._provider_error_fields(exc)
        message = self._safe_error(exc)
        if safe_response_message and safe_response_message not in message:
            message = f"HTTP {code}: {safe_response_message}" if code is not None else safe_response_message
        return CCSwitchError(
            message,
            status_code=code,
            category=category,
            cause=exc,
            provider_error_type=provider_error_type,
            provider_error_code=provider_error_code,
            safe_response_message=safe_response_message or message,
            retryable=code in {429, 500, 502, 503, 504} or is_timeout,
        )

    @staticmethod
    def _reasoning_unsupported(exc: CCSwitchError) -> bool:
        message = str(exc).lower()
        return exc.status_code in {400, 404, 405, 422} and "reasoning" in message and any(
            marker in message for marker in ("unsupported", "unknown", "invalid", "unrecognized", "not allowed")
        )

    @staticmethod
    def _can_fallback_to_chat(exc: CCSwitchError) -> bool:
        message = str(exc).lower()
        if exc.status_code in {403, 404, 405, 415, 422, 502, 503, 504}:
            return True
        return any(marker in message for marker in ("responses api", "responses endpoint", "not implemented", "unsupported"))

    def _has_chat_completions(self) -> bool:
        return bool(list(self._chat_api_candidates()))

    def _chat_api_candidates(self):
        if self._client_was_injected:
            chat_api = getattr(getattr(self.client, "chat", None), "completions", None)
            if callable(getattr(chat_api, "create", None)):
                yield chat_api
            return
        chat_base_url = self._chat_base_url(self.base_url)
        if chat_base_url != self.base_url and self._client_factory is not None and self._chat_client is None:
            try:
                self._chat_client = self._client_factory(
                    base_url=chat_base_url,
                    api_key=self.api_key or "ccswitch-local",
                    timeout=self.timeout_seconds,
                    max_retries=0,
                )
            except Exception:
                self._chat_client = None
        clients = (self._chat_client,) if chat_base_url != self.base_url else (self.client,)
        for client in clients:
            chat_api = getattr(getattr(client, "chat", None), "completions", None)
            if callable(getattr(chat_api, "create", None)):
                yield chat_api

    @staticmethod
    def _chat_base_url(base_url: str) -> str:
        normalized = str(base_url or "").rstrip("/")
        return normalized if normalized.lower().endswith("/v1") else normalized + "/v1"

    def _record_protocol_fallback(self, error: CCSwitchError):
        event = {
            "event_type": "API_PROTOCOL_FALLBACK",
            "component": "LLM_GATEWAY",
            "message": "CC Switch Responses API failed; using Chat Completions compatibility adapter",
            "metadata": {
                "from": "RESPONSES",
                "to": "CHAT_COMPLETIONS",
                "status_code": error.status_code,
                "error": self._safe_error(error),
            },
        }
        self.protocol_events.append(event)
        _emit_runtime_event(self.event_sink, event["event_type"], event["component"], event["message"], metadata=event["metadata"])

    def _record_telemetry(self, response: Any):
        usage = _field(response, "usage")
        if usage is not None:
            self.token_usage_status = "AVAILABLE"
        reasoning = _field(response, "reasoning_effort")
        if reasoning is None:
            reasoning = _field(response, "reasoning")
        if reasoning is None:
            details = _field(usage, "output_tokens_details")
            reasoning = _field(details, "reasoning_tokens")
        if reasoning is not None:
            self.reasoning_metadata_status = "AVAILABLE"

    def request_metadata(self, *, model: str, request_id: str) -> dict[str, Any]:
        request_base_url = self._chat_base_url(self.base_url) if self.protocol == "CHAT_COMPLETIONS" else self.base_url
        parsed = urlsplit(request_base_url)
        base_path = parsed.path.rstrip("/")
        endpoint = "/chat/completions" if self.protocol == "CHAT_COMPLETIONS" else "/responses"
        return {
            "provider": self.gateway,
            "endpoint_host": parsed.netloc,
            "endpoint_path": f"{base_path}{endpoint}" or endpoint,
            "protocol": self.protocol,
            "model_id": model,
            "auth_present": bool(self.api_key),
            "auth_scheme": "Bearer" if self.api_key else "none",
            "request_id": request_id,
        }

    def error_diagnostics(self, exc: Exception) -> dict[str, Any]:
        normalized = self._normalise_exception(exc)
        return normalized.diagnostics()

    @staticmethod
    def _chat_response_format(text: dict[str, Any]) -> dict[str, Any]:
        formatted = _field(text, "format", text)
        if not isinstance(formatted, dict):
            return formatted
        if formatted.get("type") == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": formatted.get("name", "structured_response"),
                    "strict": formatted.get("strict", True),
                    "schema": formatted.get("schema", {}),
                },
            }
        return formatted

    @staticmethod
    def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools or []:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                converted.append(tool)
                continue
            function = {key: value for key, value in tool.items() if key not in {"type", "name", "description", "parameters", "strict"}}
            function.update({
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters", {}),
            })
            if "strict" in tool:
                function["strict"] = tool["strict"]
            converted.append({"type": "function", "function": function})
        return converted

    @staticmethod
    def _chat_messages(instructions: str | None, input_value: Any, previous_response_id: str | None, contexts: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
        if contexts is not None and previous_response_id and previous_response_id in contexts:
            messages = deepcopy(contexts[previous_response_id])
            return messages + CCSwitchProvider._input_messages(input_value)
        messages = [{"role": "system", "content": instructions}] if instructions else []
        return messages + CCSwitchProvider._input_messages(input_value)

    @staticmethod
    def _input_messages(input_value: Any) -> list[dict[str, Any]]:
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        messages: list[dict[str, Any]] = []
        for item in input_value if isinstance(input_value, list) else [input_value]:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            if item.get("type") == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": str(item.get("output", ""))})
            elif item.get("role"):
                messages.append({key: value for key, value in item.items() if key in {"role", "content", "name", "tool_call_id", "tool_calls"}})
            elif item.get("type") == "message":
                messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
            else:
                messages.append({"role": "user", "content": json.dumps(item, ensure_ascii=False, default=str)})
        return messages

    @staticmethod
    def _assistant_message(response: Any) -> dict[str, Any] | None:
        calls = _function_calls(response)
        content = _response_text(response)
        if not calls and not content:
            return None
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if calls:
            message["tool_calls"] = [
                {
                    "id": _field(call, "call_id"),
                    "type": "function",
                    "function": {
                        "name": _field(call, "name"),
                        "arguments": _field(call, "arguments", "{}"),
                    },
                }
                for call in calls
            ]
        return message

    def _remember_response_context(self, response: Any, request: dict[str, Any], contexts: dict[str, list[dict[str, Any]]]):
        response_id = _field(response, "id")
        if not response_id:
            return
        previous_response_id = request.get("previous_response_id")
        if isinstance(request.get("messages"), list):
            # The chat request already contains every prior tool result.
            messages = deepcopy(request["messages"])
        elif previous_response_id and previous_response_id in contexts:
            messages = deepcopy(contexts[previous_response_id])
            messages.extend(self._input_messages(request.get("input")))
        else:
            messages = self._chat_messages(request.get("instructions"), request.get("input"), None)
        assistant = self._assistant_message(response)
        if assistant:
            messages.append(assistant)
        contexts[str(response_id)] = messages

    @staticmethod
    def _normalise_chat_completion(completion: Any):
        choices = _field(completion, "choices", []) or []
        choice = choices[0] if choices else None
        message = _field(choice, "message", {}) or {}
        content = _field(message, "content", "")
        if isinstance(content, list):
            content = "".join(str(_field(part, "text", "")) for part in content)
        calls = []
        for tool_call in _field(message, "tool_calls", []) or []:
            function = _field(tool_call, "function", {}) or {}
            calls.append(SimpleNamespace(
                type="function_call",
                name=_field(function, "name", ""),
                arguments=_field(function, "arguments", "{}"),
                call_id=_field(tool_call, "id"),
            ))
        return SimpleNamespace(
            id=_field(completion, "id", f"chat-{uuid4()}"),
            output=calls,
            output_text=str(content or ""),
            usage=CCSwitchProvider._normalise_chat_usage(_field(completion, "usage")),
            protocol="CHAT_COMPLETIONS",
            raw=completion,
        )

    @staticmethod
    def _normalise_chat_usage(usage: Any):
        if usage is None:
            return None
        prompt_details = _field(usage, "prompt_tokens_details")
        completion_details = _field(usage, "completion_tokens_details")
        return SimpleNamespace(
            input_tokens=_field(usage, "prompt_tokens", 0),
            output_tokens=_field(usage, "completion_tokens", 0),
            input_tokens_details=SimpleNamespace(cached_tokens=_field(prompt_details, "cached_tokens", 0)),
            output_tokens_details=SimpleNamespace(reasoning_tokens=_field(completion_details, "reasoning_tokens", 0)),
        )

    def health_check(self) -> dict[str, Any]:
        """Probe gateway health and each LLM capability independently."""
        self.reasoning_metadata_status = "UNAVAILABLE"
        self.token_usage_status = "UNAVAILABLE"
        result: dict[str, Any] = {
            "gateway": "FAIL",
            "endpoint": "FAIL",
            "model_discovery": "FAIL",
            "models": [],
            "luna_model": self.luna_model,
            "sol_model": self.sol_model,
            "luna_reasoning_effort": self.luna_reasoning_effort,
            "sol_reasoning_effort": self.sol_reasoning_effort,
            "luna": "FAIL",
            "luna_basic_call": "FAIL",
            "luna_structured_output": "FAIL",
            "sol": "FAIL",
            "sol_basic_call": "FAIL",
            "sol_structured_json": "FAIL",
            "sol_tool_calling": "FAIL",
            "sol_multi_turn": "FAIL",
            "sol_decision": "FAIL",
            "tool_calling": "FAIL",
            "structured_output": "FAIL",
            "multi_turn": "FAIL",
            "api_protocol": self.protocol,
            "protocol_fallback": "YES" if self.protocol_fallback else "NO",
            "reasoning_metadata": "UNAVAILABLE",
            "token_usage": "UNAVAILABLE",
            "errors": [],
            "error": None,
        }

        def probe(label: str, callback: Callable[[], Any]):
            try:
                return callback()
            except Exception as exc:
                safe = self._safe_error(exc)
                if label.startswith("Sol") and _status_code(exc) in {502, 503, 504}:
                    safe = "SOL_UPSTREAM_UNAVAILABLE: " + safe
                result["errors"].append(f"{label}: {safe}")
                return None

        try:
            models_api = getattr(self.client, "models", None)
            if models_api is not None and hasattr(models_api, "list"):
                models_response = models_api.list()
                model_rows = _field(models_response, "data", []) or []
                result["models"] = [str(_field(item, "id", "")) for item in model_rows if _field(item, "id")]
                result["model_discovery"] = "PASS"
                result["configured_luna_model"] = self.luna_model
                result["configured_sol_model"] = self.sol_model
                self.luna_model = _select_discovered_model(self.luna_model, result["models"], "luna")
                self.sol_model = _select_discovered_model(self.sol_model, result["models"], "sol")
                result["luna_model"] = self.luna_model
                result["sol_model"] = self.sol_model
                result["luna_available"] = self.luna_model in result["models"]
                result["sol_available"] = self.sol_model in result["models"]
            else:
                result["model_discovery"] = "UNAVAILABLE"
            result["endpoint"] = "PASS"
        except Exception as exc:
            result["errors"].append("Model Discovery: " + self._safe_error(exc))

        simple_schema = {
            "type": "json_schema",
            "name": "gateway_health",
            "strict": True,
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
        }

        luna_basic = probe("Luna Basic Call", lambda: self.create_response(model=self.luna_model, input='Return exactly "123".'))
        if luna_basic is not None and _response_text(luna_basic).strip():
            result["luna_basic_call"] = result["luna"] = "PASS"

        luna_structured_agent = LunaScreeningAgent(self.luna_model, None, provider=self, max_repair_retries=1)
        luna_structured = probe(
            "Luna Structured Output",
            lambda: self.create_response(
                model=self.luna_model,
                input='Return one candidate, NVDA, with rationale, positive_signals, risks, and screening_rationale.',
                text={"format": luna_structured_agent.output_schema()},
            ),
        )
        if luna_structured is not None:
            try:
                luna_structured_agent._parse_or_repair(luna_structured, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0})
                result["luna_structured_output"] = luna_structured_agent.structured_output_status
            except Exception as exc:
                result["luna_structured_output"] = "FAIL"
                result["errors"].append("Luna Structured Output: " + self._safe_error(exc))

        probe_tool = {
            "type": "function",
            "name": "gateway_health_probe",
            "description": "Return the supplied health probe value.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
            "strict": True,
        }
        sol_basic = probe("Sol Basic Call", lambda: self.create_response(model=self.sol_model, input='Return exactly "123".'))
        if sol_basic is not None and _response_text(sol_basic).strip():
            result["sol_basic_call"] = result["sol"] = "PASS"

        sol_json = probe(
            "Sol Structured JSON",
            lambda: self.create_response(model=self.sol_model, input="Return {ok:true} as JSON.", text={"format": simple_schema}),
        )
        if sol_json is not None:
            try:
                payload = json.loads(_response_text(sol_json))
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise ValueError("gateway returned invalid structured JSON")
                result["sol_structured_json"] = "PASS"
            except Exception as exc:
                result["errors"].append("Sol Structured JSON: " + self._safe_error(exc))

        tool_response = probe(
            "Sol Tool Calling",
            lambda: self.create_response(model=self.sol_model, input="Call gateway_health_probe with value 'ok'.", tools=[probe_tool]),
        )
        if tool_response is not None and _function_calls(tool_response):
            result["sol_tool_calling"] = result["tool_calling"] = "PASS"
            call = _function_calls(tool_response)[0]
            continuation = probe(
                "Sol Multi-turn Tool Calling",
                lambda: self.create_response(
                    model=self.sol_model,
                    previous_response_id=_field(tool_response, "id"),
                    input=[{"type": "function_call_output", "call_id": _field(call, "call_id"), "output": '{"value":"ok"}'}],
                    tools=[probe_tool],
                ),
            )
            if continuation is not None and (_response_text(continuation).strip() or not _function_calls(continuation)):
                result["sol_multi_turn"] = result["multi_turn"] = "PASS"

        sol_decision_agent = SolResearchCIOAgent(self.sol_model, None, provider=self, max_repair_retries=1)
        sol_decision = probe(
            "Sol Decision",
            lambda: self.create_response(
                model=self.sol_model,
                input="Return a valid CASH decision with complete fields and no order.",
                text={"format": sol_decision_agent.output_schema()},
            ),
        )
        if sol_decision is not None:
            try:
                sol_decision_agent._parse_or_repair(sol_decision, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0})
                result["sol_decision"] = sol_decision_agent.structured_output_status
            except Exception as exc:
                result["errors"].append("Sol Decision: " + self._safe_error(exc))

        structured_states = [result["luna_structured_output"], result["sol_structured_json"], result["sol_decision"]]
        result["structured_output"] = "FAIL" if "FAIL" in structured_states else "FALLBACK" if "FALLBACK" in structured_states else "PASS"
        result["gateway"] = result["endpoint"]
        result["api_protocol"] = self.protocol
        result["protocol_fallback"] = "YES" if self.protocol_fallback else "NO"
        result["reasoning_metadata"] = self.reasoning_metadata_status
        result["token_usage"] = self.token_usage_status
        result["error"] = result["errors"][0] if result["errors"] else None
        discovery_ok = result["model_discovery"] in {"PASS", "UNAVAILABLE"}
        usable = lambda value: value in {"PASS", "FALLBACK"}
        result["ok"] = (
            result["endpoint"] == "PASS"
            and discovery_ok
            and result["luna_basic_call"] == "PASS"
            and usable(result["luna_structured_output"])
            and result["sol_basic_call"] == "PASS"
            and result["sol_structured_json"] == "PASS"
            and result["sol_tool_calling"] == "PASS"
            and result["sol_multi_turn"] == "PASS"
            and usable(result["sol_decision"])
        )
        return result

    check_capabilities = health_check
    capability_health_check = health_check

    def _provider_error_fields(self, exc: Exception) -> tuple[str | None, str | int | None, str | None]:
        body = getattr(exc, "body", None)
        if body is None:
            response = getattr(exc, "response", None)
            body = getattr(response, "body", None)
            if body is None:
                json_method = getattr(response, "json", None)
                if callable(json_method):
                    try:
                        body = json_method()
                    except Exception:
                        body = None
        if not isinstance(body, dict):
            return None, None, None
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        error_type = error.get("type") or error.get("error_type")
        error_code = error.get("code") or error.get("error_code")
        message = error.get("message") or body.get("message")
        return (
            str(error_type) if error_type is not None else None,
            error_code,
            self._safe_text(message) if message else None,
        )

    def _safe_text(self, value: Any) -> str:
        message = str(value or "")
        if self.api_key:
            message = message.replace(self.api_key, "[REDACTED]")
        message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", message)
        return re.sub(r"(?i)\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b", "[REDACTED]", message)

    def _safe_error(self, exc: Exception) -> str:
        return self._safe_text(str(exc) or exc.__class__.__name__)


class ScreeningCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    positive_signals: list[str] = Field(default_factory=list, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=6)


class LunaScreeningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ScreeningCandidate] = Field(min_length=1, max_length=25)
    screening_rationale: str = Field(min_length=1)

    @property
    def candidate_symbols(self) -> list[str]:
        return [candidate.symbol for candidate in self.candidates]


class ComparativeDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum_relative_strength: str = "UNKNOWN"
    earnings_trend: str = "UNKNOWN"
    revenue_eps_quality: str = "UNKNOWN"
    valuation: str = "UNKNOWN"
    analyst_revisions: str = "UNKNOWN"
    catalyst: str = "UNKNOWN"
    event_risk: str = "UNKNOWN"
    volatility: str = "UNKNOWN"
    liquidity: str = "UNKNOWN"
    market_sector_fit: str = "UNKNOWN"
    downside_risk: str = "UNKNOWN"
    expected_alpha: str = "UNKNOWN"


class ComparativeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=5)
    symbol: str = Field(min_length=1)
    alpha_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    expected_alpha: float | None = None
    key_advantage: str = Field(min_length=1)
    key_weakness: str = Field(min_length=1)
    why_not_selected: list[str] = Field(default_factory=list, max_length=3)
    comparison: ComparativeDimensions = Field(default_factory=ComparativeDimensions)


class ScenarioCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(min_length=1)
    expected_return_or_direction: str = Field(min_length=1)
    key_assumptions: list[str] = Field(min_length=1, max_length=5)


EvidenceQuality = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN", "MODEL_INFERENCE"]
ImpactLevel = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


class CrossSectionalDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum: str = "UNKNOWN"
    relative_strength: str = "UNKNOWN"
    earnings_trend: str = "UNKNOWN"
    revenue_eps_quality: str = "UNKNOWN"
    valuation: str = "UNKNOWN"
    analyst_revisions: str = "UNKNOWN"
    estimate_revisions: str = "UNKNOWN"
    catalyst: str = "UNKNOWN"
    event_risk: str = "UNKNOWN"
    volatility: str = "UNKNOWN"
    liquidity: str = "UNKNOWN"
    sector_strength: str = "UNKNOWN"
    market_regime_fit: str = "UNKNOWN"
    downside_asymmetry: str = "UNKNOWN"
    expected_alpha_potential: str = "UNKNOWN"


class DeepRankingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=5)
    symbol: str = Field(min_length=1)
    preliminary_alpha_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    key_strengths: list[str] = Field(min_length=1, max_length=5)
    key_weaknesses: list[str] = Field(min_length=1, max_length=5)
    # Light decision-time Alpha/Mispricing narrative; not a score or factor.
    alpha_thesis: str = "NOT_RECORDED"
    market_expectation: str = "NOT_RECORDED"
    remaining_alpha_view: str = "NOT_RECORDED"
    research_priority: Literal["HIGH", "MEDIUM", "LOW"]
    dimensions: CrossSectionalDimensions = Field(default_factory=CrossSectionalDimensions)
    unknown_fields: list[str] = Field(default_factory=list, max_length=10)


class CrossSectionalRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking: list[DeepRankingItem] = Field(min_length=5, max_length=5)
    comparison_basis: list[str] = Field(min_length=1, max_length=16)
    missing_data_sources: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_ranking(self):
        if [item.rank for item in self.ranking] != [1, 2, 3, 4, 5]:
            raise ValueError("Stage A ranking must use ranks 1 through 5")
        symbols = [item.symbol.upper() for item in self.ranking]
        if len(set(symbols)) != 5:
            raise ValueError("Stage A ranking symbols must be unique")
        return self


class ResearchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    source: str = Field(min_length=1)
    evidence_quality: EvidenceQuality


class CausalDriver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: str = Field(min_length=1)
    category: Literal["COMPANY", "INDUSTRY", "SECTOR", "MACRO", "GEOPOLITICAL", "MARKET"]
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"]
    estimated_importance: ImpactLevel
    evidence_quality: EvidenceQuality
    persistence: Literal["SHORT", "MEDIUM", "LONG", "UNKNOWN"]
    priced_in: Literal["LOW", "PARTIAL", "HIGH", "UNKNOWN"]
    mechanism: str = Field(min_length=1)
    earnings_impact: str = Field(min_length=1)
    valuation_impact: str = Field(min_length=1)
    price_impact: str = Field(min_length=1)
    reversal_conditions: list[str] = Field(min_length=1, max_length=5)
    evidence: list[str] = Field(min_length=1, max_length=8)


class ReturnAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_specific_alpha: str
    industry_beta: str
    energy_beta: str
    broad_market_beta: str
    evidence_quality: EvidenceQuality
    explanation: str = Field(min_length=1)


class PricedInDriver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: str = Field(min_length=1)
    priced_in_level: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    remaining_alpha_potential: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class AlternativeHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(min_length=1, max_length=6)
    opposing_evidence: list[str] = Field(min_length=1, max_length=6)
    evidence_quality: EvidenceQuality


class MPCSpecificAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    likely_primary_driver: str = "DATA_UNAVAILABLE"
    war_primary_driver: Literal["YES", "PARTIAL", "MINOR", "NO", "UNKNOWN"] = "UNKNOWN"
    geopolitical_contribution: str = "DATA_UNAVAILABLE"
    mechanism: str = "DATA_UNAVAILABLE"
    oil_vs_crack_spread: str = "DATA_UNAVAILABLE"
    vlo_psx_comparison: str = "DATA_UNAVAILABLE"
    company_outperformance: str = "DATA_UNAVAILABLE"
    beta_vs_alpha: str = "DATA_UNAVAILABLE"
    earnings_revision_support: str = "DATA_UNAVAILABLE"
    priced_in_assessment: str = "DATA_UNAVAILABLE"
    twenty_day_persistence: str = "DATA_UNAVAILABLE"
    deescalation_impact: str = "DATA_UNAVAILABLE"
    maximum_downside_risk: str = "DATA_UNAVAILABLE"
    early_exit_conditions: list[str] = Field(default_factory=list, max_length=8)


class ResearchScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(min_length=1)
    key_assumptions: list[str] = Field(min_length=1, max_length=6)
    expected_direction: Literal["STRONG_UPSIDE", "MODERATE_UPSIDE", "FLAT", "MODERATE_DOWNSIDE", "SEVERE_DOWNSIDE"]
    probability_confidence: Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


class DeepDiveResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    business_driver_summary: str = Field(min_length=1)
    causal_drivers: list[CausalDriver] = Field(min_length=1, max_length=10)
    return_attribution: ReturnAttribution
    why_now: list[str] = Field(min_length=1, max_length=6)
    market_expectations: list[str] = Field(min_length=1, max_length=6)
    # The same qualitative view is retained for incumbent/challenger context;
    # old supplemental records remain readable through the sentinel default.
    alpha_thesis: str = "NOT_RECORDED"
    market_expectation: str = "NOT_RECORDED"
    remaining_alpha_view: str = "NOT_RECORDED"
    pricing_assessment: Literal["UNDERPRICED", "FAIRLY_PRICED", "OVERPRICED", "UNCERTAIN"]
    priced_in_drivers: list[PricedInDriver] = Field(min_length=1, max_length=8)
    alternative_hypotheses: list[AlternativeHypothesis] = Field(min_length=3, max_length=6)
    primary_hypothesis: str = Field(min_length=1)
    near_term_catalyst: str = Field(min_length=1)
    geopolitical_contribution: Literal["YES", "PARTIAL", "MINOR", "NO", "UNKNOWN"]
    geopolitical_mechanism: str = Field(min_length=1)
    twenty_day_persistence: Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    bull_case: ResearchScenario
    base_case: ResearchScenario
    bear_case: ResearchScenario
    key_risks: list[str] = Field(min_length=1, max_length=8)
    thesis_invalidation_conditions: list[str] = Field(min_length=1, max_length=8)
    evidence: list[ResearchEvidence] = Field(min_length=1, max_length=20)
    missing_data: list[str] = Field(default_factory=list, max_length=20)
    required_data: list[str] = Field(default_factory=list, max_length=20)
    mpc_specific_answers: MPCSpecificAnalysis = Field(default_factory=MPCSpecificAnalysis)


class BearCandidateReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    strongest_bear_argument: str = Field(min_length=1)
    failure_modes: list[str] = Field(min_length=1, max_length=10)
    what_would_make_me_change_my_mind: list[str] = Field(min_length=1, max_length=8)
    better_alternative_exists: bool
    evidence_quality: EvidenceQuality


class AdversarialReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[BearCandidateReview] = Field(min_length=3, max_length=3)


class ICScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(min_length=1)
    key_assumptions: list[str] = Field(min_length=1, max_length=6)
    expected_direction: Literal["STRONG_UPSIDE", "MODERATE_UPSIDE", "FLAT", "MODERATE_DOWNSIDE", "SEVERE_DOWNSIDE"]
    probability_confidence: Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


class PairwiseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_symbol: str
    alternative_symbol: str
    selected_advantages: list[str] = Field(min_length=1, max_length=5)
    alternative_advantages: list[str] = Field(min_length=1, max_length=5)
    higher_alpha_potential: str
    lower_downside_risk: str
    clearer_catalyst: str
    more_priced_in: str
    decision_reason: str = Field(min_length=1)


class CommitteeSizingAudit(SizingAudit):
    evidence_refs: list[Literal["STAGE_A", "STAGE_B", "STAGE_C", "TOOL_COVERAGE"]] = Field(
        min_length=1, max_length=12,
        description="Exact supplied stage IDs only; no suffixes, paths, or invented evidence IDs.",
    )


class InvestmentCommitteeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sizing_audit: CommitteeSizingAudit

    action: Literal["BUY", "HOLD_CASH", "HOLD", "REPLACE", "REDUCE", "SELL", "EXIT_TO_CASH"]
    selected_symbol: str | None
    rank: int | None = Field(default=None, ge=1, le=3)
    runner_up: str | None
    alpha_score: float | None = Field(default=None, ge=0, le=100)
    runner_up_score: float | None = Field(default=None, ge=0, le=100)
    alpha_gap: float | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_basis: list[str] = Field(min_length=1, max_length=6)
    confidence_reducers: list[str] = Field(min_length=1, max_length=6)
    expected_alpha_vs_spy: float | None = None
    expected_alpha_vs_qqq: float | None = None
    expected_alpha_basis: list[str] = Field(min_length=1, max_length=8)
    estimate_type: Literal["SOL_MODEL_ESTIMATE"] = "SOL_MODEL_ESTIMATE"
    thesis_horizon_days: int = Field(ge=1, le=90)
    target_weight: float = Field(ge=0, le=1)
    primary_alpha_drivers: list[str] = Field(min_length=1, max_length=8)
    company_specific_drivers: list[str] = Field(default_factory=list, max_length=8)
    industry_drivers: list[str] = Field(default_factory=list, max_length=8)
    macro_drivers: list[str] = Field(default_factory=list, max_length=8)
    geopolitical_drivers: list[str] = Field(default_factory=list, max_length=8)
    priced_in_assessment: str = Field(min_length=1)
    bull_case: ICScenario
    base_case: ICScenario
    bear_case: ICScenario
    thesis_invalidation_conditions: list[str] = Field(min_length=1, max_length=8)
    why_selected: list[str] = Field(min_length=1, max_length=6)
    why_not_runner_up: list[str] = Field(min_length=1, max_length=5)
    why_not_rank3: list[str] = Field(min_length=1, max_length=5)
    pairwise_comparisons: list[PairwiseComparison] = Field(min_length=2, max_length=2)
    strongest_bear_argument: str = Field(min_length=1)
    what_would_make_me_change_my_mind: list[str] = Field(min_length=1, max_length=8)
    missing_data: list[str] = Field(default_factory=list, max_length=20)
    tool_coverage: list[str] = Field(min_length=1, max_length=20)
    evidence_used: list[str] = Field(min_length=1, max_length=20)
    risk_factors: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_decision(self):
        if not set(self.sizing_audit.evidence_refs).issubset({"STAGE_A", "STAGE_B", "STAGE_C", "TOOL_COVERAGE"}):
            raise ValueError("IC sizing audit must reference supplied research stages")
        cash = self.action in {"HOLD_CASH", "EXIT_TO_CASH"}
        if cash:
            if self.selected_symbol is not None or self.target_weight != 0:
                raise ValueError("Cash IC decisions require null symbol and zero weight")
        elif not self.selected_symbol:
            raise ValueError("Invested IC decisions require selected_symbol")
        if self.alpha_score is not None and self.runner_up_score is not None:
            expected = self.alpha_score - self.runner_up_score
            if self.alpha_gap is None or abs(self.alpha_gap - expected) > 1e-6:
                raise ValueError("IC alpha_gap must equal alpha_score minus runner_up_score")
        return self


class ResearchToolCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_sources: list[str]
    missing_sources: list[str]


class SolDeepResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking: CrossSectionalRanking
    top_three: list[DeepDiveResearch] = Field(min_length=3, max_length=3)
    adversarial_review: AdversarialReview
    final_decision: InvestmentCommitteeDecision
    tool_coverage: ResearchToolCoverage


class SolDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["BUY", "HOLD", "SWITCH", "CASH"]
    symbol: str | None
    target_weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    holding_period_days: int = Field(ge=1, le=90)
    expected_excess_vs_spy: float
    expected_excess_vs_qqq: float
    thesis: list[str] = Field(min_length=1, max_length=4)
    risk_factors: list[str] = Field(min_length=1, max_length=6)
    thesis_invalidation_conditions: list[str] = Field(min_length=1, max_length=5)
    evidence_used: list[str] = Field(min_length=1, max_length=12)
    current_symbol: str | None
    new_symbol: str | None
    top_five: list[ComparativeCandidate] = Field(default_factory=list, max_length=5)
    selected_symbol: str | None = None
    runner_up_symbol: str | None = None
    selected_alpha_score: float | None = Field(default=None, ge=0, le=100)
    runner_up_alpha_score: float | None = Field(default=None, ge=0, le=100)
    alpha_gap: float | None = None
    why_selected: list[str] = Field(default_factory=list, max_length=5)
    why_selected_over_runner_up: list[str] = Field(default_factory=list, max_length=3)
    bull_case: ScenarioCase | None = None
    base_case: ScenarioCase | None = None
    bear_case: ScenarioCase | None = None
    expected_alpha_basis: list[str] = Field(default_factory=list, max_length=7)
    estimate_type: Literal["SOL_MODEL_ESTIMATE"] = "SOL_MODEL_ESTIMATE"
    confidence_basis: list[str] = Field(default_factory=list, max_length=5)
    confidence_reducers: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action == "CASH":
            if self.symbol is not None or self.target_weight != 0:
                raise ValueError("CASH decisions must have null symbol and zero target_weight")
        elif not self.symbol:
            raise ValueError("Non-CASH decisions require symbol")
        if self.action == "SWITCH":
            if not self.current_symbol or not self.new_symbol:
                raise ValueError("SWITCH decisions require current_symbol and new_symbol")
            if self.symbol.upper() != self.new_symbol.upper():
                raise ValueError("SWITCH symbol must match new_symbol")
            if self.current_symbol.upper() == self.new_symbol.upper():
                raise ValueError("SWITCH requires different current_symbol and new_symbol")
        return self

    def to_trade_intent(self, model_name: str) -> TradeIntent:
        values = {
            key: getattr(self, key)
            for key in (
                "action", "symbol", "target_weight", "confidence", "holding_period_days", "thesis",
                "risk_factors", "evidence_used", "current_symbol", "new_symbol",
            )
        }
        values["expected_alpha_vs_spy"] = self.expected_excess_vs_spy
        values["expected_alpha_vs_qqq"] = self.expected_excess_vs_qqq
        values["invalidation_conditions"] = self.thesis_invalidation_conditions
        return TradeIntent(**values, model_name=model_name)

    def validate_comparative_context(self, candidate_symbols: list[str]) -> None:
        candidates = list(dict.fromkeys(symbol.upper() for symbol in candidate_symbols))
        if len(candidates) < 5:
            return
        if len(self.top_five) != 5:
            raise ValueError("Sol comparative decision requires exactly five ranked candidates")
        ranked_symbols = [item.symbol.upper() for item in self.top_five]
        if [item.rank for item in self.top_five] != [1, 2, 3, 4, 5]:
            raise ValueError("Sol comparative ranking must use ranks 1 through 5")
        if len(set(ranked_symbols)) != 5 or not set(ranked_symbols).issubset(candidates):
            raise ValueError("Sol comparative ranking must contain unique supplied candidates")
        if self.action != "CASH":
            if self.selected_symbol != self.symbol or ranked_symbols[0] != self.symbol:
                raise ValueError("Sol comparative selected symbol must match rank 1 and the TradeIntent symbol")
            if self.runner_up_symbol != ranked_symbols[1]:
                raise ValueError("Sol comparative runner-up must match rank 2")
        if self.selected_alpha_score is None or self.runner_up_alpha_score is None or self.alpha_gap is None:
            raise ValueError("Sol comparative alpha scores and alpha gap are required")
        expected_gap = self.selected_alpha_score - self.runner_up_alpha_score
        if abs(self.alpha_gap - expected_gap) > 1e-6:
            raise ValueError("Sol comparative alpha gap must equal selected score minus runner-up score")
        if any(not item.why_not_selected for item in self.top_five[1:]):
            raise ValueError("Sol comparative ranks 2 through 5 require why_not_selected")
        if not self.why_selected or not self.why_selected_over_runner_up:
            raise ValueError("Sol comparative selection rationale is required")
        if not all((self.bull_case, self.base_case, self.bear_case)):
            raise ValueError("Sol comparative bull, base, and bear cases are required")
        if not self.expected_alpha_basis or not self.confidence_basis or not self.confidence_reducers:
            raise ValueError("Sol comparative alpha and confidence bases are required")


class SolPositionReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sizing_audit: SizingAudit

    action: Literal["HOLD", "ADD", "REDUCE", "SELL", "REPLACE", "EXIT_TO_CASH", "EXTEND_HOLD"]
    thesis_status: Literal["INTACT", "WEAKENING", "BROKEN", "IMPROVING", "EXPIRED_REVIEW_REQUIRED"]
    current_holding_score: float = Field(ge=0, le=100)
    best_alternative: str | None = None
    best_alternative_score: float | None = Field(default=None, ge=0, le=100)
    replacement_gap: float = 0.0
    confidence: float = Field(ge=0, le=1)
    new_horizon_days: int = Field(ge=1, le=180)
    reason: list[str] = Field(min_length=1, max_length=5)
    risk_level: Literal["NORMAL", "ELEVATED", "HIGH", "CRITICAL"] = "NORMAL"
    target_weight: float = Field(ge=0, le=1)
    requires_full_research: bool = False

    @model_validator(mode="after")
    def validate_comparison(self):
        if self.best_alternative_score is not None:
            expected = self.best_alternative_score - self.current_holding_score
            if abs(self.replacement_gap - expected) > 1e-6:
                raise ValueError("replacement_gap must equal best_alternative_score minus current_holding_score")
        elif self.best_alternative is not None or self.replacement_gap != 0:
            raise ValueError("Alternative symbol, score, and replacement gap must be supplied together")
        if self.action == "REPLACE" and (not self.best_alternative or self.best_alternative_score is None):
            raise ValueError("REPLACE requires a scored best alternative")
        return self


def _compact_universe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "symbol", "price", "ret_20d", "ret_60d", "ret_120d", "relative_strength_vs_spy", "relative_strength_vs_qqq",
        "annualized_volatility", "ann_vol", "market_cap", "forward_pe", "revenue_growth", "eps_growth", "earnings_days", "liquidity", "avg_dollar_volume",
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("symbol"):
            continue
        symbol = str(row["symbol"]).upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        compact = {}
        for key in fields:
            value = row.get(key)
            if value is None:
                continue
            compact[key] = round(value, 6) if isinstance(value, float) else value
        compact["symbol"] = symbol
        result.append(compact)
    return result


def _compact_candidate_summaries(candidates: Iterable[ScreeningCandidate | dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = candidate.model_dump(mode="json") if isinstance(candidate, ScreeningCandidate) else candidate
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        symbol = str(item["symbol"]).upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append({
            "symbol": symbol,
            "rationale": str(item.get("rationale") or ""),
            "positive_signals": list(item.get("positive_signals") or [])[:5],
            "risks": list(item.get("risks") or [])[:5],
        })
    return result


def _research_tools() -> list[dict[str, Any]]:
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    symbol = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False}
    symbol_days = {"type": "object", "properties": {"symbol": {"type": "string"}, "days": {"type": "integer", "minimum": 20, "maximum": 1000}}, "required": ["symbol", "days"], "additionalProperties": False}
    symbol_limit = {"type": "object", "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["symbol", "limit"], "additionalProperties": False}
    benchmark = {"type": "object", "properties": {"days": {"type": "integer", "minimum": 20, "maximum": 1000}}, "required": ["days"], "additionalProperties": False}
    definitions = [
        ("get_market_regime", "Get current broad-market regime data.", empty),
        ("get_price_history", "Get summarized price history and realized volatility for one stock.", symbol_days),
        ("get_stock_snapshot", "Get current snapshot for one stock.", symbol),
        ("get_fundamentals", "Get current fundamental metrics for one stock.", symbol),
        ("get_earnings", "Get the next earnings date and timing for one stock.", symbol),
        ("get_analyst_revisions", "Get recent analyst estimate or rating revisions for one stock.", symbol),
        ("get_news", "Get recent news for one stock.", symbol_limit),
        ("get_sec_filings", "Get recent SEC filings for one stock.", symbol_limit),
        ("get_upcoming_events", "Get earnings and known event risk for one stock.", symbol),
        ("get_portfolio", "Get the current portfolio state.", empty),
        ("get_current_positions", "Get current positions.", empty),
        ("get_benchmark_data", "Get current SPY and QQQ benchmark data.", benchmark),
    ]
    return [{"type": "function", "name": name, "description": description, "parameters": parameters, "strict": True} for name, description, parameters in definitions]


class LunaScreeningAgent:
    """Candidate generator. It has no trade or broker output fields."""

    stage = "LUNA"
    prompt_version = "luna-screening-v1"

    def __init__(self, model: str, data: DataProvider, *, provider: LLMProvider, max_candidates: int = 25, max_repair_retries: int = 1, event_sink: Callable[[dict[str, Any]], None] | None = None):
        self.model = model
        self.data = data
        self.provider = provider
        self.max_candidates = min(int(max_candidates), 25)
        self.max_repair_retries = min(2, max(0, int(max_repair_retries)))
        self.last_usage: dict[str, Any] = {}
        self.last_screening: LunaScreeningResult | None = None
        self.structured_output_status = "UNKNOWN"
        self.event_sink = event_sink

    def set_event_sink(self, event_sink: Callable[[dict[str, Any]], None] | None):
        self.event_sink = event_sink

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {"type": "json_schema", "name": "luna_screening", "strict": True, "schema": _strict_json_schema(LunaScreeningResult.model_json_schema())}

    def screen(self, universe_snapshot: list[dict[str, Any]] | None = None, *, max_candidates: int | None = None, context: dict[str, Any] | None = None) -> LunaScreeningResult:
        batch_context = context or {}
        source_rows = universe_snapshot if universe_snapshot is not None else self.data.universe_snapshot()
        rows = _compact_candidate_summaries(source_rows) if batch_context.get("final_screening") else _compact_universe(source_rows)
        if not rows:
            raise ValueError("Luna screening requires a non-empty universe snapshot")
        started = time.perf_counter()
        limit = min(int(max_candidates or self.max_candidates), self.max_candidates)
        is_final = bool(batch_context.get("final_screening"))
        start_event = (
            "LUNA_FINAL_SCREENING_STARTED"
            if is_final
            else "LUNA_BATCH_STARTED" if batch_context.get("batch_index") else "LUNA_SCREEN_STARTED"
        )
        _emit_runtime_event(
            self.event_sink,
            start_event,
            self.stage,
            f"Screening {len(rows)} stocks",
            metadata={"universe_size": len(rows), "model": self.model, "pipeline": "LUNA_SOL", **batch_context},
        )
        self.structured_output_status = "PASS"
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        if is_final:
            prompt = (
                "Globally compare these compact candidate summaries produced by prior Luna screening batches. Select only "
                "the strongest candidates for Sol research without adding new factors. Preserve the meaning of the supplied "
                f"signals and risks. Return at most {limit} candidates using the exact structured screening schema. Keep each "
                "rationale to one short sentence and use at most three short positive signals and three short risks. Return "
                "JSON only; no markdown, company descriptions, repeated input, or long-form analysis.\nCANDIDATE_SUMMARIES_JSON:\n"
                + json.dumps(rows, ensure_ascii=False, default=str, separators=(",", ":"))
            )
        else:
            prompt = (
                "Screen this compact, deduplicated S&P 500 and Nasdaq-100 snapshot. Return only the top quality candidates "
                "for a second-stage researcher. Do not produce a trade action, target weight, order, or broker command. "
                f"Return at most {limit} candidates. Use the exact structured screening schema. Keep each rationale concise; "
                "use at most three short positive signals and three short risks per stock. Do not include company introductions, "
                "markdown, long-form research, repeated input data, or macro commentary unrelated to the screen.\nUNIVERSE_JSON:\n"
                + json.dumps(rows, ensure_ascii=False, default=str, separators=(",", ":"))
            )
        try:
            response = _request_with_runtime_events(
                self.provider,
                self.model,
                self.event_sink,
                component=self.stage,
                started_event="LUNA_SCREEN_REQUEST_STARTED",
                failed_event="LUNA_SCREEN_REQUEST_FAILED",
                diagnostic_metadata={
                    "batch_id": batch_context.get("batch_id"),
                    "batch_size": len(rows),
                    "ticker_count": len(rows),
                    "split_depth": int(batch_context.get("retry_count", 0) or 0),
                    "retry_attempt": 0,
                    "final_screening_stage": batch_context.get("final_screening_stage"),
                    "final_group_index": batch_context.get("final_group_index"),
                    "final_group_total": batch_context.get("final_group_total"),
                    "input_candidate_count": batch_context.get("input_candidate_count"),
                },
                instructions="You are Luna, a screening-only analyst.",
                input=prompt,
                text={"format": self.output_schema()},
            )
            response_usage = _field(response, "usage")
            _accumulate_usage(totals, response_usage)
            result = self._parse_or_repair(response, totals)
            allowed = {row["symbol"] for row in rows}
            if len(result.candidates) > limit:
                raise ValueError("Luna returned more candidates than configured")
            if len(set(result.candidate_symbols)) != len(result.candidate_symbols):
                raise ValueError("Luna returned duplicate candidate symbols")
            unknown = [symbol for symbol in result.candidate_symbols if symbol.upper() not in allowed]
            if unknown:
                raise ValueError("Luna returned a symbol outside the supplied universe")
            result = result.model_copy(update={"candidates": [candidate.model_copy(update={"symbol": candidate.symbol.upper()}) for candidate in result.candidates]})
            self.last_screening = result
            self.last_usage = self._usage(totals, started)
            self.last_usage["available"] = response_usage is not None
            _emit_runtime_event(
                self.event_sink,
                (
                    "LUNA_FINAL_SCREENING_COMPLETED"
                    if is_final
                    else "LUNA_BATCH_COMPLETED" if batch_context.get("batch_index") else "LUNA_SCREEN_COMPLETED"
                ),
                self.stage,
                f"Screening completed: {len(result.candidates)} candidates",
                metadata={
                    "universe_size": len(rows),
                    "candidate_count": len(result.candidates),
                    "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
                    "screening_rationale": result.screening_rationale,
                    "usage": self.last_usage,
                    "model": self.model,
                    **batch_context,
                    "candidate_symbols": result.candidate_symbols,
                    "latency": self.last_usage.get("latency_ms"),
                    "input_tokens": self.last_usage.get("input_tokens") if self.last_usage.get("available") else None,
                    "output_tokens": self.last_usage.get("output_tokens") if self.last_usage.get("available") else None,
                    "cost": self.last_usage.get("estimated_cost") if self.last_usage.get("available") else None,
                    "status": "COMPLETE",
                },
            )
            return result
        except Exception as exc:
            error_event = (
                "LUNA_FINAL_SCREENING_FAILED"
                if is_final
                else "LUNA_BATCH_REQUEST_FAILED" if batch_context.get("batch_index") else "LUNA_SCREEN_ERROR"
            )
            _emit_runtime_event(self.event_sink, error_event, self.stage, str(exc), metadata={"error_type": exc.__class__.__name__, **batch_context})
            raise

    def _parse_or_repair(self, response: Any, totals: dict[str, int]) -> LunaScreeningResult:
        current = response
        last_error: Exception | None = None
        for attempt in range(self.max_repair_retries + 1):
            try:
                raw_payload = _response_text(current)
                payload = json.loads(raw_payload)
                if isinstance(payload, dict) and "candidates" not in payload and "candidate_symbols" in payload:
                    payload["candidates"] = [{"symbol": symbol, "rationale": "Candidate selected by Luna", "positive_signals": [], "risks": []} for symbol in payload.pop("candidate_symbols")]
                for candidate in payload.get("candidates", []) if isinstance(payload, dict) else []:
                    if isinstance(candidate, str):
                        payload["candidates"] = [{"symbol": item, "rationale": "Candidate selected by Luna", "positive_signals": [], "risks": []} for item in payload["candidates"]]
                        break
                if isinstance(payload, dict) and isinstance(payload.get("screening_rationale"), list):
                    payload["screening_rationale"] = "; ".join(str(item) for item in payload["screening_rationale"])
                if isinstance(payload, dict) and "screening_rationale" not in payload and payload.get("rationale"):
                    payload["screening_rationale"] = str(payload.pop("rationale"))
                if isinstance(payload, dict):
                    missing = []
                    if "screening_rationale" not in payload:
                        missing.append("screening_rationale")
                    for index, candidate in enumerate(payload.get("candidates", [])):
                        if isinstance(candidate, dict):
                            for field_name in ("positive_signals", "risks"):
                                if field_name not in candidate:
                                    missing.append(f"candidates[{index}].{field_name}")
                    if missing:
                        raise ValueError("missing required screening fields: " + ", ".join(missing))
                result = LunaScreeningResult.model_validate(payload)
                self.structured_output_status = "FALLBACK" if attempt else "PASS"
                return result
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_retries:
                    self.structured_output_status = "FAIL"
                    break
                self.structured_output_status = "FALLBACK"
                current = _request_with_runtime_events(
                    self.provider,
                    self.model,
                    self.event_sink,
                    component=self.stage,
                    started_event="LUNA_SCREEN_REQUEST_STARTED",
                    failed_event="LUNA_SCREEN_REQUEST_FAILED",
                    input=(
                        "Repair the prior response into valid JSON matching the complete Luna screening schema. "
                        "Return only the repaired JSON. Do not add action, weight, order, or broker fields.\n"
                        "ORIGINAL_JSON:\n" + raw_payload + "\n"
                        "VALIDATION_ERROR:\n" + str(exc) + "\n"
                        "COMPLETE_SCHEMA:\n" + json.dumps(self.output_schema(), ensure_ascii=False)
                    ),
                    text={"format": self.output_schema()},
                )
                _accumulate_usage(totals, _field(current, "usage"))
        raise ValueError("Luna returned invalid structured screening output; execution is blocked") from last_error

    def _usage(self, totals: dict[str, int], started: float) -> dict[str, Any]:
        cost = _estimate_cost(self.provider, totals, self.model)
        return {**totals, "latency_ms": (time.perf_counter() - started) * 1000, "estimated_cost": cost, "model": self.model, "stage": self.stage}


class SolResearchCIOAgent:
    """Deep research and final investment decision agent."""

    stage = "SOL"
    prompt_version = "sol-research-cio-v2-comparative"

    def __init__(self, model: str, data: DataProvider, *, provider: LLMProvider, max_tool_rounds: int = 12, max_repair_retries: int = 1, deep_research_enabled: bool = False, force_stock_selection: bool = False, forced_selection_high_weight: float = 1.0, forced_selection_data_limited_weight: float = 0.70, forced_selection_exploratory_weight: float = 0.35, event_sink: Callable[[dict[str, Any]], None] | None = None):
        self.model = model
        self.data = data
        self.provider = provider
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.max_repair_retries = min(2, max(0, int(max_repair_retries)))
        self.deep_research_enabled = bool(deep_research_enabled)
        self.force_stock_selection = bool(force_stock_selection)
        self.forced_selection_high_weight = _fraction(forced_selection_high_weight, "forced_selection_high_weight")
        self.forced_selection_data_limited_weight = _fraction(forced_selection_data_limited_weight, "forced_selection_data_limited_weight")
        self.forced_selection_exploratory_weight = _fraction(forced_selection_exploratory_weight, "forced_selection_exploratory_weight")
        self.last_usage: dict[str, Any] = {}
        self.last_research_evidence: list[dict[str, Any]] = []
        self.last_tool_calls: list[dict[str, Any]] = []
        self.last_decision_analysis: SolDecisionPayload | None = None
        self.last_deep_research: SolDeepResearchResult | None = None
        self.last_stage_metrics: list[dict[str, Any]] = []
        self._candidate_symbols: list[str] = []
        self.structured_output_status = "UNKNOWN"
        self.event_sink = event_sink
        # Optional, independently verified evidence supplied by the research-only skill.
        # It is deliberately empty for the production pipeline unless the caller opts in.
        self.external_verification: dict[str, Any] = {}

    def set_event_sink(self, event_sink: Callable[[dict[str, Any]], None] | None):
        self.event_sink = event_sink

    def quick_entry_review(self, payload: dict[str, Any]):
        from .entry_execution import EntryQuickReview

        schema = {"type": "json_schema", "name": "sol_quick_entry_review", "strict": True, "schema": _strict_json_schema(EntryQuickReview.model_json_schema())}
        response = _request_with_runtime_events(
            self.provider,
            self.model,
            self.event_sink,
            component=self.stage,
            started_event="SOL_ENTRY_REVIEW_STARTED",
            failed_event="SOL_ENTRY_REVIEW_FAILED",
            instructions="Review whether the original entry thesis remains actionable. Do not set an order price, quantity, or weight.",
            input="Return BUY_NOW, WAIT, or CANCEL_ENTRY using the exact schema. Use only the supplied original decision and information observed after it.\nENTRY_REVIEW_JSON:\n" + json.dumps(payload, ensure_ascii=False, default=str),
            text={"format": schema},
        )
        review = EntryQuickReview.model_validate_json(_response_text(response))
        _emit_runtime_event(self.event_sink, "SOL_ENTRY_REVIEW_COMPLETED", self.stage, review.action, symbol=payload.get("symbol"), metadata=review.model_dump(mode="json"))
        return review

    def review_position(
        self,
        position: ManagedPosition,
        portfolio: PortfolioState,
        *,
        candidate_symbols: list[str],
        review_type: str,
        event_context: dict[str, Any],
        replacement_threshold: float,
    ) -> PositionReview:
        """Review one holding against supplied alternatives and cash without a universe screen."""
        symbols = list(dict.fromkeys(str(symbol).upper() for symbol in candidate_symbols if symbol))
        evidence = {
            "holding_snapshot": self.data.stock_snapshot(position.symbol),
            "holding_price_history": self.data.price_history(position.symbol, 20),
            "holding_fundamentals": self.data.fundamentals(position.symbol),
            "holding_news": self.data.news(position.symbol, 10),
            "holding_sec_filings": self.data.sec_filings(position.symbol, 10),
            "holding_earnings": self.data.earnings(position.symbol),
            "holding_analyst_revisions": self.data.analyst_revisions(position.symbol),
            "market_regime": self.data.market_regime(),
            "benchmarks": self.data.benchmark_data(20),
            "alternatives": [
                {
                    "symbol": symbol,
                    "snapshot": self.data.stock_snapshot(symbol),
                    "fundamentals": self.data.fundamentals(symbol),
                    "analyst_revisions": self.data.analyst_revisions(symbol),
                }
                for symbol in symbols
            ],
        }
        previous = event_context.get("previous_review") or {}
        comparison = evidence_changes(previous.get("evidence_snapshot", {}), evidence)
        schema = {
            "type": "json_schema",
            "name": "sol_position_review",
            "strict": True,
            "schema": _strict_json_schema(SolPositionReviewPayload.model_json_schema()),
        }
        prompt = (
            "Current holding, supplied alternatives, and CASH are the only portfolio options in this review. "
            "Evaluate them in one common scoring context. The original horizon is a thesis horizon, not a mandatory hold. "
            "Do not create an order or set an order price. REPLACE is only a proposal; deterministic Python policy and the "
            "Risk Engine enforce the configured threshold and execution safety. A BROKEN thesis may exit to cash even when "
            "there is no replacement. Return only compact JSON matching the schema.\n"
            "SIZING AUDIT: Compare previous_review, original thesis and committee_decision with current evidence. "
            "Separate new negative facts, unchanged missing data and actual thesis changes. Missing data is UNKNOWN, "
            "not deterioration. Do not apply another haircut for the same unchanged uncertainty already reflected in "
            "the committee/previous target. This does not prevent correcting a newly identified existing risk-budget breach. "
            "A source changing is not necessarily bad news; absent history means UNKNOWN, not proof of deterioration. "
            "Past gains alone do not establish overvaluation. Explain exact target_weight and the incremental reason "
            "for any deviation from the committee target. risk_budget_basis must cite a supplied configured rule, or "
            "explicitly say UNKNOWN; never invent a limit. Describe downside qualitatively when unquantifiable. "
            "Scores/confidence are subjective and not calibrated probabilities or inputs to Kelly sizing. "
            "Use evidence_refs containing exact EVIDENCE_JSON top-level keys. No reference may be invented.\n"
            f"EVIDENCE_CHANGE_JSON: {json.dumps(comparison)}\n"
            f"REVIEW_TYPE: {review_type}\n"
            f"REPLACEMENT_THRESHOLD: {replacement_threshold}\n"
            f"POSITION_JSON: {position.model_dump_json()}\n"
            f"PORTFOLIO_JSON: {portfolio.model_dump_json()}\n"
            f"EVENT_CONTEXT_JSON: {json.dumps(event_context, ensure_ascii=False, default=str)}\n"
            f"EVIDENCE_JSON: {json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        _emit_runtime_event(
            self.event_sink,
            "SOL_POSITION_REVIEW_STARTED",
            self.stage,
            f"{review_type} review for {position.symbol}",
            symbol=position.symbol,
            metadata={"review_type": review_type, "candidate_symbols": symbols, "replacement_threshold": replacement_threshold},
        )
        response = _request_with_runtime_events(
            self.provider,
            self.model,
            self.event_sink,
            component=self.stage,
            started_event="SOL_POSITION_REVIEW_REQUEST_STARTED",
            failed_event="SOL_POSITION_REVIEW_REQUEST_FAILED",
            instructions="You are Sol performing a bounded position re-evaluation. Do not produce orders.",
            input=prompt,
            text={"format": schema},
        )
        last_error: Exception | None = None
        payload = None
        for attempt in range(self.max_repair_retries + 1):
            try:
                payload = SolPositionReviewPayload.model_validate_json(_response_text(response))
                if not set(payload.sizing_audit.evidence_refs).issubset(evidence):
                    raise ValueError("sizing_audit evidence_refs must reference supplied evidence source keys")
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_retries:
                    raise ValueError(f"Sol returned invalid position review: {exc}") from exc
                response = _request_with_runtime_events(
                    self.provider,
                    self.model,
                    self.event_sink,
                    component=self.stage,
                    started_event="SOL_POSITION_REVIEW_REPAIR_STARTED",
                    failed_event="SOL_POSITION_REVIEW_REPAIR_FAILED",
                    input=(
                        "Repair this position review. Return only JSON matching the complete schema.\n"
                        f"ORIGINAL_JSON:\n{_response_text(response)}\nVALIDATION_ERROR:\n{exc}\n"
                        f"COMPLETE_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                    text={"format": schema},
                )
        if payload is None:
            raise ValueError(f"Sol returned no position review: {last_error}")
        now = datetime.now(timezone.utc).isoformat()
        review = PositionReview(
            **payload.model_dump(),
            position_id=position.position_id,
            run_id=str(uuid4()),
            decision_id=str(uuid4()),
            review_type=review_type,
            trigger=event_context.get("event_type") or event_context.get("trigger"),
            current_holding=position.symbol,
            replacement_threshold=replacement_threshold,
            days_held=position.days_held_at(),
            original_horizon_days=position.original_thesis_horizon_days,
            reviewed_at=now,
            evidence_snapshot=evidence,
            comparison_context={
                **comparison,
                "previous_review_id": previous.get("review_id"),
                "previous_target_weight": previous.get("target_weight"),
                "current_weight": position.current_weight,
                "committee_decision": event_context.get("committee_decision"),
                "configured_risk_budget": event_context.get("configured_risk_budget", {}),
            },
        )
        _emit_runtime_event(
            self.event_sink,
            "SOL_POSITION_REVIEW_COMPLETED",
            self.stage,
            f"{review.action} / {review.thesis_status}",
            symbol=position.symbol,
            metadata={**review.model_dump(mode="json"), "run_id": review.run_id, "decision_id": review.decision_id},
        )
        return review

    @property
    def tools(self):
        return _research_tools()

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {"type": "json_schema", "name": "sol_cio_decision", "strict": True, "schema": _strict_json_schema(SolDecisionPayload.model_json_schema())}

    def decide(self, portfolio: PortfolioState, candidate_symbols: list[str] | list[ScreeningCandidate], horizon_days: int = 20, screening: LunaScreeningResult | None = None) -> TradeIntent:
        self.last_research_evidence = []
        self.last_tool_calls = []
        self.last_decision_analysis = None
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        started = time.perf_counter()
        self.structured_output_status = "PASS"
        candidates = [item.symbol if isinstance(item, ScreeningCandidate) else str(item).upper() for item in candidate_symbols]
        if not candidates:
            raise ValueError("Sol research requires candidate symbols")
        self._candidate_symbols = candidates
        self.last_deep_research = None
        self.last_stage_metrics = []
        if self.deep_research_enabled and len(candidates) >= 5:
            return self._decide_deep(portfolio, candidates, horizon_days)
        _emit_runtime_event(
            self.event_sink,
            "SOL_RESEARCH_STARTED",
            self.stage,
            f"Researching {len(candidates)} candidates",
            metadata={"candidate_symbols": candidates, "model": self.model, "horizon_days": horizon_days},
        )
        prompt = (
            "You are Sol, the research and CIO decision maker. Luna only supplied candidate symbols; its ranking is not "
            "authoritative. Independently decide which candidates deserve research, call the raw research tools as needed, "
            "and then choose BUY, HOLD, SWITCH, or CASH. You alone may produce the final decision. Do not produce orders.\n"
            f"Candidate symbols: {', '.join(candidates)}\n"
            f"Luna screening context: {json.dumps(screening.model_dump(mode='json') if screening else {}, ensure_ascii=False)}\n"
            f"Decision horizon: {horizon_days} trading days. Current portfolio JSON: {portfolio.model_dump_json()}\n"
            "Evaluate all finalists in one common cross-sectional context. For five or more candidates, return exactly five "
            "ranked candidates based only on this Sol evaluation, with rank 1 as the selected stock and rank 2 as runner-up. "
            "Do not reuse Luna scores as Sol alpha_score. Mark unavailable comparison evidence UNKNOWN and never infer it. "
            "Explain specifically why ranks 2 through 5 were not selected, why rank 1 beats rank 2, and provide bull/base/bear "
            "cases, thesis invalidation conditions, confidence basis/reducers, and expected-alpha basis. Keep thesis to 2-4 "
            "concise statements without repeating why_selected. expected_excess_vs_spy and expected_excess_vs_qqq are Sol model "
            "estimates, not statistical regression alpha; set estimate_type to SOL_MODEL_ESTIMATE. Return JSON matching the Sol CIO schema."
        )
        try:
            response = _request_with_runtime_events(
                self.provider,
                self.model,
                self.event_sink,
                component=self.stage,
                started_event="SOL_RESEARCH_REQUEST_STARTED",
                failed_event="SOL_RESEARCH_REQUEST_FAILED",
                instructions=self._instructions(),
                input=prompt,
                tools=self.tools,
                text={"format": self.output_schema()},
            )
            _accumulate_usage(totals, _field(response, "usage"))
            for _ in range(self.max_tool_rounds + 1):
                calls = _function_calls(response)
                if not calls:
                    intent = self._parse_or_repair(response, totals)
                    decision_payload = {
                        **self.last_decision_analysis.model_dump(mode="json"),
                        "expected_alpha_vs_spy": intent.expected_alpha_vs_spy,
                        "expected_alpha_vs_qqq": intent.expected_alpha_vs_qqq,
                        "timestamp": intent.timestamp,
                        "model_name": intent.model_name,
                        "decision_id": intent.decision_id,
                    }
                    self.last_usage = {**totals, "latency_ms": (time.perf_counter() - started) * 1000, "estimated_cost": _estimate_cost(self.provider, totals, self.model), "model": self.model, "stage": self.stage}
                    _emit_runtime_event(
                        self.event_sink,
                        "SOL_DECISION_COMPLETED",
                        self.stage,
                        f"{intent.action} {intent.symbol or 'CASH'}",
                        symbol=intent.symbol,
                        metadata={**decision_payload, "decision": decision_payload, "usage": self.last_usage, "model": self.model},
                    )
                    return intent
                outputs = []
                for call in calls:
                    name = str(_field(call, "name", ""))
                    raw_arguments = _field(call, "arguments", "{}") or "{}"
                    try:
                        args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                        if not isinstance(args, dict):
                            raise ValueError("tool arguments must be an object")
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        _emit_runtime_event(self.event_sink, "SOL_TOOL_RESULT", self.stage, f"{name} failed", symbol=raw_arguments if isinstance(raw_arguments, str) else None, metadata={"tool": name, "status": "ERROR", "error": str(exc)})
                        raise ValueError(f"Invalid Sol tool call: {name}") from exc
                    symbol = args.get("symbol") if isinstance(args.get("symbol"), str) else None
                    _emit_runtime_event(self.event_sink, "SOL_TOOL_CALL", self.stage, f"Calling {name}", symbol=symbol, metadata={"tool": name, "arguments": args})
                    try:
                        result = self._dispatch(name, args)
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        _emit_runtime_event(self.event_sink, "SOL_TOOL_RESULT", self.stage, f"{name} failed", symbol=symbol, metadata={"tool": name, "status": "ERROR", "error": str(exc)})
                        raise ValueError(f"Invalid Sol tool call: {name}") from exc
                    call_record = {"name": name, "arguments": args, "call_id": _field(call, "call_id")}
                    self.last_tool_calls.append(call_record)
                    self.last_research_evidence.append({"stage": self.stage, "tool": name, "arguments": args, "result": result})
                    _emit_runtime_event(self.event_sink, "SOL_TOOL_RESULT", self.stage, f"{name} succeeded", symbol=symbol, metadata={"tool": name, "status": "SUCCESS", "result": result})
                    outputs.append({"type": "function_call_output", "call_id": _field(call, "call_id"), "output": json.dumps(result, ensure_ascii=False, default=str)})
                response = _request_with_runtime_events(
                    self.provider,
                    self.model,
                    self.event_sink,
                    component=self.stage,
                    started_event="SOL_TOOL_CONTINUATION_REQUEST_STARTED",
                    failed_event="SOL_TOOL_CONTINUATION_REQUEST_FAILED",
                    previous_response_id=_field(response, "id"),
                    instructions=self._instructions(),
                    input=outputs,
                    tools=self.tools,
                    text={"format": self.output_schema()},
                )
                _accumulate_usage(totals, _field(response, "usage"))
            raise RuntimeError("Sol exceeded maximum tool rounds")
        except Exception as exc:
            _emit_runtime_event(self.event_sink, "SOL_RESEARCH_ERROR", self.stage, str(exc), metadata={"error_type": exc.__class__.__name__})
            raise

    def decide_from_persisted_ranking_evidence(
        self,
        portfolio: PortfolioState,
        candidate_symbols: list[str],
        evidence_rows: list[dict[str, Any]],
        *,
        horizon_days: int = 20,
        source_run_id: str,
    ) -> TradeIntent:
        """Resume Deep Research after a failed Stage A request without rerunning Luna or its research tools."""
        candidates = list(dict.fromkeys(str(symbol).upper() for symbol in candidate_symbols if symbol))
        if len(candidates) < 5:
            raise ValueError("Persisted Sol resume requires at least five Final Candidates")
        required_tools = ("get_price_history", "get_fundamentals", "get_analyst_revisions", "get_upcoming_events")
        persisted: dict[tuple[str, str], dict[str, Any]] = {}
        for row in evidence_rows:
            tool = str(row.get("tool") or "")
            arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            symbol = str(arguments.get("symbol") or row.get("symbol") or "").upper()
            if symbol not in candidates or tool not in required_tools:
                continue
            key = (symbol, tool)
            if key in persisted:
                raise ValueError(f"Persisted Sol evidence contains duplicate {tool} for {symbol}")
            if "result" not in row:
                raise ValueError(f"Persisted Sol evidence has no result for {tool} {symbol}")
            persisted[key] = row
        missing = [f"{symbol}:{tool}" for symbol in candidates for tool in required_tools if (symbol, tool) not in persisted]
        if missing:
            raise ValueError(f"Persisted Sol ranking evidence is incomplete: {', '.join(missing[:10])}")

        original_ranking_evidence = self._ranking_evidence

        def replay_ranking_evidence(symbol: str) -> dict[str, Any]:
            normalized = symbol.upper()
            result = {"symbol": normalized}
            for tool in required_tools:
                row = persisted[(normalized, tool)]
                arguments = {"symbol": normalized}
                self.last_tool_calls.append({"name": tool, "arguments": arguments, "reused_from_run_id": source_run_id})
                self.last_research_evidence.append({"stage": self.stage, "tool": tool, "arguments": arguments, "result": row["result"], "reused_from_run_id": source_run_id})
                _emit_runtime_event(
                    self.event_sink,
                    "SOL_TOOL_RESULT_REUSED",
                    self.stage,
                    f"Reused {tool} from prior run",
                    symbol=normalized,
                    metadata={"tool": tool, "status": "REUSED", "source_run_id": source_run_id},
                )
                result[tool.removeprefix("get_")] = row["result"]
            return result

        _emit_runtime_event(
            self.event_sink,
            "SOL_RESEARCH_RESUMED",
            self.stage,
            f"Resuming Sol decision from {len(persisted)} persisted tool results",
            metadata={"source_run_id": source_run_id, "candidate_symbols": candidates, "evidence_count": len(persisted)},
        )
        self._ranking_evidence = replay_ranking_evidence
        try:
            return self._decide_deep(portfolio, candidates, horizon_days)
        finally:
            self._ranking_evidence = original_ranking_evidence

    def _decide_deep(self, portfolio: PortfolioState, candidates: list[str], horizon_days: int) -> TradeIntent:
        started = time.perf_counter()
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        self.last_tool_calls = []
        self.last_research_evidence = []
        _emit_runtime_event(
            self.event_sink,
            "SOL_RESEARCH_STARTED",
            self.stage,
            f"Deep research across {len(candidates)} finalists",
            metadata={"candidate_symbols": candidates, "model": self.model, "horizon_days": horizon_days, "pipeline": "DEEP_RESEARCH_IC"},
        )
        coverage = ResearchToolCoverage(
            available_sources=[
                "price history", "stock snapshot", "fundamentals", "earnings", "analyst revisions",
                "news", "SEC filings", "upcoming events", "SPY/QQQ benchmarks", "market regime",
            ],
            missing_sources=["Historical consensus valuation series"],
        )
        external_verification = getattr(self, "external_verification", {}) or {}
        external_verification_json = json.dumps(external_verification, ensure_ascii=False, default=str)
        external_verification_note = (
            " VERIFIED_EXTERNAL_EVIDENCE_JSON contains independently retrieved evidence. Use only entries with status "
            "VERIFIED as facts; entries marked CONFLICT or UNAVAILABLE remain unknown. Cite the evidence by describing "
            "the source and reporting period, not by inventing a stage or source.\n"
            f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{external_verification_json}\n"
            if external_verification else
            " No independent external verification packet was supplied; keep unavailable fields UNKNOWN.\n"
        )
        stage_a_evidence = [self._ranking_evidence(symbol) for symbol in candidates]
        ranking = self._structured_deep_stage(
            CrossSectionalRanking,
            "sol_cross_sectional_ranking",
            "SOL_STAGE_A_RANKING",
            (
                "Rank all supplied finalists in one common cross-sectional context and return exactly Top 5. Compare momentum, "
                "relative strength, earnings/revenue/EPS quality, valuation, revisions, catalysts, event risk, volatility, "
                "liquidity, sector strength, regime fit, downside asymmetry, and alpha potential. Mark unavailable facts UNKNOWN. "
                "preliminary_alpha_score is comparable only within this run. Do not make a trade decision.\n"
                f"CANDIDATE_EVIDENCE_JSON:\n{json.dumps(stage_a_evidence, ensure_ascii=False, default=str)}\n"
                f"KNOWN_MISSING_SOURCES:\n{json.dumps(coverage.missing_sources, ensure_ascii=False)}\n"
                f"{external_verification_note}"
            ),
            totals,
            metadata={"candidate_count": len(candidates)},
        )
        supplied = set(candidates)
        if any(item.symbol.upper() not in supplied for item in ranking.ranking):
            raise ValueError("Stage A returned a symbol outside Final Candidates")
        top_symbols = [item.symbol.upper() for item in ranking.ranking[:3]]
        if set(top_symbols) & {"MPC", "VLO", "PSX"}:
            coverage = coverage.model_copy(update={
                "missing_sources": [
                    *coverage.missing_sources,
                    "Reliable current crack spread series", "Refinery utilization", "Refinery outage feed",
                    "Crude differentials", "Refined-product inventories", "Realtime commodity and product prices",
                    "Shipping disruption data", "Geopolitical causal attribution dataset",
                ]
            })
        deep_dives: list[DeepDiveResearch] = []
        for index, symbol in enumerate(top_symbols, start=1):
            evidence = self._deep_evidence(symbol, candidates)
            research = self._structured_deep_stage(
                DeepDiveResearch,
                "sol_deep_dive",
                "SOL_DEEP_DIVE",
                (
                    "Perform causal investment research for this one Top3 finalist. Momentum is an outcome, not a cause. "
                    "Attribute company alpha versus industry/sector/market beta, test at least three alternative hypotheses, "
                    "explain why now, market expectations, priced-in level, remaining alpha, and observable invalidation conditions. "
                    "Include distinct bull, base, and bear scenarios with explicit assumptions and directional outcomes. "
                    "Never invent missing current data; write DATA_UNAVAILABLE and list required data. For MPC, directly test oil, "
                    "crack spreads, peers VLO/PSX, energy beta, geopolitical/war hypotheses, earnings revisions, valuation, and the "
                    "20-day persistence question without presuming war is causal. Return compact structured evidence, not an essay.\n"
                    f"SYMBOL: {symbol}\nRESEARCH_EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False, default=str)}\n"
                    f"KNOWN_MISSING_SOURCES:\n{json.dumps(coverage.missing_sources, ensure_ascii=False)}\n"
                    f"{external_verification_note}"
                ),
                totals,
                symbol=symbol,
                metadata={"deep_dive_index": index, "deep_dive_total": 3},
            )
            if research.symbol.upper() != symbol:
                raise ValueError(f"Deep Dive returned {research.symbol}, expected {symbol}")
            deep_dives.append(research)
        adversarial = self._structured_deep_stage(
            AdversarialReview,
            "sol_adversarial_review",
            "SOL_STAGE_C_ADVERSARIAL",
            (
                "Act as the investment committee stress reviewer without presuming acceptance or rejection. Identify the strongest "
                "bear case, thesis/data/priced-in/momentum/mean-reversion/earnings/macro/sector/geopolitical/valuation failure modes, "
                "whether the positive evidence outweighs them, whether a better alternative exists, and what evidence changes your mind. "
                "Do not make the final trade decision.\n"
                f"TOP3_RESEARCH_JSON:\n{json.dumps([item.model_dump(mode='json') for item in deep_dives], ensure_ascii=False)}"
                f"\n{external_verification_note}"
            ),
            totals,
            metadata={"symbols": top_symbols},
        )
        if {item.symbol.upper() for item in adversarial.reviews} != set(top_symbols):
            raise ValueError("Adversarial review must cover exactly the Top3 symbols")
        final = self._structured_deep_stage(
            InvestmentCommitteeDecision,
            "sol_investment_committee_decision",
            "SOL_STAGE_D_IC_DECISION",
            (
                "Make the final investment committee decision only now. Assume the portfolio starts in cash unless current portfolio "
                "state says otherwise. Cash is valid for an explicitly negative base case. Compare #1 directly with #2 and #3. "
                "Unavailable data is uncertainty, not negative evidence: describe its confidence impact, not an automatic "
                "weight haircut or cash decision. Do not repeatedly penalize unchanged missing evidence. Past gains alone "
                "are not a sell signal. In sizing_audit explain the precise target, new negatives versus unchanged unknowns, "
                "downside scenario and incremental rationale. If no configured budget or prior decision is supplied, mark "
                "that basis UNKNOWN rather than inventing a limit or evidence change. Ranking/confidence are not calibrated "
                "probabilities; no Kelly sizing from confidence. sizing_audit.evidence_refs must contain only exact IDs "
                "STAGE_A, STAGE_B, STAGE_C or TOOL_COVERAGE (for example [\"STAGE_B\", \"STAGE_C\"]). "
                "STAGE_A_JSON is supplied under ID STAGE_A; likewise STAGE_B_JSON, STAGE_C_JSON and TOOL_COVERAGE_JSON. "
                "Do not append _JSON, symbols, claim descriptions or paths to these IDs. Select only when causal evidence, "
                "remaining alpha, downside asymmetry, and catalyst clarity justify concentration. Expected alpha may be null when "
                "evidence is insufficient and must be labeled SOL_MODEL_ESTIMATE when supplied. Do not output an order or price.\n"
                f"PRIOR_SIZING_CONTEXT_JSON: {json.dumps(getattr(self, 'sizing_context', {}), ensure_ascii=False, default=str)}\n"
                f"HORIZON_DAYS: {horizon_days}\nPORTFOLIO_JSON: {portfolio.model_dump_json()}\n"
                f"STAGE_A_JSON: {ranking.model_dump_json()}\n"
                f"STAGE_B_JSON: {json.dumps([item.model_dump(mode='json') for item in deep_dives], ensure_ascii=False)}\n"
                f"STAGE_C_JSON: {adversarial.model_dump_json()}\n"
                f"TOOL_COVERAGE_JSON: {coverage.model_dump_json()}\n"
                f"{external_verification_note}"
            ),
            totals,
            metadata={"symbols": top_symbols},
        )
        if final.selected_symbol and final.selected_symbol.upper() not in set(top_symbols):
            raise ValueError("Investment Committee selected a symbol outside Top3")
        if final.selected_symbol:
            selected = final.selected_symbol.upper()
            expected_rank = top_symbols.index(selected) + 1
            alternatives = {symbol for symbol in top_symbols if symbol != selected}
            comparisons = final.pairwise_comparisons
            if final.rank != expected_rank:
                raise ValueError("Investment Committee rank does not match Stage A")
            if any(item.selected_symbol.upper() != selected for item in comparisons):
                raise ValueError("Pairwise comparisons must use the selected symbol")
            if {item.alternative_symbol.upper() for item in comparisons} != alternatives:
                raise ValueError("Pairwise comparisons must cover the other Top3 symbols")
        elif (
            self.force_stock_selection
            and final.action == "HOLD_CASH"
            and final.base_case.expected_direction not in {"MODERATE_DOWNSIDE", "SEVERE_DOWNSIDE"}
        ):
            selected = top_symbols[0]
            runner_up = top_symbols[1]
            ranked = {item.symbol.upper(): item for item in ranking.ranking}
            deep_by_symbol = {item.symbol.upper(): item for item in deep_dives}
            selected_research = deep_by_symbol[selected]
            evidence_incomplete = bool(selected_research.missing_data or selected_research.required_data)
            selected_rank = ranked[selected]
            selected_score = selected_rank.preliminary_alpha_score
            runner_score = ranked[runner_up].preliminary_alpha_score
            if not evidence_incomplete and selected_score >= 85 and selected_rank.confidence >= 0.75:
                selection_tier, target_weight = "HIGH_CONVICTION", self.forced_selection_high_weight
            elif selected_score >= 75 and selected_rank.confidence >= 0.65:
                selection_tier = "DATA_LIMITED" if evidence_incomplete else "BALANCED"
                target_weight = self.forced_selection_data_limited_weight
            else:
                selection_tier, target_weight = "EXPLORATORY", self.forced_selection_exploratory_weight
            normalized_pairs = [
                item.model_copy(update={"selected_symbol": selected, "alternative_symbol": alternative})
                for item, alternative in zip(final.pairwise_comparisons, (runner_up, top_symbols[2]))
            ]
            selection_reasons = [
                f"Stage A ranked {selected} first with alpha score {selected_score:.1f}",
                f"Alpha-validation policy applied {selection_tier} sizing",
                *selected_rank.key_strengths,
            ][:6]
            final = final.model_copy(update={
                "action": "BUY",
                "selected_symbol": selected,
                "rank": 1,
                "runner_up": runner_up,
                "alpha_score": selected_score,
                "runner_up_score": runner_score,
                "alpha_gap": selected_score - runner_score,
                "confidence": selected_rank.confidence,
                "target_weight": target_weight,
                "pairwise_comparisons": normalized_pairs,
                "why_selected": selection_reasons,
            })
            _emit_runtime_event(
                self.event_sink,
                "SOL_FORCED_STOCK_SELECTION",
                self.stage,
                f"Selected {selected} at {target_weight:.0%} because CASH is disallowed by policy",
                symbol=selected,
                metadata={
                    "reason": "force_stock_selection",
                    "original_action": "HOLD_CASH",
                    "evidence_incomplete": evidence_incomplete,
                    "selection_tier": selection_tier,
                    "selected_symbol": selected,
                    "requested_weight": target_weight,
                },
            )
        result = SolDeepResearchResult(
            ranking=ranking,
            top_three=deep_dives,
            adversarial_review=adversarial,
            final_decision=final,
            tool_coverage=coverage,
        )
        self.last_deep_research = result
        self.last_usage = {
            **totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "estimated_cost": _estimate_cost(self.provider, totals, self.model),
            "model": self.model,
            "stage": self.stage,
        }
        intent = self._deep_trade_intent(final, portfolio)
        _emit_runtime_event(
            self.event_sink,
            "SOL_IC_DECISION_COMPLETED",
            self.stage,
            f"{intent.action} {intent.symbol or 'CASH'}",
            symbol=intent.symbol,
            metadata={"decision": final.model_dump(mode="json"), "deep_research": result.model_dump(mode="json"), "usage": self.last_usage, "model": self.model},
        )
        return intent

    def _structured_deep_stage(self, model_type, schema_name: str, event_prefix: str, prompt: str, totals: dict[str, int], *, symbol: str | None = None, metadata: dict[str, Any] | None = None):
        stage_started = time.perf_counter()
        usage_before = dict(totals)
        repair_count = 0
        schema = {"type": "json_schema", "name": schema_name, "strict": True, "schema": _strict_json_schema(model_type.model_json_schema())}
        _emit_runtime_event(self.event_sink, f"{event_prefix}_STARTED", self.stage, f"{event_prefix} started", symbol=symbol, metadata=metadata or {})
        response = _request_with_runtime_events(
            self.provider,
            self.model,
            self.event_sink,
            component=self.stage,
            started_event=f"{event_prefix}_REQUEST_STARTED",
            failed_event=f"{event_prefix}_REQUEST_FAILED",
            instructions="You are Sol. Return auditable structured investment research only; never reveal private chain-of-thought and never create orders.",
            input=prompt,
            text={"format": schema},
        )
        _accumulate_usage(totals, _field(response, "usage"))
        last_error: Exception | None = None
        for attempt in range(self.max_repair_retries + 1):
            try:
                result = model_type.model_validate_json(_response_text(response))
                metric = {
                    "stage": event_prefix,
                    "symbol": symbol,
                    "latency_ms": (time.perf_counter() - stage_started) * 1000,
                    "input_tokens": totals["input_tokens"] - usage_before["input_tokens"],
                    "output_tokens": totals["output_tokens"] - usage_before["output_tokens"],
                    "cached_tokens": totals["cached_tokens"] - usage_before["cached_tokens"],
                    "reasoning_tokens": totals["reasoning_tokens"] - usage_before["reasoning_tokens"],
                    "repair_count": repair_count,
                    "status": "PASS" if repair_count == 0 else "REPAIRED",
                }
                self.last_stage_metrics.append(metric)
                _emit_runtime_event(self.event_sink, f"{event_prefix}_COMPLETED", self.stage, f"{event_prefix} completed", symbol=symbol, metadata={**(metadata or {}), "result": result.model_dump(mode="json"), "metrics": metric})
                return result
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_retries:
                    break
                repair_count += 1
                self.structured_output_status = "FALLBACK"
                response = _request_with_runtime_events(
                    self.provider,
                    self.model,
                    self.event_sink,
                    component=self.stage,
                    started_event=f"{event_prefix}_REPAIR_STARTED",
                    failed_event=f"{event_prefix}_REPAIR_FAILED",
                    input=(
                        "Repair the prior response. Return only JSON matching the complete schema. Do not add unsupported facts.\n"
                        + (
                            "sizing_audit.evidence_refs must contain only exact IDs STAGE_A, STAGE_B, STAGE_C, TOOL_COVERAGE. "
                            "Use only stages that actually support the claim; do not substitute an arbitrary allowed ID. "
                            "The supplied research context follows so you can verify references.\n"
                            f"ORIGINAL_RESEARCH_CONTEXT:\n{prompt}\n"
                            if model_type is InvestmentCommitteeDecision else ""
                        )
                        +
                        f"ORIGINAL_JSON:\n{_response_text(response)}\nVALIDATION_ERROR:\n{exc}\nCOMPLETE_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                    text={"format": schema},
                )
                _accumulate_usage(totals, _field(response, "usage"))
        raise ValueError(f"{event_prefix} returned invalid structured output: {last_error}") from last_error

    def _record_research_call(self, name: str, symbol: str | None, callback):
        arguments = {"symbol": symbol} if symbol else {}
        _emit_runtime_event(self.event_sink, "SOL_TOOL_CALL", self.stage, f"Calling {name}", symbol=symbol, metadata={"tool": name, "arguments": arguments})
        try:
            result = callback()
            status = "SUCCESS"
        except Exception as exc:
            result = {"status": "DATA_UNAVAILABLE", "error": str(exc)}
            status = "DATA_UNAVAILABLE"
        self.last_tool_calls.append({"name": name, "arguments": arguments})
        self.last_research_evidence.append({"stage": self.stage, "tool": name, "arguments": arguments, "result": result})
        _emit_runtime_event(self.event_sink, "SOL_TOOL_RESULT", self.stage, f"{name} {status.lower()}", symbol=symbol, metadata={"tool": name, "status": status, "result": result})
        return result

    def _ranking_evidence(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "price_history": self._record_research_call("get_price_history", symbol, lambda: self.data.price_history(symbol, 120)),
            "fundamentals": self._record_research_call("get_fundamentals", symbol, lambda: self.data.fundamentals(symbol)),
            "analyst_revisions": self._record_research_call("get_analyst_revisions", symbol, lambda: self.data.analyst_revisions(symbol)),
            "upcoming_events": self._record_research_call("get_upcoming_events", symbol, lambda: self.data.upcoming_events(symbol)),
        }

    def _deep_evidence(self, symbol: str, finalists: list[str]) -> dict[str, Any]:
        evidence = {
            "symbol": symbol,
            "stock_snapshot": self._record_research_call("get_stock_snapshot", symbol, lambda: self.data.stock_snapshot(symbol)),
            "price_history": self._record_research_call("get_price_history", symbol, lambda: self.data.price_history(symbol, 252)),
            "fundamentals": self._record_research_call("get_fundamentals", symbol, lambda: self.data.fundamentals(symbol)),
            "earnings": self._record_research_call("get_earnings", symbol, lambda: self.data.earnings(symbol)),
            "analyst_revisions": self._record_research_call("get_analyst_revisions", symbol, lambda: self.data.analyst_revisions(symbol)),
            "news": self._record_research_call("get_news", symbol, lambda: self.data.news(symbol, 10)),
            "sec_filings": self._record_research_call("get_sec_filings", symbol, lambda: self.data.sec_filings(symbol, 10)),
            "upcoming_events": self._record_research_call("get_upcoming_events", symbol, lambda: self.data.upcoming_events(symbol)),
            "market_regime": self._record_research_call("get_market_regime", None, self.data.market_regime),
            "benchmarks": self._record_research_call("get_benchmark_data", None, lambda: self.data.benchmark_data(252)),
        }
        if symbol == "MPC":
            peers = [peer for peer in ("VLO", "PSX") if peer in set(finalists)]
            evidence["refining_peers"] = {
                peer: self._record_research_call("get_price_history", peer, lambda peer=peer: self.data.price_history(peer, 120))
                for peer in peers
            }
            evidence["energy_sector"] = self._record_research_call("get_price_history", "XLE", lambda: self.data.price_history("XLE", 120))
        return evidence

    def _deep_trade_intent(self, final: InvestmentCommitteeDecision, portfolio: PortfolioState) -> TradeIntent:
        current = portfolio.current_symbol
        if final.action in {"HOLD_CASH", "EXIT_TO_CASH", "SELL"}:
            action, symbol, current_symbol, new_symbol, weight = "CASH", None, None, None, 0.0
        elif final.action == "REPLACE":
            action, symbol, current_symbol, new_symbol, weight = "SWITCH", final.selected_symbol, current, final.selected_symbol, final.target_weight
        elif final.action == "REDUCE":
            action, symbol, current_symbol, new_symbol, weight = "HOLD", current or final.selected_symbol, None, None, final.target_weight
        elif final.action == "HOLD":
            action, symbol, current_symbol, new_symbol, weight = "HOLD", current or final.selected_symbol, None, None, final.target_weight
        else:
            action, symbol, current_symbol, new_symbol, weight = "BUY", final.selected_symbol, None, None, final.target_weight
        thesis = list(dict.fromkeys([*final.why_selected, *final.primary_alpha_drivers]))[:6]
        return TradeIntent(
            action=action,
            symbol=symbol,
            current_symbol=current_symbol,
            new_symbol=new_symbol,
            target_weight=weight,
            confidence=final.confidence,
            holding_period_days=final.thesis_horizon_days,
            expected_alpha_vs_spy=final.expected_alpha_vs_spy or 0.0,
            expected_alpha_vs_qqq=final.expected_alpha_vs_qqq or 0.0,
            thesis=thesis,
            risk_factors=final.risk_factors[:6],
            invalidation_conditions=final.thesis_invalidation_conditions[:6],
            evidence_used=final.evidence_used[:12],
            model_name=self.model,
        )

    def _parse_or_repair(self, response: Any, totals: dict[str, int]) -> TradeIntent:
        current = response
        last_error: Exception | None = None
        for attempt in range(self.max_repair_retries + 1):
            try:
                payload = json.loads(_response_text(current))
                if isinstance(payload, dict):
                    if "expected_excess_vs_spy" not in payload and "expected_alpha_vs_spy" in payload:
                        payload["expected_excess_vs_spy"] = payload.pop("expected_alpha_vs_spy")
                    if "expected_excess_vs_qqq" not in payload and "expected_alpha_vs_qqq" in payload:
                        payload["expected_excess_vs_qqq"] = payload.pop("expected_alpha_vs_qqq")
                    if "thesis_invalidation_conditions" not in payload and "invalidation_conditions" in payload:
                        payload["thesis_invalidation_conditions"] = payload.pop("invalidation_conditions")
                analysis = SolDecisionPayload.model_validate(payload)
                analysis.validate_comparative_context(self._candidate_symbols)
                result = analysis.to_trade_intent(self.model)
                self.last_decision_analysis = analysis
                self.structured_output_status = "FALLBACK" if attempt else "PASS"
                return result
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_retries:
                    self.structured_output_status = "FAIL"
                    break
                self.structured_output_status = "FALLBACK"
                current = _request_with_runtime_events(
                    self.provider,
                    self.model,
                    self.event_sink,
                    component=self.stage,
                    started_event="SOL_DECISION_REPAIR_REQUEST_STARTED",
                    failed_event="SOL_DECISION_REPAIR_REQUEST_FAILED",
                    input=(
                        "Repair the prior response into valid JSON matching the complete Sol CIO decision schema. "
                        "Return only the repaired JSON. Do not create an order.\n"
                        "ORIGINAL_JSON:\n" + _response_text(current) + "\n"
                        "VALIDATION_ERROR:\n" + str(exc) + "\n"
                        "COMPLETE_SCHEMA:\n" + json.dumps(self.output_schema(), ensure_ascii=False)
                    ),
                    text={"format": self.output_schema()},
                )
                _accumulate_usage(totals, _field(current, "usage"))
        raise ValueError(f"Sol returned invalid structured decision output; no TradeIntent was created: {last_error}") from last_error

    def _dispatch(self, name: str, args: dict[str, Any]):
        if name == "get_market_regime": return self.data.market_regime()
        if name == "get_price_history": return self.data.price_history(args["symbol"], args["days"])
        if name == "get_stock_snapshot": return self.data.stock_snapshot(args["symbol"])
        if name == "get_fundamentals": return self.data.fundamentals(args["symbol"])
        if name == "get_earnings": return self.data.earnings(args["symbol"])
        if name == "get_analyst_revisions": return self.data.analyst_revisions(args["symbol"])
        if name == "get_news": return self.data.news(args["symbol"], args["limit"])
        if name == "get_sec_filings": return self.data.sec_filings(args["symbol"], args["limit"])
        if name == "get_upcoming_events": return self.data.upcoming_events(args["symbol"])
        if name == "get_portfolio": return self.data.portfolio()
        if name == "get_current_positions": return self.data.current_positions()
        if name == "get_benchmark_data": return self.data.benchmark_data(args["days"])
        raise KeyError(name)

    @staticmethod
    def _instructions() -> str:
        prompt_files = ("system_prompt.md", "research_prompt.md", "portfolio_manager_prompt.md", "risk_review_prompt.md")
        return "\n\n".join((ROOT / "prompts" / filename).read_text(encoding="utf-8") for filename in prompt_files)


def _estimate_cost(provider: Any, totals: dict[str, int], model: str) -> float | None:
    estimate = getattr(provider, "estimate_cost", None)
    if not estimate:
        return None
    try:
        return estimate(totals["input_tokens"], totals["output_tokens"], model=model)
    except TypeError:
        return estimate(totals["input_tokens"], totals["output_tokens"])


class PipelineError(RuntimeError):
    pass


class LunaSolPipeline:
    """Deterministic Python orchestration for LUNA_SOL and SOL_ONLY."""

    def __init__(
        self,
        data: DataProvider,
        provider: LLMProvider,
        runtime: LLMRuntimeConfig | None = None,
        *,
        max_tool_rounds: int = 12,
        max_candidates: int = 25,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.data = data
        self.runtime = runtime or LLMRuntimeConfig.from_env()
        self.provider = provider
        self.event_sink = event_sink
        if hasattr(provider, "set_event_sink"):
            provider.set_event_sink(event_sink)
        self.luna = LunaScreeningAgent(self.runtime.luna_model, data, provider=provider, max_candidates=max_candidates, event_sink=event_sink)
        self.sol = SolResearchCIOAgent(
            self.runtime.sol_model,
            data,
            provider=provider,
            max_tool_rounds=max_tool_rounds,
            deep_research_enabled=self.runtime.deep_research_enabled,
            force_stock_selection=self.runtime.force_stock_selection,
            forced_selection_high_weight=self.runtime.forced_selection_high_weight,
            forced_selection_data_limited_weight=self.runtime.forced_selection_data_limited_weight,
            forced_selection_exploratory_weight=self.runtime.forced_selection_exploratory_weight,
            event_sink=event_sink,
        )
        self.last_usage: dict[str, Any] = {}
        self.last_research_evidence: list[dict[str, Any]] = []
        self.last_pipeline_metadata: dict[str, Any] = {}
        self.last_fallback_events: list[dict[str, Any]] = []
        self.last_runtime_exact_probe: dict[str, Any] = {}
        self.last_luna_batches: list[dict[str, Any]] = []
        self.last_luna_final_screening: dict[str, Any] = {}
        self.model = self.runtime.sol_model
        self.prompt_version = "luna-sol-v1"

    def set_event_sink(self, event_sink: Callable[[dict[str, Any]], None] | None):
        self.event_sink = event_sink
        self.luna.set_event_sink(event_sink)
        self.sol.set_event_sink(event_sink)
        if hasattr(self.provider, "set_event_sink"):
            self.provider.set_event_sink(event_sink)

    def probe_runtime_llm_exact_path(self, universe_snapshot: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Exercise the same Luna agent/provider path used by a production run."""
        started = time.perf_counter()
        rows = _compact_universe(universe_snapshot if universe_snapshot is not None else self.data.universe_snapshot())[:1]
        if not rows:
            result = {"status": "FAIL", "ok": False, "model": self.runtime.luna_model, "error": "probe universe is empty"}
            self.last_runtime_exact_probe = result
            return result
        _emit_runtime_event(
            self.event_sink,
            "LLM_RUNTIME_EXACT_PROBE_STARTED",
            "LLM",
            "Runtime exact Luna path probe started",
            metadata={
                "model": self.runtime.luna_model,
                "protocol": getattr(self.provider, "protocol", self.runtime.api_protocol),
                "universe_size": len(rows),
            },
        )
        try:
            screening = self.luna.screen(rows)
            result = {
                "status": "PASS",
                "ok": True,
                "model": self.runtime.luna_model,
                "protocol": getattr(self.provider, "protocol", self.runtime.api_protocol),
                "candidate_symbols": screening.candidate_symbols,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
            _emit_runtime_event(
                self.event_sink,
                "LLM_RUNTIME_EXACT_PROBE_COMPLETED",
                "LLM",
                "Runtime exact Luna path probe passed",
                metadata=result,
            )
        except Exception as exc:
            result = {
                "status": "FAIL",
                "ok": False,
                "model": self.runtime.luna_model,
                "protocol": getattr(self.provider, "protocol", self.runtime.api_protocol),
                "error": getattr(self.provider, "_safe_error", lambda error: str(error))(exc),
                "diagnostics": _provider_error_metadata(self.provider, exc),
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
            _emit_runtime_event(
                self.event_sink,
                "LLM_RUNTIME_EXACT_PROBE_FAILED",
                "LLM",
                "Runtime exact Luna path probe failed",
                metadata=result,
            )
        self.last_runtime_exact_probe = result
        return result

    def decide(self, portfolio: PortfolioState, horizon_days: int = 20) -> TradeIntent:
        started = time.perf_counter()
        self.last_research_evidence = []
        self.last_fallback_events = []
        self.last_pipeline_metadata = {}
        self.last_luna_batches = []
        self.last_luna_final_screening = {}
        compact = _compact_universe(self.data.universe_snapshot())
        if not compact:
            raise PipelineError("Universe snapshot is empty; no TradeIntent was created")
        screening: LunaScreeningResult | None = None
        configured_pipeline = self.runtime.pipeline
        actual_pipeline = configured_pipeline
        if configured_pipeline == "LUNA_SOL":
            try:
                screening = self._screen_batched(compact) if len(compact) >= self.runtime.luna_screen_batch_size else self.luna.screen(compact)
                candidates: list[str] | list[ScreeningCandidate] = screening.candidates
                self.last_research_evidence.append({"stage": "LUNA", "tool": "universe_snapshot", "arguments": {}, "result": compact})
                self.last_research_evidence.append({"stage": "LUNA", "tool": "luna_screening", "arguments": {}, "result": screening.model_dump(mode="json")})
            except Exception as exc:
                if isinstance(exc, ScreeningIncompleteError):
                    raise PipelineError(f"Luna screening incomplete: {exc}") from exc
                if not self.runtime.fallback_to_sol_only:
                    raise PipelineError(f"Luna screening failed: {exc}") from exc
                actual_pipeline = "SOL_ONLY"
                self.last_fallback_events.append(self._fallback_event(str(exc)))
                _emit_runtime_event(self.event_sink, "PIPELINE_FALLBACK", "PIPELINE", "Luna failed; falling back to Sol only", metadata=self.last_fallback_events[-1])
                candidates = [row["symbol"] for row in compact[:25]]
        else:
            candidates = [row["symbol"] for row in compact[:25]]
        try:
            self._progress("SOL_RESEARCH", 55, initial_candidates=len(candidates), sol_candidates_reviewed=0, sol_tool_calls=0, sol_tool_results=0)
            intent = self.sol.decide(portfolio, candidates, horizon_days=horizon_days, screening=screening)
            self._progress("SOL_DECISION", 92, initial_candidates=len(candidates), sol_tool_calls=len(self.sol.last_tool_calls), sol_tool_results=len(self.sol.last_research_evidence), sol_usage=self.sol.last_usage)
        except Exception as exc:
            raise PipelineError(f"Sol research/CIO failed: {exc}") from exc
        self.last_research_evidence.extend(self.sol.last_research_evidence)
        self.last_usage = self._aggregate_usage(started)
        self.last_pipeline_metadata = {
            "pipeline": actual_pipeline,
            "configured_pipeline": configured_pipeline,
            "gateway": getattr(self.provider, "gateway", "ccswitch"),
            "api_protocol": getattr(self.provider, "protocol", self.runtime.api_protocol),
            "protocol_fallback": "YES" if getattr(self.provider, "protocol_fallback", self.runtime.protocol_fallback) else "NO",
            "luna_model": self.runtime.luna_model,
            "sol_model": self.runtime.sol_model,
            "luna_reasoning_effort": self.runtime.luna_reasoning_effort,
            "sol_reasoning_effort": self.runtime.sol_reasoning_effort,
            "luna_candidates": screening.candidate_symbols if screening else [],
            "candidate_symbols": [item.symbol if isinstance(item, ScreeningCandidate) else item for item in candidates],
            "luna_rationale": screening.screening_rationale if screening else None,
            "sol_tool_calls": self.sol.last_tool_calls,
            "sol_research_evidence": self.sol.last_research_evidence,
            "sol_stage_a": self._deep_metadata("ranking"),
            "sol_top_three": self._deep_metadata("top_three", []),
            "sol_adversarial_review": self._deep_metadata("adversarial_review"),
            "tool_coverage": self._deep_metadata("tool_coverage"),
            "missing_data_sources": self._deep_missing_data(),
            "total_sol_tool_calls": len(self.sol.last_tool_calls),
            "sol_stage_metrics": list(self.sol.last_stage_metrics),
            "sol_final_decision": self._final_decision_metadata(intent),
            "final_decision": self._final_decision_metadata(intent),
            "confidence": intent.confidence,
            "expected_excess_vs_spy": intent.expected_alpha_vs_spy,
            "expected_excess_vs_qqq": intent.expected_alpha_vs_qqq,
            "prompt_versions": {"luna": self.luna.prompt_version, "sol": self.sol.prompt_version},
            "structured_output_status": "FALLBACK" if "FALLBACK" in {self.luna.structured_output_status, self.sol.structured_output_status} else "PASS",
            "tool_calling_status": "PASS",
            "luna_usage": self.luna.last_usage,
            "luna_batches": list(self.last_luna_batches),
            "luna_final_screening": dict(self.last_luna_final_screening),
            "sol_usage": self.sol.last_usage,
            "usage": {"luna": self.luna.last_usage, "sol": self.sol.last_usage},
            "fallback_events": list(self.last_fallback_events),
            "latency_ms": self.last_usage.get("latency_ms"),
            "total_decision_cost": self.last_usage.get("estimated_cost"),
        }
        return intent.model_copy(update={"model_name": self.runtime.sol_model})

    def resume_sol_decision(
        self,
        portfolio: PortfolioState,
        candidate_symbols: list[str],
        evidence_rows: list[dict[str, Any]],
        *,
        source_run_id: str,
        horizon_days: int = 20,
    ) -> TradeIntent:
        started = time.perf_counter()
        self.last_research_evidence = []
        self.last_fallback_events = []
        self.last_pipeline_metadata = {}
        try:
            intent = self.sol.decide_from_persisted_ranking_evidence(
                portfolio,
                candidate_symbols,
                evidence_rows,
                horizon_days=horizon_days,
                source_run_id=source_run_id,
            )
        except Exception as exc:
            raise PipelineError(f"Resumed Sol research/CIO failed: {exc}") from exc
        self.last_research_evidence.extend(self.sol.last_research_evidence)
        self.last_usage = self._aggregate_usage(started)
        self.last_pipeline_metadata = {
            "pipeline": "LUNA_SOL_RESUMED",
            "configured_pipeline": self.runtime.pipeline,
            "gateway": getattr(self.provider, "gateway", "ccswitch"),
            "api_protocol": getattr(self.provider, "protocol", self.runtime.api_protocol),
            "protocol_fallback": "YES" if getattr(self.provider, "protocol_fallback", self.runtime.protocol_fallback) else "NO",
            "luna_model": self.runtime.luna_model,
            "sol_model": self.runtime.sol_model,
            "luna_reasoning_effort": self.runtime.luna_reasoning_effort,
            "sol_reasoning_effort": self.runtime.sol_reasoning_effort,
            "luna_candidates": candidate_symbols,
            "candidate_symbols": candidate_symbols,
            "sol_tool_calls": self.sol.last_tool_calls,
            "sol_research_evidence": self.sol.last_research_evidence,
            "sol_stage_a": self._deep_metadata("ranking"),
            "sol_top_three": self._deep_metadata("top_three", []),
            "sol_adversarial_review": self._deep_metadata("adversarial_review"),
            "tool_coverage": self._deep_metadata("tool_coverage"),
            "missing_data_sources": self._deep_missing_data(),
            "total_sol_tool_calls": len(self.sol.last_tool_calls),
            "sol_stage_metrics": list(self.sol.last_stage_metrics),
            "sol_final_decision": self._final_decision_metadata(intent),
            "final_decision": self._final_decision_metadata(intent),
            "confidence": intent.confidence,
            "expected_excess_vs_spy": intent.expected_alpha_vs_spy,
            "expected_excess_vs_qqq": intent.expected_alpha_vs_qqq,
            "prompt_versions": {"luna": self.luna.prompt_version, "sol": self.sol.prompt_version},
            "structured_output_status": self.sol.structured_output_status if self.sol.structured_output_status in {"PASS", "FALLBACK"} else "PASS",
            "tool_calling_status": "PASS",
            "luna_usage": {},
            "sol_usage": self.sol.last_usage,
            "usage": {"sol": self.sol.last_usage},
            "fallback_events": [],
            "resume_source_run_id": source_run_id,
            "latency_ms": self.last_usage.get("latency_ms"),
            "total_decision_cost": self.last_usage.get("estimated_cost"),
        }
        return intent.model_copy(update={"model_name": self.runtime.sol_model})

    def _final_decision_metadata(self, intent: TradeIntent) -> dict[str, Any]:
        if self.sol.last_deep_research is not None:
            return {
                **self.sol.last_deep_research.final_decision.model_dump(mode="json"),
                "timestamp": intent.timestamp,
                "model_name": self.runtime.sol_model,
                "decision_id": intent.decision_id,
            }
        analysis = self.sol.last_decision_analysis
        if analysis is None:
            return intent.model_dump(mode="json")
        return {
            **analysis.model_dump(mode="json"),
            "expected_alpha_vs_spy": intent.expected_alpha_vs_spy,
            "expected_alpha_vs_qqq": intent.expected_alpha_vs_qqq,
            "timestamp": intent.timestamp,
            "model_name": self.runtime.sol_model,
            "decision_id": intent.decision_id,
        }

    def _deep_metadata(self, field: str, default: Any = None) -> Any:
        research = self.sol.last_deep_research
        if research is None:
            return default if default is not None else {}
        value = getattr(research, field)
        if isinstance(value, list):
            return [item.model_dump(mode="json") for item in value]
        return value.model_dump(mode="json")

    def _deep_missing_data(self) -> list[str]:
        research = self.sol.last_deep_research
        if research is None:
            return []
        values = [*research.ranking.missing_data_sources, *research.tool_coverage.missing_sources]
        for item in research.top_three:
            values.extend(item.missing_data)
            values.extend(item.required_data)
        values.extend(research.final_decision.missing_data)
        return list(dict.fromkeys(value for value in values if value))

    def _screen_batched(self, rows: list[dict[str, Any]]) -> LunaScreeningResult:
        batch_size = self.runtime.luna_screen_batch_size
        top_k = self.runtime.luna_screen_batch_top_k
        minimum = self.runtime.luna_min_batch_size
        final_top_k = self.runtime.luna_final_top_k
        final_max_candidates = self.runtime.luna_final_max_candidates
        if not (
            1 <= minimum <= batch_size
            and 1 <= top_k <= 25
            and 1 <= final_top_k <= final_max_candidates <= 25
        ):
            raise ValueError("Invalid Luna batch screening configuration")
        initial_batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
        covered: set[str] = set()
        merged: dict[str, ScreeningCandidate] = {}
        usages: list[dict[str, Any]] = []
        call_count = 0
        retry_calls = 0
        self._progress("LUNA_BATCH_SCREENING", 5, universe_total=len(rows), universe_processed=0, batch_index=0, batch_total=len(initial_batches), luna_batch_calls=0, luna_retry_calls=0, luna_final_screening_calls=0)

        def run_batch(batch_rows: list[dict[str, Any]], batch_index: int, batch_id: str, parent_batch_id: str | None = None, retry_count: int = 0):
            nonlocal call_count, retry_calls
            call_count += 1
            if retry_count:
                retry_calls += 1
            symbols = [row["symbol"] for row in batch_rows]
            context = {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "batch_total": len(initial_batches),
                "parent_batch_id": parent_batch_id,
                "batch_size": len(batch_rows),
                "symbols": symbols,
                "model": self.runtime.luna_model,
                "prompt_version": self.luna.prompt_version,
                "retry_count": retry_count,
                "status": "RUNNING",
            }
            try:
                result = self.luna.screen(batch_rows, max_candidates=top_k, context=context)
            except Exception as exc:
                if self._can_split_batch(exc) and len(batch_rows) >= minimum * 2:
                    midpoint = len(batch_rows) // 2
                    left, right = batch_rows[:midpoint], batch_rows[midpoint:]
                    if len(left) >= minimum and len(right) >= minimum:
                        _emit_runtime_event(self.event_sink, "LUNA_BATCH_SPLIT", "LUNA", f"Adaptive split: {len(batch_rows)} -> {len(left)} + {len(right)}", metadata={**context, "status": "SPLIT", "original_batch_size": len(batch_rows), "new_batch_size": len(left), "split_sizes": [len(left), len(right)], "reason": str(exc), "retry_count": retry_count + 1})
                        run_batch(left, batch_index, batch_id + ".1", batch_id, retry_count + 1)
                        run_batch(right, batch_index, batch_id + ".2", batch_id, retry_count + 1)
                        return
                self._progress(
                    "LUNA_BATCH_SCREENING",
                    5 + 40 * len(covered) / len(rows),
                    universe_total=len(rows),
                    universe_processed=len(covered),
                    batch_index=batch_index,
                    batch_total=len(initial_batches),
                    current_batch_size=len(batch_rows),
                    merged_candidates=len(merged),
                    batch_id=batch_id,
                    luna_batch_calls=call_count,
                    luna_retry_calls=retry_calls,
                    luna_final_screening_calls=0,
                    error_stage="LUNA_BATCH_SCREENING",
                    error_message=str(exc),
                )
                _emit_runtime_event(self.event_sink, "LUNA_BATCH_FAILED", "LUNA", str(exc), metadata={**context, "status": "FAILED", "error": str(exc)})
                raise ScreeningIncompleteError(f"batch {batch_id} failed at size {len(batch_rows)}: {exc}") from exc
            covered.update(symbols)
            usages.append(dict(self.luna.last_usage))
            for candidate in result.candidates:
                merged.setdefault(candidate.symbol, candidate)
            record = {**context, "status": "COMPLETE", "candidate_symbols": result.candidate_symbols, "latency": self.luna.last_usage.get("latency_ms"), "input_tokens": self.luna.last_usage.get("input_tokens"), "output_tokens": self.luna.last_usage.get("output_tokens"), "cost": self.luna.last_usage.get("estimated_cost")}
            self.last_luna_batches.append(record)
            processed = len(covered)
            usage_so_far = self._aggregate_stage_usage(usages, self.runtime.luna_model, "LUNA")
            self._progress("LUNA_BATCH_SCREENING", 5 + 40 * processed / len(rows), universe_total=len(rows), universe_processed=processed, batch_index=batch_index, batch_total=len(initial_batches), current_batch_size=len(batch_rows), batch_candidates=len(result.candidates), merged_candidates=len(merged), batch_id=batch_id, luna_batch_calls=call_count, luna_retry_calls=retry_calls, luna_final_screening_calls=0, luna_usage=usage_so_far)

        for index, batch_rows in enumerate(initial_batches, start=1):
            run_batch(batch_rows, index, f"batch-{index}")
        missing = sorted({row["symbol"] for row in rows} - covered)
        if missing:
            raise ScreeningIncompleteError(f"universe coverage incomplete: {len(missing)} symbol(s) missing")
        self._progress("LUNA_FINAL_SCREENING", 45, universe_total=len(rows), universe_processed=len(covered), batch_total=len(initial_batches), merged_candidates=len(merged), luna_batch_calls=call_count, luna_retry_calls=retry_calls, luna_final_screening_calls=0)
        final = self._screen_final_candidates(list(merged.values()), usages)
        self.luna.last_usage = self._aggregate_stage_usage(usages, self.runtime.luna_model, "LUNA")
        self._progress("LUNA_FINAL_SCREENING", 55, universe_total=len(rows), universe_processed=len(covered), batch_total=len(initial_batches), merged_candidates=len(merged), final_luna_candidates=len(final.candidates), covered=len(covered), missing=0, luna_batch_calls=call_count, luna_retry_calls=retry_calls, luna_final_screening_calls=self.last_luna_final_screening["call_count"], luna_usage=self.luna.last_usage)
        return final

    def _screen_final_candidates(
        self,
        candidates: list[ScreeningCandidate],
        usages: list[dict[str, Any]],
    ) -> LunaScreeningResult:
        final_top_k = self.runtime.luna_final_top_k
        final_max = self.runtime.luna_final_max_candidates
        if not (1 <= final_top_k <= final_max <= 25):
            raise ValueError("Invalid Luna final screening configuration")

        def screen_group(
            group: list[ScreeningCandidate],
            limit: int,
            *,
            stage: str,
            group_index: int | None = None,
            group_total: int | None = None,
        ) -> LunaScreeningResult:
            result = self.luna.screen(
                [candidate.model_dump(mode="json") for candidate in group],
                max_candidates=limit,
                context={
                    "final_screening": True,
                    "final_screening_stage": stage,
                    "final_group_index": group_index,
                    "final_group_total": group_total,
                    "input_candidate_count": len(group),
                    "status": "RUNNING",
                },
            )
            usages.append(dict(self.luna.last_usage))
            return result

        if len(candidates) <= final_max:
            final = screen_group(candidates, final_top_k, stage="GLOBAL_TIE_BREAK")
            self.last_luna_final_screening = {
                "input_candidate_count": len(candidates),
                "partition_sizes": [],
                "partition_keep_counts": [],
                "global_input_candidate_count": len(candidates),
                "call_count": 1,
            }
            return final

        group_count = (len(candidates) + final_max - 1) // final_max
        partitions = [candidates[index::group_count] for index in range(group_count)]
        keep_per_group = (final_max + group_count - 1) // group_count
        staged: dict[str, ScreeningCandidate] = {}
        partition_keep_counts: list[int] = []
        for index, partition in enumerate(partitions, start=1):
            result = screen_group(
                partition,
                min(keep_per_group, len(partition)),
                stage="PARTITION",
                group_index=index,
                group_total=group_count,
            )
            partition_keep_counts.append(len(result.candidates))
            for candidate in result.candidates:
                staged.setdefault(candidate.symbol, candidate)

        global_candidates = list(staged.values())[:final_max]
        final = screen_group(global_candidates, final_top_k, stage="GLOBAL_TIE_BREAK")
        self.last_luna_final_screening = {
            "input_candidate_count": len(candidates),
            "partition_sizes": [len(partition) for partition in partitions],
            "partition_keep_counts": partition_keep_counts,
            "global_input_candidate_count": len(global_candidates),
            "call_count": group_count + 1,
        }
        return final

    @staticmethod
    def _can_split_batch(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        category = str(getattr(exc, "category", "")).lower()
        message = str(exc).lower()
        if status in {401, 403, 429} or category in {"authentication", "rate_limit", "invalid_request"}:
            return False
        if status == 413 or any(marker in message for marker in ("payload too large", "request too large", "context length", "token limit")):
            return True
        return status in {500, 502, 503, 504} or category in {"upstream", "timeout"} or any(
            marker in message
            for marker in ("server overload", "connection reset", "client connection lost", "timeout awaiting response headers")
        )

    @staticmethod
    def _aggregate_stage_usage(items: list[dict[str, Any]], model: str, stage: str) -> dict[str, Any]:
        available = bool(items) and all(item.get("available", False) for item in items)
        result = {name: sum(int(item.get(name, 0) or 0) for item in items) if available else None for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")}
        result["latency_ms"] = sum(float(item.get("latency_ms", 0) or 0) for item in items)
        costs = [item.get("estimated_cost") for item in items]
        result["estimated_cost"] = None if not available or not costs or any(value is None for value in costs) else sum(float(value) for value in costs)
        result.update({"model": model, "stage": stage, "call_count": len(items), "available": available})
        return result

    def _progress(self, stage: str, progress_percent: float, **metadata):
        _emit_runtime_event(self.event_sink, "AI_PROGRESS", "PIPELINE", stage, metadata={"stage": stage, "progress_percent": round(float(progress_percent), 2), "updated_at": datetime.now(timezone.utc).isoformat(), **metadata})

    def _aggregate_usage(self, started: float) -> dict[str, Any]:
        stages = (self.luna.last_usage, self.sol.last_usage)
        totals = {name: sum(int(item.get(name, 0) or 0) for item in stages) for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")}
        totals["latency_ms"] = (time.perf_counter() - started) * 1000
        totals["estimated_cost"] = sum(float(item.get("estimated_cost") or 0) for item in stages) or None
        totals["model"] = self.runtime.sol_model
        totals["stage"] = "PIPELINE"
        return totals

    @staticmethod
    def _fallback_event(reason: str) -> dict[str, Any]:
        return {"event": "PIPELINE_FALLBACK", "from": "LUNA_SOL", "to": "SOL_ONLY", "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}


LLMPipeline = LunaSolPipeline


class OpenAIProvider(LLMProvider):
    """Legacy name retained for old callers; it still requires the CC Switch URL."""

    def __init__(self, model: str):
        runtime = LLMRuntimeConfig.from_env()
        self.model = model
        self._provider = CCSwitchProvider(runtime=runtime, model=model)

    def estimate_cost(self, input_tokens: int | None, output_tokens: int | None):
        return self._provider.estimate_cost(input_tokens, output_tokens, model=self.model)

    def create_response(self, **kwargs):
        return self._provider.create_response(model=self.model, **kwargs)


class LLMPortfolioManager:
    """Compatibility single-stage manager used by older tests and integrations."""

    def __init__(self, model: str, data: DataProvider, max_tool_rounds: int = 12, provider: LLMProvider | None = None, prompt_version: str = "v1"):
        if not model or model == "YOUR_TOOL_CAPABLE_MODEL":
            raise ValueError("Set a tool-capable LLM model available through the configured gateway.")
        self.model = model
        self.data = data
        self.max_tool_rounds = max_tool_rounds
        self.provider = provider or OpenAIProvider(model)
        self.prompt_version = prompt_version
        prompt_files = ("system_prompt.md", "research_prompt.md", "portfolio_manager_prompt.md", "risk_review_prompt.md")
        self.instructions = "\n\n".join((ROOT / "prompts" / filename).read_text(encoding="utf-8") for filename in prompt_files)
        self.last_research_evidence: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] = {}

    @property
    def tools(self):
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        symbol = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False}
        symbol_days = {"type": "object", "properties": {"symbol": {"type": "string"}, "days": {"type": "integer", "minimum": 20, "maximum": 1000}}, "required": ["symbol", "days"], "additionalProperties": False}
        symbol_limit = {"type": "object", "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["symbol", "limit"], "additionalProperties": False}
        return [
            self._tool("get_market_regime", "Get current broad-market regime data.", empty),
            self._tool("get_universe", "Get the current deduplicated S&P 500 and Nasdaq-100 universe snapshot for screening.", empty),
            self._tool("get_stock_snapshot", "Get current snapshot for one stock.", symbol),
            self._tool("get_price_history", "Get summarized price history and realized volatility for one stock.", symbol_days),
            self._tool("get_fundamentals", "Get current fundamental metrics for one stock.", symbol),
            self._tool("get_earnings", "Get the next earnings date and timing for one stock.", symbol),
            self._tool("get_analyst_revisions", "Get recent analyst estimate or rating revisions for one stock.", symbol),
            self._tool("get_news", "Get recent news for one stock.", symbol_limit),
            self._tool("get_sec_filings", "Get recent SEC filings for one stock.", symbol_limit),
            self._tool("get_portfolio", "Get the current portfolio state.", empty),
            self._tool("get_current_positions", "Get current positions.", empty),
            self._tool("get_benchmark_data", "Get current SPY and QQQ benchmark data.", {"type": "object", "properties": {"days": {"type": "integer", "minimum": 20, "maximum": 1000}}, "required": ["days"], "additionalProperties": False}),
            self._tool("get_upcoming_events", "Get earnings and known event risk for one stock.", symbol),
        ]

    def decide(self, portfolio: PortfolioState, horizon_days: int = 20) -> TradeIntent:
        self.last_research_evidence = []
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        started = time.perf_counter()
        prompt = "Make the next portfolio decision. Use tools to research the current portfolio and universe. " + f"Decision horizon: {horizon_days} trading days. Current portfolio JSON: {portfolio.model_dump_json()}. Return final JSON matching the required schema."
        response = self.provider.create_response(instructions=self.instructions, input=prompt, tools=self.tools, text={"format": self._trade_schema()})
        _accumulate_usage(totals, _field(response, "usage"))
        for _ in range(self.max_tool_rounds + 1):
            calls = _function_calls(response)
            if not calls:
                try:
                    decision = LLMDecision.model_validate_json(_response_text(response))
                except Exception as exc:
                    raise ValueError("LLM returned invalid TradeIntent JSON; execution is blocked") from exc
                self.last_usage = {**totals, "latency_ms": (time.perf_counter() - started) * 1000, "estimated_cost": _estimate_cost(self.provider, totals, self.model)}
                return decision.to_trade_intent(self.model)
            outputs = []
            for call in calls:
                try:
                    args = json.loads(_field(call, "arguments", "{}") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be an object")
                    result = self._dispatch(_field(call, "name"), args)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid tool call from LLM: {_field(call, 'name')}") from exc
                self.last_research_evidence.append({"tool": _field(call, "name"), "arguments": args, "result": result})
                outputs.append({"type": "function_call_output", "call_id": _field(call, "call_id"), "output": json.dumps(result, ensure_ascii=False, default=str)})
            response = self.provider.create_response(previous_response_id=_field(response, "id"), instructions=self.instructions, input=outputs, tools=self.tools, text={"format": self._trade_schema()})
            _accumulate_usage(totals, _field(response, "usage"))
        raise RuntimeError("Agent exceeded maximum tool rounds")

    def _dispatch(self, name: str, args: dict[str, Any]):
        if name == "get_market_regime": return self.data.market_regime()
        if name == "get_universe": return self.data.universe_snapshot()
        if name == "get_stock_snapshot": return self.data.stock_snapshot(args["symbol"])
        if name == "get_price_history": return self.data.price_history(args["symbol"], args["days"])
        if name == "get_fundamentals": return self.data.fundamentals(args["symbol"])
        if name == "get_earnings": return self.data.earnings(args["symbol"])
        if name == "get_analyst_revisions": return self.data.analyst_revisions(args["symbol"])
        if name == "get_news": return self.data.news(args["symbol"], args["limit"])
        if name == "get_sec_filings": return self.data.sec_filings(args["symbol"], args["limit"])
        if name == "get_portfolio": return self.data.portfolio()
        if name == "get_current_positions": return self.data.current_positions()
        if name == "get_benchmark_data": return self.data.benchmark_data(args["days"])
        if name == "get_upcoming_events": return self.data.upcoming_events(args["symbol"])
        raise KeyError(name)

    @staticmethod
    def _tool(name: str, description: str, parameters: dict[str, Any]):
        return {"type": "function", "name": name, "description": description, "parameters": parameters, "strict": True}

    @staticmethod
    def _trade_schema():
        return {"type": "json_schema", "name": "llm_decision", "strict": True, "schema": LLMDecision.model_json_schema()}
