"""FIRST_RUN_OBSERVE health checks and the non-technical diagnostic entrypoint."""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import env
from .setup_wizard import discover_paper_port, is_paper_account, read_dotenv_value, redact_sensitive, write_diagnostic_report


MARKET_DATA_STATES = frozenset({
    "LIVE",
    "FROZEN",
    "DELAYED",
    "DELAYED_FROZEN",
    "MARKET_CLOSED",
    "PERMISSION_DENIED",
    "UNAVAILABLE",
})


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    message: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "message": self.message}


@dataclass
class FirstRunResult:
    state: str
    checks: list[HealthCheck]
    user_message: str
    details: dict[str, Any]

    @property
    def ready(self) -> bool:
        return self.state == "READY_FOR_OBSERVE"

    def check(self, name: str) -> str | None:
        aliases = {"LLM API": "LLM Gateway"}
        requested = aliases.get(name, name)
        for item in self.checks:
            if item.name == requested or (name == "LLM API" and item.name == "LLM API"):
                return item.status
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "ready": self.ready,
            "user_message": self.user_message,
            "checks": [item.as_dict() for item in self.checks],
            "details": self.details,
        }


def _is_waiting_for_login(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    if "official ibkr" in message or "incompatible" in message:
        return False
    return any(
        marker in message
        for marker in (
            "socket refused",
            "connection refused",
            "nextvalidid timeout",
            "managedaccounts timeout",
            "not logged",
            "could not connect",
        )
    )


def is_waiting_for_tws(exc: Exception) -> bool:
    return _is_waiting_for_login(exc)


def _paper_account_check(broker) -> HealthCheck:
    raw_broker = getattr(broker, "_broker", broker)
    app = getattr(raw_broker, "app", None)
    accounts = getattr(app, "managed_accounts", None)
    if accounts is not None:
        if is_paper_account(accounts):
            return HealthCheck("Paper Account", "PASS", "one DU Paper account detected")
        return HealthCheck("Paper Account", "FAIL", "non-Paper or ambiguous account detected")
    if str(getattr(broker, "mode", "")).upper() == "IBKR_PAPER":
        return HealthCheck("Paper Account", "PASS", "IBKR Paper account was accepted by the broker adapter")
    return HealthCheck("Paper Account", "FAIL", "Paper broker source is not active")


def _llm_check_for_config(base_url: str, api_key: str, luna_model: str, sol_model: str) -> tuple[str, str]:
    if not base_url:
        return "WARN", "CC Switch gateway URL is not configured"
    result = _llm_diagnostic_for_config(
        base_url,
        api_key,
        luna_model,
        sol_model,
        api_protocol=env("LLM_API_PROTOCOL"),
        protocol_fallback=env("LLM_PROTOCOL_FALLBACK"),
    )
    endpoint = str(result.get("endpoint", "FAIL"))
    return endpoint, _llm_capability_summary(result)


def _llm_check_for_key(api_key: str) -> tuple[str, str]:
    """Compatibility helper for older callers; it never targets an official endpoint."""
    return _llm_check_for_config(
        env("LLM_BASE_URL", ""),
        api_key or env("LLM_API_KEY", ""),
        env("LLM_LUNA_MODEL", "gpt-5.6-luna"),
        env("LLM_SOL_MODEL", "gpt-5.6-sol"),
    )


def _default_llm_check() -> tuple[str, str]:
    return _llm_check_for_config(
        env("LLM_BASE_URL", ""),
        env("LLM_API_KEY", "") or env("OPENAI_API_KEY", ""),
        env("LLM_LUNA_MODEL", "gpt-5.6-luna"),
        env("LLM_SOL_MODEL", "gpt-5.6-sol"),
    )


def _llm_diagnostic_for_config(
    base_url: str,
    api_key: str,
    luna_model: str,
    sol_model: str,
    *,
    api_protocol: str | None = None,
    protocol_fallback: bool | str | None = None,
) -> dict[str, Any]:
    if not str(base_url or "").strip():
        return {"endpoint": "WARN", "model_discovery": "WARN", "ok": False, "error": "CC Switch gateway URL is missing"}
    try:
        from .llm_agent import CCSwitchProvider

        return CCSwitchProvider(
            base_url=base_url,
            api_key=api_key,
            luna_model=luna_model,
            sol_model=sol_model,
            api_protocol=api_protocol,
            protocol_fallback=protocol_fallback,
        ).health_check()
    except Exception as exc:
        return {"endpoint": "FAIL", "model_discovery": "FAIL", "ok": False, "error": redact_sensitive(str(exc))}


def _llm_capability_summary(result: dict[str, Any]) -> str:
    fields = (
        ("CC Switch Gateway", "endpoint"),
        ("Model Discovery", "model_discovery"),
        ("Luna Basic Call", "luna_basic_call"),
        ("Luna Structured Output", "luna_structured_output"),
        ("Sol Basic Call", "sol_basic_call"),
        ("Sol Structured JSON", "sol_structured_json"),
        ("Sol Tool Calling", "sol_tool_calling"),
        ("Sol Decision", "sol_decision"),
    )
    return " | ".join(f"{label}: {result.get(key, 'UNKNOWN')}" for label, key in fields)


def _llm_health_checks(result: dict[str, Any]) -> list[HealthCheck]:
    stage_fields = (
        ("CC Switch Gateway", "endpoint", "gateway endpoint is reachable"),
        ("Model Discovery", "model_discovery", "configured model list was queried"),
        ("Luna Basic Call", "luna_basic_call", "Luna text capability"),
        ("Luna Structured Output", "luna_structured_output", "Luna structured screening capability"),
        ("Sol Basic Call", "sol_basic_call", "Sol text capability"),
        ("Sol Structured JSON", "sol_structured_json", "Sol JSON capability"),
        ("Sol Tool Calling", "sol_tool_calling", "Sol single-tool capability"),
        ("Sol Multi-turn Tool Calling", "sol_multi_turn", "Sol tool continuation capability"),
        ("Sol Decision", "sol_decision", "Sol structured decision capability"),
    )
    checks = [HealthCheck(name, str(result.get(key, "UNKNOWN")), message) for name, key, message in stage_fields]
    overall = "PASS" if result.get("ok") else "WARN" if result.get("endpoint") == "WARN" else "FAIL"
    checks.append(HealthCheck("LLM API", overall, _llm_capability_summary(result)))
    return checks


def _database_check(database_path: str | Path | None) -> HealthCheck:
    if database_path is None:
        return HealthCheck("Database", "WARN", "database path was not supplied")
    try:
        from .storage import SQLiteStore

        with SQLiteStore(database_path) as store:
            store.runtime_snapshot()
    except Exception:
        return HealthCheck("Database", "FAIL", "database could not be opened")
    return HealthCheck("Database", "PASS", "SQLite database is writable")


def _scheduler_check(scheduler_config: dict | None) -> HealthCheck:
    try:
        from .scheduler import SchedulerConfig

        SchedulerConfig.from_mapping(scheduler_config or {"timezone": "America/New_York"})
    except Exception:
        return HealthCheck("Scheduler", "FAIL", "scheduler configuration is invalid")
    return HealthCheck("Scheduler", "PASS", "scheduler configuration is valid")


def _health_quote(broker, symbol: str):
    try:
        return broker.get_execution_quote(symbol, allow_delayed=True)
    except TypeError as exc:
        if "allow_delayed" not in str(exc):
            raise
        return broker.get_execution_quote(symbol)


_PROBE_MARKET_DATA_TYPE_NAMES = {
    1: "LIVE",
    2: "FROZEN",
    3: "DELAYED",
    4: "DELAYED_FROZEN",
}


def _probe_has_price(probe: dict[str, Any] | None) -> bool:
    if not isinstance(probe, dict):
        return False
    for field in ("bid", "ask", "last", "close"):
        try:
            if float(probe.get(field, 0)) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _probe_market_data_state(probe: dict[str, Any] | None) -> str:
    if not _probe_has_price(probe):
        return "UNAVAILABLE"
    data_type = str((probe or {}).get("data_type", "")).upper().strip()
    if data_type in {"FROZEN", "DELAYED_FROZEN"}:
        return "MARKET_CLOSED"
    if data_type == "REALTIME":
        return "LIVE"
    if data_type == "DELAYED":
        return "DELAYED"
    return "UNAVAILABLE"


def _probe_failure_state(probe_result: dict[str, Any]) -> str:
    permission_codes = {354, 10090, 10186}
    permission_markers = ("not subscribed", "market data denied", "permission")
    for probe in probe_result.get("probes", []):
        code = probe.get("error_code")
        try:
            code = int(code) if code is not None else None
        except (TypeError, ValueError):
            code = None
        message = str(probe.get("error_message", "")).lower()
        if code in permission_codes or any(marker in message for marker in permission_markers):
            return "PERMISSION_DENIED"
    return "UNAVAILABLE"


def _probe_quote(probe: dict[str, Any], symbol: str) -> dict[str, Any]:
    def positive(field: str):
        try:
            value = float(probe.get(field, 0))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    bid, ask, last, close = (positive(field) for field in ("bid", "ask", "last", "close"))
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    return {
        "symbol": probe.get("symbol", symbol),
        "price": mid or last or close or bid or ask,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        "close": close,
        "bid_size": probe.get("bid_size"),
        "ask_size": probe.get("ask_size"),
        "last_size": probe.get("last_size"),
        "timestamp": probe.get("timestamp", probe.get("broker_timestamp")),
        "market_status": "CLOSED" if probe.get("data_type") in {"FROZEN", "DELAYED_FROZEN"} else "OPEN",
        "source": "IBKR",
        "data_type": probe.get("data_type", "UNKNOWN"),
    }


def _broker_market_data_probe(broker, symbol: str):
    raw_broker = getattr(broker, "_broker", broker)
    probe = getattr(raw_broker, "probe_market_data", None)
    return probe(symbol) if callable(probe) else None


def _broker_historical_fallback(broker, symbol: str):
    raw_broker = getattr(broker, "_broker", broker)
    fallback = getattr(raw_broker, "get_historical_fallback", None)
    if not callable(fallback):
        raise RuntimeError("IBKR historical market data fallback is unavailable")
    return fallback(symbol)


def _streaming_probe_state(probe_result: dict[str, Any]) -> str:
    probes = probe_result.get("probes", [])
    if any(item.get("result") == "NO_CURRENT_TICK" for item in probes):
        return "NO_CURRENT_TICK"
    return str(probes[-1].get("result", "UNAVAILABLE")) if probes else "UNAVAILABLE"


def _closed_delayed_probe_has_no_tick(probe_result: dict[str, Any]) -> bool:
    return any(
        item.get("requested_type") in {3, 4}
        and item.get("callback_market_data_type") in {3, 4}
        and item.get("result") == "NO_CURRENT_TICK"
        and item.get("error_code") is None
        for item in probe_result.get("probes", [])
    )


def _quote_is_fresh(quote: dict[str, Any], at: datetime | None, max_age_minutes: float) -> bool:
    try:
        timestamp = datetime.fromisoformat(str(quote.get("timestamp", "")).replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    current = at or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_minutes = (current.astimezone(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 60
    return 0 <= age_minutes <= float(max_age_minutes)


def format_market_data_probe_report(probe_result: dict[str, Any] | None) -> str:
    """Render raw four-mode market-data evidence without credentials."""
    if not isinstance(probe_result, dict):
        return ""
    symbol = probe_result.get("symbol", "SPY")
    lines = [str(symbol), ""]
    if probe_result.get("market_session"):
        lines.extend([
            f"Market Session: {probe_result['market_session']}",
            f"Streaming Quote: {probe_result.get('streaming_quote', 'UNKNOWN')}",
        ])
        historical = probe_result.get("historical_fallback") or {}
        lines.extend([
            f"Historical Fallback: {historical.get('result', 'NOT_USED')}",
            f"Last Close: {historical.get('last_close')}",
            f"Last Bar Time: {historical.get('last_bar_time')}",
            f"Historical Error: {historical.get('error', '')}",
            "",
        ])
    for probe in probe_result.get("probes", []):
        requested_type = probe.get("requested_type")
        name = probe.get("requested_type_name") or _PROBE_MARKET_DATA_TYPE_NAMES.get(requested_type, "UNKNOWN")
        callback_type = probe.get("callback_market_data_type")
        lines.extend([
            f"{name}:",
            f"requested_type={requested_type}",
            f"callback_type={callback_type}",
            f"callback_market_data_type={callback_type}",
            f"bid={probe.get('bid')}",
            f"ask={probe.get('ask')}",
            f"last={probe.get('last')}",
            f"close={probe.get('close')}",
            f"open={probe.get('open')}",
            f"high={probe.get('high')}",
            f"low={probe.get('low')}",
            f"volume={probe.get('volume')}",
            f"bid_size={probe.get('bid_size')}",
            f"ask_size={probe.get('ask_size')}",
            f"last_size={probe.get('last_size')}",
            f"broker_timestamp={probe.get('broker_timestamp', probe.get('timestamp'))}",
            f"received_tick={probe.get('received_tick', False)}",
            f"error_code={probe.get('error_code')}",
            f"error_message={probe.get('error_message', '')}",
            f"warning_code={probe.get('warning_code')}",
            f"warning_message={probe.get('warning_message', '')}",
            f"elapsed_ms={probe.get('elapsed_ms')}",
            f"result={probe.get('result', 'UNKNOWN')}",
            "",
        ])
    lines.append(f"FINAL MARKET DATA STATE: {probe_result.get('final_state', 'UNAVAILABLE')}")
    return "\n".join(lines).rstrip()


def _first_run_report_details(result: FirstRunResult) -> str:
    lines = [result.user_message]
    probe = result.details.get("market_data_probe")
    rendered_probe = format_market_data_probe_report(probe)
    if rendered_probe:
        lines.extend(["", rendered_probe])
    return "\n".join(lines)


def classify_market_data_state(quote: dict[str, Any]) -> str:
    """Classify quote quality without treating a valid closed-market quote as failure."""
    try:
        if float(quote.get("price", 0)) <= 0 or not quote.get("timestamp"):
            return "UNAVAILABLE"
    except (TypeError, ValueError):
        return "UNAVAILABLE"

    data_type = str(quote.get("data_type", "")).upper().strip()
    quality = {
        "REALTIME": "LIVE",
        "LIVE": "LIVE",
        "FROZEN": "FROZEN",
        "DELAYED": "DELAYED",
        "DELAYED_FROZEN": "DELAYED_FROZEN",
    }.get(data_type)
    market_status = str(quote.get("market_status", quote.get("market_state", ""))).upper().strip()
    if market_status in {"CLOSED", "MARKET_CLOSED"}:
        return "MARKET_CLOSED"
    return quality or "UNAVAILABLE"


def classify_market_data_error(exc: Exception) -> str:
    message = str(exc).lower()
    if any(marker in message for marker in ("permission", "not subscribed", "market data denied")):
        return "PERMISSION_DENIED"
    if any(marker in message for marker in ("error 354", "error 10090", "error 10167", "error 10186", "error code 354", "error code 10090", "error code 10167", "error code 10186")):
        return "PERMISSION_DENIED"
    return "UNAVAILABLE"


def _market_data_health_check(state: str) -> HealthCheck:
    if state == "LIVE":
        return HealthCheck("Market Data", "PASS", "IBKR live market data quote received")
    if state in {"FROZEN", "DELAYED", "DELAYED_FROZEN", "MARKET_CLOSED"}:
        return HealthCheck("Market Data", "WARN", f"IBKR market data state is {state}; quote is not live")
    if state == "PERMISSION_DENIED":
        return HealthCheck("Market Data", "FAIL", "IBKR market data permission or subscription is unavailable")
    return HealthCheck("Market Data", "FAIL", "IBKR market data quote is unavailable")


def run_first_run_observe(
    broker,
    *,
    llm_check: Callable[[], bool | tuple[str, str]] | None = None,
    database_path: str | Path | None = None,
    scheduler_config: dict | None = None,
    quote_symbol: str = "SPY",
    api_capabilities: dict | None = None,
    dashboard_checker: Callable[[], bool | tuple[str, str]] | None = None,
    market_clock=None,
    market_time: datetime | None = None,
    max_quote_age_minutes: float = 30.0,
) -> FirstRunResult:
    """Read the broker and one liquid quote without submitting or cancelling."""
    checks: list[HealthCheck] = []
    details: dict[str, Any] = {}
    if api_capabilities is not None:
        api_check = HealthCheck(
            "IBKR API",
            "PASS" if api_capabilities.get("ok") else "FAIL",
            "official callback capabilities are available" if api_capabilities.get("ok") else "official TWS API is missing or incompatible",
        )
        checks.append(api_check)
        details["api_version"] = api_capabilities.get("api_version")
        if api_check.status == "FAIL":
            return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. Install the official IBKR TWS API before continuing.", details)
    try:
        if not getattr(broker, "connected", False):
            broker.connect()
        if not getattr(broker, "connected", False):
            raise ConnectionError("broker socket is not connected")
        checks.extend([
            HealthCheck("TWS", "PASS", "broker connection is available"),
            HealthCheck("Socket", "PASS", "socket API connection is available"),
        ])
    except Exception as exc:
        if _is_waiting_for_login(exc):
            checks.extend([
                HealthCheck("TWS", "WARN", "TWS is not connected or Paper login is not complete"),
                HealthCheck("Socket", "WARN", "waiting for the TWS Paper socket"),
            ])
            return FirstRunResult(
                state="WAITING_FOR_TWS_PAPER_LOGIN",
                checks=checks,
                user_message="WAITING FOR TWS PAPER LOGIN. Please log in to the Paper account in TWS and run detection again.",
                details={"connection_error": str(exc)},
            )
        checks.extend([
            HealthCheck("TWS", "FAIL", "TWS/API connection failed"),
            HealthCheck("Socket", "FAIL", "socket API is unavailable"),
        ])
        return FirstRunResult(
            state="SAFE_MODE",
            checks=checks,
            user_message="SAFE_MODE. Setup checks failed; trading is disabled.",
            details={"connection_error": str(exc)},
        )

    checks.append(_paper_account_check(broker))
    if checks[-1].status == "FAIL":
        return FirstRunResult(
            state="SAFE_MODE",
            checks=checks,
            user_message="SAFE_MODE. A non-Paper or ambiguous account was detected; the system refused to continue.",
            details={},
        )

    try:
        position_symbols = list(broker.position_symbols())
        position_prices: dict[str, float] = {}
        for symbol in position_symbols:
            position_quote = _health_quote(broker, symbol)
            position_prices[symbol] = float(position_quote["price"])
        snapshot = broker.reconcile(position_prices)
        details.update({
            "equity": snapshot.equity,
            "cash": snapshot.cash,
            "positions": len(snapshot.positions),
            "open_orders": len(snapshot.open_orders),
        })
        checks.extend([
            HealthCheck("Account", "PASS", "account summary and equity were read"),
            HealthCheck("Positions", "PASS", f"{len(snapshot.positions)} position(s) read"),
            HealthCheck("Open Orders", "PASS", f"{len(snapshot.open_orders)} open order(s) read"),
        ])
    except Exception as exc:
        checks.append(HealthCheck("Account", "FAIL", "broker reconciliation failed"))
        return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. Broker reconciliation failed; trading is disabled.", {"reconciliation_error": str(exc)})

    try:
        if market_clock is None:
            from .market_clock import USEquityMarketClock

            market_clock = USEquityMarketClock()
        market_open = bool(market_clock.is_open(market_time) if market_time is not None else market_clock.is_open())
        details["market_session"] = "OPEN" if market_open else "CLOSED"
        broker.resolve_instrument(quote_symbol)
        probe_result = _broker_market_data_probe(broker, quote_symbol)
        if probe_result is not None:
            details["market_data_probe"] = probe_result
            streaming_quote = _streaming_probe_state(probe_result)
            details["streaming_quote"] = streaming_quote
            probe_result["market_session"] = details["market_session"]
            probe_result["streaming_quote"] = streaming_quote
            selected_probe = probe_result.get("selected")
            if not selected_probe:
                historical = None
                if not market_open and _closed_delayed_probe_has_no_tick(probe_result):
                    try:
                        bar = _broker_historical_fallback(broker, quote_symbol)
                        completed_session = market_clock.latest_completed_session(market_time) if market_time is not None else market_clock.latest_completed_session()
                        valid = (
                            str(bar.get("bar_date", "")) == completed_session["date"]
                            and float(bar.get("close", 0)) > 0
                        )
                        historical = {
                            "result": "PASS" if valid else "FAIL",
                            "last_close": bar.get("close"),
                            "last_bar_time": completed_session["close"] if valid else bar.get("bar_date"),
                            "bar_date": bar.get("bar_date"),
                            "source": bar.get("source", "IBKR historical"),
                            "warning_code": bar.get("warning_code"),
                            "warning_message": bar.get("warning_message", ""),
                        }
                    except Exception as exc:
                        historical = {"result": "FAIL", "error": redact_sensitive(str(exc))}
                if historical and historical["result"] == "PASS":
                    details["historical_fallback"] = historical
                    probe_result["historical_fallback"] = historical
                    probe_result["final_state"] = "MARKET_CLOSED"
                    market_data_state = "MARKET_CLOSED"
                    quote = {
                        "symbol": quote_symbol,
                        "price": historical["last_close"],
                        "close": historical["last_close"],
                        "timestamp": historical["last_bar_time"],
                        "market_status": "CLOSED",
                        "source": historical["source"],
                        "data_type": "FROZEN",
                    }
                    data_type = "FROZEN"
                else:
                    if historical:
                        details["historical_fallback"] = historical
                        probe_result["historical_fallback"] = historical
                    market_data_state = _probe_failure_state(probe_result)
                    details["market_data_state"] = market_data_state
                    details["market_data_errors"] = [
                        {"error_code": item.get("error_code"), "error_message": redact_sensitive(item.get("error_message", ""))}
                        for item in probe_result.get("probes", [])
                        if item.get("error_code") is not None or item.get("error_message")
                    ]
                    checks.append(_market_data_health_check(market_data_state))
                    checks.append(HealthCheck("Contract Resolution", "PASS", f"{quote_symbol} resolved through the broker adapter"))
                    return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. Market data checks failed; trading is disabled.", details)
            else:
                quote = _probe_quote(selected_probe, quote_symbol)
                data_type = str(quote.get("data_type", "UNKNOWN")).upper()
                market_data_state = _probe_market_data_state(selected_probe)
        else:
            quote = _health_quote(broker, quote_symbol)
            data_type = str(quote.get("data_type", "UNKNOWN")).upper()
            market_data_state = classify_market_data_state(quote)
        if market_open and (data_type not in {"REALTIME", "DELAYED"} or not _quote_is_fresh(quote, market_time, max_quote_age_minutes)):
            details["market_data_state"] = "UNAVAILABLE"
            checks.append(_market_data_health_check("UNAVAILABLE"))
            checks.append(HealthCheck("Contract Resolution", "PASS", f"{quote_symbol} resolved through the broker adapter"))
            return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. Market data checks failed; trading is disabled.", details)
        details["market_data_state"] = market_data_state
        details["quote"] = {
            "symbol": quote.get("symbol", quote_symbol),
            "data_type": data_type,
            "market_status": quote.get("market_status", quote.get("market_state", "UNKNOWN")),
            "source": quote.get("source", "IBKR"),
        }
        if probe_result is not None:
            details["quote"].update({
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "last": quote.get("last"),
                "close": quote.get("close"),
                "bid_size": quote.get("bid_size"),
                "ask_size": quote.get("ask_size"),
                "last_size": quote.get("last_size"),
                "timestamp": quote.get("timestamp"),
            })
        checks.append(_market_data_health_check(market_data_state))
        checks.append(HealthCheck("Contract Resolution", "PASS", f"{quote_symbol} resolved through the broker adapter"))
    except Exception as exc:
        market_data_state = classify_market_data_error(exc)
        details["market_data_state"] = market_data_state
        checks.append(_market_data_health_check(market_data_state))
        return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. Market data checks failed; trading is disabled.", {**details, "market_data_error": str(exc)})

    mutations_allowed = bool(getattr(broker, "mutations_allowed", True))
    checks.append(HealthCheck("Broker Mutation", "FAIL" if mutations_allowed else "PASS", "OBSERVE orders are disabled" if not mutations_allowed else "broker mutation is enabled"))
    if mutations_allowed:
        return FirstRunResult("SAFE_MODE", checks, "SAFE_MODE. First-run checks require OBSERVE with broker mutation disabled.", details)

    if llm_check is None:
        llm_result = _llm_diagnostic_for_config(
            env("LLM_BASE_URL", ""),
            env("LLM_API_KEY", "") or env("OPENAI_API_KEY", ""),
            env("LLM_LUNA_MODEL", "gpt-5.6-luna"),
            env("LLM_SOL_MODEL", "gpt-5.6-sol"),
            api_protocol=env("LLM_API_PROTOCOL"),
            protocol_fallback=env("LLM_PROTOCOL_FALLBACK"),
        )
        details["llm_capabilities"] = {
            key: value for key, value in llm_result.items() if key not in {"errors", "error"}
        }
        checks.extend(_llm_health_checks(llm_result))
        if llm_result.get("errors"):
            details["llm_errors"] = [redact_sensitive(error) for error in llm_result["errors"]]
    else:
        try:
            llm_result = llm_check()
            if isinstance(llm_result, tuple):
                llm_status, llm_message = llm_result
            else:
                llm_status, llm_message = ("PASS", "CC Switch health check passed") if llm_result else ("FAIL", "CC Switch health check failed")
        except Exception:
            llm_status, llm_message = "FAIL", "CC Switch health check failed"
        checks.append(HealthCheck("LLM Gateway", llm_status, llm_message))
    checks.append(_database_check(database_path))
    try:
        dashboard_result = dashboard_checker() if dashboard_checker else bool(importlib.util.find_spec("streamlit"))
        if isinstance(dashboard_result, tuple):
            dashboard_status, dashboard_message = dashboard_result
        else:
            dashboard_status, dashboard_message = (
                ("PASS", "Streamlit is available") if dashboard_result else ("FAIL", "Streamlit is not installed")
            )
    except Exception as exc:
        dashboard_status, dashboard_message = "FAIL", f"Dashboard dependency check failed: {exc}"
    checks.append(HealthCheck("Dashboard", dashboard_status, dashboard_message))
    checks.append(_scheduler_check(scheduler_config))

    ready = all(item.status != "FAIL" for item in checks)
    state = "READY_FOR_OBSERVE" if ready else "SAFE_MODE"
    message = "READY FOR OBSERVE" if ready else "SAFE_MODE. One or more setup checks failed; trading is disabled."
    return FirstRunResult(state, checks, message, details)


def persist_first_run_result(store, result: FirstRunResult) -> None:
    """Publish a sanitized health result to the Dashboard runtime state."""
    store.set_runtime("first_run_state", result.state)
    store.set_runtime("first_run_ready", result.ready)
    store.set_runtime("safe_mode", result.state == "SAFE_MODE")
    store.set_runtime("trading_enabled", False)
    store.set_runtime("scheduler_status", "STOPPED")
    store.set_runtime("service_status", result.state)
    store.set_runtime("first_run_message", result.user_message)
    store.set_runtime(
        "broker_status",
        "CONNECTED" if result.ready else "WAITING_FOR_LOGIN" if result.state == "WAITING_FOR_TWS_PAPER_LOGIN" else "DISCONNECTED",
    )
    market_data = next((item.status for item in result.checks if item.name == "Market Data"), "UNKNOWN")
    market_data_state = result.details.get("market_data_state", market_data)
    quote_details = result.details.get("quote") if isinstance(result.details.get("quote"), dict) else {}
    historical_details = result.details.get("historical_fallback") if isinstance(result.details.get("historical_fallback"), dict) else {}
    check_status = {item.name: item.status for item in result.checks}
    llm_details = result.details.get("llm_capabilities") if isinstance(result.details.get("llm_capabilities"), dict) else {}
    gateway_status = check_status.get("CC Switch Gateway", check_status.get("LLM Gateway", "UNKNOWN"))
    capability_status = check_status.get("LLM API", gateway_status)

    def agent_status(check_name: str) -> str:
        status = check_status.get(check_name, "UNKNOWN")
        return "ONLINE" if status in {"PASS", "FALLBACK"} else "ERROR" if status == "FAIL" else status

    store.set_runtime("market_data_status", market_data_state)
    store.set_runtime("market_data_state", market_data_state)
    store.set_runtime("market_data_quality", quote_details.get("data_type", "UNKNOWN"))
    store.set_runtime("market_session", result.details.get("market_session", quote_details.get("market_status", "UNKNOWN")))
    store.set_runtime("market_data_source", quote_details.get("source", historical_details.get("source", "UNKNOWN")))
    store.set_runtime("market_data_last_bar_time", historical_details.get("last_bar_time", quote_details.get("timestamp")))
    store.set_runtime("llm_status", capability_status)
    store.set_runtime("llm_gateway_status", gateway_status)
    store.set_runtime("llm_capability_status", capability_status)
    store.set_runtime("llm_model_discovery_status", check_status.get("Model Discovery", "UNKNOWN"))
    store.set_runtime("llm_luna_basic_status", check_status.get("Luna Basic Call", "UNKNOWN"))
    store.set_runtime("llm_luna_structured_status", check_status.get("Luna Structured Output", "UNKNOWN"))
    store.set_runtime("llm_sol_basic_status", check_status.get("Sol Basic Call", "UNKNOWN"))
    store.set_runtime("llm_sol_structured_status", check_status.get("Sol Structured JSON", "UNKNOWN"))
    store.set_runtime("llm_sol_tool_status", check_status.get("Sol Tool Calling", "UNKNOWN"))
    store.set_runtime("llm_sol_decision_status", check_status.get("Sol Decision", "UNKNOWN"))
    store.set_runtime("llm_luna_status", agent_status("Luna Basic Call"))
    store.set_runtime("llm_sol_status", agent_status("Sol Basic Call"))
    store.set_runtime("llm_provider", env("LLM_PROVIDER", "ccswitch"))
    store.set_runtime("llm_api_protocol", llm_details.get("api_protocol", "UNKNOWN"))
    store.set_runtime("llm_protocol_fallback", llm_details.get("protocol_fallback", "NO"))
    store.set_runtime("llm_reasoning_metadata", llm_details.get("reasoning_metadata", "UNAVAILABLE"))
    store.set_runtime("llm_token_usage", llm_details.get("token_usage", "UNAVAILABLE"))
    store.set_runtime("setup_details", result.details)
    store.set_runtime("setup.tws_installed", check_status.get("TWS", "UNKNOWN"))
    store.set_runtime("setup.tws_running", check_status.get("TWS", "UNKNOWN"))
    store.set_runtime("setup.tws_connected", check_status.get("Socket", "UNKNOWN"))
    store.set_runtime("setup.paper_account", check_status.get("Paper Account", "UNKNOWN"))
    store.set_runtime("setup.socket_port", env("IBKR_PORT", "unknown"))
    store.set_runtime("setup.socket_port_status", check_status.get("Socket", "UNKNOWN"))
    store.set_runtime("setup.ibkr_api_version", result.details.get("api_version", "unknown"))
    store.set_runtime("setup.market_data_type", market_data_state)
    store.set_runtime(
        "setup.market_data_permission",
        "FAIL" if market_data_state == "PERMISSION_DENIED" else "WARN" if market_data_state != "LIVE" else "PASS",
    )
    for check in result.checks:
        key = "setup." + check.name.lower().replace(" ", "_")
        store.set_runtime(key, check.status)
    store.set_runtime("llm_pipeline", llm_details.get("pipeline", env("LLM_PIPELINE", "LUNA_SOL")))
    store.set_runtime("llm_luna_model", llm_details.get("luna_model", env("LLM_LUNA_MODEL", "gpt-5.6-luna")))
    store.set_runtime("llm_sol_model", llm_details.get("sol_model", env("LLM_SOL_MODEL", "gpt-5.6-sol")))


def local_diagnostic_checks(project_root: str | Path) -> list[dict[str, str]]:
    root = Path(project_root)
    checks: list[dict[str, str]] = []
    checks.append({"name": "Python", "status": "PASS", "message": "Python interpreter is running"})
    dependencies = ("yaml", "dotenv", "pydantic", "pandas", "streamlit")
    missing = []
    for dependency in dependencies:
        if importlib.util.find_spec(dependency) is None:
            missing.append(dependency)
    checks.append({"name": "Dependencies", "status": "PASS" if not missing else "FAIL", "message": "all required packages are available" if not missing else "missing: " + ", ".join(missing)})
    env_path = root / ".env"
    checks.append({"name": "Configuration", "status": "PASS" if env_path.exists() else "WARN", "message": ".env exists" if env_path.exists() else ".env has not been generated"})
    try:
        from .ibkr_diagnostics import inspect_api_capabilities

        api_result = inspect_api_capabilities()
        checks.append({"name": "IBKR API", "status": "PASS" if api_result["ok"] else "FAIL", "message": "official callback capabilities are available" if api_result["ok"] else "official TWS API is missing or incompatible"})
    except Exception:
        checks.append({"name": "IBKR API", "status": "FAIL", "message": "official TWS API check failed"})
    if os.name == "nt":
        try:
            tasklist = subprocess.run(["tasklist", "/FI", "IMAGENAME eq tws.exe"], capture_output=True, text=True, timeout=2, check=False)
            tws_running = "tws.exe" in tasklist.stdout.lower()
        except (OSError, subprocess.SubprocessError):
            tws_running = False
    else:
        tws_running = False
    checks.append({"name": "TWS", "status": "PASS" if tws_running else "WARN", "message": "TWS process is running" if tws_running else "TWS is not running"})
    paper_port = discover_paper_port(timeout=0.1)
    checks.append({"name": "Socket", "status": "PASS" if paper_port else "WARN", "message": f"safe Paper port {paper_port} is reachable" if paper_port else "no safe Paper port is reachable"})
    configured_key = read_dotenv_value(env_path, "LLM_API_KEY") or read_dotenv_value(env_path, "OPENAI_API_KEY") or env("LLM_API_KEY", "") or env("OPENAI_API_KEY", "")
    configured_base_url = read_dotenv_value(env_path, "LLM_BASE_URL") or env("LLM_BASE_URL", "")
    configured_luna = read_dotenv_value(env_path, "LLM_LUNA_MODEL") or env("LLM_LUNA_MODEL", "gpt-5.6-luna")
    configured_sol = read_dotenv_value(env_path, "LLM_SOL_MODEL") or env("LLM_SOL_MODEL", "gpt-5.6-sol")
    if configured_base_url:
        llm_result = _llm_diagnostic_for_config(configured_base_url, configured_key, configured_luna, configured_sol)
        llm_status = str(llm_result.get("endpoint", "FAIL"))
        llm_message = _llm_capability_summary(llm_result)
        checks.extend(item.as_dict() for item in _llm_health_checks(llm_result))
    else:
        llm_status, llm_message = "WARN", "CC Switch gateway URL is missing"
    checks.append({"name": "CC Switch", "status": llm_status, "message": llm_message})
    if not configured_base_url:
        checks.append({"name": "LLM API", "status": "WARN", "message": "Compatibility alias: " + llm_message})
    database_path = root / "data" / "ai_fund_manager.sqlite3"
    checks.append({"name": "Database", "status": "PASS" if database_path.exists() else "WARN", "message": "SQLite database is present" if database_path.exists() else "database will be created on start"})
    try:
        from .risk_engine import RiskEngine
        from .scheduler import SchedulerConfig

        _ = RiskEngine, SchedulerConfig
        risk_status, risk_message = "PASS", "Risk Engine is importable"
    except Exception:
        risk_status, risk_message = "FAIL", "Risk Engine or scheduler import failed"
    checks.append({"name": "Risk Engine", "status": risk_status, "message": risk_message})
    checks.append({"name": "Execution Mode", "status": "PASS", "message": "OBSERVE is the only first-run execution mode"})
    checks.append({"name": "Broker Mutation", "status": "PASS", "message": "disabled for FIRST_RUN_OBSERVE"})
    checks.append({"name": "Process Lock", "status": "PASS" if not (root / "data" / "backend.lock").exists() else "WARN", "message": "no active lock file" if not (root / "data" / "backend.lock").exists() else "backend lock exists; another service may be running"})
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Fund Manager first-run checks")
    parser.add_argument("--first-run-observe", action="store_true")
    parser.add_argument("--troubleshoot", action="store_true")
    parser.add_argument("--report", default="diagnostic_report.txt")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root or Path(__file__).resolve().parents[1])

    if args.troubleshoot:
        checks = local_diagnostic_checks(root)
        report = write_diagnostic_report(root / args.report, checks)
        for check in checks:
            print(f"{check['name']:<20} {check['status']}")
        print(f"REPORT={report}")
        return 0 if not any(item["status"] == "FAIL" for item in checks) else 1

    if not args.first_run_observe:
        parser.error("use --first-run-observe or --troubleshoot")

    from .config import load_config
    from .ibkr_diagnostics import inspect_api_capabilities
    from .main import build_executor
    from .storage import SQLiteStore

    cfg = load_config(root / "config.yaml")
    database_path = env("DATABASE_PATH", cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3"))
    if not Path(database_path).is_absolute():
        database_path = root / database_path
    with SQLiteStore(database_path) as store:
        api_result = inspect_api_capabilities()
        api_check = HealthCheck(
            "IBKR API",
            "PASS" if api_result["ok"] else "FAIL",
            "official callback capabilities are available" if api_result["ok"] else "official TWS API is missing or incompatible",
        )
        if not api_result["ok"]:
            result = FirstRunResult(
                "SAFE_MODE",
                [api_check, HealthCheck("Broker Mutation", "PASS", "disabled for FIRST_RUN_OBSERVE")],
                "SAFE_MODE. Install the official IBKR TWS API before continuing.",
                {"api_version": api_result.get("api_version")},
            )
            persist_first_run_result(store, result)
            checks = [item.as_dict() for item in result.checks]
            report = write_diagnostic_report(root / args.report, checks, details=_first_run_report_details(result))
            for check in checks:
                print(f"{check['name']:<20} {check['status']}")
            print(result.user_message)
            print(f"REPORT={report}")
            return 1
        broker = build_executor(cfg, store, broker_source="IBKR_PAPER", execution_mode="OBSERVE")
        result = run_first_run_observe(
            broker,
            database_path=database_path,
            scheduler_config=cfg.get("scheduler"),
        )
        result.checks.insert(0, api_check)
        result.details.setdefault("api_version", api_result.get("api_version"))
        persist_first_run_result(store, result)
        checks = [item.as_dict() for item in result.checks]
        report = write_diagnostic_report(root / args.report, checks, details=_first_run_report_details(result))
        for check in checks:
            print(f"{check['name']:<20} {check['status']}")
        print(result.user_message)
        print(f"REPORT={report}")
        disconnect = getattr(broker, "disconnect", None)
        if disconnect:
            disconnect()
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
