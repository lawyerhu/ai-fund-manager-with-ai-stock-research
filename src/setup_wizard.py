"""Small, safe helpers used by the Windows first-run wizard."""
from __future__ import annotations

import argparse
import re
import socket
from pathlib import Path
from typing import Any, Callable, Iterable


SAFE_PAPER_PORTS = (7947, 7497, 4002)
LIVE_PORTS = (7496, 7946, 4001)
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_SOL_MODEL = DEFAULT_MODEL
OFFICIAL_TWS_URL = "https://www.interactivebrokers.com/en/trading/tws.php"
OFFICIAL_TWS_API_URL = "https://www.interactivebrokers.com/en/trading/tws-api.php"


def validate_paper_port(port: int | str) -> int:
    try:
        value = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("IBKR port must be an integer") from exc
    if value in LIVE_PORTS:
        raise ValueError(f"Live IBKR port {value} is blocked")
    if not 1 <= value <= 65535:
        raise ValueError("IBKR port must be between 1 and 65535")
    return value


def discover_paper_port(
    host: str = "127.0.0.1",
    ports: Iterable[int] = SAFE_PAPER_PORTS,
    timeout: float = 0.4,
    connector: Callable = socket.create_connection,
) -> int | None:
    """Return the first reachable safe Paper port, never a live port."""
    seen: set[int] = set()
    for candidate in ports:
        try:
            port = validate_paper_port(candidate)
        except ValueError:
            continue
        if port in seen:
            continue
        seen.add(port)
        connection = None
        try:
            connection = connector((host, port), timeout)
            return port
        except (OSError, TimeoutError):
            continue
        finally:
            if connection is not None:
                close = getattr(connection, "close", None)
                if close:
                    close()
    return None


def is_paper_account(accounts: Iterable[str]) -> bool:
    normalized = [str(account).strip().upper() for account in accounts if str(account).strip()]
    return len(normalized) == 1 and normalized[0].startswith("DU")


def read_dotenv_value(path: str | Path, name: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    prefix = f"{name}="
    for raw_line in file_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value
    return ""


def _env_value(value: str | None) -> str:
    value = str(value or "")
    if "\r" in value or "\n" in value:
        raise ValueError("configuration values cannot contain newlines")
    return value


def _protocol_value(value: str | None) -> str:
    normalized = str(value or "RESPONSES").strip().upper().replace("-", "_").replace(" ", "_")
    if normalized not in {"RESPONSES", "CHAT_COMPLETIONS"}:
        raise ValueError("LLM_API_PROTOCOL must be RESPONSES or CHAT_COMPLETIONS")
    return normalized


def _yes_no(value: bool | str | None) -> str:
    return "YES" if str(value or "").strip().lower() in {"1", "true", "yes", "on"} else "NO"


def ccswitch_health_check(
    base_url: str,
    api_key: str = "",
    *,
    luna_model: str = DEFAULT_LUNA_MODEL,
    sol_model: str = DEFAULT_SOL_MODEL,
    api_protocol: str | None = None,
    protocol_fallback: bool | str | None = None,
) -> dict[str, Any]:
    """Probe the configured gateway without exposing its credentials."""
    if not str(base_url or "").strip():
        return {"ok": False, "endpoint": "WARN", "error": "CC Switch gateway URL is missing"}
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
        return {"ok": False, "endpoint": "FAIL", "error": redact_sensitive(str(exc))}


def write_observe_env(
    path: str | Path,
    *,
    port: int | str,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    base_url: str = "",
    luna_model: str = DEFAULT_LUNA_MODEL,
    sol_model: str | None = None,
    pipeline: str = "LUNA_SOL",
    timeout_seconds: int | float = 300,
    host: str = "127.0.0.1",
    client_id: int = 41,
) -> Path:
    """Write the wizard-owned safe defaults atomically.

    The explicit execution mode is deliberately written alongside the legacy
    mode so stale settings cannot turn the first run into a mutating run.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_port = validate_paper_port(port)
    selected_sol_model = sol_model or model or DEFAULT_SOL_MODEL
    values = {
        "TRADING_MODE": "IBKR_PAPER",
        "BROKER_SOURCE": "IBKR_PAPER",
        "EXECUTION_MODE": "OBSERVE",
        "FIRST_RUN_MODE": "FIRST_RUN_OBSERVE",
        "IBKR_HOST": _env_value(host),
        "IBKR_PORT": str(safe_port),
        "IBKR_CLIENT_ID": str(int(client_id)),
        "DATA_PROVIDER": "yahoo",
        "LLM_PROVIDER": "ccswitch",
        "LLM_BASE_URL": _env_value(base_url),
        "LLM_API_KEY": _env_value(api_key),
        "LLM_LUNA_MODEL": _env_value(luna_model or DEFAULT_LUNA_MODEL),
        "LLM_SOL_MODEL": _env_value(selected_sol_model),
        "LLM_LUNA_REASONING_EFFORT": "max",
        "LLM_SOL_REASONING_EFFORT": "medium",
        "LLM_PIPELINE": _env_value(pipeline or "LUNA_SOL"),
        "LLM_API_PROTOCOL": "RESPONSES",
        "LLM_PROTOCOL_FALLBACK": "NO",
        "LLM_TIMEOUT_SECONDS": _env_value(str(timeout_seconds)),
        # Legacy aliases keep older integrations readable; runtime uses LLM_*.
        "OPENAI_API_KEY": _env_value(api_key),
        "OPENAI_MODEL": _env_value(selected_sol_model),
        "DATABASE_PATH": "data/ai_fund_manager.sqlite3",
    }
    lines = [
        "# Generated by AI Fund Manager setup wizard. Keep this file private.",
        *[f"{key}={value}" for key, value in values.items()],
    ]
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def persist_detected_llm_config(
    path: str | Path,
    *,
    luna_model: str,
    sol_model: str,
    provider: str = "ccswitch",
    pipeline: str = "LUNA_SOL",
    api_protocol: str | None = None,
    protocol_fallback: bool | str | None = None,
) -> Path:
    """Persist capability-test values while preserving secrets and unrelated settings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    detected_protocol = _protocol_value(api_protocol or read_dotenv_value(target, "LLM_API_PROTOCOL") or "RESPONSES")
    detected_fallback = _yes_no(
        protocol_fallback
        if protocol_fallback is not None
        else read_dotenv_value(target, "LLM_PROTOCOL_FALLBACK")
    )
    updates = {
        "LLM_PROVIDER": _env_value(provider),
        "LLM_LUNA_MODEL": _env_value(luna_model or read_dotenv_value(target, "LLM_LUNA_MODEL") or DEFAULT_LUNA_MODEL),
        "LLM_SOL_MODEL": _env_value(sol_model or read_dotenv_value(target, "LLM_SOL_MODEL") or DEFAULT_SOL_MODEL),
        "LLM_PIPELINE": _env_value(pipeline or read_dotenv_value(target, "LLM_PIPELINE") or "LUNA_SOL"),
        "LLM_API_PROTOCOL": detected_protocol,
        "LLM_PROTOCOL_FALLBACK": detected_fallback,
    }
    lines = target.read_text(encoding="utf-8-sig").splitlines() if target.exists() else []
    seen: set[str] = set()
    rewritten: list[str] = []
    for line in lines:
        key, separator, _ = line.partition("=")
        normalized_key = key.strip() if separator else ""
        if normalized_key in updates:
            rewritten.append(f"{normalized_key}={updates[normalized_key]}")
            seen.add(normalized_key)
        else:
            rewritten.append(line)
    if rewritten and rewritten[-1] != "":
        rewritten.append("")
    rewritten.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    temporary = target.with_name(f".{target.name}.llm.tmp")
    temporary.write_text("\n".join(rewritten).rstrip("\n") + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def redact_sensitive(text: str) -> str:
    """Remove credentials and account identifiers from support output."""
    redacted = re.sub(
        r"(?im)(OPENAI_API_KEY\s*[=:]\s*)[^\r\n]*",
        r"\1[REDACTED]",
        str(text),
    )
    redacted = re.sub(
        r"(?i)\b(api[_ -]?key|password|token|secret)\b(\s*[=:]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        redacted,
    )
    return re.sub(r"\b(?:DU|U)\d{4,}\b", "[ACCOUNT_REDACTED]", redacted, flags=re.IGNORECASE)


def write_diagnostic_report(
    path: str | Path,
    checks: Iterable[dict],
    *,
    details: str | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["AI Fund Manager diagnostic report", ""]
    for check in checks:
        name = check.get("name", "Unknown")
        status = check.get("status", "UNKNOWN")
        message = check.get("message", "")
        lines.append(f"{name}: {status} - {message}".rstrip())
    if details:
        lines.extend(["", "Advanced Details", details])
    target.write_text(redact_sensitive("\n".join(lines) + "\n"), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Fund Manager setup helpers")
    parser.add_argument("--discover-port", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.discover_port:
        port = discover_paper_port(host=args.host)
        print(port or "")
        return 0 if port else 1
    parser.error("one of the setup helper actions is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
