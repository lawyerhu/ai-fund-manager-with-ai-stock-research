from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .analytics import compare_to_benchmarks, compute_metrics
from .config import env
from .storage import redact_sensitive


def dashboard_refresh_seconds(cfg: dict | None = None) -> float:
    configured = env("DASHBOARD_REFRESH_SECONDS")
    if configured in (None, ""):
        configured = ((cfg or {}).get("dashboard", {}) or {}).get("refresh_seconds", 3)
    try:
        value = float(configured)
    except (TypeError, ValueError):
        return 3.0
    return value if value > 0 else 3.0


def _next_weekly(day_name: str, hhmm: str, timezone_name: str) -> str:
    weekdays = {name: index for index, name in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
    now = datetime.now(ZoneInfo(timezone_name))
    hour, minute = (int(value) for value in hhmm.split(":"))
    days = (weekdays[day_name.lower()] - now.weekday()) % 7
    candidate = (now + timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate.isoformat()


def _event_metadata(row: dict) -> dict:
    raw = row.get("metadata_json", "{}")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _event_timestamp(row: dict) -> str:
    return str(row.get("timestamp") or row.get("recorded_at") or "")


def _timeline_row(row: dict) -> dict:
    metadata = _event_metadata(row)
    return {
        "id": row.get("id"),
        "timestamp": _event_timestamp(row),
        "run_id": row.get("run_id"),
        "decision_id": row.get("decision_id"),
        "component": row.get("component"),
        "event_type": row.get("event_type"),
        "symbol": row.get("symbol"),
        "message": row.get("message", ""),
        "metadata": metadata,
    }


def _initial_ai_stage() -> dict:
    return {"status": "WAITING", "started_at": None, "completed_at": None}


def _build_ai_activity(events: list[dict], runtime: dict, pipeline_rows: list[dict]) -> dict:
    progress = runtime.get("ai_research_progress") if isinstance(runtime.get("ai_research_progress"), dict) else {}
    timeline = sorted((_timeline_row(row) for row in events), key=lambda row: (row["timestamp"], row["id"] or 0))
    current_run_id = runtime.get("ai.current_run_id")
    if not current_run_id:
        cycle = runtime.get("ai_cycle")
        current_run_id = cycle.get("run_id") if isinstance(cycle, dict) else None
    if not current_run_id:
        current_run_id = next((row.get("run_id") for row in reversed(timeline) if row.get("run_id")), None)
    run_events = [row for row in timeline if row.get("run_id") == current_run_id] if current_run_id else []
    if not run_events:
        return {
            "run_id": current_run_id,
            "status": "IDLE",
            "started_at": None,
            "completed_at": None,
            "pipeline": runtime.get("llm_pipeline"),
            "luna_model": runtime.get("llm_luna_model"),
            "sol_model": runtime.get("llm_sol_model"),
            "stages": {name: _initial_ai_stage() for name in ("Luna", "Sol", "Risk Engine", "Execution")},
            "final_decision": {},
            "progress": progress,
        }

    stages = {name: _initial_ai_stage() for name in ("Luna", "Sol", "Risk Engine", "Execution")}
    activity = {
        "run_id": current_run_id,
        "status": "RUNNING",
        "started_at": run_events[0]["timestamp"],
        "completed_at": None,
        "pipeline": None,
        "luna_model": None,
        "sol_model": None,
        "stages": stages,
        "final_decision": {},
        "progress": progress,
    }
    for row in run_events:
        event_type = str(row.get("event_type") or "")
        metadata = row.get("metadata") or {}
        if event_type == "AI_RUN_STARTED":
            activity.update({key: metadata[key] for key in ("pipeline", "luna_model", "sol_model") if key in metadata})
        elif event_type == "AI_RUN_COMPLETED":
            activity["status"] = metadata.get("status", "COMPLETED")
            activity["completed_at"] = row["timestamp"]
        elif event_type in {"AI_RUN_FAILED", "LLM_ERROR"}:
            activity["status"] = "ERROR"
            activity["completed_at"] = row["timestamp"]
        elif event_type in {"LUNA_SCREEN_STARTED", "LUNA_BATCH_STARTED", "LUNA_BATCH_SPLIT", "LUNA_BATCH_REQUEST_FAILED"}:
            stages["Luna"].update({"status": "RUNNING", "started_at": row["timestamp"], **metadata})
        elif event_type == "LUNA_BATCH_COMPLETED":
            stages["Luna"].update({"status": "RUNNING", **metadata})
        elif event_type == "LUNA_SCREEN_COMPLETED":
            stages["Luna"].update({"status": "COMPLETE", "completed_at": row["timestamp"], **metadata})
        elif event_type in {"LUNA_SCREEN_ERROR", "LUNA_BATCH_FAILED"}:
            stages["Luna"].update({"status": "ERROR", "completed_at": row["timestamp"], "error": row["message"]})
        elif event_type == "SOL_RESEARCH_STARTED":
            stages["Sol"].update({"status": "RUNNING", "started_at": row["timestamp"], "tool_calls": [], "tool_results": [], **metadata})
        elif event_type == "SOL_TOOL_CALL":
            stages["Sol"].setdefault("tool_calls", []).append({"timestamp": row["timestamp"], "tool": metadata.get("tool", row["message"]), "symbol": row.get("symbol"), "arguments": metadata.get("arguments", {})})
        elif event_type == "SOL_TOOL_RESULT":
            stages["Sol"].setdefault("tool_results", []).append({"timestamp": row["timestamp"], "tool": metadata.get("tool", row["message"]), "symbol": row.get("symbol"), "status": metadata.get("status", "SUCCESS"), "result": metadata.get("result", {})})
        elif event_type in {"SOL_DECISION_COMPLETED", "SOL_IC_DECISION_COMPLETED"}:
            stages["Sol"].update({"status": "COMPLETE", "completed_at": row["timestamp"]})
            activity["final_decision"] = metadata.get("decision", metadata)
            if metadata.get("deep_research"):
                activity["deep_research"] = metadata["deep_research"]
        elif event_type == "SOL_RESEARCH_ERROR":
            stages["Sol"].update({"status": "ERROR", "completed_at": row["timestamp"], "error": row["message"]})
        elif event_type == "RISK_EVALUATION_STARTED":
            stages["Risk Engine"].update({"status": "RUNNING", "started_at": row["timestamp"], **metadata})
        elif event_type == "RISK_EVALUATED":
            stages["Risk Engine"].update({"status": "COMPLETE", "completed_at": row["timestamp"], **metadata})
        elif event_type == "RISK_EVALUATION_ERROR":
            stages["Risk Engine"].update({"status": "ERROR", "completed_at": row["timestamp"], "error": row["message"]})
        elif event_type.startswith("WOULD_"):
            stages["Execution"].update({"status": "WOULD_EXECUTE", "event_type": event_type, "timestamp": row["timestamp"], "message": row["message"], **metadata})
        elif event_type == "EXECUTION_BLOCKED":
            stages["Execution"].update({"status": "BLOCKED", "timestamp": row["timestamp"], "message": row["message"], **metadata})
        elif event_type == "EXECUTION_COMPLETED":
            execution_status = "WOULD_EXECUTE" if metadata.get("order_not_sent") else "BLOCKED" if metadata.get("status") in {"ERROR", "REJECTED"} else "COMPLETE"
            stages["Execution"].update({"status": execution_status, "completed_at": row["timestamp"], "message": row["message"], **metadata})

    if activity["pipeline"] is None and pipeline_rows:
        activity["pipeline"] = pipeline_rows[0].get("pipeline")
    decision_id = (activity.get("final_decision") or {}).get("decision_id")
    pipeline_row = next(
        (row for row in pipeline_rows if row.get("decision_id") == decision_id),
        pipeline_rows[0] if pipeline_rows else None,
    )
    if pipeline_row:
        try:
            candidate_symbols = json.loads(pipeline_row.get("candidate_symbols_json") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            candidate_symbols = []
        evidence = (activity.get("final_decision") or {}).get("evidence_used") or []
        comparative_notes = [
            str(item) for item in evidence
            if str(item).strip().lower().startswith("comparative screen:")
        ]
        if candidate_symbols or comparative_notes:
            activity["comparative_context"] = {
                "candidate_symbols": candidate_symbols,
                "comparative_notes": comparative_notes,
                "structured_ranking_available": bool((activity.get("final_decision") or {}).get("top_five")),
            }
    return activity


def _normalize(values):
    values = [float(value) for value in values]
    if not values or values[0] == 0:
        return []
    return [100.0 * value / values[0] for value in values]


def _configured(name: str, placeholder: str | None = None) -> str:
    value = env(name, "")
    return "Configured" if value and value != placeholder else "Missing"


def _health_status(value, *, default="UNKNOWN"):
    value = str(value or default).upper()
    value = {
        "ONLINE": "PASS",
        "OK": "PASS",
        "CONNECTED": "PASS",
        "REALTIME": "PASS",
        "LIVE": "PASS",
        "FROZEN": "WARN",
        "DELAYED": "WARN",
        "DELAYED_FROZEN": "WARN",
        "MARKET_CLOSED": "WARN",
        "PERMISSION_DENIED": "FAIL",
        "UNAVAILABLE": "FAIL",
        "OFFLINE": "FAIL",
        "ERROR": "FAIL",
    }.get(value, value)
    return value if value in {"PASS", "WARN", "FAIL", "FALLBACK", "UNAVAILABLE", "UNKNOWN"} else default


def _gateway_display_status(value) -> str:
    value = str(value or "UNKNOWN").upper()
    return {"PASS": "ONLINE", "OK": "ONLINE", "CONNECTED": "ONLINE", "FAIL": "OFFLINE", "ERROR": "OFFLINE"}.get(value, value)


def _capability_display_status(value, *, default="UNKNOWN") -> str:
    value = str(value or default).upper()
    return {"ONLINE": "PASS", "OK": "PASS", "CONNECTED": "PASS", "OFFLINE": "FAIL", "ERROR": "FAIL"}.get(value, value)


def _agent_display_status(value) -> str:
    value = str(value or "UNKNOWN").upper()
    return {"PASS": "ONLINE", "OK": "ONLINE", "CONNECTED": "ONLINE", "FAIL": "ERROR", "OFFLINE": "ERROR", "ERROR": "ERROR"}.get(value, value)


def build_setup_health(runtime: dict, cfg: dict) -> dict:
    setup_details = redact_sensitive(runtime.get("setup_details")) if isinstance(runtime.get("setup_details"), dict) else {}
    quote_details = setup_details.get("quote") if isinstance(setup_details.get("quote"), dict) else {}
    broker_source = runtime.get("broker_source", cfg.get("execution", {}).get("broker_source", "LOCAL"))
    execution_mode = runtime.get("execution_mode", cfg.get("execution", {}).get("execution_mode", "OBSERVE"))
    broker_status = runtime.get("broker_status", "DISCONNECTED")
    scheduler_status = runtime.get("scheduler_status", "STOPPED")
    market_data_state = runtime.get("market_data_state", runtime.get("market_data_status", quote_details.get("data_type", "UNKNOWN")))
    llm_cfg = cfg.get("llm", {}) or {}
    llm_pipeline = runtime.get("llm_pipeline", env("LLM_PIPELINE", llm_cfg.get("pipeline", "LUNA_SOL")))
    llm_gateway_status = runtime.get("llm_gateway_status", runtime.get("llm_status", "UNKNOWN"))
    luna_status = runtime.get("llm_luna_status", "UNKNOWN")
    sol_status = runtime.get("llm_sol_status", "UNKNOWN")
    database_path = env("DATABASE_PATH", cfg.get("dashboard", {}).get("database_path", cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3")))
    checks = {
        "TWS Installed": _health_status(runtime.get("setup.tws_installed", runtime.get("setup.tws", "UNKNOWN"))),
        "TWS Running": _health_status(runtime.get("setup.tws_running", runtime.get("setup.tws", "UNKNOWN"))),
        "TWS Connected": _health_status(runtime.get("setup.tws_connected", runtime.get("setup.tws", "PASS" if broker_status == "CONNECTED" else "UNKNOWN"))),
        "Paper Account": _health_status(runtime.get("setup.paper_account", "UNKNOWN")),
        "Socket Port": _health_status(runtime.get("setup.socket_port_status", runtime.get("setup.socket", "PASS" if env("IBKR_PORT") else "UNKNOWN"))),
        "IBKR API Version": _health_status(runtime.get("setup.ibkr_api_version", "UNKNOWN")),
        "Market Data Type": _health_status(runtime.get("setup.market_data_type", runtime.get("setup.market_data", "PASS" if quote_details.get("data_type") else "UNKNOWN"))),
        "Market Data State": _health_status(market_data_state),
        "Market Data Permission": _health_status(runtime.get("setup.market_data_permission", runtime.get("setup.market_data", "UNKNOWN"))),
        "LLM Gateway": _health_status(llm_gateway_status, default="WARN"),
        "LLM Capability": _health_status(runtime.get("llm_capability_status", "UNKNOWN")),
        "CC Switch Gateway": _health_status(runtime.get("llm_gateway_status", "UNKNOWN"), default="WARN"),
        "Model Discovery": _health_status(runtime.get("llm_model_discovery_status", "UNKNOWN")),
        "Luna Basic Call": _health_status(runtime.get("llm_luna_basic_status", "UNKNOWN")),
        "Luna Structured Output": _health_status(runtime.get("llm_luna_structured_status", "UNKNOWN")),
        "Sol Basic Call": _health_status(runtime.get("llm_sol_basic_status", "UNKNOWN")),
        "Sol Structured JSON": _health_status(runtime.get("llm_sol_structured_status", "UNKNOWN")),
        "Sol Tool Calling": _health_status(runtime.get("llm_sol_tool_status", "UNKNOWN")),
        "Sol Decision": _health_status(runtime.get("llm_sol_decision_status", "UNKNOWN")),
        "Reasoning Metadata": _health_status(runtime.get("llm_reasoning_metadata", "UNKNOWN")),
        "Token Usage": _health_status(runtime.get("llm_token_usage", "UNKNOWN")),
        "Luna": _health_status(luna_status),
        "Sol": _health_status(sol_status),
        "Tool Calling": _health_status(runtime.get("llm_tool_calling", "UNKNOWN")),
        "Structured Output": _health_status(runtime.get("llm_structured_output", "UNKNOWN")),
        "Execution Mode": "PASS" if str(execution_mode).upper() == "OBSERVE" else "FAIL",
        "Broker Source": "PASS" if str(broker_source).upper() == "IBKR_PAPER" else "FAIL",
        "Scheduler": "PASS" if str(scheduler_status).upper() == "RUNNING" else "WARN" if str(scheduler_status).upper() == "STOPPED" else _health_status(scheduler_status),
        "Database": "PASS" if os.path.exists(database_path) else "WARN",
        "SAFE_MODE": "FAIL" if bool(runtime.get("safe_mode", False)) else "WARN" if runtime.get("service_status") == "WAITING_FOR_TWS_PAPER_LOGIN" else "PASS",
    }
    values = {
        "Socket Port": runtime.get("setup.socket_port", env("IBKR_PORT", "unknown")),
        "IBKR API Version": runtime.get("setup.ibkr_api_version", "unknown"),
        "Market Data Type": quote_details.get("data_type", runtime.get("setup.market_data_type", "unknown")),
        "Market Data State": market_data_state,
        "Market Data Source": runtime.get("market_data_source", quote_details.get("source", "unknown")),
        "Last Completed Market Data": runtime.get("market_data_last_bar_time", quote_details.get("timestamp", "unknown")),
        "Execution Mode": str(execution_mode).upper(),
        "Broker Source": str(broker_source).upper(),
        "Scheduler": str(scheduler_status).upper(),
        "LLM Gateway": "CC Switch",
        "Pipeline": str(llm_pipeline),
        "Screening Model": runtime.get("llm_luna_model", env("LLM_LUNA_MODEL", llm_cfg.get("luna_model", "gpt-5.6-luna"))),
        "Research + CIO": runtime.get("llm_sol_model", env("LLM_SOL_MODEL", llm_cfg.get("sol_model", "gpt-5.6-sol"))),
        "State": runtime.get("first_run_state", runtime.get("service_status", "OFFLINE")),
    }
    return {
        "state": runtime.get("first_run_state", runtime.get("service_status", "NOT_ACTIVE")),
        "checks": checks,
        "values": values,
        "details": setup_details,
        "execution_mode": str(execution_mode).upper(),
        "broker_source": str(broker_source).upper(),
    }


def build_dashboard_payload(store, cfg: dict) -> dict:
    runtime = store.runtime_snapshot()
    llm_cfg = cfg.get("llm", {}) or {}
    pipeline_rows = redact_sensitive(store.pipeline_history(50))
    latest_pipeline = pipeline_rows[0] if pipeline_rows else {}
    pipeline = runtime.get("llm_pipeline", latest_pipeline.get("pipeline", env("LLM_PIPELINE", llm_cfg.get("pipeline", "LUNA_SOL"))))
    luna_model = runtime.get("llm_luna_model", latest_pipeline.get("luna_model", env("LLM_LUNA_MODEL", llm_cfg.get("luna_model", "gpt-5.6-luna"))))
    sol_model = runtime.get("llm_sol_model", latest_pipeline.get("sol_model", env("LLM_SOL_MODEL", llm_cfg.get("sol_model", "gpt-5.6-sol"))))
    gateway_status = _gateway_display_status(runtime.get("llm_gateway_status", runtime.get("llm_status", "UNKNOWN")))
    luna_status = _agent_display_status(runtime.get("llm_luna_status", "ONLINE" if latest_pipeline else "UNKNOWN"))
    sol_status = _agent_display_status(runtime.get("llm_sol_status", "ONLINE" if latest_pipeline else "UNKNOWN"))
    tool_status = _capability_display_status(runtime.get("llm_tool_calling", latest_pipeline.get("tool_calling_status", "PASS") if latest_pipeline else "UNKNOWN"))
    structured_status = _capability_display_status(runtime.get("llm_structured_output", latest_pipeline.get("structured_output_status", "PASS") if latest_pipeline else "UNKNOWN"))
    latest = store.latest_portfolio()
    state, positions = latest if latest else (None, [])
    equity_rows = store.equity_history()
    equity = [float(row["equity"]) for row in equity_rows]
    benchmarks = store.benchmark_history()
    performance_rows = store.daily_performance_history()
    dates = [row["date"] for row in performance_rows]
    daily_ai = [float(row["ai_nav"]) for row in performance_rows]
    daily_spy = [float(row["spy_close"]) for row in performance_rows]
    daily_qqq = [float(row["qqq_close"]) for row in performance_rows]
    comparison = compare_to_benchmarks(daily_ai, daily_spy, daily_qqq) if performance_rows else None
    strategy_series = _normalize(daily_ai)
    spy_series = _normalize(daily_spy)
    qqq_series = _normalize(daily_qqq)
    trading_costs = store.trading_cost_total()
    llm_costs = store.llm_api_cost_total()
    net_strategy_return = comparison.get("total_return") if comparison else None
    spy_return = compute_metrics(daily_spy, periods_per_year=252)["total_return"]
    qqq_return = compute_metrics(daily_qqq, periods_per_year=252)["total_return"]
    gross_strategy_return = (
        (daily_ai[-1] + trading_costs) / daily_ai[0] - 1
        if len(daily_ai) >= 2 and daily_ai[0] > 0 else None
    )
    returns = {
        "gross_strategy_return": gross_strategy_return,
        "trading_costs": trading_costs,
        "llm_costs": llm_costs,
        "net_strategy_return": net_strategy_return,
        "spy_return": spy_return,
        "qqq_return": qqq_return,
        "net_excess_return_vs_spy": net_strategy_return - spy_return if net_strategy_return is not None and spy_return is not None else None,
        "net_excess_return_vs_qqq": net_strategy_return - qqq_return if net_strategy_return is not None and qqq_return is not None else None,
    }
    macro_policy = cfg.get("risk", {}).get("event_risk", {}).get("macro_unknown_policy", "cap")
    default_macro_status = "UNKNOWN — POSITION CAPPED" if macro_policy == "cap" else "UNKNOWN — NEW EXPOSURE BLOCKED" if macro_policy == "reject_new" else "UNKNOWN — ALLOWED"
    runtime_events = store.runtime_events(500)
    timeline = sorted((_timeline_row(row) for row in runtime_events), key=lambda row: (row["timestamp"], row["id"] or 0))
    ai_activity = _build_ai_activity(runtime_events, runtime, pipeline_rows)
    decision_rows = redact_sensitive(store.recent("llm_decisions", 50))
    evidence_rows = redact_sensitive(store.recent("research_evidence", 200))
    risk_rows = redact_sensitive(store.recent("risk_decisions", 1))
    command_rows = redact_sensitive(store.recent("control_commands", 20))
    current_worker_id = runtime.get("command_worker_id")
    active_ai_research_command = next(
        (
            row for row in command_rows
            if row.get("command") in {"RUN_AI_RESEARCH", "RUN_FULL_AI_RESEARCH"}
            and (
                str(row.get("status") or "").upper() in {"QUEUED", "PENDING"}
                or (
                    str(row.get("status") or "").upper() in {"CLAIMED", "RUNNING"}
                    and row.get("worker_id") == current_worker_id
                )
            )
        ),
        None,
    )
    last_reconciliation_command = next(
        (row for row in command_rows if row.get("command") == "RUN_RECONCILIATION"),
        None,
    )
    latest_decision = json.loads(decision_rows[0]["decision_json"]) if decision_rows else {}
    managed_position = store.active_managed_position(state.current_symbol if state else None)
    position_reviews = store.position_reviews(managed_position.position_id, 20) if managed_position else []
    position_triggers = store.position_triggers(managed_position.position_id, 20) if managed_position else []
    series_metrics = {
        "AI Strategy": compute_metrics(daily_ai, periods_per_year=252),
        "SPY Buy & Hold": compute_metrics(daily_spy, periods_per_year=252),
        "QQQ Buy & Hold": compute_metrics(daily_qqq, periods_per_year=252),
    }
    scheduler_cfg = cfg.get("scheduler", {}) or {}
    research_day = scheduler_cfg.get("full_research_day", scheduler_cfg.get("weekly_day", "saturday"))
    research_time = scheduler_cfg.get("full_research_time", scheduler_cfg.get("weekly_time", "10:00"))
    scheduler_timezone = scheduler_cfg.get("timezone", "America/New_York")
    broker_status = runtime.get("broker_status", "DISCONNECTED")
    monitor_connected = runtime.get("broker_monitor_connected")
    monitor_reconciled = runtime.get("broker_reconciliation_ready")
    broker_display_status = (
        "CONNECTED"
        if monitor_connected is True and monitor_reconciled is True
        else "RECONCILIATION_REQUIRED"
        if monitor_connected is True and monitor_reconciled is False
        else broker_status
    )
    return {
        "status": {
            "system": runtime.get("service_status", "OFFLINE"),
            "trading_mode": runtime.get("trading_mode", str(cfg.get("execution", {}).get("mode", "OBSERVE")).upper()),
            "broker_source": runtime.get("broker_source", str(cfg.get("execution", {}).get("broker_source", "LOCAL")).upper()),
            "execution_mode": runtime.get("execution_mode", str(cfg.get("execution", {}).get("execution_mode", "OBSERVE")).upper()),
            "broker": broker_display_status,
            "llm": runtime.get("llm_status", "UNKNOWN"),
            "market_data": runtime.get("market_data_status", "UNKNOWN"),
            "market_session": runtime.get("market_session", "UNKNOWN"),
            "market_data_source": runtime.get("market_data_source", "UNKNOWN"),
            "market_data_last_bar_time": runtime.get("market_data_last_bar_time"),
            "macro_data": runtime.get("macro_data_status", default_macro_status),
            "scheduler": runtime.get("scheduler_status", "STOPPED"),
            "command_worker": runtime.get("command_worker_status", "STOPPED"),
            "command_worker_heartbeat": runtime.get("command_worker_heartbeat"),
            "broker_monitor": runtime.get("broker_monitor_status", "STOPPED"),
            "risk_state": runtime.get("risk_state", state.risk_state if state else "NORMAL"),
            "last_reconciliation": runtime.get("last_reconciliation"),
            "last_ai_decision": runtime.get("last_ai_decision"),
            "first_run_state": runtime.get("first_run_state", "NOT_ACTIVE"),
            "first_run_ready": bool(runtime.get("first_run_ready", False)),
            "first_run_message": runtime.get("first_run_message"),
            "safe_mode": bool(runtime.get("safe_mode", False)),
            "manual_halt": bool(runtime.get("manual_halt", False)),
            "last_error": runtime.get("last_error"),
            "tws": runtime.get("tws_status", runtime.get("broker_status", "DISCONNECTED")),
            "cc_switch": gateway_status,
            "luna": str(luna_status).upper(),
            "sol": str(sol_status).upper(),
            "current_position": state.current_symbol if state and state.current_symbol else "CASH",
            "last_ai_decision_id": latest_decision.get("decision_id"),
            "latest_command": command_rows[0] if command_rows else None,
            "active_ai_research_command": active_ai_research_command,
            "last_reconciliation_command": last_reconciliation_command,
            "acknowledged_failed_command_id": runtime.get("dashboard.acknowledged_failed_command_id"),
        },
        "operations": {
            "full_research_frequency": cfg.get("strategy", {}).get("full_research_frequency", "WEEKLY"),
            "next_full_research": _next_weekly(research_day, research_time, scheduler_timezone),
            "last_full_research": runtime.get("last_full_research") or runtime.get("scheduler.last_weekly_run"),
            "daily_review_time": scheduler_cfg.get("daily_review_time", scheduler_cfg.get("daily_time", "16:05")),
            "last_daily_review": runtime.get("last_daily_review") or runtime.get("scheduler.last_daily_run"),
            "risk_interval_minutes": scheduler_cfg.get("risk_monitor_interval_minutes", 15),
            "last_full_risk_check": runtime.get("last_full_risk_check") or runtime.get("scheduler.last_risk_run"),
            "last_immediate_risk_check": runtime.get("last_immediate_risk_check"),
            "broker_monitor": runtime.get("broker_monitor_status", "STOPPED"),
            "broker_monitor_heartbeat": runtime.get("broker_monitor_heartbeat"),
            "risk_data_freshness_seconds": runtime.get("risk_data_age_seconds"),
            "market_data_type": (runtime.get("current_market_data") or {}).get("market_data_type", runtime.get("market_data_status", "UNKNOWN")),
            "current_position_review": runtime.get("current_position_review", "REVIEW_REQUIRED"),
        },
        "configuration": {
            "LLM_PROVIDER": _configured("LLM_PROVIDER"),
            "LLM_BASE_URL": _configured("LLM_BASE_URL"),
            "LLM_API_KEY": _configured("LLM_API_KEY"),
            "LLM_LUNA_MODEL": _configured("LLM_LUNA_MODEL"),
            "LLM_SOL_MODEL": _configured("LLM_SOL_MODEL"),
            "LLM_LUNA_REASONING_EFFORT": _configured("LLM_LUNA_REASONING_EFFORT"),
            "LLM_SOL_REASONING_EFFORT": _configured("LLM_SOL_REASONING_EFFORT"),
            "LLM_PIPELINE": _configured("LLM_PIPELINE"),
            "OPENAI_API_KEY": _configured("OPENAI_API_KEY"),
            "OPENAI_MODEL": _configured("OPENAI_MODEL", "YOUR_TOOL_CAPABLE_MODEL"),
            "IBKR_HOST": _configured("IBKR_HOST"),
            "IBKR_PORT": _configured("IBKR_PORT"),
        },
        "setup": build_setup_health(runtime, cfg),
        "llm": {
            "gateway": "CC Switch",
            "gateway_status": str(gateway_status).upper(),
            "pipeline": pipeline,
            "pipeline_label": "Luna -> Sol" if pipeline == "LUNA_SOL" else "Sol only" if pipeline == "SOL_ONLY" else str(pipeline),
            "screening_model": luna_model,
            "research_cio_model": sol_model,
            "luna_status": str(luna_status).upper(),
            "sol_status": str(sol_status).upper(),
            "tool_calling": str(tool_status).upper(),
            "structured_output": str(structured_status).upper(),
            "model_discovery": _capability_display_status(runtime.get("llm_model_discovery_status", "UNKNOWN")),
            "luna_basic_call": _capability_display_status(runtime.get("llm_luna_basic_status", "UNKNOWN")),
            "luna_structured_output": _capability_display_status(runtime.get("llm_luna_structured_status", "UNKNOWN")),
            "sol_basic_call": _capability_display_status(runtime.get("llm_sol_basic_status", "UNKNOWN")),
            "sol_structured_json": _capability_display_status(runtime.get("llm_sol_structured_status", "UNKNOWN")),
            "sol_tool_calling": _capability_display_status(runtime.get("llm_sol_tool_status", "UNKNOWN")),
            "sol_decision": _capability_display_status(runtime.get("llm_sol_decision_status", "UNKNOWN")),
            "reasoning_metadata": str(runtime.get("llm_reasoning_metadata", "UNAVAILABLE")).upper(),
            "token_usage": str(runtime.get("llm_token_usage", "UNAVAILABLE")).upper(),
            "api_protocol": str(runtime.get("llm_api_protocol", "UNKNOWN")).upper(),
            "protocol_fallback": str(runtime.get("llm_protocol_fallback", "NO")).upper(),
            "latest_run": latest_pipeline,
            "history": pipeline_rows,
            "total_decision_cost": llm_costs,
        },
        "account": {
            "equity": state.equity if state else None,
            "cash": state.cash if state else None,
            "invested_value": state.invested_value if state else None,
            "current_drawdown": state.drawdown if state else None,
            "historical_max_drawdown": state.historical_max_drawdown if state else None,
            "peak_equity": state.peak_equity if state else None,
            "realized_pnl": state.realized_pnl if state else None,
            "unrealized_pnl": state.unrealized_pnl if state else None,
            "positions": positions,
        },
        "risk": risk_rows,
        "decisions": decision_rows,
        "evidence": evidence_rows,
        "pipeline_history": pipeline_rows,
        "journal": store.journal_history(100),
        "orders": redact_sensitive(store.recent("order_records", 200)),
        "executions": redact_sensitive(store.recent("executions", 200)),
        "pending_entries": redact_sensitive([entry.model_dump(mode="json") for entry in store.pending_entries()]),
        "position_manager": redact_sensitive({
            "reduction_shadows": store.reduction_shadows(),
            "position": {**managed_position.model_dump(mode="json"), "days_held": managed_position.days_held_at()} if managed_position else None,
            "latest_review": position_reviews[0].model_dump(mode="json") if position_reviews else None,
            "reviews": [review.model_dump(mode="json") for review in position_reviews],
            "latest_trigger": position_triggers[0].model_dump(mode="json") if position_triggers else None,
            "triggers": [trigger.model_dump(mode="json") for trigger in position_triggers],
        }),
        "errors": redact_sensitive(store.recent("errors", 200)),
        "state_events": redact_sensitive(store.recent("state_events", 200)),
        "runtime_events": redact_sensitive(runtime_events),
        "commands": command_rows,
        "timeline": redact_sensitive(timeline),
        "ai_activity": redact_sensitive(ai_activity),
        "benchmarks": {
            "dates": dates,
            "series": {"AI Strategy": strategy_series, "SPY Buy & Hold": spy_series, "QQQ Buy & Hold": qqq_series},
            "metrics": comparison or compute_metrics(daily_ai, periods_per_year=252),
            "series_metrics": series_metrics,
            "returns": returns,
            "llm_api_cost": llm_costs,
            "trading_costs": trading_costs,
        },
    }
