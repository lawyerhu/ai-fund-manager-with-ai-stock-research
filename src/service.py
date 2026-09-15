from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import os
import re
import threading
import time
import uuid

from .process_lock import SingletonLock
from .first_run import is_waiting_for_tws, persist_first_run_result, run_first_run_observe
from .ibkr_diagnostics import inspect_api_capabilities
from .llm_agent import PipelineError
from .models import TradeIntent
from .runner import BrokerReadinessError


class BackendService:
    """Owns the long-lived backend lifecycle and gates scheduling on reconciliation."""

    def __init__(self, cfg, runner, store, scheduler=None, lock=None, use_llm=False, initialization_error=None, first_run_observe=False):
        self.cfg = cfg
        self.runner = runner
        self.store = store
        self.scheduler = scheduler
        lock_path = Path("backend.lock") if str(store.path) == ":memory:" else store.path.parent / "backend.lock"
        self.lock = lock or SingletonLock(lock_path)
        self._started = False
        self._stop_event = threading.Event()
        self._heartbeat_stop_event = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._cycle_lock = threading.RLock()
        self._command_process_lock = threading.Lock()
        self.use_llm = use_llm
        self.initialization_error = initialization_error
        self.first_run_observe = bool(first_run_observe) and not bool(store.get_runtime("first_run_completed", False))
        self._pending_weekly_intent = None
        self._next_broker_reconnect_at = 0.0
        self._broker_auto_recovery_blocked = False

    def start(self):
        self.lock.acquire()
        self._started = True
        executor = getattr(self.runner, "executor", None)
        macro_policy = self.cfg.get("risk", {}).get("event_risk", {}).get("macro_unknown_policy", "cap")
        macro_status = "UNKNOWN — POSITION CAPPED" if macro_policy == "cap" else "UNKNOWN — NEW EXPOSURE BLOCKED" if macro_policy == "reject_new" else "UNKNOWN — ALLOWED"
        self._set_runtime(service_status="STARTING", trading_enabled=False, safe_mode=False, trading_mode=getattr(executor, "mode", "UNKNOWN"), broker_source=getattr(executor, "broker_source", "UNKNOWN"), execution_mode=getattr(executor, "execution_mode", "UNKNOWN"), broker_status="DISCONNECTED", llm_status="ONLINE" if self.use_llm and self.initialization_error is None else "ERROR" if self.use_llm else "DISABLED", macro_data_status=macro_status, first_run_state="STARTING" if self.first_run_observe else self.store.get_runtime("first_run_state", "NOT_ACTIVE"))
        self._restore_llm_runtime_state()
        if self.first_run_observe:
            try:
                result = run_first_run_observe(
                    executor,
                    database_path=self.store.path,
                    scheduler_config=self.cfg.get("scheduler"),
                    api_capabilities=inspect_api_capabilities(),
                )
                persist_first_run_result(self.store, result)
                if result.ready and hasattr(self.runner, "reconcile_portfolio"):
                    portfolio = self.runner.reconcile_portfolio()
                    self._set_runtime(last_reconciliation=portfolio.as_of, risk_state=portfolio.risk_state, last_error=None)
                if self.scheduler:
                    self.scheduler.stop()
                self._start_supervisor()
                return self.status()
            except Exception as exc:
                self._enter_safe_mode("first-run", exc)
                if self.scheduler:
                    self.scheduler.stop()
                self._start_supervisor()
                return self.status()
        try:
            portfolio = self.runner.reconcile_portfolio()
        except Exception as exc:
            if is_waiting_for_tws(exc):
                self.store.save_error("startup", "Waiting for TWS Paper login", {"technical_error": str(exc)}, severity="WARNING", component="backend")
                self._set_runtime(service_status="WAITING_FOR_TWS_PAPER_LOGIN", trading_enabled=False, safe_mode=False, broker_status="WAITING_FOR_LOGIN", last_error="Please log in to the IBKR Paper account in TWS", scheduler_status="STOPPED")
            else:
                self.store.save_error("startup", str(exc), severity="ERROR", component="backend")
                self._set_runtime(service_status="SAFE_MODE", trading_enabled=False, safe_mode=True, last_error=str(exc), scheduler_status="STOPPED")
            if self.scheduler:
                self.scheduler.stop()
            self._start_supervisor()
            return self.status()

        self._set_runtime(
            broker_status="CONNECTED",
            market_data_status=self.store.get_runtime("market_data_status", "OK"),
            last_error=None,
        )
        if self.initialization_error is not None:
            self._enter_safe_mode("startup", self.initialization_error)
            if self.scheduler:
                self.scheduler.stop()
            self._start_supervisor()
            return self.status()

        if self._needs_startup_risk_check(portfolio):
            try:
                risk_result = self.runner.run_daily_risk_check()
                portfolio = risk_result["portfolio"]
            except Exception as exc:
                self._enter_safe_mode("startup-risk", exc)
                if self.scheduler:
                    self.scheduler.stop()
                self._start_supervisor()
                return self.status()

        manual_halt = bool(self.store.get_runtime("manual_halt", False))
        if manual_halt:
            self._set_runtime(service_status="MANUAL_HALT", trading_enabled=False, safe_mode=False, risk_state=portfolio.risk_state, scheduler_status="STOPPED")
        else:
            if self._scheduler_enabled():
                self.scheduler.start()
            self._set_runtime(service_status="ONLINE", trading_enabled=portfolio.risk_state != "RISK_HALTED" and self._market_allows_orders(), safe_mode=False, risk_state=portfolio.risk_state, scheduler_status=self._scheduler_status())
        self._start_supervisor()
        return self.status()

    def stop(self):
        if self.scheduler:
            self.scheduler.stop()
        self._stop_event.set()
        self._heartbeat_stop_event.set()
        self._set_runtime(service_status="STOPPED", trading_enabled=False, scheduler_status="STOPPED")
        if self._control_thread and self._control_thread is not threading.current_thread():
            self._control_thread.join(timeout=2)
        if self._heartbeat_thread and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(timeout=2)
        self._control_thread = None
        self._heartbeat_thread = None
        self._started = False
        self.lock.release()

    def _start_supervisor(self):
        if self._control_thread and self._control_thread.is_alive():
            return
        self._stop_event.clear()
        stale_before = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - 60, timezone.utc).isoformat()
        self.store.recover_stale_claims(stale_before)
        abandoned = self.store.fail_abandoned_running_commands(self._worker_id)
        if abandoned:
            self.store.save_runtime_event(
                "COMMAND_WORKER_RECOVERY",
                "SERVICE",
                f"Closed {abandoned} abandoned RUNNING command(s); no command was replayed",
                metadata={"abandoned_count": abandoned, "worker_id": self._worker_id},
            )
        self._set_runtime(command_worker_status="STARTING", command_worker_id=self._worker_id)
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="ai-fund-worker-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        self._control_thread = threading.Thread(target=self._supervise, name="ai-fund-command-worker", daemon=True)
        self._control_thread.start()

    def _heartbeat_loop(self):
        while not self._heartbeat_stop_event.is_set() and not self._stop_event.is_set():
            try:
                self._write_worker_heartbeat()
            except Exception as exc:
                try:
                    self.store.save_error("worker-heartbeat", str(exc), severity="ERROR", component="backend")
                except Exception:
                    # The database may be unavailable; keep the worker alive to recover.
                    logging.getLogger(__name__).error("Heartbeat persistence failed; retrying without database error logging")
            if self._heartbeat_stop_event.wait(1.0):
                break

    def _supervise(self):
        while not self._stop_event.is_set():
            try:
                self._set_runtime(command_worker_status="RUNNING", command_worker_id=self._worker_id)
                self.process_commands()
                self.heartbeat()
            except Exception as exc:
                try:
                    self.store.save_error("supervisor", str(exc), severity="ERROR", component="backend")
                    self._set_runtime(command_worker_status="ERROR", command_worker_error=str(exc))
                except Exception:
                    logging.getLogger(__name__).error("Supervisor persistence failed; retrying without database error logging")
            self._stop_event.wait(1.0)
        self._set_runtime(command_worker_status="STOPPED")

    def heartbeat(self):
        self._write_worker_heartbeat()
        self.monitor_broker_state()
        return self.status()

    def _write_worker_heartbeat(self):
        now = datetime.now(timezone.utc).isoformat()
        self.store.set_runtime("heartbeat", now)
        self.store.set_runtime("command_worker_heartbeat", now)
        self.store.set_runtime("scheduler_status", self._scheduler_status())

    def monitor_broker_state(self):
        executor = getattr(self.runner, "executor", None)
        connected = bool(getattr(executor, "connected", False))
        reconciled = bool(getattr(executor, "broker_state_known", False))
        previous = self.store.get_runtime("broker_monitor_connected")
        previous_reconciled = self.store.get_runtime("broker_reconciliation_ready")
        recoverable_safe_mode = bool(self.store.get_runtime("safe_mode", False)) and self._is_recoverable_broker_error(
            self.store.get_runtime("last_error", "")
        )
        if (
            getattr(executor, "broker_source", None) == "IBKR_PAPER"
            and (not connected or not reconciled or recoverable_safe_mode)
            and not self._broker_auto_recovery_blocked
            and time.monotonic() >= self._next_broker_reconnect_at
        ):
            connected, reconciled = self._recover_ibkr_after_login(executor)
        broker_status = "CONNECTED" if connected and reconciled else "RECONCILIATION_REQUIRED" if connected else "DISCONNECTED"
        self._set_runtime(
            broker_monitor_status="RUNNING",
            broker_monitor_heartbeat=datetime.now(timezone.utc).isoformat(),
            broker_monitor_connected=connected,
            broker_reconciliation_ready=reconciled,
            broker_status=broker_status,
        )
        if connected and reconciled and previous_reconciled is not True:
            if previous_reconciled is not None:
                event_type = "BROKER_RECONNECTED"
                self.trigger_risk_event(event_type, source="BROKER", severity="HIGH", metadata={"connected": connected, "reconciled": reconciled})
            self._queue_pending_execution_resume()
        elif previous is not None and bool(previous) and not connected:
            self.trigger_risk_event("BROKER_DISCONNECTED", source="BROKER", severity="HIGH", metadata={"connected": False, "reconciled": False})
        return {"connected": connected, "reconciled": reconciled}

    def _recover_ibkr_after_login(self, executor):
        interval = float(self.cfg.get("execution", {}).get("broker_reconnect_interval_seconds", 15))
        self._next_broker_reconnect_at = time.monotonic() + max(0.0, interval)
        if self.scheduler:
            self.scheduler.stop()
        self._set_runtime(
            service_status="RECONNECTING",
            trading_enabled=False,
            broker_status="RECONCILIATION_REQUIRED" if getattr(executor, "connected", False) else "DISCONNECTED",
            scheduler_status="STOPPED",
        )
        try:
            with self._cycle_lock:
                if not bool(getattr(executor, "connected", False)):
                    executor.reconnect()
                portfolio = self.runner.reconcile_portfolio()
        except Exception as exc:
            if is_waiting_for_tws(exc) and not self._is_recoverable_broker_error(exc):
                self._set_runtime(
                    service_status="WAITING_FOR_TWS_PAPER_LOGIN",
                    safe_mode=False,
                    trading_enabled=False,
                    broker_status="WAITING_FOR_LOGIN",
                    last_error="Please log in to the IBKR Paper account in TWS",
                )
            elif self._is_recoverable_broker_error(exc):
                self._enter_safe_mode("broker-recovery", exc)
            else:
                self._broker_auto_recovery_blocked = True
                self._enter_safe_mode("broker-recovery", exc)
            return bool(getattr(executor, "connected", False)), False

        manual_halt = bool(self.store.get_runtime("manual_halt", False))
        llm_failed = bool(self.use_llm and self.store.get_runtime("llm_status") == "ERROR")
        risk_halted = portfolio.risk_state == "RISK_HALTED"
        if not manual_halt and self._scheduler_enabled():
            self.scheduler.start()
        self._set_runtime(
            service_status="MANUAL_HALT" if manual_halt else "ONLINE",
            safe_mode=False,
            trading_enabled=not manual_halt and not llm_failed and not risk_halted and self._market_allows_orders(),
            broker_status="CONNECTED",
            broker_reconciliation_ready=True,
            risk_state=portfolio.risk_state,
            last_reconciliation=getattr(portfolio, "as_of", datetime.now(timezone.utc).isoformat()),
            last_error=None,
            scheduler_status=self._scheduler_status(),
        )
        self.store.save_runtime_event(
            "BROKER_RECOVERY_COMPLETED",
            "BROKER",
            "TWS connection restored and full broker reconciliation completed",
            metadata={"scheduler_status": self._scheduler_status(), "risk_state": portfolio.risk_state},
        )
        return True, True

    @staticmethod
    def _is_recoverable_broker_error(error) -> bool:
        message = str(error or "").lower()
        hard_failures = (
            "unknown broker open order",
            "live account",
            "multiple long positions",
            "short positions",
            "unsupported security",
            "unsupported currency",
            "paper account",
        )
        if any(marker in message for marker in hard_failures):
            return False
        return any(
            marker in message
            for marker in (
                "execution quote timed out",
                "market data",
                "stale quote",
                "quote unavailable",
                "is disconnected",
                "ibkr transport is not connected after readiness callbacks",
            )
        )

    def trigger_risk_event(self, event_type: str, *, source: str, symbol: str | None = None, severity: str = "MEDIUM", metadata: dict | None = None) -> int:
        details = {**(metadata or {}), "source": source, "severity": severity, "risk_action": "PENDING"}
        event_id = self.store.save_runtime_event(event_type, source, f"{event_type} triggered immediate risk review", symbol=symbol, metadata=details)
        self.store.enqueue_command(
            "RUN_IMMEDIATE_RISK_CHECK",
            {"event_id": event_id, "event_type": event_type, "source": source, "symbol": symbol, "severity": severity, "metadata": metadata or {}},
            source="EVENT",
            trigger_reason=event_type,
        )
        return event_id

    def trigger_position_event(
        self,
        event_id: str,
        event_type: str,
        *,
        symbol: str | None = None,
        source: str = "MONITOR",
        severity: str = "MEDIUM",
        metadata: dict | None = None,
    ) -> bool:
        manager = getattr(self.runner, "position_manager", None)
        if manager is None:
            raise RuntimeError("Position Manager is not configured")
        position = self.store.active_managed_position(symbol)
        if position is None:
            return False
        accepted = manager.register_trigger(
            position,
            event_id,
            event_type,
            source=source,
            severity=severity,
            metadata=metadata,
        )
        if accepted:
            self.store.enqueue_command(
                "RUN_EVENT_SOL_REVIEW",
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "source": source,
                    "symbol": position.symbol,
                    "severity": severity,
                    "metadata": metadata or {},
                },
                source="POSITION_MONITOR",
                trigger_reason=event_type,
            )
        return accepted

    def process_commands(self):
        with self._command_process_lock:
            while command := self.store.claim_next_command(self._worker_id):
                command_id = command["id"]
                name = command["command"].upper()
                if not self.store.mark_command_running(command_id, self._worker_id):
                    continue
                try:
                    result = self._handle_command(name, json.loads(command["payload_json"]))
                    self.store.complete_command(command_id, result=result, worker_id=self._worker_id)
                except Exception as exc:
                    self.store.save_error("control", str(exc), {"command": name}, severity="ERROR", component="backend")
                    self.store.complete_command(command_id, error=str(exc), error_stage=name, worker_id=self._worker_id)
            return self.status()

    def run_reconciliation_now(self):
        try:
            with self._cycle_lock:
                portfolio = self.runner.reconcile_portfolio()
        except Exception as exc:
            self._enter_safe_mode("reconciliation", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise
        manual_halt = bool(self.store.get_runtime("manual_halt", False))
        llm_failed = bool(self.use_llm and self.store.get_runtime("llm_status") == "ERROR")
        self._set_runtime(
            service_status="MANUAL_HALT" if manual_halt else "ONLINE",
            safe_mode=False,
            trading_enabled=not manual_halt and not llm_failed and portfolio.risk_state != "RISK_HALTED" and self._market_allows_orders(),
            risk_state=portfolio.risk_state,
            last_error=None,
        )
        self._queue_pending_execution_resume()
        return portfolio.model_dump(mode="json")

    def run_risk_check_now(self):
        try:
            with self._cycle_lock:
                result = self.runner.run_daily_risk_check()
        except Exception as exc:
            self._enter_safe_mode("risk-check", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise
        pending_entries = self.run_pending_entries() if hasattr(self.runner, "process_pending_entries") else []
        self._set_runtime(risk_state=result["portfolio"].risk_state)
        self.store.set_runtime("last_full_risk_check", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event("FULL_RISK_CHECK", "RISK_ENGINE", f"Full risk check completed: {self._risk_action(result)}", metadata={"risk_action": self._risk_action(result)})
        output = self._json_result(result)
        output["pending_entries"] = pending_entries
        return output

    def run_ai_research_now(self):
        self._ensure_transaction_allowed()
        self.store.save_runtime_event("FULL_RESEARCH_STARTED", "PIPELINE", "Full Luna -> Sol research started")
        try:
            self._ensure_runtime_llm_probe()
            with self._cycle_lock:
                result = self.runner.run(use_llm=self.use_llm)
        except PipelineError as exc:
            self._fail_closed_llm(exc)
            raise
        state = str(result.get("state"))
        if state == "WAITING_FOR_BROKER":
            self._pending_weekly_intent = result.get("intent")
            self._persist_runner_waiting_execution(result, "MANUAL_FULL_RESEARCH")
            self._mark_waiting_for_broker(result, "MANUAL_FULL_RESEARCH")
            self.store.set_runtime("last_full_research", datetime.now(timezone.utc).isoformat())
            self.store.save_runtime_event(
                "FULL_RESEARCH_WAITING_FOR_BROKER",
                "PIPELINE",
                "Luna -> Sol research completed; execution is waiting for broker recovery",
                decision_id=getattr(result.get("intent"), "decision_id", None),
            )
            return self._json_result(result)
        if self.first_run_observe:
            self.store.set_runtime("first_run_completed", True)
            self._set_runtime(first_run_state="OBSERVE_RESEARCH_COMPLETED", first_run_ready=False, trading_enabled=False, scheduler_status="STOPPED")
        self._set_runtime(risk_state=result["portfolio"].risk_state, last_ai_decision=result["intent"].timestamp)
        self._persist_runner_waiting_execution(result, "MANUAL_FULL_RESEARCH")
        self.store.set_runtime("last_full_research", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event("FULL_RESEARCH_COMPLETED", "PIPELINE", "Full Luna -> Sol research completed", decision_id=result["intent"].decision_id)
        return self._json_result(result)

    def run_sol_decision_resume(self, payload: dict):
        self._ensure_transaction_allowed()
        source_run_id = str((payload or {}).get("source_run_id") or "").strip()
        if not source_run_id:
            raise ValueError("RUN_SOL_DECISION_RESUME requires source_run_id")
        self.store.save_runtime_event(
            "SOL_RESUME_STARTED",
            "PIPELINE",
            "Resuming Sol from persisted Luna candidates and research evidence",
            metadata={"source_run_id": source_run_id},
        )
        try:
            self._ensure_runtime_llm_probe()
            with self._cycle_lock:
                result = self.runner.run(use_llm=self.use_llm, resume_source_run_id=source_run_id)
        except PipelineError as exc:
            self._fail_closed_llm(exc)
            raise
        self._set_runtime(risk_state=result["portfolio"].risk_state, last_ai_decision=result["intent"].timestamp)
        self._persist_runner_waiting_execution(result, "SOL_DECISION_RESUME")
        self.store.set_runtime("last_full_research", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event(
            "SOL_RESUME_COMPLETED",
            "PIPELINE",
            "Resumed Sol decision completed",
            decision_id=result["intent"].decision_id,
            metadata={"source_run_id": source_run_id},
        )
        return self._json_result(result)

    def emergency_stop(self):
        self.store.set_runtime("manual_halt", True)
        if self.scheduler:
            self.scheduler.stop()
        try:
            with self._cycle_lock:
                result = self.runner.emergency_stop()
            portfolio = result["portfolio"]
            self._set_runtime(service_status="MANUAL_HALT", trading_enabled=False, safe_mode=False, scheduler_status="STOPPED", risk_state=portfolio.risk_state)
            return self._json_result(result)
        except Exception as exc:
            self._set_runtime(service_status="MANUAL_HALT", trading_enabled=False, safe_mode=True, scheduler_status="STOPPED", last_error=str(exc))
            raise

    def resume_manual_halt(self):
        if self.store.get_runtime("safe_mode", False):
            raise RuntimeError("Cannot resume while backend is in SAFE_MODE")
        with self._cycle_lock:
            portfolio = self.runner.reconcile_portfolio()
        if portfolio.risk_state == "RISK_HALTED":
            raise RuntimeError("Risk halt remains active; manual halt cannot resume trading")
        self.store.set_runtime("manual_halt", False)
        if self._scheduler_enabled():
            self.scheduler.start()
        self._set_runtime(
            service_status="ONLINE",
            trading_enabled=not (self.use_llm and self.store.get_runtime("llm_status") == "ERROR"),
            risk_state=portfolio.risk_state,
            scheduler_status=self._scheduler_status(),
        )
        return self.status()

    def status(self):
        current = self.store.runtime_snapshot()
        current["scheduler_status"] = self._scheduler_status()
        current.setdefault("trading_enabled", False)
        return current

    def _scheduler_status(self):
        if not self.scheduler:
            return "STOPPED"
        return self.scheduler.status().get("status", "STOPPED")

    def _scheduler_enabled(self):
        if not self.scheduler:
            return False
        config = getattr(self.scheduler, "config", None)
        return bool(getattr(config, "enabled", True))

    def _market_allows_orders(self) -> bool:
        return self.store.get_runtime("market_data_status", "OK") not in {"MARKET_CLOSED", "STALE", "ERROR", "UNAVAILABLE"}

    def _needs_startup_risk_check(self, portfolio) -> bool:
        if getattr(portfolio, "risk_state", None) == "RISK_HALTED":
            return True
        try:
            drawdown = float(getattr(portfolio, "drawdown", 0.0))
        except (TypeError, ValueError):
            return False
        risk_cfg = self.cfg.get("risk", {})
        thresholds = [
            float(item["drawdown"])
            for item in risk_cfg.get("drawdown_tiers", [])
            if float(item.get("max_weight", 1.0)) < 1.0
        ]
        thresholds.append(float(risk_cfg.get("hard_drawdown_limit", 0.25)))
        return bool(thresholds) and drawdown >= min(thresholds) - float(risk_cfg.get("comparison_tolerance", 1e-9))

    def _ensure_transaction_allowed(self):
        if self.store.get_runtime("safe_mode", False):
            raise RuntimeError("AI transaction cycle is disabled in SAFE_MODE")
        if self.store.get_runtime("manual_halt", False):
            raise RuntimeError("AI transaction cycle is disabled in MANUAL_HALT")
        if self.store.get_runtime("risk_state", "NORMAL") == "RISK_HALTED":
            raise RuntimeError("AI transaction cycle is disabled in RISK_HALTED")
        if self.store.get_runtime("service_status") == "WAITING_FOR_TWS_PAPER_LOGIN":
            raise RuntimeError("AI transaction cycle is disabled while waiting for TWS Paper login")

    def _ensure_runtime_llm_probe(self):
        """Require a passing probe through the same Luna runtime path before AI research."""
        if not self.use_llm:
            return
        probe = getattr(self.runner, "probe_runtime_llm_exact_path", None)
        if not callable(probe):
            return
        signature = self._llm_runtime_signature()
        stored = self.store.get_runtime("llm_runtime_exact_probe")
        if isinstance(stored, dict) and stored.get("status") == "PASS" and stored.get("signature") == signature:
            self._mark_llm_runtime_ready()
            return
        agent = getattr(self.runner, "agent", None)
        try:
            if agent is not None and hasattr(agent, "set_event_sink") and hasattr(self.runner, "_agent_event_sink"):
                agent.set_event_sink(self.runner._agent_event_sink)
            result = probe()
        except Exception as exc:
            result = {"status": "FAIL", "ok": False, "error": str(exc)}
        if not isinstance(result, dict) or result.get("status") != "PASS" or result.get("ok") is not True:
            failure = result.get("error", "probe did not pass") if isinstance(result, dict) else "probe did not return a valid result"
            persisted = dict(result) if isinstance(result, dict) else {"error": str(failure)}
            persisted.update({"status": "FAIL", "ok": False, "signature": signature})
            self.store.set_runtime("llm_runtime_exact_probe", persisted)
            raise PipelineError(f"runtime exact probe failed: {failure}")
        passed = dict(result)
        passed["status"] = "PASS"
        passed["ok"] = True
        passed["signature"] = signature
        self.store.set_runtime("llm_runtime_exact_probe", passed)
        self._mark_llm_runtime_ready()

    def _llm_runtime_signature(self) -> dict:
        agent = getattr(self.runner, "agent", None)
        runtime = getattr(agent, "runtime", None)
        provider = getattr(agent, "provider", None)
        return {
            "luna_model": getattr(runtime, "luna_model", None),
            "sol_model": getattr(runtime, "sol_model", None),
            "api_protocol": getattr(runtime, "api_protocol", getattr(provider, "protocol", None)),
            "protocol_fallback": getattr(runtime, "protocol_fallback", getattr(provider, "protocol_fallback", None)),
            "timeout_seconds": getattr(runtime, "timeout_seconds", getattr(provider, "timeout_seconds", None)),
        }

    def _mark_llm_runtime_ready(self, *, clear_previous_error: bool = True):
        previous_llm_status = self.store.get_runtime("llm_status")
        previous_gateway_status = self.store.get_runtime("llm_gateway_status")
        self._set_runtime(
            llm_status="ONLINE",
            llm_gateway_status="ONLINE",
            llm_luna_status="ONLINE",
        )
        if self.store.get_runtime("llm_sol_status") in {"ERROR", "OFFLINE", None}:
            self.store.set_runtime("llm_sol_status", "ONLINE")
        if clear_previous_error and self.store.get_runtime("last_error") and (
            previous_llm_status in {"ERROR", "OFFLINE"}
            or previous_gateway_status in {"ERROR", "OFFLINE"}
        ):
            self.store.set_runtime("last_error", None)

    def _restore_llm_runtime_state(self):
        """Do not expose a previous transient LLM failure as the current state after restart."""
        if not self.use_llm or self.initialization_error is not None:
            return
        signature = self._llm_runtime_signature()
        stored_probe = self.store.get_runtime("llm_runtime_exact_probe")
        probe_verified = (
            isinstance(stored_probe, dict)
            and stored_probe.get("status") == "PASS"
            and stored_probe.get("signature") == signature
        )
        capability_verified = self.store.get_runtime("llm_capability_status") == "PASS"
        if probe_verified:
            self._mark_llm_runtime_ready(clear_previous_error=False)
            if capability_verified:
                self.store.set_runtime("llm_sol_status", "ONLINE")
            return
        if capability_verified:
            self._set_runtime(
                llm_status="ONLINE",
                llm_gateway_status="UNKNOWN",
                llm_luna_status="UNKNOWN",
                llm_sol_status="ONLINE",
            )
            return
        self._set_runtime(
            llm_status="ONLINE",
            llm_gateway_status="UNKNOWN",
            llm_luna_status="UNKNOWN",
            llm_sol_status="UNKNOWN",
        )

    def _set_runtime(self, **values):
        for key, value in values.items():
            self.store.set_runtime(key, value)

    def _handle_command(self, name: str, payload: dict):
        if name == "START_SCHEDULER":
            if self.store.get_runtime("safe_mode", False) or self.store.get_runtime("manual_halt", False):
                raise RuntimeError("Scheduler cannot start in SAFE_MODE or MANUAL_HALT")
            if self._scheduler_enabled():
                self.scheduler.start()
            self._set_runtime(
                service_status="ONLINE",
                trading_enabled=self.store.get_runtime("risk_state", "NORMAL") != "RISK_HALTED" and not (self.use_llm and self.store.get_runtime("llm_status") == "ERROR"),
                scheduler_status=self._scheduler_status(),
            )
            return self.status()
        if name == "STOP_SCHEDULER":
            if self.scheduler:
                self.scheduler.stop()
            self._set_runtime(scheduler_status="STOPPED")
            return self.status()
        if name == "RUN_RECONCILIATION":
            return self.run_reconciliation_now()
        if name in {"RUN_RISK_CHECK", "RUN_FULL_RISK_CHECK"}:
            return self.run_risk_check_now()
        if name == "RUN_FULL_AI_RESEARCH":
            return self.run_weekly_cycle(use_llm=self.use_llm)
        if name == "RUN_AI_RESEARCH":
            return self.run_ai_research_now()
        if name == "RUN_SOL_DECISION_RESUME":
            return self.run_sol_decision_resume(payload)
        if name == "RUN_IMMEDIATE_RISK_CHECK":
            return self.run_immediate_risk_check(payload)
        if name == "RUN_DAILY_POSITION_REVIEW":
            return self.run_position_review(event_context=None)
        if name == "RUN_EVENT_SOL_REVIEW":
            return self.run_position_review(event_context=payload)
        if name == "RUN_PRE_EXECUTION_REVALIDATION":
            return self.run_pre_execution_revalidation(payload)
        if name == "RUN_PENDING_ENTRIES":
            return self.run_pending_entries()
        if name == "RUN_POSITION_MONITOR":
            return self.run_position_monitor()
        if name == "EMERGENCY_STOP":
            return self.emergency_stop()
        if name == "RESUME_MANUAL_HALT":
            return self.resume_manual_halt()
        raise ValueError(f"Unknown backend command: {name}")

    def run_pending_entries(self):
        results = self.runner.process_pending_entries() if hasattr(self.runner, "process_pending_entries") else []
        if any(result.get("status") == "DECISION_EXPIRED" for result in results):
            self._queue_research_refresh("ENTRY_EXECUTION")
        pending = self.store.get_runtime("pending_execution_intent")
        if isinstance(pending, dict) and pending.get("intent"):
            clock = getattr(self.runner, "market_clock", None)
            if clock is not None and not clock.is_open():
                results.append({"decision_id": pending["intent"].get("decision_id"), "status": "WAITING_FOR_MARKET", "reason": "Market is closed"})
                return results
            intent = TradeIntent.model_validate(pending["intent"])
            if not self._decision_is_fresh(intent):
                self.store.set_runtime("pending_execution_intent", None)
                self._queue_research_refresh("PENDING_ADJUSTMENT")
                results.append({"decision_id": intent.decision_id, "status": "DECISION_EXPIRED", "reason": "Fresh AI research is required"})
                return results
            try:
                with self._cycle_lock:
                    execution = self.runner.run(intent=intent, use_llm=False)
            except BrokerReadinessError as exc:
                self._mark_waiting_for_broker({"intent": intent, "broker_waiting_reason": str(exc)}, "BROKER_RECOVERY")
                results.append({"decision_id": intent.decision_id, "status": "WAITING_FOR_BROKER", "reason": str(exc)})
                return results
            state = str(execution.get("state"))
            self._persist_runner_waiting_execution(execution, "BROKER_RECOVERY")
            if state == "WAITING_FOR_BROKER":
                self._mark_waiting_for_broker(execution, "BROKER_RECOVERY")
            if state not in {"WAITING_FOR_MARKET", "WAITING_FOR_BROKER", "PENDING_ENTRY", "REJECTED"}:
                self.store.set_runtime("pending_execution_intent", None)
            results.append({"decision_id": intent.decision_id, "status": state, "result": self._json_result(execution)})
        return results

    def _queue_research_refresh(self, source: str):
        active_research = any(
            row.get("command") in {"RUN_AI_RESEARCH", "RUN_FULL_AI_RESEARCH"}
            and row.get("status") in {"QUEUED", "CLAIMED", "RUNNING"}
            for row in self.store.recent("control_commands", 50)
        )
        if active_research:
            return
        command_id = self.store.enqueue_command(
            "RUN_FULL_AI_RESEARCH",
            source=source,
            trigger_reason="PENDING_DECISION_EXPIRED",
        )
        self.store.save_runtime_event(
            "AI_RESEARCH_REFRESH_QUEUED",
            "ENTRY_EXECUTION",
            "Expired pending decision requires fresh AI research",
            metadata={"command_id": command_id},
        )

    def _queue_pending_execution_resume(self):
        pending = self.store.get_runtime("pending_execution_intent")
        if not isinstance(pending, dict) or not pending.get("intent"):
            return None
        command_id, created = self.store.enqueue_command_unless_active(
            "RUN_PENDING_ENTRIES",
            active_commands=("RUN_PENDING_ENTRIES",),
            source="BROKER_RECOVERY",
            trigger_reason="BROKER_RECONNECTED",
        )
        if created:
            self.store.save_runtime_event(
                "PENDING_EXECUTION_RESUME_QUEUED",
                "BROKER",
                "Broker recovery queued the deferred execution revalidation",
                decision_id=pending["intent"].get("decision_id"),
                symbol=pending["intent"].get("symbol"),
                metadata={"command_id": command_id},
            )
        return command_id

    def _mark_waiting_for_broker(self, result: dict, source: str):
        reason = str(result.get("broker_waiting_reason") or result.get("reason") or "Broker readiness is required before execution")
        self._enter_safe_mode("broker-execution", BrokerReadinessError(reason))
        executor = getattr(self.runner, "executor", None)
        connected = bool(getattr(executor, "connected", False))
        reconciled = bool(getattr(executor, "broker_state_known", False))
        self._set_runtime(
            broker_status="CONNECTED" if connected and reconciled else "RECONCILIATION_REQUIRED" if connected else "DISCONNECTED",
            broker_reconciliation_ready=connected and reconciled,
            scheduler_status="STOPPED",
        )
        if self.scheduler:
            self.scheduler.stop()
        intent = result.get("intent")
        if isinstance(intent, TradeIntent):
            decision_id = intent.decision_id
            symbol = intent.symbol
        elif isinstance(intent, dict):
            decision_id = intent.get("decision_id")
            symbol = intent.get("symbol")
        else:
            decision_id = None
            symbol = None
        self.store.save_runtime_event(
            "BROKER_EXECUTION_WAITING",
            "BROKER",
            "Execution is waiting for broker recovery and full reconciliation",
            decision_id=decision_id,
            symbol=symbol,
            metadata={"source": source, "reason": reason},
        )

    def _persist_waiting_execution(self, result: dict, source: str):
        decision = result.get("decision")
        execution = result.get("execution")
        if not isinstance(decision, dict) or not isinstance(execution, dict):
            return
        state = str(execution.get("state") or "")
        risk = execution.get("risk")
        reason = risk.get("reason", "") if isinstance(risk, dict) else str(getattr(risk, "reason", ""))
        transient = state in {"WAITING_FOR_MARKET", "WAITING_FOR_BROKER", "PENDING_ENTRY"} or (
            state == "REJECTED" and any(marker in reason.lower() for marker in ("market", "quote", "stale", "行情", "休市"))
        )
        if not transient:
            return
        self.store.set_runtime("pending_execution_intent", {
            "intent": decision,
            "source": source,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason or state,
        })
        self.store.save_runtime_event(
            "EXECUTION_DEFERRED",
            "EXECUTION",
            f"{decision.get('action', 'ADJUSTMENT')} deferred to the next valid market window",
            decision_id=decision.get("decision_id"),
            symbol=decision.get("symbol"),
            metadata={"source": source, "state": state, "reason": reason},
        )

    def _persist_runner_waiting_execution(self, result: dict, source: str):
        intent = result.get("intent")
        if not isinstance(intent, TradeIntent):
            return
        state = str(result.get("state") or "")
        if intent.action == "BUY" and state != "WAITING_FOR_BROKER":
            return
        risk = result.get("risk")
        risk_values = (
            risk.model_dump(mode="json")
            if hasattr(risk, "model_dump")
            else {"reason": getattr(risk, "reason", "")}
        )
        self._persist_waiting_execution(
            {
                "decision": intent.model_dump(mode="json"),
                "execution": {
                    "state": str(result.get("state") or ""),
                    "risk": risk_values,
                },
            },
            source,
        )

    def run_position_review(self, event_context: dict | None = None):
        event_type = "EVENT_SOL_REVIEW" if event_context else "DAILY_POSITION_REVIEW"
        self.store.save_runtime_event(event_type, "SOL", f"{event_type} started", metadata=event_context or {})
        with self._cycle_lock:
            result = self.runner.run_position_review(event_context=event_context)
        now = datetime.now(timezone.utc).isoformat()
        key = "last_event_sol_review" if event_context else "last_daily_review"
        self._persist_waiting_execution(result, event_type)
        self._set_runtime(**{key: now, "current_position_review": result.get("review_action", "REVIEW_REQUIRED")})
        self.store.save_runtime_event(event_type, "SOL", f"{event_type} completed: {result.get('review_action', 'REVIEW_REQUIRED')}", metadata=result)
        return result

    def run_position_monitor(self):
        try:
            with self._cycle_lock:
                result = self.runner.run_position_monitor()
        except Exception as exc:
            self._enter_safe_mode("position-monitor", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise
        execution = result.get("execution") if isinstance(result, dict) else None
        if isinstance(execution, dict):
            self._persist_runner_waiting_execution(execution, "POSITION_MONITOR")
        if result.get("monitor_action") == "REVIEW_REQUIRED":
            result["review_queued"] = self.trigger_position_event(
                result["event_id"],
                result["event_type"],
                symbol=result.get("symbol"),
                source="POSITION_MONITOR",
                severity=result.get("severity", "HIGH"),
                metadata=result.get("metadata") or {},
            )
        self.store.set_runtime("last_position_monitor", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event(
            "POSITION_MONITOR_COMPLETED",
            "POSITION_MONITOR",
            f"Position monitor completed: {result.get('monitor_action', 'UNKNOWN')}",
            symbol=result.get("symbol"),
            metadata={key: value for key, value in result.items() if key != "execution"},
        )
        return self._json_result(result)

    def run_pre_execution_revalidation(self, payload: dict | None = None):
        if hasattr(self.runner, "process_pending_entries") and self.store.pending_entries(("PENDING_ENTRY", "ENTRY_REVIEW_REQUIRED")):
            result = self.run_pending_entries()
            self.store.set_runtime("last_pre_execution_revalidation", datetime.now(timezone.utc).isoformat())
            return result
        intent = self._pending_weekly_intent
        result = self.runner.pre_execution_revalidate(intent=intent, context=payload or {})
        self.store.set_runtime("last_pre_execution_revalidation", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event("PRE_EXECUTION_REVALIDATION", "EXECUTION", result.get("reason", "pre-execution revalidation completed"), decision_id=getattr(intent, "decision_id", None), metadata=result)
        return result

    def run_immediate_risk_check(self, payload: dict | None = None):
        result = self.run_risk_check_now()
        action = self._risk_action(result)
        metadata = {**(payload or {}), "risk_action": action}
        self.store.set_runtime("last_immediate_risk_check", datetime.now(timezone.utc).isoformat())
        self.store.save_runtime_event("IMMEDIATE_RISK_CHECK", "RISK_ENGINE", f"Immediate risk action: {action}", symbol=(payload or {}).get("symbol"), metadata=metadata)
        return {**result, "risk_action": action}

    @staticmethod
    def _risk_action(result: dict) -> str:
        portfolio = result.get("portfolio")
        risk = result.get("risk")
        if getattr(portfolio, "risk_state", None) == "RISK_HALTED":
            return "FORCE_CASH"
        if risk is None:
            return "NO_ACTION"
        if getattr(risk, "forced_cash", False):
            return "FORCE_CASH"
        if not getattr(risk, "approved", True):
            return "BLOCK_NEW_ENTRY"
        if getattr(risk, "approved_weight", 1.0) < getattr(risk, "requested_weight", 0.0):
            return "REDUCE_MAX_WEIGHT"
        return "HOLD"

    def run_weekly_cycle(self, use_llm: bool = True):
        self._ensure_transaction_allowed()
        try:
            self._ensure_runtime_llm_probe()
            with self._cycle_lock:
                pending = self._pending_weekly_intent
                if pending is not None and self._decision_is_fresh(pending):
                    result = self.runner.run(intent=pending, use_llm=False)
                else:
                    self._pending_weekly_intent = None
                    result = self.runner.run(use_llm=use_llm)
            state = str(result["state"])
            self._pending_weekly_intent = result["intent"] if state in {"WAITING_FOR_MARKET", "WAITING_FOR_BROKER", "PENDING_ENTRY"} else None
            self._persist_runner_waiting_execution(result, "WEEKLY_FULL_RESEARCH")
            if state == "WAITING_FOR_BROKER":
                self._mark_waiting_for_broker(result, "WEEKLY_FULL_RESEARCH")
                return result
            self._set_runtime(
                service_status="ONLINE",
                safe_mode=False,
                trading_enabled=result["portfolio"].risk_state != "RISK_HALTED",
                llm_status="ONLINE" if use_llm else self.store.get_runtime("llm_status"),
                llm_gateway_status="ONLINE" if use_llm else self.store.get_runtime("llm_gateway_status"),
                risk_state=result["portfolio"].risk_state,
                last_ai_decision=result["intent"].timestamp,
            )
            return result
        except Exception as exc:
            if isinstance(exc, PipelineError):
                self._fail_closed_llm(exc)
                raise
            self._enter_safe_mode("weekly", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise

    def _decision_is_fresh(self, intent) -> bool:
        try:
            timestamp = datetime.fromisoformat(intent.timestamp.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (AttributeError, TypeError, ValueError):
            return False
        limit = float(self.cfg.get("risk", {}).get("max_decision_age_minutes_for_execution", 60))
        age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 60
        return 0 <= age <= limit

    def run_daily_cycle(self):
        try:
            with self._cycle_lock:
                result = self.runner.run_daily_risk_check()
            manual_halt = bool(self.store.get_runtime("manual_halt", False))
            llm_failed = bool(self.use_llm and self.store.get_runtime("llm_status") == "ERROR")
            self._set_runtime(service_status="MANUAL_HALT" if manual_halt else "ONLINE", safe_mode=False, trading_enabled=not manual_halt and not llm_failed and result["portfolio"].risk_state != "RISK_HALTED", risk_state=result["portfolio"].risk_state)
            return result
        except Exception as exc:
            self._enter_safe_mode("daily", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise

    def run_risk_monitor_cycle(self):
        try:
            with self._cycle_lock:
                result = self.runner.run_daily_risk_check(record_performance=False)
            self._set_runtime(risk_state=result["portfolio"].risk_state)
            return result
        except Exception as exc:
            self._enter_safe_mode("risk-monitor", exc)
            if self.scheduler:
                self.scheduler.stop()
            raise

    def _enter_safe_mode(self, stage: str, exc: Exception):
        message = str(exc)
        self.store.save_error(stage, message, severity="ERROR", component="backend")
        self.store.save_runtime_event(
            "BROKER_ERROR",
            "BROKER",
            f"{stage}: {message}",
            run_id=getattr(self.runner, "_active_run_id", None),
            decision_id=getattr(self.runner, "_active_decision_id", None),
            metadata={"stage": stage, "error_type": exc.__class__.__name__},
        )
        self.store.save_runtime_event(
            "SAFE_MODE",
            "SERVICE",
            f"{stage}: {message}",
            run_id=getattr(self.runner, "_active_run_id", None),
            decision_id=getattr(self.runner, "_active_decision_id", None),
            metadata={"stage": stage, "error_type": exc.__class__.__name__},
        )
        self._set_runtime(service_status="SAFE_MODE", safe_mode=True, trading_enabled=False, last_error=message, scheduler_status="STOPPED")

    def _fail_closed_llm(self, exc: Exception):
        """Disable new AI decisions while leaving intraday risk monitoring alive."""
        details = self._llm_failure_details(exc)
        message = details["safe_response_message"]
        self.store.save_error("llm", message, details, severity="ERROR", component="llm")
        self.store.save_runtime_event(
            "LLM_ERROR",
            "LLM",
            message,
            run_id=getattr(self.runner, "_active_run_id", None),
            decision_id=getattr(self.runner, "_active_decision_id", None),
            metadata=details,
        )
        gateway_status = "PASS" if details.get("http_status") is not None else "OFFLINE"
        stage = str(details.get("stage", ""))
        luna_status = "ERROR" if stage.startswith("LUNA") else self.store.get_runtime("llm_luna_status", "UNKNOWN")
        sol_status = "ERROR" if stage.startswith("SOL") else self.store.get_runtime("llm_sol_status", "UNKNOWN")
        self._set_runtime(
            service_status="ONLINE",
            safe_mode=False,
            trading_enabled=False,
            llm_status="ERROR",
            llm_gateway_status=gateway_status,
            llm_luna_status=luna_status,
            llm_sol_status=sol_status,
            last_error=message,
        )

    def _llm_failure_details(self, exc: Exception) -> dict:
        root = exc
        while getattr(root, "__cause__", None) is not None:
            root = root.__cause__
        agent = getattr(self.runner, "agent", None)
        provider = getattr(agent, "provider", None)
        diagnostics = getattr(provider, "error_diagnostics", None)
        details = diagnostics(root) if callable(diagnostics) else {}
        if not isinstance(details, dict):
            details = {}
        text = str(root) or str(exc) or exc.__class__.__name__
        status = details.get("http_status")
        if status is None:
            match = re.search(r"\bHTTP(?:\s+status)?[=: ]+(\d{3})\b", str(exc), re.IGNORECASE)
            status = int(match.group(1)) if match else None
        runtime = getattr(agent, "runtime", None)
        lower = str(exc).lower()
        if "luna" in lower:
            stage = "LUNA_SCREENING"
            model = getattr(runtime, "luna_model", None)
        elif "sol" in lower:
            stage = "SOL_RESEARCH"
            model = getattr(runtime, "sol_model", None)
        elif "probe" in lower:
            stage = "RUNTIME_EXACT_PROBE"
            model = getattr(runtime, "luna_model", None)
        else:
            stage = "LLM"
            model = getattr(runtime, "sol_model", None)
        safe_message = details.get("safe_response_message") or text
        if status is not None and not re.search(r"\b(?:HTTP|status)\b", safe_message, re.IGNORECASE):
            safe_message = f"HTTP {status}: {safe_message}"
        return {
            "stage": stage,
            "provider": getattr(provider, "gateway", "ccswitch"),
            "model": model,
            "protocol": getattr(provider, "protocol", getattr(runtime, "api_protocol", "UNKNOWN")),
            "http_status": status,
            "provider_error_type": details.get("provider_error_type"),
            "provider_error_code": details.get("provider_error_code"),
            "safe_response_message": safe_message,
            "retryable": bool(details.get("retryable", False)),
            "error_type": exc.__class__.__name__,
        }

    @staticmethod
    def _json_result(result):
        if isinstance(result, dict):
            return {key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value for key, value in result.items()}
        return result
