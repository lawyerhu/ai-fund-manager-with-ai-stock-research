from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from functools import wraps
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .models import BrokerSnapshot, ExecutionReport, ManagedPosition, OrderRequest, PendingEntry, PortfolioState, PositionReview, PositionTrigger, RiskDecision, TradeIntent


_SECRET_ENV_NAMES = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "CC_SWITCH_TOKEN",
    "IBKR_PASSWORD",
    "IBKR_USERNAME",
    "IBKR_ACCOUNT",
)
_SECRET_KEY_PATTERN = re.compile(r"(?:api[_-]?key|token|secret|password|credential|authorization|account[_-]?(?:number|id|code))", re.IGNORECASE)
_SECRET_TOKEN_PATTERN = re.compile(r"(?i)\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b")
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def _redact(value: Any, key: str | None = None, secrets: tuple[str, ...] | None = None):
    if secrets is None:
        secrets = tuple(secret for env_name in _SECRET_ENV_NAMES if (secret := os.getenv(env_name)))
    if key:
        normalized_key = key.lower()
        if any(marker in normalized_key for marker in ("key", "token", "secret", "password", "credential", "authorization", "account")) and _SECRET_KEY_PATTERN.search(key):
            return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _redact(item_value, str(item_key), secrets) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        sanitized = value
        for secret in secrets:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        normalized = sanitized.lower()
        if any(prefix in normalized for prefix in ("sk-", "rk-", "pk-")):
            sanitized = _SECRET_TOKEN_PATTERN.sub("[REDACTED]", sanitized)
        if "bearer" in normalized:
            sanitized = _BEARER_PATTERN.sub(r"\1[REDACTED]", sanitized)
        return sanitized
    return value


def redact_sensitive(value: Any):
    """Return a display/audit-safe copy without credentials or bearer tokens."""
    return _redact(value)


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._db_lock:
            return method(self, *args, **kwargs)
    return wrapper


class SQLiteStore:
    """Small append-oriented audit store for decisions, risk and execution."""

    def __init__(self, path: str | Path = "data/ai_fund_manager.sqlite3", *, read_only: bool = False):
        self._db_lock = threading.RLock()
        self.path = Path(path)
        if read_only and str(self.path) == ":memory:":
            raise ValueError("read-only SQLiteStore requires a file database")
        if read_only and not self.path.exists():
            raise FileNotFoundError(self.path)
        if str(self.path) != ":memory:" and not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            self.connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.row_factory = sqlite3.Row
        if not read_only:
            self.create_schema()

    @_synchronized
    def create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS portfolio_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                equity REAL NOT NULL, peak_equity REAL NOT NULL, cash REAL NOT NULL,
                current_symbol TEXT, current_weight REAL NOT NULL, risk_state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reduction_shadows (
                review_id TEXT PRIMARY KEY, proposal_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OBSERVING'
            );
            CREATE TABLE IF NOT EXISTS reduction_shadow_observations (
                review_id TEXT NOT NULL, date TEXT NOT NULL, observation_json TEXT NOT NULL,
                PRIMARY KEY(review_id, date)
            );
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                symbol TEXT NOT NULL, quantity REAL NOT NULL, market_price REAL,
                market_value REAL, weight REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS llm_decisions (
                decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL, decision_json TEXT NOT NULL, latency_ms REAL,
                input_tokens INTEGER, output_tokens INTEGER, estimated_cost REAL,
                pipeline TEXT, gateway TEXT, luna_model TEXT, sol_model TEXT,
                luna_candidates_json TEXT, luna_rationale TEXT, sol_tool_calls_json TEXT,
                sol_research_evidence_json TEXT, fallback_events_json TEXT, total_decision_cost REAL
            );
            CREATE TABLE IF NOT EXISTS research_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_intents (
                decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, intent_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS risk_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, risk_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS order_records (
                client_order_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, order_json TEXT NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, client_order_id TEXT NOT NULL,
                decision_id TEXT NOT NULL, recorded_at TEXT NOT NULL, execution_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                symbol TEXT NOT NULL, value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_performance_observations (
                date TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
                ai_nav REAL NOT NULL, spy_close REAL NOT NULL, qqq_close REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER,
                output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER,
                latency_ms REAL, estimated_cost REAL, stage TEXT, gateway TEXT, pipeline TEXT
            );
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, pipeline TEXT NOT NULL,
                configured_pipeline TEXT, gateway TEXT NOT NULL, luna_model TEXT, sol_model TEXT,
                luna_candidates_json TEXT NOT NULL, candidate_symbols_json TEXT NOT NULL,
                luna_rationale TEXT, sol_tool_calls_json TEXT NOT NULL,
                sol_research_evidence_json TEXT NOT NULL, final_decision_json TEXT NOT NULL,
                fallback_events_json TEXT NOT NULL, prompt_versions_json TEXT NOT NULL,
                confidence REAL, expected_excess_vs_spy REAL, expected_excess_vs_qqq REAL,
                luna_input_tokens INTEGER, luna_output_tokens INTEGER, luna_cached_tokens INTEGER,
                luna_reasoning_tokens INTEGER, luna_latency_ms REAL, luna_api_cost REAL,
                sol_input_tokens INTEGER, sol_output_tokens INTEGER, sol_cached_tokens INTEGER,
                sol_reasoning_tokens INTEGER, sol_latency_ms REAL, sol_api_cost REAL,
                total_input_tokens INTEGER, total_output_tokens INTEGER, total_cached_tokens INTEGER,
                total_reasoning_tokens INTEGER, total_latency_ms REAL, total_decision_cost REAL,
                structured_output_status TEXT, tool_calling_status TEXT
            );
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                stage TEXT NOT NULL, message TEXT NOT NULL, details_json TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'ERROR', component TEXT,
                decision_id TEXT, order_id TEXT
            );
            CREATE TABLE IF NOT EXISTS state_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                decision_id TEXT, state TEXT NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS risk_state_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                decision_id TEXT, previous_state TEXT NOT NULL, risk_state TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                state_json TEXT NOT NULL, positions_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS local_broker_state (
                singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                updated_at TEXT NOT NULL, cash REAL NOT NULL, positions_json TEXT NOT NULL,
                prices_json TEXT NOT NULL, orders_json TEXT NOT NULL, reports_json TEXT NOT NULL,
                average_costs_json TEXT NOT NULL DEFAULT '{}', realized_pnl REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY, updated_at TEXT NOT NULL, value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                command TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL,
                completed_at TEXT, result_json TEXT, error TEXT,
                worker_id TEXT, queued_at TEXT, claimed_at TEXT, started_at TEXT,
                error_stage TEXT, error_message TEXT, source TEXT, trigger_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS runtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                run_id TEXT, decision_id TEXT, component TEXT NOT NULL,
                event_type TEXT NOT NULL, symbol TEXT, message TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_events_timestamp ON runtime_events(timestamp, id);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_run_id ON runtime_events(run_id, id);
            CREATE TABLE IF NOT EXISTS pending_entries (
                decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
                entry_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pending_entries_status ON pending_entries(status, updated_at);
            CREATE TABLE IF NOT EXISTS managed_positions (
                position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, status TEXT NOT NULL,
                updated_at TEXT NOT NULL, position_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_managed_positions_symbol ON managed_positions(symbol, status);
            CREATE TABLE IF NOT EXISTS position_reviews (
                review_id TEXT PRIMARY KEY, position_id TEXT NOT NULL, reviewed_at TEXT NOT NULL,
                action TEXT NOT NULL, thesis_status TEXT NOT NULL, review_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_position_reviews_position ON position_reviews(position_id, reviewed_at);
            CREATE TABLE IF NOT EXISTS position_triggers (
                event_id TEXT PRIMARY KEY, position_id TEXT NOT NULL, event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL, trigger_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_position_triggers_position ON position_triggers(position_id, occurred_at);
            CREATE TABLE IF NOT EXISTS decision_journal (
                decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, symbol TEXT,
                action TEXT NOT NULL, requested_weight REAL NOT NULL, approved_weight REAL,
                entry_price REAL, filled_price REAL, return_1d REAL, return_5d REAL,
                return_20d REAL, spy_return_1d REAL, spy_return_5d REAL, spy_return_20d REAL,
                qqq_return_1d REAL, qqq_return_5d REAL, qqq_return_20d REAL,
                alpha_vs_spy REAL, alpha_vs_qqq REAL, journal_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS journal_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, stock_price REAL, spy_value REAL, qqq_value REAL,
                observation_index INTEGER NOT NULL
            );
            """
        )
        self._ensure_column("llm_decisions", "cached_tokens", "INTEGER")
        self._ensure_column("llm_decisions", "reasoning_tokens", "INTEGER")
        self._ensure_column("llm_decisions", "pipeline", "TEXT")
        self._ensure_column("llm_decisions", "gateway", "TEXT")
        self._ensure_column("llm_decisions", "luna_model", "TEXT")
        self._ensure_column("llm_decisions", "sol_model", "TEXT")
        self._ensure_column("llm_decisions", "luna_candidates_json", "TEXT")
        self._ensure_column("llm_decisions", "luna_rationale", "TEXT")
        self._ensure_column("llm_decisions", "sol_tool_calls_json", "TEXT")
        self._ensure_column("llm_decisions", "sol_research_evidence_json", "TEXT")
        self._ensure_column("llm_decisions", "fallback_events_json", "TEXT")
        self._ensure_column("llm_decisions", "total_decision_cost", "REAL")
        self._ensure_column("model_usage", "cached_tokens", "INTEGER")
        self._ensure_column("model_usage", "reasoning_tokens", "INTEGER")
        self._ensure_column("model_usage", "stage", "TEXT")
        self._ensure_column("model_usage", "gateway", "TEXT")
        self._ensure_column("model_usage", "pipeline", "TEXT")
        self._ensure_column("pipeline_runs", "structured_output_status", "TEXT")
        self._ensure_column("pipeline_runs", "tool_calling_status", "TEXT")
        self._ensure_column("errors", "severity", "TEXT NOT NULL DEFAULT 'ERROR'")
        self._ensure_column("errors", "component", "TEXT")
        self._ensure_column("errors", "decision_id", "TEXT")
        self._ensure_column("errors", "order_id", "TEXT")
        self._ensure_column("order_records", "broker_order_id", "INTEGER")
        self._ensure_column("order_records", "perm_id", "INTEGER")
        self._ensure_column("control_commands", "worker_id", "TEXT")
        self._ensure_column("control_commands", "queued_at", "TEXT")
        self._ensure_column("control_commands", "claimed_at", "TEXT")
        self._ensure_column("control_commands", "started_at", "TEXT")
        self._ensure_column("control_commands", "error_stage", "TEXT")
        self._ensure_column("control_commands", "error_message", "TEXT")
        self._ensure_column("control_commands", "source", "TEXT")
        self._ensure_column("control_commands", "trigger_reason", "TEXT")
        self._ensure_column("local_broker_state", "average_costs_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("local_broker_state", "realized_pnl", "REAL NOT NULL DEFAULT 0")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str):
        columns = {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @_synchronized
    def set_runtime(self, key: str, value: Any):
        self.connection.execute(
            "INSERT OR REPLACE INTO runtime_state(key,updated_at,value_json) VALUES(?,?,?)",
            (key, self._now(), self._json(value)),
        )
        self.connection.commit()

    @_synchronized
    def get_runtime(self, key: str, default: Any = None):
        row = self.connection.execute("SELECT value_json FROM runtime_state WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

    @_synchronized
    def runtime_snapshot(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key,value_json FROM runtime_state").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    @_synchronized
    def enqueue_command(self, command: str, payload: dict[str, Any] | None = None, *, source: str | None = None, trigger_reason: str | None = None) -> int:
        queued_at = self._now()
        cursor = self.connection.execute(
            "INSERT INTO control_commands(created_at,command,payload_json,status,queued_at,source,trigger_reason) VALUES(?,?,?,?,?,?,?)",
            (queued_at, command, self._json(payload or {}), "QUEUED", queued_at, source, trigger_reason),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @_synchronized
    def enqueue_command_unless_active(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        *,
        active_commands: tuple[str, ...] | None = None,
        source: str | None = None,
        trigger_reason: str | None = None,
    ) -> tuple[int, bool]:
        """Atomically enqueue a command unless the current worker already has one active."""
        commands = tuple(active_commands or (command,))
        placeholders = ",".join("?" for _ in commands)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            worker_row = self.connection.execute(
                "SELECT value_json FROM runtime_state WHERE key='command_worker_id'"
            ).fetchone()
            worker_id = json.loads(worker_row["value_json"]) if worker_row else None
            active = self.connection.execute(
                f"SELECT id FROM control_commands WHERE command IN ({placeholders}) AND "
                "(status IN ('QUEUED','PENDING') OR "
                "(status IN ('CLAIMED','RUNNING') AND worker_id=?)) ORDER BY id DESC LIMIT 1",
                (*commands, worker_id),
            ).fetchone()
            if active is not None:
                self.connection.commit()
                return int(active["id"]), False
            queued_at = self._now()
            cursor = self.connection.execute(
                "INSERT INTO control_commands(created_at,command,payload_json,status,queued_at,source,trigger_reason) VALUES(?,?,?,?,?,?,?)",
                (queued_at, command, self._json(payload or {}), "QUEUED", queued_at, source, trigger_reason),
            )
            self.connection.commit()
            return int(cursor.lastrowid), True
        except Exception:
            self.connection.rollback()
            raise

    @_synchronized
    def pending_commands(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM control_commands WHERE status IN ('QUEUED','PENDING') ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def command(self, command_id: int) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM control_commands WHERE id=?", (int(command_id),)).fetchone()
        return dict(row) if row else None

    @_synchronized
    def claim_next_command(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically assign the oldest queued command to one worker."""
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT * FROM control_commands WHERE status IN ('QUEUED','PENDING') ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            claimed_at = self._now()
            cursor = self.connection.execute(
                "UPDATE control_commands SET status='CLAIMED',worker_id=?,claimed_at=?,queued_at=COALESCE(queued_at,created_at) "
                "WHERE id=? AND status IN ('QUEUED','PENDING')",
                (worker_id, claimed_at, row["id"]),
            )
            if cursor.rowcount != 1:
                self.connection.rollback()
                return None
            self.connection.commit()
            claimed = self.connection.execute("SELECT * FROM control_commands WHERE id=?", (row["id"],)).fetchone()
            return dict(claimed)
        except Exception:
            self.connection.rollback()
            raise

    @_synchronized
    def mark_command_running(self, command_id: int, worker_id: str) -> bool:
        cursor = self.connection.execute(
            "UPDATE control_commands SET status='RUNNING',started_at=? WHERE id=? AND status='CLAIMED' AND worker_id=?",
            (self._now(), command_id, worker_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    @_synchronized
    def recover_stale_claims(self, stale_before: str) -> int:
        """Requeue claims that never reached RUNNING; running commands are never replayed."""
        cursor = self.connection.execute(
            "UPDATE control_commands SET status='QUEUED',worker_id=NULL,claimed_at=NULL "
            "WHERE status='CLAIMED' AND started_at IS NULL AND claimed_at < ?",
            (stale_before,),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    @_synchronized
    def fail_abandoned_running_commands(self, active_worker_id: str) -> int:
        """Close RUNNING commands owned by a worker that is no longer active.

        A command that reached RUNNING may already have caused an external side effect,
        so it is recorded as failed and never replayed automatically.
        """
        message = "Worker exited before command completion; command was not replayed"
        cursor = self.connection.execute(
            "UPDATE control_commands SET status='FAILED',completed_at=?,error_stage=?,"
            "error=?,error_message=? WHERE status='RUNNING' AND (worker_id IS NULL OR worker_id<>?)",
            (self._now(), "WORKER_RECOVERY", message, message, active_worker_id),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    @_synchronized
    def complete_command(self, command_id: int, result: Any = None, error: str | None = None, *, error_stage: str | None = None, worker_id: str | None = None):
        status = "FAILED" if error else "SUCCEEDED"
        worker_clause = " AND worker_id=?" if worker_id is not None else ""
        parameters = [status, self._now(), self._json(_redact(result)) if result is not None else None, _redact(error), _redact(error_stage), _redact(error)]
        parameters.append(command_id)
        if worker_id is not None:
            parameters.append(worker_id)
        self.connection.execute(
            "UPDATE control_commands SET status=?,completed_at=?,result_json=?,error=?,error_stage=?,error_message=? WHERE id=?" + worker_clause,
            parameters,
        )
        self.connection.commit()

    @_synchronized
    def save_runtime_event(
        self,
        event_type: str,
        component: str,
        message: str,
        *,
        run_id: str | None = None,
        decision_id: str | None = None,
        symbol: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO runtime_events(timestamp,run_id,decision_id,component,event_type,symbol,message,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                timestamp or self._now(),
                run_id,
                decision_id,
                _redact(component),
                _redact(event_type),
                _redact(symbol),
                _redact(message),
                self._json(_redact(metadata or {})),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @_synchronized
    def runtime_events(self, limit: int = 500, run_id: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        if run_id is None:
            rows = self.connection.execute("SELECT * FROM runtime_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.connection.execute("SELECT * FROM runtime_events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def save_pending_entry(self, entry: PendingEntry) -> PendingEntry:
        existing = self.connection.execute(
            "SELECT entry_json FROM pending_entries WHERE decision_id=?", (entry.decision_id,)
        ).fetchone()
        if existing:
            return PendingEntry.model_validate_json(existing["entry_json"])
        self.connection.execute(
            "INSERT INTO pending_entries(decision_id,run_id,symbol,created_at,updated_at,status,entry_json) VALUES(?,?,?,?,?,?,?)",
            (
                entry.decision_id,
                entry.run_id,
                entry.symbol,
                entry.decision_time,
                entry.updated_at,
                entry.status,
                entry.model_dump_json(),
            ),
        )
        self.connection.commit()
        return entry

    @_synchronized
    def pending_entry(self, decision_id: str) -> PendingEntry | None:
        row = self.connection.execute(
            "SELECT entry_json FROM pending_entries WHERE decision_id=?", (decision_id,)
        ).fetchone()
        return PendingEntry.model_validate_json(row["entry_json"]) if row else None

    @_synchronized
    def pending_entries(self, statuses: tuple[str, ...] | None = None) -> list[PendingEntry]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.connection.execute(
                f"SELECT entry_json FROM pending_entries WHERE status IN ({placeholders}) ORDER BY created_at",
                statuses,
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT entry_json FROM pending_entries ORDER BY created_at").fetchall()
        return [PendingEntry.model_validate_json(row["entry_json"]) for row in rows]

    @_synchronized
    def update_pending_entry(self, entry: PendingEntry) -> PendingEntry:
        cursor = self.connection.execute(
            "UPDATE pending_entries SET updated_at=?,status=?,entry_json=? WHERE decision_id=?",
            (entry.updated_at, entry.status, entry.model_dump_json(), entry.decision_id),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown pending entry: {entry.decision_id}")
        return entry

    @_synchronized
    def save_managed_position(self, position: ManagedPosition) -> ManagedPosition:
        self.connection.execute(
            """INSERT INTO managed_positions(position_id,symbol,status,updated_at,position_json)
               VALUES(?,?,?,?,?) ON CONFLICT(position_id) DO UPDATE SET
               symbol=excluded.symbol,status=excluded.status,updated_at=excluded.updated_at,
               position_json=excluded.position_json""",
            (position.position_id, position.symbol, position.monitoring_status, position.updated_at, position.model_dump_json()),
        )
        self.connection.commit()
        return position

    @_synchronized
    def managed_position(self, position_id: str) -> ManagedPosition | None:
        row = self.connection.execute("SELECT position_json FROM managed_positions WHERE position_id=?", (position_id,)).fetchone()
        return ManagedPosition.model_validate_json(row["position_json"]) if row else None

    @_synchronized
    def active_managed_position(self, symbol: str | None = None) -> ManagedPosition | None:
        if symbol:
            row = self.connection.execute(
                "SELECT position_json FROM managed_positions WHERE status='ACTIVE' AND symbol=? ORDER BY updated_at DESC LIMIT 1",
                (symbol.upper(),),
            ).fetchone()
        else:
            row = self.connection.execute("SELECT position_json FROM managed_positions WHERE status='ACTIVE' ORDER BY updated_at DESC LIMIT 1").fetchone()
        return ManagedPosition.model_validate_json(row["position_json"]) if row else None

    @_synchronized
    def active_managed_positions(self) -> list[ManagedPosition]:
        rows = self.connection.execute("SELECT position_json FROM managed_positions WHERE status='ACTIVE' ORDER BY updated_at DESC").fetchall()
        return [ManagedPosition.model_validate_json(row["position_json"]) for row in rows]

    @_synchronized
    def save_position_review(self, review: PositionReview) -> PositionReview:
        self.connection.execute(
            "INSERT INTO position_reviews(review_id,position_id,reviewed_at,action,thesis_status,review_json) VALUES(?,?,?,?,?,?)",
            (review.review_id, review.position_id, review.reviewed_at, review.action, review.thesis_status, self._json(_redact(review.model_dump(mode="json")))),
        )
        self.connection.commit()
        return review

    @_synchronized
    def position_reviews(self, position_id: str, limit: int = 100) -> list[PositionReview]:
        rows = self.connection.execute(
            "SELECT review_json FROM position_reviews WHERE position_id=? ORDER BY reviewed_at DESC LIMIT ?",
            (position_id, max(1, int(limit))),
        ).fetchall()
        return [PositionReview.model_validate_json(row["review_json"]) for row in rows]

    @_synchronized
    def position_trigger(self, event_id: str) -> PositionTrigger | None:
        row = self.connection.execute("SELECT trigger_json FROM position_triggers WHERE event_id=?", (event_id,)).fetchone()
        return PositionTrigger.model_validate_json(row["trigger_json"]) if row else None

    @_synchronized
    def save_position_trigger(self, trigger: PositionTrigger) -> PositionTrigger:
        self.connection.execute(
            "INSERT INTO position_triggers(event_id,position_id,event_type,occurred_at,trigger_json) VALUES(?,?,?,?,?)",
            (trigger.event_id, trigger.position_id, trigger.event_type, trigger.occurred_at, trigger.model_dump_json()),
        )
        self.connection.commit()
        return trigger

    @_synchronized
    def position_triggers(self, position_id: str, limit: int = 100) -> list[PositionTrigger]:
        rows = self.connection.execute(
            "SELECT trigger_json FROM position_triggers WHERE position_id=? ORDER BY occurred_at DESC LIMIT ?",
            (position_id, max(1, int(limit))),
        ).fetchall()
        return [PositionTrigger.model_validate_json(row["trigger_json"]) for row in rows]

    @_synchronized
    def save_decision_journal(self, intent: TradeIntent, risk: RiskDecision | None, entry_price: float | None, filled_price: float | None = None, candidates: list[str] | None = None):
        risk_values = risk.model_dump(mode="json") if risk else {}
        payload = {
            "decision": intent.model_dump(mode="json"),
            "risk": risk_values,
            "candidates": candidates or [],
        }
        self.connection.execute(
            """INSERT OR REPLACE INTO decision_journal(
                decision_id,recorded_at,symbol,action,requested_weight,approved_weight,
                entry_price,filled_price,journal_json
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (intent.decision_id, intent.timestamp, intent.symbol, intent.action, intent.target_weight,
             risk.approved_weight if risk else None, entry_price, filled_price, self._json(payload)),
        )
        self.connection.execute(
            "DELETE FROM journal_observations WHERE decision_id=?", (intent.decision_id,)
        )
        self.connection.execute(
            "INSERT INTO journal_observations(decision_id,recorded_at,stock_price,spy_value,qqq_value,observation_index) VALUES(?,?,?,?,?,?)",
            (intent.decision_id, self._now(), filled_price or entry_price, 100.0, 100.0, 0),
        )
        self.connection.commit()

    @_synchronized
    def record_journal_observation(self, decision_id: str, stock_price: float | None, spy_value: float | None, qqq_value: float | None):
        row = self.connection.execute("SELECT entry_price FROM decision_journal WHERE decision_id=?", (decision_id,)).fetchone()
        if not row:
            return
        previous = self.connection.execute("SELECT MAX(observation_index) AS value FROM journal_observations WHERE decision_id=?", (decision_id,)).fetchone()
        index = int(previous["value"] if previous and previous["value"] is not None else -1) + 1
        self.connection.execute(
            "INSERT INTO journal_observations(decision_id,recorded_at,stock_price,spy_value,qqq_value,observation_index) VALUES(?,?,?,?,?,?)",
            (decision_id, self._now(), stock_price, spy_value, qqq_value, index),
        )
        entry = self.connection.execute("SELECT entry_price FROM decision_journal WHERE decision_id=?", (decision_id,)).fetchone()
        entry_price = float(entry["entry_price"]) if entry and entry["entry_price"] else None
        if entry_price and stock_price:
            updates: dict[str, float] = {}
            for threshold, field in ((1, "return_1d"), (5, "return_5d"), (20, "return_20d")):
                if index == threshold:
                    updates[field] = float(stock_price) / entry_price - 1
                    if spy_value:
                        updates[f"spy_{field}"] = float(spy_value) / 100.0 - 1
                    if qqq_value:
                        updates[f"qqq_{field}"] = float(qqq_value) / 100.0 - 1
            for field, value in updates.items():
                self.connection.execute(f"UPDATE decision_journal SET {field}=? WHERE decision_id=?", (value, decision_id))
            if index in {1, 5, 20}:
                self.connection.execute(
                    "UPDATE decision_journal SET alpha_vs_spy=COALESCE(return_20d,return_5d,return_1d)-COALESCE(spy_return_20d,spy_return_5d,spy_return_1d), alpha_vs_qqq=COALESCE(return_20d,return_5d,return_1d)-COALESCE(qqq_return_20d,qqq_return_5d,qqq_return_1d) WHERE decision_id=?",
                    (decision_id,),
                )
        self.connection.commit()

    @_synchronized
    def journal_history(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM decision_journal ORDER BY recorded_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def llm_api_cost_total(self) -> float:
        row = self.connection.execute("SELECT COALESCE(SUM(estimated_cost),0) AS value FROM model_usage").fetchone()
        return float(row["value"] or 0.0)

    @_synchronized
    def trading_cost_total(self) -> float:
        rows = self.connection.execute("SELECT client_order_id,execution_json FROM executions ORDER BY id").fetchall()
        latest_by_order = {row["client_order_id"]: json.loads(row["execution_json"]) for row in rows}
        total = 0.0
        for report in latest_by_order.values():
            broker_cost = float(report.get("total_execution_cost", 0) or 0)
            if broker_cost <= 0:
                broker_cost = float(report.get("commission", 0) or 0) + float(report.get("fees", 0) or 0)
            total += broker_cost + float(report.get("slippage", 0) or 0)
        return total

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @_synchronized
    def save_equity(self, equity: float, peak_equity: float, cash: float, current_symbol: str | None, current_weight: float, risk_state: str):
        self.connection.execute("INSERT INTO portfolio_equity(recorded_at,equity,peak_equity,cash,current_symbol,current_weight,risk_state) VALUES(?,?,?,?,?,?,?)", (self._now(), equity, peak_equity, cash, current_symbol, current_weight, risk_state))
        self.connection.commit()

    @_synchronized
    def save_portfolio(self, state: PortfolioState, positions: list[dict[str, Any]]):
        recorded_at = self._now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO portfolio_snapshots(recorded_at,state_json,positions_json) VALUES(?,?,?)",
                (recorded_at, self._json(state.model_dump(mode="json")), self._json(positions)),
            )
            self.connection.execute(
                "INSERT INTO portfolio_equity(recorded_at,equity,peak_equity,cash,current_symbol,current_weight,risk_state) VALUES(?,?,?,?,?,?,?)",
                (recorded_at, state.equity, state.peak_equity, state.cash, state.current_symbol, state.current_weight, state.risk_state),
            )
            for position in positions:
                self.connection.execute(
                    "INSERT INTO positions(recorded_at,symbol,quantity,market_price,market_value,weight) VALUES(?,?,?,?,?,?)",
                    (recorded_at, position["symbol"], position["quantity"], position.get("market_price"), position.get("market_value"), position.get("weight", 0.0)),
                )

    @_synchronized
    def latest_portfolio(self) -> tuple[PortfolioState, list[dict[str, Any]]] | None:
        row = self.connection.execute("SELECT state_json,positions_json FROM portfolio_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            return PortfolioState.model_validate_json(row["state_json"]), json.loads(row["positions_json"])
        return None

    @_synchronized
    def save_local_broker_state(self, cash: float, positions: dict[str, float], prices: dict[str, float], orders: dict[str, Any], reports: dict[str, Any], average_costs: dict[str, float] | None = None, realized_pnl: float = 0.0):
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO local_broker_state(singleton_id,updated_at,cash,positions_json,prices_json,orders_json,reports_json,average_costs_json,realized_pnl) VALUES(1,?,?,?,?,?,?,?,?)",
                (self._now(), cash, self._json(positions), self._json(prices), self._json(orders), self._json(reports), self._json(average_costs or {}), realized_pnl),
            )

    @_synchronized
    def load_local_broker_state(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT cash,positions_json,prices_json,orders_json,reports_json,average_costs_json,realized_pnl FROM local_broker_state WHERE singleton_id=1").fetchone()
        if not row:
            return None
        return {
            "cash": float(row["cash"]),
            "positions": json.loads(row["positions_json"]),
            "prices": json.loads(row["prices_json"]),
            "orders": json.loads(row["orders_json"]),
            "reports": json.loads(row["reports_json"]),
            "average_costs": json.loads(row["average_costs_json"]),
            "realized_pnl": float(row["realized_pnl"]),
        }

    @_synchronized
    def save_position(self, symbol: str, quantity: float, market_price: float | None, market_value: float | None, weight: float):
        self.connection.execute("INSERT INTO positions(recorded_at,symbol,quantity,market_price,market_value,weight) VALUES(?,?,?,?,?,?)", (self._now(), symbol, quantity, market_price, market_value, weight))
        self.connection.commit()

    @_synchronized
    def save_decision(self, intent: TradeIntent, prompt_version: str = "v1", latency_ms: float | None = None, input_tokens: int | None = None, output_tokens: int | None = None, cached_tokens: int | None = None, reasoning_tokens: int | None = None, estimated_cost: float | None = None, pipeline_metadata: dict[str, Any] | None = None, **_ignored):
        payload = intent.model_dump(mode="json")
        metadata = pipeline_metadata or {}
        pipeline = metadata.get("pipeline")
        gateway = metadata.get("gateway")
        luna_usage = metadata.get("luna_usage") or {}
        sol_usage = metadata.get("sol_usage") or {}
        total_cost = metadata.get("total_decision_cost", estimated_cost)
        self.connection.execute(
            """INSERT OR REPLACE INTO llm_decisions(
                decision_id,recorded_at,model_name,prompt_version,decision_json,latency_ms,
                input_tokens,output_tokens,cached_tokens,reasoning_tokens,estimated_cost,
                pipeline,gateway,luna_model,sol_model,luna_candidates_json,luna_rationale,
                sol_tool_calls_json,sol_research_evidence_json,fallback_events_json,total_decision_cost
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                intent.decision_id, intent.timestamp, intent.model_name, prompt_version, self._json(payload), latency_ms,
                input_tokens, output_tokens, cached_tokens, reasoning_tokens, estimated_cost,
                pipeline, gateway, metadata.get("luna_model"), metadata.get("sol_model"),
                self._json(metadata.get("luna_candidates", [])), metadata.get("luna_rationale"),
                self._json(metadata.get("sol_tool_calls", [])), self._json(metadata.get("sol_research_evidence", [])),
                self._json(metadata.get("fallback_events", [])), total_cost,
            ),
        )
        self.connection.execute("INSERT OR REPLACE INTO trade_intents(decision_id,recorded_at,intent_json) VALUES(?,?,?)", (intent.decision_id, intent.timestamp, self._json(payload)))
        self.connection.execute("DELETE FROM model_usage WHERE decision_id=?", (intent.decision_id,))
        stage_usage = []
        if luna_usage:
            stage_usage.append(("LUNA", luna_usage))
        if sol_usage:
            stage_usage.append(("SOL", sol_usage))
        if not stage_usage:
            stage_usage.append(("SINGLE", {"model": intent.model_name, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens, "reasoning_tokens": reasoning_tokens, "latency_ms": latency_ms, "estimated_cost": estimated_cost}))
        for stage, usage in stage_usage:
            self.connection.execute(
                "INSERT INTO model_usage(decision_id,recorded_at,model_name,input_tokens,output_tokens,cached_tokens,reasoning_tokens,latency_ms,estimated_cost,stage,gateway,pipeline) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    intent.decision_id, self._now(), usage.get("model", intent.model_name), usage.get("input_tokens"), usage.get("output_tokens"),
                    usage.get("cached_tokens"), usage.get("reasoning_tokens"), usage.get("latency_ms"), usage.get("estimated_cost"),
                    stage, gateway, pipeline,
                ),
            )
        if metadata:
            self._save_pipeline_run(intent, metadata)
        self.connection.commit()

    def _save_pipeline_run(self, intent: TradeIntent, metadata: dict[str, Any]) -> None:
        luna_usage = metadata.get("luna_usage") or {}
        sol_usage = metadata.get("sol_usage") or {}
        total_usage = metadata.get("usage") or {}
        self.connection.execute(
            """INSERT OR REPLACE INTO pipeline_runs(
                decision_id,recorded_at,pipeline,configured_pipeline,gateway,luna_model,sol_model,
                luna_candidates_json,candidate_symbols_json,luna_rationale,sol_tool_calls_json,
                sol_research_evidence_json,final_decision_json,fallback_events_json,prompt_versions_json,
                confidence,expected_excess_vs_spy,expected_excess_vs_qqq,
                luna_input_tokens,luna_output_tokens,luna_cached_tokens,luna_reasoning_tokens,luna_latency_ms,luna_api_cost,
                sol_input_tokens,sol_output_tokens,sol_cached_tokens,sol_reasoning_tokens,sol_latency_ms,sol_api_cost,
                total_input_tokens,total_output_tokens,total_cached_tokens,total_reasoning_tokens,total_latency_ms,total_decision_cost
                ,structured_output_status,tool_calling_status
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                intent.decision_id, self._now(), metadata.get("pipeline", "UNKNOWN"), metadata.get("configured_pipeline"), metadata.get("gateway", "unknown"),
                metadata.get("luna_model"), metadata.get("sol_model"), self._json(metadata.get("luna_candidates", [])), self._json(metadata.get("candidate_symbols", [])),
                metadata.get("luna_rationale"), self._json(metadata.get("sol_tool_calls", [])), self._json(metadata.get("sol_research_evidence", [])),
                self._json(metadata.get("final_decision", intent.model_dump(mode="json"))), self._json(metadata.get("fallback_events", [])), self._json(metadata.get("prompt_versions", {})),
                metadata.get("confidence", intent.confidence), metadata.get("expected_excess_vs_spy", intent.expected_alpha_vs_spy), metadata.get("expected_excess_vs_qqq", intent.expected_alpha_vs_qqq),
                luna_usage.get("input_tokens"), luna_usage.get("output_tokens"), luna_usage.get("cached_tokens"), luna_usage.get("reasoning_tokens"), luna_usage.get("latency_ms"), luna_usage.get("estimated_cost"),
                sol_usage.get("input_tokens"), sol_usage.get("output_tokens"), sol_usage.get("cached_tokens"), sol_usage.get("reasoning_tokens"), sol_usage.get("latency_ms"), sol_usage.get("estimated_cost"),
                total_usage.get("input_tokens"), total_usage.get("output_tokens"), total_usage.get("cached_tokens"), total_usage.get("reasoning_tokens"), total_usage.get("latency_ms"), metadata.get("total_decision_cost", total_usage.get("estimated_cost")),
                metadata.get("structured_output_status", "UNKNOWN"), metadata.get("tool_calling_status", "UNKNOWN"),
            ),
        )

    @_synchronized
    def pipeline_history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM pipeline_runs ORDER BY recorded_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def save_evidence(self, decision_id: str, evidence: list[dict[str, Any]]):
        self.connection.executemany("INSERT INTO research_evidence(decision_id,recorded_at,tool_name,arguments_json,result_json) VALUES(?,?,?,?,?)", [(decision_id, self._now(), item.get("tool", "unknown"), self._json(item.get("arguments", {})), self._json(item.get("result", {}))) for item in evidence])
        self.connection.commit()

    @_synchronized
    def save_risk(self, decision_id: str, decision: RiskDecision):
        self.connection.execute("INSERT INTO risk_decisions(decision_id,recorded_at,risk_json) VALUES(?,?,?)", (decision_id, self._now(), self._json(decision.model_dump(mode="json"))))
        self.connection.commit()

    @_synchronized
    def save_order(self, request: OrderRequest, status: str = "PENDING"):
        self.connection.execute("INSERT OR REPLACE INTO order_records(client_order_id,decision_id,recorded_at,order_json,status) VALUES(?,?,?,?,?)", (request.client_order_id, request.decision_id, self._now(), self._json(request.model_dump(mode="json")), status))
        self.connection.commit()

    @_synchronized
    def update_order_status(self, client_order_id: str, status: str):
        self.connection.execute("UPDATE order_records SET status=? WHERE client_order_id=?", (status, client_order_id))
        self.connection.commit()

    @_synchronized
    def update_order_broker_ids(self, client_order_id: str, broker_order_id: int | None, perm_id: int | None):
        self.connection.execute(
            "UPDATE order_records SET broker_order_id=COALESCE(?,broker_order_id), perm_id=COALESCE(?,perm_id) WHERE client_order_id=?",
            (broker_order_id, perm_id, client_order_id),
        )
        self.connection.commit()

    @_synchronized
    def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM order_records WHERE client_order_id=?", (client_order_id,)).fetchone()
        return dict(row) if row else None

    @_synchronized
    def order_by_broker_ids(self, broker_order_id: int | None, perm_id: int | None = None) -> dict[str, Any] | None:
        if perm_id is not None:
            row = self.connection.execute("SELECT * FROM order_records WHERE perm_id=? ORDER BY recorded_at DESC LIMIT 1", (perm_id,)).fetchone()
            if row:
                return dict(row)
        if broker_order_id is None:
            return None
        row = self.connection.execute("SELECT * FROM order_records WHERE broker_order_id=? ORDER BY recorded_at DESC LIMIT 1", (broker_order_id,)).fetchone()
        return dict(row) if row else None

    @_synchronized
    def active_orders(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM order_records WHERE status IN ('PENDING','PARTIALLY_FILLED') ORDER BY recorded_at").fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def save_execution(self, report: ExecutionReport):
        self.connection.execute("INSERT INTO executions(client_order_id,decision_id,recorded_at,execution_json) VALUES(?,?,?,?)", (report.client_order_id, report.decision_id, report.timestamp, self._json(report.model_dump(mode="json"))))
        self.update_order_status(report.client_order_id, report.status)
        self.update_order_broker_ids(report.client_order_id, report.broker_order_id, report.perm_id)
        self.connection.commit()

    @_synchronized
    def save_benchmark(self, symbol: str, value: float):
        self.connection.execute("INSERT INTO benchmark_values(recorded_at,symbol,value) VALUES(?,?,?)", (self._now(), symbol, value))
        self.connection.commit()

    @_synchronized
    def save_daily_performance(self, date: str, ai_nav: float, spy_close: float, qqq_close: float):
        self.connection.execute(
            "INSERT INTO daily_performance_observations(date,recorded_at,ai_nav,spy_close,qqq_close) VALUES(?,?,?,?,?) "
            "ON CONFLICT(date) DO UPDATE SET recorded_at=excluded.recorded_at, ai_nav=excluded.ai_nav, spy_close=excluded.spy_close, qqq_close=excluded.qqq_close",
            (date, self._now(), ai_nav, spy_close, qqq_close),
        )
        self.connection.commit()

    @_synchronized
    def daily_performance_history(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT date,ai_nav,spy_close,qqq_close FROM daily_performance_observations ORDER BY date"
        ).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def save_reduction_shadow(self, review_id: str, proposal: dict):
        self.connection.execute("INSERT OR IGNORE INTO reduction_shadows(review_id,proposal_json) VALUES(?,?)",
                                (review_id, self._json(_redact(proposal))))
        self.connection.commit()

    @_synchronized
    def save_shadow_observation(self, review_id: str, observation: dict):
        self.connection.execute("INSERT OR IGNORE INTO reduction_shadow_observations VALUES(?,?,?)",
                                (review_id, observation["date"], self._json(observation)))
        self.connection.commit()

    @_synchronized
    def complete_reduction_shadow(self, review_id: str):
        self.connection.execute("UPDATE reduction_shadows SET status='COMPLETE' WHERE review_id=?", (review_id,))
        self.connection.commit()

    @_synchronized
    def reduction_shadows(self) -> list[dict]:
        # A Dashboard running against a pre-upgrade DB remains read-only.
        if not self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='reduction_shadows'").fetchone():
            return []
        result = []
        for row in self.connection.execute("SELECT * FROM reduction_shadows ORDER BY rowid DESC"):
            observations = self.connection.execute(
                "SELECT observation_json FROM reduction_shadow_observations WHERE review_id=? ORDER BY date", (row["review_id"],)
            ).fetchall()
            result.append({"review_id": row["review_id"], "status": row["status"],
                           "proposal": json.loads(row["proposal_json"]),
                           "observations": [json.loads(o[0]) for o in observations]})
        return result

    @_synchronized
    def save_error(self, stage: str, message: str, details: dict[str, Any] | None = None, severity: str = "ERROR", component: str | None = None, decision_id: str | None = None, order_id: str | None = None):
        self.connection.execute("INSERT INTO errors(recorded_at,stage,message,details_json,severity,component,decision_id,order_id) VALUES(?,?,?,?,?,?,?,?)", (_redact(self._now()), _redact(stage), _redact(message), self._json(_redact(details or {})), _redact(severity), _redact(component or stage), _redact(decision_id), _redact(order_id)))
        self.connection.commit()

    @_synchronized
    def save_state(self, state: str, reason: str, decision_id: str | None = None):
        self.connection.execute("INSERT INTO state_events(recorded_at,decision_id,state,reason) VALUES(?,?,?,?)", (self._now(), decision_id, state, reason))
        self.connection.commit()

    @_synchronized
    def latest_state(self) -> str | None:
        row = self.connection.execute("SELECT state FROM state_events ORDER BY id DESC LIMIT 1").fetchone()
        return row["state"] if row else None

    @_synchronized
    def save_risk_state_event(self, previous_state: str, risk_state: str, reason: str, decision_id: str | None = None):
        self.connection.execute(
            "INSERT INTO risk_state_events(recorded_at,decision_id,previous_state,risk_state,reason) VALUES(?,?,?,?,?)",
            (self._now(), decision_id, previous_state, risk_state, reason),
        )
        self.connection.commit()

    @_synchronized
    def risk_state_history(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT recorded_at,decision_id,previous_state,risk_state,reason FROM risk_state_events ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def equity_history(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT recorded_at,equity,peak_equity,cash,current_symbol,current_weight,risk_state FROM portfolio_equity ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def benchmark_history(self) -> dict[str, list[float]]:
        rows = self.connection.execute("SELECT symbol,value FROM benchmark_values ORDER BY id").fetchall()
        values: dict[str, list[float]] = {}
        for row in rows:
            values.setdefault(row["symbol"], []).append(float(row["value"]))
        return values

    @_synchronized
    def recent(self, table: str, limit: int = 20) -> list[dict[str, Any]]:
        allowed = {"llm_decisions", "research_evidence", "risk_decisions", "order_records", "executions", "errors", "benchmark_values", "state_events", "control_commands", "pipeline_runs", "runtime_events", "pending_entries", "managed_positions", "position_reviews", "position_triggers"}
        if table not in allowed:
            raise ValueError("Unsupported table")
        order_column = {"runtime_events": "timestamp", "control_commands": "created_at", "pending_entries": "created_at", "managed_positions": "updated_at", "position_reviews": "reviewed_at", "position_triggers": "occurred_at"}.get(table, "recorded_at")
        rows = self.connection.execute(f"SELECT * FROM {table} ORDER BY {order_column} DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    @_synchronized
    def purge_old_runtime_events(self, keep_days: int = 14, min_keep_rows: int = 1000) -> int:
        """Prune old runtime events while preserving recent audit history."""
        total_rows = self.connection.execute("SELECT COUNT(*) AS count FROM runtime_events").fetchone()["count"]
        if total_rows <= min_keep_rows:
            return 0
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=max(0, keep_days))).isoformat()
        cursor = self.connection.execute(
            "DELETE FROM runtime_events WHERE timestamp <= ? AND id NOT IN (SELECT id FROM runtime_events ORDER BY id DESC LIMIT ?)",
            (cutoff_date, min_keep_rows),
        )
        self.connection.commit()
        return cursor.rowcount

    @_synchronized
    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
