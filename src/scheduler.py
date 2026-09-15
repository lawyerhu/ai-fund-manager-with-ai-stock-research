from __future__ import annotations

import threading
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from .market_clock import USEquityMarketClock


_WEEKDAYS = {name: index for index, name in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}


@dataclass
class SchedulerConfig:
    enabled: bool = True
    weekly_day: str = "monday"
    weekly_time: str = "09:45"
    daily_time: str = "16:05"
    full_research_day: str | None = None
    full_research_time: str | None = None
    pre_execution_time: str = "09:00"
    daily_review_time: str | None = None
    timezone: str = "America/New_York"
    risk_monitor_interval_minutes: int = 15
    position_monitor_interval_minutes: int = 30
    entry_check_interval_seconds: int = 30
    poll_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, values: dict | None):
        values = values or {}
        keys = (
            "enabled", "weekly_day", "weekly_time", "daily_time", "full_research_day",
            "full_research_time", "pre_execution_time", "daily_review_time", "timezone",
            "risk_monitor_interval_minutes", "position_monitor_interval_minutes", "entry_check_interval_seconds", "poll_seconds",
        )
        return cls(**{key: values[key] for key in keys if key in values})

    def __post_init__(self):
        self.weekly_day = self.weekly_day.lower()
        if self.full_research_day is not None:
            self.full_research_day = self.full_research_day.lower()
        if self.weekly_day not in _WEEKDAYS:
            raise ValueError(f"Unknown scheduler weekday: {self.weekly_day}")
        if self.full_research_day is not None and self.full_research_day not in _WEEKDAYS:
            raise ValueError(f"Unknown full research weekday: {self.full_research_day}")
        for value in (self.weekly_time, self.daily_time, self.full_research_time, self.pre_execution_time, self.daily_review_time):
            if value is None:
                continue
            datetime.strptime(value, "%H:%M")
        if self.poll_seconds <= 0:
            raise ValueError("scheduler poll_seconds must be positive")
        if self.risk_monitor_interval_minutes <= 0:
            raise ValueError("risk_monitor_interval_minutes must be positive")
        if self.position_monitor_interval_minutes <= 0:
            raise ValueError("position_monitor_interval_minutes must be positive")
        if self.entry_check_interval_seconds <= 0:
            raise ValueError("entry_check_interval_seconds must be positive")
        ZoneInfo(self.timezone)


class PaperScheduler:
    """Threaded clock that invokes weekly research and daily risk callbacks once per period."""

    def __init__(self, config: SchedulerConfig | None = None, weekly_callback: Callable[[], object] | None = None, daily_callback: Callable[[], object] | None = None, store=None, control_callback: Callable[[], object] | None = None, risk_callback: Callable[[], object] | None = None, market_clock=None, pre_execution_callback: Callable[[], object] | None = None, entry_callback: Callable[[], object] | None = None, position_callback: Callable[[], object] | None = None):
        self.config = config or SchedulerConfig()
        self.weekly_callback = weekly_callback or (lambda: None)
        self.daily_callback = daily_callback or (lambda: None)
        self.pre_execution_callback = pre_execution_callback
        self.risk_callback = risk_callback
        self.entry_callback = entry_callback
        self.position_callback = position_callback
        self.market_clock = market_clock or USEquityMarketClock()
        self.control_callback = control_callback
        self.store = store
        self._timezone = ZoneInfo(self.config.timezone)
        self._last_weekly_run = self.store.get_runtime("scheduler.last_weekly_run") if self.store else None
        self._last_daily_run = self.store.get_runtime("scheduler.last_daily_run") if self.store else None
        self._last_pre_execution_run = self.store.get_runtime("scheduler.last_pre_execution_run") if self.store else None
        self._last_risk_run = self.store.get_runtime("scheduler.last_risk_run") if self.store else None
        self._last_entry_run = self.store.get_runtime("scheduler.last_entry_run") if self.store else None
        self._last_position_run = self.store.get_runtime("scheduler.last_position_run") if self.store else None
        self._weekly_status = self.store.get_runtime("scheduler.weekly_status", "COMPLETED") if self.store else "COMPLETED"
        self._weekly_command_id = self.store.get_runtime("scheduler.weekly_command_id") if self.store else None
        self._weekly_command_day = self.store.get_runtime("scheduler.weekly_command_day") if self.store else None
        self._status = "STOPPED"
        self._last_error = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not self.config.enabled:
            self._status = "STOPPED"
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._status = "RUNNING"
        self._thread = threading.Thread(target=self._loop, name="ai-fund-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._status = "STOPPED"
        self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None

    def run_once(self, now: datetime | None = None):
        if self.control_callback:
            self.control_callback()
        current = now or datetime.now(self._timezone)
        if current.tzinfo is None:
            current = current.replace(tzinfo=self._timezone)
        else:
            current = current.astimezone(self._timezone)
        day = current.date().isoformat()
        self._refresh_weekly_command()
        interval = self.config.risk_monitor_interval_minutes
        risk_bucket = f"{day}T{current.hour:02d}:{(current.minute // interval) * interval:02d}"
        market_open = self.market_clock.is_open(current)
        entry_interval = self.config.entry_check_interval_seconds
        seconds = current.hour * 3600 + current.minute * 60 + current.second
        entry_bucket = f"{day}T{seconds // entry_interval}"
        position_interval = self.config.position_monitor_interval_minutes
        position_bucket = f"{day}T{current.hour:02d}:{(current.minute // position_interval) * position_interval:02d}"
        if self.store:
            self.store.set_runtime("market_session", "OPEN" if market_open else "CLOSED")
        if self.position_callback and market_open and self._last_position_run != position_bucket:
            self.position_callback()
            self._last_position_run = position_bucket
            if self.store:
                self.store.set_runtime("scheduler.last_position_run", position_bucket)
        if self.entry_callback and market_open and self._last_entry_run != entry_bucket:
            self.entry_callback()
            self._last_entry_run = entry_bucket
            if self.store:
                self.store.set_runtime("scheduler.last_entry_run", entry_bucket)
        if self.risk_callback and market_open and self._last_risk_run != risk_bucket:
            self.risk_callback()
            self._last_risk_run = risk_bucket
            if self.store:
                self.store.set_runtime("scheduler.last_risk_run", risk_bucket)
        research_day = self.config.full_research_day or self.config.weekly_day
        research_time = self.config.full_research_time or self.config.weekly_time
        daily_review_time = self.config.daily_review_time or self.config.daily_time
        is_session_day = self._is_session_day(current)
        scheduled_weekly_due = (
            self._weekly_command_id is None
            and self._weekly_status != "WAITING_FOR_MARKET"
            and (self._weekly_status != "FAILED" or self._weekly_command_day != day)
            and current.weekday() == _WEEKDAYS[research_day]
            and self._at_or_after(current, research_time)
            and self._last_weekly_run != day
        )
        waiting_retry_due = self._weekly_status == "WAITING_FOR_MARKET" and self.market_clock.is_open(current)
        if scheduled_weekly_due or waiting_retry_due:
            try:
                result = self.weekly_callback()
                command_id = self._command_id(result)
                if command_id is not None:
                    self._weekly_command_id = command_id
                    self._weekly_command_day = day
                    self._weekly_status = "QUEUED"
                    if self.store:
                        self.store.set_runtime("scheduler.weekly_command_id", command_id)
                        self.store.set_runtime("scheduler.weekly_command_day", day)
                        self.store.set_runtime("scheduler.weekly_status", self._weekly_status)
                    return
                state = str(result.get("state")) if isinstance(result, dict) and result.get("state") is not None else "COMPLETED"
                self._weekly_status = "WAITING_FOR_MARKET" if state == "WAITING_FOR_MARKET" else "COMPLETED"
                if self._weekly_status == "COMPLETED":
                    self._last_weekly_run = day
                    if self.store:
                        self.store.set_runtime("scheduler.last_weekly_run", day)
                if self.store:
                    self.store.set_runtime("scheduler.weekly_status", self._weekly_status)
            except Exception:
                self._weekly_status = "FAILED"
                if self.store:
                    self.store.set_runtime("scheduler.weekly_status", self._weekly_status)
                raise
        if self.pre_execution_callback and is_session_day and self._at_or_after(current, self.config.pre_execution_time) and self._last_pre_execution_run != day:
            self.pre_execution_callback()
            self._last_pre_execution_run = day
            if self.store:
                self.store.set_runtime("scheduler.last_pre_execution_run", day)
        if is_session_day and self._at_or_after(current, daily_review_time) and self._last_daily_run != day:
            self.daily_callback()
            self._last_daily_run = day
            if self.store:
                self.store.set_runtime("scheduler.last_daily_run", day)

    def _refresh_weekly_command(self):
        if self.store is None or self._weekly_command_id is None:
            return
        command = self.store.command(self._weekly_command_id)
        if command is None:
            self._weekly_status = "FAILED"
            self._weekly_command_id = None
            self.store.set_runtime("scheduler.weekly_status", self._weekly_status)
            return
        status = str(command.get("status") or "")
        if status in {"QUEUED", "CLAIMED", "RUNNING"}:
            self._weekly_status = status
        elif status == "FAILED":
            self._weekly_status = "FAILED"
            self._weekly_command_id = None
            self.store.set_runtime("scheduler.weekly_command_id", None)
        elif status == "SUCCEEDED":
            try:
                result = json.loads(command.get("result_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                result = {}
            state = str(result.get("state") or "COMPLETED") if isinstance(result, dict) else "COMPLETED"
            self._weekly_status = "WAITING_FOR_MARKET" if state in {"WAITING_FOR_MARKET", "PENDING_ENTRY"} else "COMPLETED"
            self._weekly_command_id = None
            self.store.set_runtime("scheduler.weekly_command_id", None)
            if self._weekly_status == "COMPLETED":
                self._last_weekly_run = self._weekly_command_day or self._last_weekly_run
                self.store.set_runtime("scheduler.last_weekly_run", self._last_weekly_run)
        self.store.set_runtime("scheduler.weekly_status", self._weekly_status)

    @staticmethod
    def _command_id(result):
        if isinstance(result, int):
            return result
        if isinstance(result, dict):
            value = result.get("command_id")
            return int(value) if value is not None else None
        return None

    def status(self) -> dict:
        return {"status": self._status, "weekly_status": self._weekly_status, "last_weekly_run": self._last_weekly_run, "last_daily_run": self._last_daily_run, "last_pre_execution_run": self._last_pre_execution_run, "last_risk_run": self._last_risk_run, "last_entry_run": self._last_entry_run, "last_position_run": self._last_position_run, "last_error": self._last_error}

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_once()
                self._last_error = None
            except Exception as exc:  # Keep the supervisor alive and expose the failure.
                self._last_error = str(exc)
                if self.store:
                    self.store.set_runtime("scheduler.last_error", self._last_error)
                    self.store.save_error("scheduler", self._last_error, severity="ERROR", component="scheduler")
            self._stop_event.wait(self.config.poll_seconds)

    @staticmethod
    def _at_or_after(now: datetime, hhmm: str) -> bool:
        hour, minute = (int(value) for value in hhmm.split(":"))
        return (now.hour, now.minute) >= (hour, minute)

    def _is_session_day(self, current: datetime) -> bool:
        checker = getattr(self.market_clock, "is_session_day", None)
        return bool(checker(current)) if callable(checker) else current.weekday() < 5
