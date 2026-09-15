from datetime import datetime, timedelta, timezone
import sqlite3
from types import SimpleNamespace

import pytest

from src.service import BackendService

from src.storage import SQLiteStore
from src.worker_health import invalidate_worker_health, worker_is_ready


def test_transport_loss_is_recoverable_but_account_failure_is_not():
    assert BackendService._is_recoverable_broker_error("IBKR transport is not connected after readiness callbacks")
    assert not BackendService._is_recoverable_broker_error("live account: IBKR transport is not connected")


@pytest.mark.parametrize("loop", ["heartbeat", "supervisor"])
def test_worker_recovers_when_error_recording_also_fails(tmp_path, monkeypatch, loop):
    with SQLiteStore(tmp_path / "recovery.sqlite3") as store:
        service = BackendService({}, SimpleNamespace(), store)
        attempts = []

        def fail_recording(*args, **kwargs):
            raise sqlite3.OperationalError("database or disk is full")

        def write_then_recover(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise sqlite3.OperationalError("database or disk is full")
            service._stop_event.set()

        monkeypatch.setattr(store, "save_error", fail_recording)
        monkeypatch.setattr(service._stop_event, "wait", lambda timeout: False)
        monkeypatch.setattr(service._heartbeat_stop_event, "wait", lambda timeout: False)
        if loop == "heartbeat":
            monkeypatch.setattr(service, "_write_worker_heartbeat", write_then_recover)
            service._heartbeat_loop()
        else:
            monkeypatch.setattr(service, "_set_runtime", write_then_recover)
            monkeypatch.setattr(service, "process_commands", lambda: None)
            monkeypatch.setattr(service, "heartbeat", lambda: None)
            service._supervise()
        assert len(attempts) >= 2


def test_worker_health_requires_fresh_running_heartbeat(tmp_path, monkeypatch):
    database = tmp_path / "worker-health.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database))
    with SQLiteStore(database) as store:
        store.set_runtime("command_worker_status", "RUNNING")
        store.set_runtime("command_worker_heartbeat", datetime.now(timezone.utc).isoformat())
    assert worker_is_ready() is True

    with SQLiteStore(database) as store:
        store.set_runtime("command_worker_heartbeat", (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat())
    assert worker_is_ready() is False

    invalidate_worker_health()
    with SQLiteStore(database, read_only=True) as store:
        assert store.get_runtime("command_worker_status") == "STOPPED"
        assert store.get_runtime("command_worker_heartbeat") is None
