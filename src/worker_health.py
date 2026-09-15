from __future__ import annotations

from datetime import datetime, timezone
import sys

from .config import env
from .storage import SQLiteStore


def worker_is_ready() -> bool:
    try:
        with SQLiteStore(env("DATABASE_PATH", "data/ai_fund_manager.sqlite3"), read_only=True) as store:
            status = store.get_runtime("command_worker_status")
            heartbeat = store.get_runtime("command_worker_heartbeat")
        if status != "RUNNING" or not heartbeat:
            return False
        timestamp = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() <= 10
    except Exception:
        return False


def invalidate_worker_health() -> None:
    with SQLiteStore(env("DATABASE_PATH", "data/ai_fund_manager.sqlite3")) as store:
        store.set_runtime("command_worker_status", "STOPPED")
        store.set_runtime("command_worker_heartbeat", None)


if __name__ == "__main__":
    if "--invalidate" in sys.argv:
        invalidate_worker_health()
        sys.exit(0)
    sys.exit(0 if worker_is_ready() else 1)
