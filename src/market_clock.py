from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class TradingSessionProvider(Protocol):
    def is_open(self, at: datetime | None = None) -> bool:
        ...


class USEquityMarketClock:
    """NYSE regular-session calendar, including holidays and early closes."""

    def __init__(self, now=None):
        self._now = now or (lambda: datetime.now(timezone.utc))

    def is_open(self, at: datetime | None = None) -> bool:
        try:
            import exchange_calendars as xcals
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("exchange-calendars is required when market-hours enforcement is enabled") from exc
        timestamp = pd.Timestamp(at or self._now()).tz_convert("UTC").floor("min")
        return bool(xcals.get_calendar("XNYS").is_open_on_minute(timestamp))

    def latest_completed_session(self, at: datetime | None = None) -> dict[str, str]:
        try:
            import exchange_calendars as xcals
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("exchange-calendars is required when market-hours enforcement is enabled") from exc
        timestamp = pd.Timestamp(at or self._now())
        timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
        calendar = xcals.get_calendar("XNYS")
        session_date = timestamp.normalize().tz_localize(None)
        session = calendar.date_to_session(session_date, direction="previous")
        if calendar.session_close(session) > timestamp:
            session = calendar.previous_session(session)
        close = calendar.session_close(session)
        return {"date": session.date().isoformat(), "close": close.isoformat()}

    def is_session_day(self, at: datetime | None = None) -> bool:
        try:
            import exchange_calendars as xcals
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("exchange-calendars is required when market-hours enforcement is enabled") from exc
        timestamp = pd.Timestamp(at or self._now())
        timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
        return bool(xcals.get_calendar("XNYS").is_session(timestamp.date()))

    def session_open(self, at: datetime | None = None) -> datetime:
        try:
            import exchange_calendars as xcals
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("exchange-calendars is required when market-hours enforcement is enabled") from exc
        timestamp = pd.Timestamp(at or self._now())
        timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
        session = xcals.get_calendar("XNYS").date_to_session(timestamp.date(), direction="none")
        return xcals.get_calendar("XNYS").session_open(session).to_pydatetime()
