from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from math import sqrt
from statistics import stdev
from typing import Any
from urllib.request import Request, urlopen

from .models import BenchmarkData, PortfolioState


class MacroEventProvider(ABC):
    @abstractmethod
    def upcoming_events(self) -> dict[str, Any]:
        raise NotImplementedError


class UnsupportedMacroEventProvider(MacroEventProvider):
    """Explicitly unavailable macro calendar used by Yahoo until a source is added."""

    def upcoming_events(self) -> dict[str, Any]:
        return {
            "supported": False,
            "events": [],
            "next_event": None,
            "hours_to_event": None,
            "risk_level": "UNKNOWN",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }


class DataProvider(ABC):
    """Read-only research contract used by both mock and real providers."""

    @abstractmethod
    def market_regime(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def universe_snapshot(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def stock_snapshot(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def price_history(self, symbol: str, days: int = 252) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fundamentals(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def earnings(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def analyst_revisions(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def news(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sec_filings(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def benchmark_data(self, days: int = 252) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def upcoming_events(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    def set_portfolio_context(self, state: PortfolioState, positions: list[dict[str, Any]]):
        self._portfolio_context = {**state.model_dump(mode="json"), "positions": positions, "source": "reconciled_broker"}

    def portfolio(self) -> dict[str, Any]:
        return getattr(self, "_portfolio_context", {"positions": [], "cash": None, "source": "unreconciled"})

    def current_positions(self) -> list[dict[str, Any]]:
        return self.portfolio().get("positions", [])


class MockDataProvider(DataProvider):
    """Small deterministic sample. Replace before making investment claims."""

    def __init__(self):
        self.now = datetime.now(timezone.utc).isoformat()
        self.rows = {
            "NVDA": {"price": 180.0, "ret_20d": 0.08, "ret_60d": 0.19, "ret_120d": 0.31, "ann_vol": 0.48, "revenue_growth": 0.31, "eps_growth": 0.36, "forward_pe": 31.0, "market_cap": 4_000_000_000_000, "avg_dollar_volume": 2_000_000_000, "earnings_days": 18, "gap_pct": 0.01},
            "META": {"price": 690.0, "ret_20d": 0.05, "ret_60d": 0.15, "ret_120d": 0.24, "ann_vol": 0.31, "revenue_growth": 0.18, "eps_growth": 0.22, "forward_pe": 25.0, "market_cap": 1_700_000_000_000, "avg_dollar_volume": 1_500_000_000, "earnings_days": 40, "gap_pct": 0.01},
            "AVGO": {"price": 335.0, "ret_20d": 0.07, "ret_60d": 0.17, "ret_120d": 0.28, "ann_vol": 0.39, "revenue_growth": 0.24, "eps_growth": 0.27, "forward_pe": 29.0, "market_cap": 1_500_000_000_000, "avg_dollar_volume": 900_000_000, "earnings_days": 28, "gap_pct": 0.02},
            "MSFT": {"price": 515.0, "ret_20d": 0.03, "ret_60d": 0.10, "ret_120d": 0.16, "ann_vol": 0.24, "revenue_growth": 0.15, "eps_growth": 0.17, "forward_pe": 30.0, "market_cap": 3_800_000_000_000, "avg_dollar_volume": 1_200_000_000, "earnings_days": 35, "gap_pct": 0.005},
        }
        self.benchmark_series = {"SPY": [100.0, 101.0, 102.515, 104.5653], "QQQ": [100.0, 101.5, 104.0375, 108.1995]}

    def market_regime(self):
        return {"as_of": self.now, "spy_above_200d": True, "qqq_above_200d": True, "vix": 17.8, "breadth": "positive"}

    def universe_snapshot(self):
        spy_20d = 0.02
        qqq_20d = 0.04
        return [{"symbol": s, **v, "relative_strength_vs_spy": v["ret_20d"] - spy_20d, "relative_strength_vs_qqq": v["ret_20d"] - qqq_20d, "as_of": self.now, "universe_sources": ["SP500", "NASDAQ100"]} for s, v in self.rows.items()]

    def stock_snapshot(self, symbol: str):
        symbol = symbol.upper()
        row = self.rows[symbol]
        return {"symbol": symbol, **row, "as_of": self.now, "quote_as_of": self.now, "last_bar_at": self.now, "trading_halted": False, "price_available": True}

    def price_history(self, symbol: str, days: int = 252):
        row = self.rows[symbol.upper()]
        return {"symbol": symbol.upper(), "days": days, "ret_20d": row["ret_20d"], "ret_60d": row["ret_60d"], "ret_120d": row["ret_120d"], "ann_vol": row["ann_vol"], "avg_dollar_volume": row["avg_dollar_volume"], "gap_pct": row["gap_pct"], "as_of": self.now}

    def fundamentals(self, symbol: str):
        row = self.rows[symbol.upper()]
        return {"symbol": symbol.upper(), "revenue_growth": row["revenue_growth"], "eps_growth": row["eps_growth"], "forward_pe": row["forward_pe"], "as_of": self.now}

    def earnings(self, symbol: str):
        row = self.rows[symbol.upper()]
        date = (datetime.now(timezone.utc) + timedelta(days=row["earnings_days"])).date().isoformat()
        return {"symbol": symbol.upper(), "next_earnings_date": date, "days_to_earnings": row["earnings_days"], "as_of": self.now}

    def analyst_revisions(self, symbol: str):
        return {"symbol": symbol.upper(), "upgrades_30d": 2, "downgrades_30d": 0, "eps_revision_direction": "positive", "as_of": self.now}

    def news(self, symbol: str, limit: int = 10):
        return {"symbol": symbol.upper(), "items": ["Mock news only; connect a real news source before trading."], "as_of": self.now}

    def sec_filings(self, symbol: str, limit: int = 10):
        return {"symbol": symbol.upper(), "items": [], "as_of": self.now}

    def benchmark_data(self, days: int = 252):
        count = len(self.benchmark_series["SPY"])
        cursor = datetime.fromisoformat(self.now).date()
        dates = []
        while len(dates) < count:
            if cursor.weekday() < 5:
                dates.append(cursor.isoformat())
            cursor -= timedelta(days=1)
        dates.reverse()
        return BenchmarkData(days=days, dates=dates, series=self.benchmark_series, as_of=self.now, source="mock").model_dump(mode="json")

    def upcoming_events(self, symbol: str):
        return {"symbol": symbol.upper(), "earnings_days": self.rows[symbol.upper()]["earnings_days"], "macro_event": False, "macro_event_supported": True, "as_of": self.now}


class WikipediaUniverseProvider:
    """Fetch and deduplicate S&P 500 and Nasdaq-100 constituents on demand."""

    def fetch(self) -> list[str]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas to fetch the Wikipedia universe") from exc

        urls = ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"]
        symbols: set[str] = set()
        for url in urls:
            request = Request(url, headers={"User-Agent": "AI-Fund-Manager/1.0 (local research)"})
            with urlopen(request, timeout=20) as response:
                tables = pd.read_html(StringIO(response.read().decode("utf-8")))
            column = "Symbol"
            table = next((t for t in tables if column in t.columns), None)
            if table is None:
                raise RuntimeError(f"Could not find {column} in {url}")
            symbols.update(str(value).replace(".", "-").upper() for value in table[column].dropna())
        nasdaq_request = Request(
            "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
            },
        )
        with urlopen(nasdaq_request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = (((payload.get("data") or {}).get("data") or {}).get("rows") or [])
        if not rows:
            raise RuntimeError("Could not fetch Nasdaq-100 constituents from Nasdaq")
        symbols.update(str(row["symbol"]).replace(".", "-").upper() for row in rows if row.get("symbol"))
        return sorted(symbols)


class YahooFinanceDataProvider(DataProvider):
    """Low-cost Yahoo Finance provider. It is opt-in and intentionally not used by mock mode."""

    def __init__(self, universe_provider: WikipediaUniverseProvider | None = None, macro_event_provider: MacroEventProvider | None = None):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance and pandas to use the real data provider") from exc
        self._yf = yf
        self.universe_provider = universe_provider or WikipediaUniverseProvider()
        self.macro_event_provider = macro_event_provider or UnsupportedMacroEventProvider()
        self._now = lambda: datetime.now(timezone.utc).isoformat()

    def market_regime(self):
        spy = self.price_history("SPY", 260)
        qqq = self.price_history("QQQ", 260)
        return {"as_of": self._now(), "spy_above_200d": spy.get("above_200d"), "qqq_above_200d": qqq.get("above_200d"), "source": "yahoo_finance"}

    def universe_snapshot(self):
        as_of = self._now()
        symbols = self.universe_provider.fetch()
        history = self._yf.download(symbols, period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        benchmark_returns = {symbol: self.price_history(symbol, 130).get("ret_20d") for symbol in ("SPY", "QQQ")}
        rows = []
        missing = []
        for symbol in symbols:
            try:
                frame = history[symbol] if getattr(history.columns, "nlevels", 1) > 1 else history
                rows.append(self._universe_row(symbol, frame, benchmark_returns, as_of))
            except (KeyError, TypeError, ValueError, IndexError):
                missing.append(symbol)
        for symbol in missing:
            try:
                fallback = self._yf.download([symbol], period="1y", interval="1d", group_by="ticker", auto_adjust=False, threads=False, progress=False)
                frame = fallback[symbol] if getattr(fallback.columns, "nlevels", 1) > 1 else fallback
                rows.append(self._universe_row(symbol, frame, benchmark_returns, as_of))
            except (KeyError, TypeError, ValueError, IndexError):
                pass
        completed = {row["symbol"] for row in rows}
        unresolved = [symbol for symbol in symbols if symbol not in completed]
        if unresolved:
            raise RuntimeError(f"Universe snapshot incomplete: {len(unresolved)} symbol(s) missing: {', '.join(unresolved)}")
        return rows

    def _universe_row(self, symbol, frame, benchmark_returns, as_of):
        closes = self._series_values(frame)
        if len(closes) < 21:
            raise ValueError(f"Insufficient universe history for {symbol}")
        returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
        volumes = frame["Volume"] if "Volume" in frame else None
        ret_20d = closes[-1] / closes[-21] - 1
        return {
            "symbol": symbol, "price": closes[-1], "ret_20d": ret_20d,
            "ret_60d": closes[-1] / closes[-61] - 1 if len(closes) > 60 else None,
            "ret_120d": closes[-1] / closes[-121] - 1 if len(closes) > 120 else None,
            "ann_vol": stdev(returns[-252:]) * sqrt(252) if len(returns) > 1 else None,
            "relative_strength_vs_spy": ret_20d - benchmark_returns["SPY"] if benchmark_returns["SPY"] is not None else None,
            "relative_strength_vs_qqq": ret_20d - benchmark_returns["QQQ"] if benchmark_returns["QQQ"] is not None else None,
            "avg_dollar_volume": float((frame["Close"] * volumes).tail(60).mean()) if volumes is not None else None,
            "market_cap": None, "forward_pe": None, "revenue_growth": None, "eps_growth": None,
            "earnings_days": None, "as_of": as_of, "universe_sources": ["SP500_OR_NASDAQ100"],
        }

    def _ticker(self, symbol: str):
        return self._yf.Ticker(symbol.upper())

    @staticmethod
    def _series_values(history):
        return [float(value) for value in history["Close"].dropna().tolist()]

    def price_history(self, symbol: str, days: int = 252):
        history = self._ticker(symbol).history(period="2y", auto_adjust=False)
        closes = self._series_values(history)
        if len(closes) < 21:
            raise RuntimeError(f"Insufficient price history for {symbol}")
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        sample = returns[-min(days, len(returns)):]
        vol = stdev(sample) * sqrt(252) if len(sample) > 1 else None
        last_bar_at = history["Close"].dropna().index[-1].isoformat()
        return {"symbol": symbol.upper(), "days": days, "price": closes[-1], "prices": closes[-days:], "ret_20d": closes[-1] / closes[-21] - 1, "ret_60d": closes[-1] / closes[-61] - 1 if len(closes) > 60 else None, "ret_120d": closes[-1] / closes[-121] - 1 if len(closes) > 120 else None, "ann_vol": vol, "avg_dollar_volume": float((history["Close"] * history["Volume"]).tail(days).mean()) if "Volume" in history else None, "gap_pct": float(history["Open"].iloc[-1] / history["Close"].iloc[-2] - 1) if "Open" in history else None, "above_200d": closes[-1] > sum(closes[-200:]) / min(200, len(closes)), "as_of": self._now(), "quote_as_of": last_bar_at, "last_bar_at": last_bar_at}

    def stock_snapshot(self, symbol: str):
        return {**self.price_history(symbol), **self.fundamentals(symbol), **self.earnings(symbol), "symbol": symbol.upper()}

    def fundamentals(self, symbol: str):
        info = self._ticker(symbol).info
        return {"symbol": symbol.upper(), "revenue_growth": info.get("revenueGrowth"), "eps_growth": info.get("earningsQuarterlyGrowth"), "forward_pe": info.get("forwardPE"), "as_of": self._now()}

    def earnings(self, symbol: str):
        dates = self._ticker(symbol).get_earnings_dates(limit=8)
        future = [date for date in dates.index if date.to_pydatetime().astimezone(timezone.utc) > datetime.now(timezone.utc)]
        days = (future[0].to_pydatetime().date() - datetime.now(timezone.utc).date()).days if future else None
        return {"symbol": symbol.upper(), "next_earnings_date": future[0].date().isoformat() if future else None, "days_to_earnings": days, "as_of": self._now()}

    def analyst_revisions(self, symbol: str):
        rows = self._ticker(symbol).upgrades_downgrades
        return {"symbol": symbol.upper(), "recent_revisions": rows.head(10).reset_index().to_dict("records") if rows is not None else [], "as_of": self._now()}

    def news(self, symbol: str, limit: int = 10):
        return {"symbol": symbol.upper(), "items": self._ticker(symbol).news[:limit], "as_of": self._now()}

    def sec_filings(self, symbol: str, limit: int = 10):
        filings = getattr(self._ticker(symbol), "sec_filings", None)
        return {"symbol": symbol.upper(), "items": filings[:limit] if filings else [], "as_of": self._now()}

    def benchmark_data(self, days: int = 252):
        closes_by_symbol = {}
        for symbol in ("SPY", "QQQ"):
            history = self._ticker(symbol).history(period="2y", auto_adjust=False)
            closes = history["Close"].dropna()
            if closes.empty:
                raise RuntimeError(f"Missing benchmark history for {symbol}")
            closes_by_symbol[symbol] = closes
        common_dates = closes_by_symbol["SPY"].index.intersection(closes_by_symbol["QQQ"].index).sort_values()[-days:]
        if len(common_dates) == 0:
            raise RuntimeError("SPY and QQQ have no common benchmark trading dates")
        series = {}
        for symbol, closes in closes_by_symbol.items():
            values = [float(value) for value in closes.loc[common_dates].tolist()]
            series[symbol] = [100.0 * value / values[0] for value in values]
        dates = [value.date().isoformat() if hasattr(value, "date") else str(value)[:10] for value in common_dates]
        return BenchmarkData(days=days, dates=dates, series=series, as_of=self._now(), source="yahoo_finance").model_dump(mode="json")

    def upcoming_events(self, symbol: str):
        macro = self.macro_event_provider.upcoming_events()
        return {
            **self.earnings(symbol),
            "symbol": symbol.upper(),
            "macro_event": macro.get("next_event"),
            "macro_event_supported": macro["supported"],
            "macro_events": macro["events"],
            "macro_hours_to_event": macro["hours_to_event"],
            "macro_risk_level": macro["risk_level"],
            "macro_as_of": macro["as_of"],
        }
