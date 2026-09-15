from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data_provider import MockDataProvider, UnsupportedMacroEventProvider, WikipediaUniverseProvider, YahooFinanceDataProvider


SCREENING_FIELDS = {
    "symbol", "price", "ret_20d", "ret_60d", "ret_120d", "ann_vol",
    "relative_strength_vs_spy", "relative_strength_vs_qqq", "avg_dollar_volume",
    "market_cap", "forward_pe", "revenue_growth", "eps_growth", "earnings_days",
}


def test_universe_rows_have_compact_screening_contract():
    row = MockDataProvider().universe_snapshot()[0]
    assert SCREENING_FIELDS <= set(row)


def test_mock_and_yahoo_benchmarks_share_spy_qqq_series_schema():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")

    class FakeTicker:
        def __init__(self, multiplier):
            self.multiplier = multiplier

        def history(self, **kwargs):
            return pd.DataFrame({"Close": [(100 + index) * self.multiplier for index in range(30)]}, index=dates)

    yahoo = YahooFinanceDataProvider.__new__(YahooFinanceDataProvider)
    yahoo._ticker = lambda symbol: FakeTicker(1 if symbol == "SPY" else 2)
    yahoo._now = lambda: datetime.now(timezone.utc).isoformat()

    mock_result = MockDataProvider().benchmark_data(20)
    yahoo_result = yahoo.benchmark_data(20)

    for result in (mock_result, yahoo_result):
        assert set(result) == {"days", "dates", "series", "as_of", "source"}
        assert set(result["series"]) == {"SPY", "QQQ"}
        assert all(values for values in result["series"].values())
        assert len(result["dates"]) == len(result["series"]["SPY"]) == len(result["series"]["QQQ"])

    assert yahoo_result["dates"][-1] == dates[-1].date().isoformat()


def test_yahoo_universe_uses_one_batch_download_for_all_constituents():
    symbols = ["AAA", "BBB"]
    dates = pd.date_range("2026-01-01", periods=130, freq="D")
    columns = pd.MultiIndex.from_product([symbols, ["Close", "Volume"]])
    values = []
    for index in range(130):
        values.append([100 + index, 1_000_000, 200 + index, 2_000_000])
    frame = pd.DataFrame(values, index=dates, columns=columns)

    class FakeYF:
        def __init__(self):
            self.calls = []

        def download(self, requested, **kwargs):
            self.calls.append(list(requested))
            return frame

    yahoo = YahooFinanceDataProvider.__new__(YahooFinanceDataProvider)
    yahoo._yf = FakeYF()
    yahoo.universe_provider = type("Universe", (), {"fetch": lambda self: symbols})()
    yahoo._now = lambda: datetime.now(timezone.utc).isoformat()
    yahoo.price_history = lambda symbol, days=252: {"ret_20d": 0.01 if symbol == "SPY" else 0.02}

    rows = yahoo.universe_snapshot()

    assert yahoo._yf.calls == [symbols]
    assert [row["symbol"] for row in rows] == symbols
    assert all(SCREENING_FIELDS <= set(row) for row in rows)


def test_yahoo_universe_retries_symbols_missing_from_batch_download():
    symbols = ["AAA", "BBB"]
    dates = pd.date_range("2026-01-01", periods=130, freq="D")
    bulk = pd.DataFrame(
        [[100 + index, 1_000_000] for index in range(130)],
        index=dates,
        columns=pd.MultiIndex.from_product([["AAA"], ["Close", "Volume"]]),
    )
    fallback = pd.DataFrame(
        {"Close": [200 + index for index in range(130)], "Volume": [2_000_000] * 130},
        index=dates,
    )

    class FakeYF:
        def __init__(self): self.calls = []
        def download(self, requested, **kwargs):
            self.calls.append(list(requested))
            return bulk if len(requested) > 1 else fallback

    yahoo = YahooFinanceDataProvider.__new__(YahooFinanceDataProvider)
    yahoo._yf = FakeYF()
    yahoo.universe_provider = type("Universe", (), {"fetch": lambda self: symbols})()
    yahoo._now = lambda: datetime.now(timezone.utc).isoformat()
    yahoo.price_history = lambda symbol, days=252: {"ret_20d": 0.01}

    rows = yahoo.universe_snapshot()

    assert yahoo._yf.calls == [symbols, ["BBB"]]
    assert [row["symbol"] for row in rows] == symbols


def test_yahoo_universe_fails_closed_when_individual_retry_is_missing():
    symbols = ["AAA"]
    empty = pd.DataFrame(columns=["Close", "Volume"])

    class FakeYF:
        def download(self, requested, **kwargs): return empty

    yahoo = YahooFinanceDataProvider.__new__(YahooFinanceDataProvider)
    yahoo._yf = FakeYF()
    yahoo.universe_provider = type("Universe", (), {"fetch": lambda self: symbols})()
    yahoo._now = lambda: datetime.now(timezone.utc).isoformat()
    yahoo.price_history = lambda symbol, days=252: {"ret_20d": 0.01}

    with pytest.raises(RuntimeError, match="Universe snapshot incomplete.*AAA"):
        yahoo.universe_snapshot()


def test_wikipedia_universe_uses_explicit_user_agent(monkeypatch):
    requests = []

    class Response:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return self.body.encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if "api.nasdaq.com" in request.full_url:
            return Response('{"data":{"data":{"rows":[{"symbol":"AAPL"}]}}}')
        return Response("<table><tr><th>Symbol</th></tr><tr><td>BRK.B</td></tr></table>")

    monkeypatch.setattr("src.data_provider.urlopen", fake_urlopen)

    assert WikipediaUniverseProvider().fetch() == ["AAPL", "BRK-B"]
    assert len(requests) == 2
    assert requests[0][0].get_header("User-agent") == "AI-Fund-Manager/1.0 (local research)"
    assert requests[1][0].get_header("User-agent") == "Mozilla/5.0"
    assert all(timeout == 20 for _, timeout in requests)


def test_yahoo_quote_timestamp_comes_from_last_market_bar():
    dates = pd.date_range("2026-08-20", periods=30, freq="D", tz="UTC")
    history = pd.DataFrame(
        {"Open": range(100, 130), "Close": range(101, 131), "Volume": [1_000_000] * 30},
        index=dates,
    )

    class FakeTicker:
        def history(self, **kwargs):
            return history

    yahoo = YahooFinanceDataProvider.__new__(YahooFinanceDataProvider)
    yahoo._ticker = lambda symbol: FakeTicker()
    yahoo._now = lambda: datetime.now(timezone.utc).isoformat()

    result = yahoo.price_history("NVDA", 20)

    assert result["quote_as_of"] == dates[-1].isoformat()
    assert result["last_bar_at"] == dates[-1].isoformat()


def test_unsupported_macro_provider_has_explicit_fail_closed_contract():
    result = UnsupportedMacroEventProvider().upcoming_events()

    assert set(result) == {"supported", "events", "next_event", "hours_to_event", "risk_level", "as_of"}
    assert result["supported"] is False
    assert result["events"] == []
    assert result["risk_level"] == "UNKNOWN"
