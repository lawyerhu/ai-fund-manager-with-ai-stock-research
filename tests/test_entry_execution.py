from datetime import datetime, timedelta, timezone

import pytest

from src.entry_execution import EntryExecutionEngine
from src.models import BrokerSnapshot, ExecutionQuote, ExecutionReport, OrderRequest, PositionSnapshot, RiskDecision, TradeIntent
from src.data_provider import MockDataProvider
from src.execution.observe import ObserveExecutor
from src.execution.paper import LocalPaperExecutor
from src.runner import StrategyRunner
from src.scheduler import PaperScheduler, SchedulerConfig
from src.storage import SQLiteStore


class ClosedMarketClock:
    def is_open(self, at=None):
        return False


def buy_intent(**updates):
    values = {
        "action": "BUY",
        "symbol": "MPC",
        "target_weight": 0.98,
        "confidence": 0.82,
        "holding_period_days": 20,
        "expected_alpha_vs_spy": 0.04,
        "expected_alpha_vs_qqq": 0.03,
        "thesis": ["Refining margins are improving"],
        "risk_factors": ["Oil-price volatility"],
        "invalidation_conditions": ["Guidance reduction"],
        "evidence_used": ["fundamentals", "earnings"],
        "timestamp": datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc).isoformat(),
        "decision_id": "decision-mpc-1",
        "model_name": "gpt-5.6-sol",
    }
    values.update(updates)
    return TradeIntent(**values)


def test_closed_market_buy_is_persisted_as_pending_entry_without_broker_mutation(tmp_path):
    with SQLiteStore(tmp_path / "entry.sqlite3") as store:
        engine = EntryExecutionEngine(
            {"execution": {"regular_hours_only": True}},
            store=store,
            market_clock=ClosedMarketClock(),
        )

        entry = engine.capture(
            buy_intent(),
            run_id="run-1",
            decision_reference_price=193.20,
            market_regime_at_decision="NEUTRAL",
        )

        assert entry.status == "PENDING_ENTRY"
        assert entry.decision_id == "decision-mpc-1"
        assert entry.symbol == "MPC"
        assert entry.decision_reference_price == 193.20
        assert entry.positive_factors == ["Refining margins are improving"]
        assert entry.risks == ["Oil-price volatility"]
        assert store.pending_entry("decision-mpc-1").status == "PENDING_ENTRY"
        assert store.recent("order_records", 10) == []
        assert store.recent("executions", 10) == []


def test_open_entry_revalidation_accepts_small_gap_and_calculates_marketable_limit(tmp_path):
    class OpenMarketClock:
        def is_open(self, at=None):
            return True

    now = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)
    quote = ExecutionQuote(
        symbol="MPC", price=101.0, bid=100.99, ask=101.01, mid=101.0,
        timestamp=now.isoformat(), market_status="OPEN", source="IBKR",
        data_type="REALTIME",
    )
    with SQLiteStore(tmp_path / "entry-open.sqlite3") as store:
        engine = EntryExecutionEngine(
            {"execution": {"entry_delay_after_open_seconds": 0, "limit_slippage_bps": 5, "entry_revalidation": {"max_price_gap_pct_without_review": 0.03, "max_spread_bps": 30}}},
            store=store, market_clock=OpenMarketClock(),
        )
        entry = engine.capture(buy_intent(), run_id="run-2", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        result = engine.revalidate(entry, quote, now=now)
        assert result.entry.status == "ENTRY_VALID"
        assert result.checks["price_gap_pct"] == 0.01
        assert engine.marketable_limit_price(quote) == 101.07
        assert engine.marketable_limit_price(quote, action="SELL") == 100.93


def test_large_gap_or_stale_quote_never_becomes_executable(tmp_path):
    class OpenMarketClock:
        def is_open(self, at=None): return True

    now = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)
    with SQLiteStore(tmp_path / "entry-guards.sqlite3") as store:
        engine = EntryExecutionEngine(
            {"execution": {"entry_delay_after_open_seconds": 0, "entry_max_quote_age_seconds": 10, "entry_revalidation": {"max_price_gap_pct_without_review": 0.03, "max_spread_bps": 30}}},
            store=store, market_clock=OpenMarketClock(),
        )
        entry = engine.capture(buy_intent(decision_id="decision-mpc-2"), run_id="run-3", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        stale = ExecutionQuote(symbol="MPC", price=101, bid=100.9, ask=101.1, mid=101, timestamp="2026-08-31T13:00:00+00:00", market_status="OPEN", source="IBKR", data_type="REALTIME")
        assert engine.revalidate(entry, stale, now=now).entry.status == "ENTRY_FAILED"
        fresh_gap = stale.model_copy(update={"price": 108, "bid": 107.9, "ask": 108.1, "mid": 108, "timestamp": now.isoformat()})
        assert engine.revalidate(entry, fresh_gap, now=now).entry.status == "ENTRY_REVIEW_REQUIRED"


def test_runner_routes_closed_buy_to_pending_entry_and_never_submits(tmp_path):
    class ClosedData(MockDataProvider):
        def stock_snapshot(self, symbol):
            return {**super().stock_snapshot(symbol), "price": 193.20, "quote_as_of": self.now}

    class ClosedQuote:
        def get_quote(self, symbol):
            return ExecutionQuote(
                symbol=symbol, price=193.20, timestamp="2026-08-28T20:00:00+00:00",
                market_status="CLOSED", source="IBKR historical", data_type="FROZEN",
            )

    cfg = {
        "portfolio": {"starting_equity": 1000, "max_positions": 1},
        "execution": {"entry_engine_enabled": True, "enforce_market_hours": True},
        "agent": {"decision_horizon_days": 20},
        "risk": {"hard_drawdown_limit": 0.25, "drawdown_tiers": [], "event_risk": {}},
    }
    with SQLiteStore(tmp_path / "runner-pending.sqlite3") as store:
        broker = ObserveExecutor(starting_cash=1000)
        runner = StrategyRunner(cfg, ClosedData(), broker, store, market_clock=ClosedMarketClock(), quote_provider=ClosedQuote())
        result = runner.run(intent=buy_intent(symbol="NVDA", decision_id="decision-nvda-closed"), use_llm=False)

        assert str(result["state"]) == "PENDING_ENTRY"
        assert result["pending_entry"].status == "PENDING_ENTRY"
        assert broker.place_order_calls == 0
        assert broker.cancel_order_calls == 0
        assert store.recent("order_records", 10) == []


def test_entry_builds_marketable_limit_only_after_risk_and_persists_actual_fill(tmp_path):
    class OpenMarketClock:
        def is_open(self, at=None): return True

    now = datetime.now(timezone.utc).isoformat()
    quote = ExecutionQuote(symbol="NVDA", price=100, bid=99.99, ask=100.01, mid=100, timestamp=now, market_status="OPEN", source="IBKR", data_type="REALTIME")
    portfolio = BrokerSnapshot(equity=1000, cash=1000, positions=[], open_orders=[], source="IBKR_PAPER")
    risk = RiskDecision(approved=True, requested_weight=0.5, approved_weight=0.5, reason="approved")
    with SQLiteStore(tmp_path / "entry-order.sqlite3") as store:
        engine = EntryExecutionEngine({"execution": {"entry_delay_after_open_seconds": 0, "limit_slippage_bps": 5, "max_total_slippage_bps": 30}}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA", decision_id="decision-nvda-order"), run_id="run-order", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        entry = entry.model_copy(update={"status": "ENTRY_VALID", "current_price": 100})
        store.update_pending_entry(entry)
        order = engine.build_order(entry, quote, portfolio, risk, broker_connected=True, broker_reconciled=True, execution_mode="PAPER")
        assert order.order_type == "LMT"
        assert order.limit_price == 100.07
        assert order.quantity == 1000 * 0.5 / 100.07
        report = ExecutionReport(client_order_id=order.client_order_id, decision_id=entry.decision_id, status="FILLED", filled_quantity=order.quantity, remaining_quantity=0, average_price=100.05, message="Paper fill")
        filled = engine.apply_execution(entry, report, quote_mid=100)
        assert filled.status == "ENTRY_FILLED"
        assert filled.entry_price == 100.05
        assert filled.filled_quantity == order.quantity
        assert filled.slippage_vs_decision == pytest.approx(0.0005)


def open_quote(price=100, *, bid=99.99, ask=100.01, timestamp=None):
    return ExecutionQuote(symbol="NVDA", price=price, bid=bid, ask=ask, mid=(bid + ask) / 2, timestamp=timestamp or datetime.now(timezone.utc).isoformat(), market_status="OPEN", source="IBKR", data_type="REALTIME")


class OpenMarketClock:
    def is_open(self, at=None): return True


@pytest.mark.parametrize("bid,ask,expected_status,expected_spread", [
    (99.99, 100.01, "ENTRY_VALID", 2.0),
    (99.0, 101.0, "ENTRY_REVIEW_REQUIRED", 200.0),
    (None, None, "ENTRY_VALID", None),
    (None, 100.01, "ENTRY_VALID", None),
])
def test_missing_mid_uses_available_prices_and_preserves_spread_check(tmp_path, bid, ask, expected_status, expected_spread):
    now = datetime.now(timezone.utc)
    quote = ExecutionQuote(symbol="NVDA", price=100, bid=bid, ask=ask,
                           timestamp=now.isoformat(), market_status="OPEN", source="IBKR", data_type="REALTIME")
    with SQLiteStore(tmp_path / "missing-mid.sqlite3") as store:
        engine = EntryExecutionEngine({}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        result = engine.revalidate(entry, quote, now=now)
        assert result.entry.status == expected_status
        assert result.checks["current_price"] == 100
        if expected_spread is None:
            assert result.checks["spread_bps"] is None
        else:
            assert result.checks["spread_bps"] == pytest.approx(expected_spread)
        if ask is None:
            with pytest.raises(ValueError, match="ask"):
                engine.marketable_limit_price(quote)


@pytest.mark.parametrize("age,expected_status", [
    (-86400, "ENTRY_FAILED"), (-1.001, "ENTRY_FAILED"),
    (-1, "ENTRY_VALID"), (0, "ENTRY_VALID"),
    (10, "ENTRY_VALID"), (10.001, "ENTRY_FAILED"),
])
def test_quote_time_boundaries(tmp_path, age, expected_status):
    now = datetime.now(timezone.utc)
    quote = open_quote(timestamp=(now - timedelta(seconds=age)).isoformat())
    with SQLiteStore(tmp_path / "quote-time.sqlite3") as store:
        engine = EntryExecutionEngine({"execution": {"entry_max_quote_age_seconds": 10}}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        result = engine.revalidate(entry, quote, now=now)
        assert result.entry.status == expected_status
        assert result.checks["quote_age_seconds"] == pytest.approx(age)


def test_entry_waits_for_configured_post_open_delay(tmp_path):
    now = datetime.now(timezone.utc)
    with SQLiteStore(tmp_path / "delay.sqlite3") as store:
        engine = EntryExecutionEngine({"execution": {"entry_delay_after_open_seconds": 120}}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA", decision_id="delay"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        result = engine.revalidate(entry, open_quote(timestamp=now.isoformat()), now=now, opened_at=now - timedelta(seconds=119))
        assert result.entry.status == "PENDING_ENTRY"


def test_wide_spread_requires_review(tmp_path):
    with SQLiteStore(tmp_path / "spread.sqlite3") as store:
        engine = EntryExecutionEngine({"execution": {"entry_delay_after_open_seconds": 0, "entry_revalidation": {"max_spread_bps": 30}}}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA", decision_id="spread"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        assert engine.revalidate(entry, open_quote(bid=99, ask=101)).entry.status == "ENTRY_REVIEW_REQUIRED"


def test_major_news_requires_review_even_when_price_gap_is_small(tmp_path):
    with SQLiteStore(tmp_path / "news.sqlite3") as store:
        engine = EntryExecutionEngine({"execution": {"entry_delay_after_open_seconds": 0}}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA", decision_id="news"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        assert engine.revalidate(entry, open_quote(), material_events={"major_news": True}).entry.status == "ENTRY_REVIEW_REQUIRED"


def test_sol_cancel_entry_closes_pending_intent(tmp_path):
    with SQLiteStore(tmp_path / "cancel.sqlite3") as store:
        engine = EntryExecutionEngine({}, store=store, market_clock=OpenMarketClock())
        entry = engine.capture(buy_intent(symbol="NVDA", decision_id="cancel"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        cancelled = engine.apply_quick_review(entry, {"action": "CANCEL_ENTRY", "confidence": 0.9, "thesis_status": "BROKEN", "max_acceptable_price_deviation": 0.0, "reasons": ["Guidance was reduced"]})
        assert cancelled.status == "ENTRY_CANCELLED"


def valid_entry_and_context(store, decision_id="guards"):
    engine = EntryExecutionEngine({"execution": {"limit_slippage_bps": 5, "max_total_slippage_bps": 30}}, store=store, market_clock=OpenMarketClock())
    entry = engine.capture(buy_intent(symbol="NVDA", decision_id=decision_id), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL").model_copy(update={"status": "ENTRY_VALID", "current_price": 100})
    store.update_pending_entry(entry)
    risk = RiskDecision(approved=True, requested_weight=0.5, approved_weight=0.5, reason="approved")
    return engine, entry, risk


@pytest.mark.parametrize("connected,reconciled", [(False, True), (True, False)])
def test_connection_or_reconciliation_gap_blocks_entry(tmp_path, connected, reconciled):
    with SQLiteStore(tmp_path / f"connection-{connected}-{reconciled}.sqlite3") as store:
        engine, entry, risk = valid_entry_and_context(store)
        with pytest.raises(RuntimeError, match="connection and reconciliation"):
            engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=1000, source="IBKR_PAPER"), risk, broker_connected=connected, broker_reconciled=reconciled, execution_mode="PAPER")


def test_live_execution_mode_is_blocked(tmp_path):
    with SQLiteStore(tmp_path / "live.sqlite3") as store:
        engine, entry, risk = valid_entry_and_context(store)
        with pytest.raises(RuntimeError, match="only in PAPER"):
            engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=1000, source="IBKR_PAPER"), risk, broker_connected=True, broker_reconciled=True, execution_mode="LIVE")


def test_margin_and_gross_exposure_are_blocked(tmp_path):
    with SQLiteStore(tmp_path / "exposure.sqlite3") as store:
        engine, entry, risk = valid_entry_and_context(store)
        with pytest.raises(RuntimeError, match="Margin"):
            engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=0, source="IBKR_PAPER"), risk, broker_connected=True, broker_reconciled=True, execution_mode="PAPER")
        held = PositionSnapshot(symbol="MSFT", quantity=6, market_price=100, market_value=600)
        with pytest.raises(RuntimeError, match="Gross exposure"):
            engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=400, positions=[held], invested_value=600, source="IBKR_PAPER"), risk, broker_connected=True, broker_reconciled=True, execution_mode="PAPER")


def test_risk_rejection_and_maximum_chase_block_order(tmp_path):
    with SQLiteStore(tmp_path / "risk-slippage.sqlite3") as store:
        engine, entry, risk = valid_entry_and_context(store)
        rejected = risk.model_copy(update={"approved": False, "approved_weight": 0})
        with pytest.raises(RuntimeError, match="Risk Engine"):
            engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=1000, source="IBKR_PAPER"), rejected, broker_connected=True, broker_reconciled=True, execution_mode="PAPER")
        with pytest.raises(RuntimeError, match="slippage"):
            engine.build_order(entry, open_quote(101, bid=100.99, ask=101.01), BrokerSnapshot(equity=1000, cash=1000, source="IBKR_PAPER"), risk, broker_connected=True, broker_reconciled=True, execution_mode="PAPER")


def test_non_fractional_quantity_rounds_down(tmp_path):
    with SQLiteStore(tmp_path / "whole.sqlite3") as store:
        engine, entry, risk = valid_entry_and_context(store)
        order = engine.build_order(entry, open_quote(), BrokerSnapshot(equity=1000, cash=1000, source="IBKR_PAPER"), risk, broker_connected=True, broker_reconciled=True, execution_mode="PAPER", fractional_shares=False)
        assert order.quantity == 4


def test_partial_fill_and_restart_restore_pending_state(tmp_path):
    path = tmp_path / "restart.sqlite3"
    with SQLiteStore(path) as store:
        engine, entry, _ = valid_entry_and_context(store, "partial")
        report = ExecutionReport(client_order_id="order", decision_id=entry.decision_id, status="PARTIALLY_FILLED", filled_quantity=3, remaining_quantity=1.9, average_price=100.05)
        assert engine.apply_execution(entry, report, quote_mid=100).status == "ENTRY_PARTIALLY_FILLED"
    with SQLiteStore(path) as reopened:
        restored = reopened.pending_entry("partial")
        assert restored.status == "ENTRY_PARTIALLY_FILLED"
        assert restored.filled_quantity == 3
        assert restored.remaining_quantity == 1.9


def test_duplicate_capture_is_idempotent(tmp_path):
    with SQLiteStore(tmp_path / "duplicate.sqlite3") as store:
        engine = EntryExecutionEngine({}, store=store, market_clock=ClosedMarketClock())
        first = engine.capture(buy_intent(symbol="NVDA", decision_id="same"), run_id="run", decision_reference_price=100, market_regime_at_decision="NEUTRAL")
        second = engine.capture(buy_intent(symbol="NVDA", decision_id="same"), run_id="other", decision_reference_price=110, market_regime_at_decision="RISK_OFF")
        assert first == second
        assert len(store.pending_entries()) == 1


def entry_runner_config(max_requotes=2):
    return {
        "portfolio": {"starting_equity": 1000, "max_positions": 1},
        "execution": {"entry_engine_enabled": True, "entry_delay_after_open_seconds": 0, "entry_max_quote_age_seconds": 10, "limit_slippage_bps": 5, "max_total_slippage_bps": 30, "entry_order_timeout_seconds": 0.01, "max_entry_requotes": max_requotes, "fractional_shares": True},
        "risk": {"hard_drawdown_limit": 0.25, "halt_recovery_drawdown": 0.20, "target_annualized_vol": 0.25, "absolute_max_weight": 1, "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_decision_age_minutes_for_execution": 60, "max_annualized_volatility": 1, "min_avg_dollar_volume": 0, "max_gap_pct": 0.08, "drawdown_tiers": [], "event_risk": {}},
    }


class FreshNVDAQuote:
    def get_quote(self, symbol):
        return open_quote(180, bid=179.99, ask=180.01).model_copy(update={"symbol": symbol})


class CapturingPaperBroker(LocalPaperExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted_orders = []

    def submit_order(self, request, **kwargs):
        self.last_submitted_order = request
        self.submitted_orders.append(request)
        return super().submit_order(request, **kwargs)


def seed_pending_entry(runner, decision_id):
    intent = buy_intent(symbol="NVDA", decision_id=decision_id, timestamp=datetime.now(timezone.utc).isoformat())
    return runner.entry_engine.capture(intent, run_id="synthetic-run", decision_reference_price=180, market_regime_at_decision="NEUTRAL")


def test_open_market_synthetic_local_paper_entry_reaches_actual_fill(tmp_path):
    with SQLiteStore(tmp_path / "synthetic-fill.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        runner = StrategyRunner(entry_runner_config(), MockDataProvider(), broker, store, market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote())
        seed_pending_entry(runner, "synthetic-fill")
        result = runner.process_pending_entries()
        assert result[0]["status"] == "ENTRY_FILLED"
        assert store.pending_entry("synthetic-fill").entry_price == pytest.approx(180)
        assert broker.current_positions()[0]["symbol"] == "NVDA"


def test_pending_entry_rechecks_holdings_immediately_before_submit(tmp_path):
    class PositionAppearsBroker(LocalPaperExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.position_snapshot_calls = 0
            self.submit_calls = 0

        def position_symbols(self, timeout_seconds=None):
            self.position_snapshot_calls += 1
            if self.position_snapshot_calls == 2:
                self._positions["NVDA"] = 1.0
                self._prices["NVDA"] = 180.0
                self._average_costs["NVDA"] = 180.0
                self._cash -= 180.0
            return super().position_symbols(timeout_seconds)

        def submit_order(self, request, **kwargs):
            self.submit_calls += 1
            return super().submit_order(request, **kwargs)

    with SQLiteStore(tmp_path / "entry-final-holdings-check.sqlite3") as store:
        broker = PositionAppearsBroker(starting_cash=1000, store=store)
        runner = StrategyRunner(
            entry_runner_config(), MockDataProvider(), broker, store,
            market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote(),
        )
        seed_pending_entry(runner, "entry-final-holdings-check")

        result = runner.process_pending_entries()

        assert result[0]["status"] == "ENTRY_FAILED"
        assert broker.submit_calls == 0


def test_open_market_buy_uses_the_durable_entry_engine_and_limit_order(tmp_path):
    with SQLiteStore(tmp_path / "open-run-entry.sqlite3") as store:
        broker = CapturingPaperBroker(starting_cash=1000, store=store)
        runner = StrategyRunner(
            entry_runner_config(), MockDataProvider(), broker, store,
            market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote(),
        )
        intent = buy_intent(
            symbol="NVDA",
            decision_id="open-run-entry",
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_weight=0.5,
        )

        result = runner.run(intent=intent)

        assert result["state"] == "FILLED"
        assert broker.last_submitted_order.order_type == "LMT"
        assert store.pending_entry(intent.decision_id).status == "ENTRY_FILLED"


def test_entry_engine_adds_only_the_approved_target_delta(tmp_path):
    with SQLiteStore(tmp_path / "entry-add-delta.sqlite3") as store:
        broker = CapturingPaperBroker(starting_cash=1000, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=2, reference_price=180))
        runner = StrategyRunner(
            entry_runner_config(), MockDataProvider(), broker, store,
            market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote(),
        )
        intent = buy_intent(
            symbol="NVDA",
            decision_id="entry-add-delta",
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_weight=0.5,
        )

        result = runner.run(intent=intent)

        assert result["portfolio"].current_weight == pytest.approx(0.5, abs=0.002)
        assert broker.last_submitted_order.order_type == "LMT"


def test_cash_liquidation_uses_realtime_bid_and_limit_order(tmp_path):
    with SQLiteStore(tmp_path / "entry-cash-limit.sqlite3") as store:
        broker = CapturingPaperBroker(starting_cash=1000, store=store)
        broker.connect()
        broker.submit_order(OrderRequest(decision_id="seed", symbol="NVDA", action="BUY", quantity=2, reference_price=180))
        runner = StrategyRunner(
            entry_runner_config(), MockDataProvider(), broker, store,
            market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote(),
        )
        intent = TradeIntent(
            action="CASH", symbol=None, target_weight=0.0, confidence=1.0,
            holding_period_days=1, expected_alpha_vs_spy=0.0, expected_alpha_vs_qqq=0.0,
            thesis=["Exit"], risk_factors=["Risk"], invalidation_conditions=["Resume"],
            evidence_used=["position_review"], decision_id="entry-cash-limit",
        )

        result = runner.run(intent=intent)

        sell = broker.submitted_orders[-1]
        assert result["state"] == "FILLED"
        assert sell.action == "SELL"
        assert sell.order_type == "LMT"
        assert sell.limit_price < 179.99


class TimeoutThenFillBroker(LocalPaperExecutor):
    def __init__(self, *args, always_timeout=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.always_timeout = always_timeout
        self.submit_calls = 0
        self.cancel_calls = 0

    def submit_order(self, request, **kwargs):
        self.submit_calls += 1
        if not self.always_timeout and self.submit_calls >= 2:
            return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="FILLED", filled_quantity=request.quantity, average_price=request.limit_price, message="Synthetic requote fill")
        return ExecutionReport(client_order_id=request.client_order_id, decision_id=request.decision_id, status="PENDING", remaining_quantity=request.quantity, message="Synthetic pending")

    def await_order(self, client_order_id, timeout_seconds=None):
        return ExecutionReport(client_order_id=client_order_id, decision_id="synthetic", status="TIMEOUT", message="Synthetic timeout")

    def cancel_order(self, client_order_id):
        self.cancel_calls += 1
        return ExecutionReport(client_order_id=client_order_id, decision_id="synthetic", status="CANCELLED", message="Synthetic cancel")


def test_entry_timeout_cancels_and_requotes_once_before_fill(tmp_path):
    with SQLiteStore(tmp_path / "requote-fill.sqlite3") as store:
        broker = TimeoutThenFillBroker(starting_cash=1000, store=store)
        runner = StrategyRunner(entry_runner_config(max_requotes=1), MockDataProvider(), broker, store, market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote())
        seed_pending_entry(runner, "requote-fill")
        result = runner.process_pending_entries()
        assert result[0]["status"] == "ENTRY_FILLED"
        assert broker.submit_calls == 2
        assert broker.cancel_calls == 1


def test_entry_stops_after_requote_limit_is_exhausted(tmp_path):
    with SQLiteStore(tmp_path / "requote-stop.sqlite3") as store:
        broker = TimeoutThenFillBroker(starting_cash=1000, store=store, always_timeout=True)
        runner = StrategyRunner(entry_runner_config(max_requotes=1), MockDataProvider(), broker, store, market_clock=OpenMarketClock(), quote_provider=FreshNVDAQuote())
        seed_pending_entry(runner, "requote-stop")
        result = runner.process_pending_entries()
        assert result[0]["status"] == "ENTRY_REVIEW_REQUIRED"
        assert broker.submit_calls == 2
        assert broker.cancel_calls == 2


def test_pending_entry_monitor_waits_120_seconds_without_quote_or_broker_call(tmp_path):
    now = datetime.now(timezone.utc)

    class StabilizingClock:
        def is_open(self, at=None): return True
        def session_open(self, at=None): return now - timedelta(seconds=119)

    class ForbiddenQuote:
        def get_quote(self, symbol):
            raise AssertionError("quote must not be requested during stabilization")

    with SQLiteStore(tmp_path / "stabilizing.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        cfg = entry_runner_config()
        cfg["execution"]["entry_delay_after_open_seconds"] = 120
        runner = StrategyRunner(cfg, MockDataProvider(), broker, store, market_clock=StabilizingClock(), quote_provider=ForbiddenQuote())
        seed_pending_entry(runner, "stabilizing")
        result = runner.process_pending_entries(now)
        assert result[0]["status"] == "PENDING_ENTRY"
        assert broker.connected is False


def test_expired_pending_entry_requires_fresh_research_before_quote_or_broker_call(tmp_path):
    now = datetime.now(timezone.utc)

    class ForbiddenQuote:
        def get_quote(self, symbol):
            raise AssertionError("expired decision must not request a quote")

    with SQLiteStore(tmp_path / "expired-entry.sqlite3") as store:
        broker = LocalPaperExecutor(starting_cash=1000, store=store)
        runner = StrategyRunner(
            entry_runner_config(), MockDataProvider(), broker, store,
            market_clock=OpenMarketClock(), quote_provider=ForbiddenQuote(),
        )
        intent = buy_intent(
            symbol="MPC",
            decision_id="expired-entry",
            timestamp=(now - timedelta(minutes=61)).isoformat(),
        )
        runner.entry_engine.capture(
            intent, run_id="expired-run", decision_reference_price=100,
            market_regime_at_decision="NEUTRAL",
        )

        result = runner.process_pending_entries(now)

        assert result[0]["status"] == "DECISION_EXPIRED"
        assert store.pending_entry("expired-entry").status == "ENTRY_CANCELLED"
        assert broker.connected is False


def test_scheduler_entry_monitor_ticks_every_30_seconds_only_in_open_session():
    calls = []

    class Clock:
        def is_open(self, at=None): return at.hour == 10
        def is_session_day(self, at=None): return True

    scheduler = PaperScheduler(SchedulerConfig(entry_check_interval_seconds=30, timezone="UTC"), entry_callback=lambda: calls.append("ENTRY"), market_clock=Clock())
    base = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)
    scheduler.run_once(base)
    scheduler.run_once(base + timedelta(seconds=10))
    scheduler.run_once(base + timedelta(seconds=30))
    scheduler.run_once(base.replace(hour=11))
    assert calls == ["ENTRY", "ENTRY"]
