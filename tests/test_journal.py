import pytest

from src.models import RiskDecision, TradeIntent
from src.storage import SQLiteStore


def test_decision_journal_records_realized_benchmark_comparison():
    intent = TradeIntent(
        action="BUY", symbol="NVDA", target_weight=0.5, confidence=0.8,
        holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01,
        thesis=["growth"], risk_factors=["volatility"], invalidation_conditions=["thesis breaks"],
        evidence_used=["universe"], decision_id="journal-decision",
    )
    risk = RiskDecision(approved=True, requested_weight=0.5, approved_weight=0.4, reason="volatility cap")

    with SQLiteStore(":memory:") as store:
        store.save_decision_journal(intent, risk, entry_price=100, filled_price=100, candidates=["NVDA", "META"])
        store.record_journal_observation("journal-decision", stock_price=110, spy_value=105, qqq_value=108)
        store.record_journal_observation("journal-decision", stock_price=120, spy_value=110, qqq_value=115)
        store.record_journal_observation("journal-decision", stock_price=130, spy_value=120, qqq_value=125)
        entry = store.journal_history()[0]

        assert entry["approved_weight"] == pytest.approx(0.4)
        assert entry["return_1d"] == pytest.approx(0.1)
        assert entry["spy_return_1d"] == pytest.approx(0.05)
        assert entry["alpha_vs_spy"] == pytest.approx(0.05)
        assert entry["return_5d"] is None
        assert "META" in entry["journal_json"]
