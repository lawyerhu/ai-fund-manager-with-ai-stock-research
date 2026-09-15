import pytest

from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest


def test_local_paper_tracks_average_cost_and_realized_unrealized_pnl():
    executor = LocalPaperExecutor(starting_cash=1000.0)
    executor.connect()
    executor.submit_order(OrderRequest(decision_id="buy", symbol="NVDA", action="BUY", quantity=5, reference_price=100))

    marked = executor.reconcile({"NVDA": 120})
    assert marked.positions[0].average_cost == pytest.approx(100)
    assert marked.positions[0].unrealized_pnl == pytest.approx(100)
    assert marked.realized_pnl == pytest.approx(0)

    executor.submit_order(OrderRequest(decision_id="sell", symbol="NVDA", action="SELL", quantity=2, reference_price=120))
    final = executor.reconcile({"NVDA": 120})

    assert final.cash == pytest.approx(740)
    assert final.positions[0].quantity == pytest.approx(3)
    assert final.positions[0].realized_pnl == pytest.approx(40)
    assert final.positions[0].unrealized_pnl == pytest.approx(60)
    assert final.realized_pnl == pytest.approx(40)
