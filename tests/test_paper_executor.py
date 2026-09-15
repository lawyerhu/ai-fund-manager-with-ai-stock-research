import pytest

from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest


def test_partial_fill_duplicate_and_reconnect():
    executor = LocalPaperExecutor(starting_cash=1000)
    request = OrderRequest(decision_id="decision-1", symbol="NVDA", action="BUY", quantity=2.5, reference_price=100)
    assert executor.submit_order(request).status == "ERROR"
    executor.connect()
    partial = executor.submit_order(request, fill_ratio=0.4)
    assert partial.status == "PARTIALLY_FILLED"
    assert partial.filled_quantity == 1.0
    final = executor.fill_order(request.client_order_id)
    assert final.status == "FILLED" and final.filled_quantity == 2.5
    duplicate = executor.submit_order(OrderRequest(decision_id="decision-1", symbol="NVDA", action="BUY", quantity=2.5, reference_price=100))
    assert duplicate.status == "REJECTED"
    executor.reconnect()
    assert executor.connected


def test_fractional_and_minimum_notional_guards():
    executor = LocalPaperExecutor(fractional_shares=False, minimum_order_notional=100)
    executor.connect()
    fractional = executor.submit_order(OrderRequest(decision_id="d1", symbol="MSFT", action="BUY", quantity=1.5, reference_price=100))
    too_small = executor.submit_order(OrderRequest(decision_id="d2", symbol="MSFT", action="BUY", quantity=0.5, reference_price=100))
    assert fractional.status == "REJECTED"
    assert too_small.status == "REJECTED"


def test_rejected_client_order_is_idempotent_and_partial_fills_use_weighted_average():
    executor = LocalPaperExecutor(starting_cash=1000)
    executor.connect()

    rejected_request = OrderRequest(decision_id="insufficient", symbol="NVDA", action="BUY", quantity=20, reference_price=100)
    first_rejection = executor.submit_order(rejected_request)
    second_rejection = executor.submit_order(rejected_request)
    assert first_rejection.status == second_rejection.status == "REJECTED"
    assert second_rejection.message == first_rejection.message

    request = OrderRequest(decision_id="weighted", symbol="NVDA", action="BUY", quantity=8, reference_price=100)
    partial = executor.submit_order(request, fill_ratio=0.5, fill_price=100)
    final = executor.fill_order(request.client_order_id, fill_ratio=1.0, fill_price=120)
    assert partial.average_price == 100
    assert final.average_price == 110
    assert executor.cancel_order(request.client_order_id).status == "FILLED"


def test_local_paper_slippage_and_commission_reduce_net_equity():
    executor = LocalPaperExecutor(
        starting_cash=1000,
        commission_per_order=1.0,
        commission_per_share=0.01,
        minimum_commission=1.0,
        slippage_bps=10,
    )
    executor.connect()

    report = executor.submit_order(OrderRequest(decision_id="costs", symbol="NVDA", action="BUY", quantity=5, reference_price=100))
    snapshot = executor.reconcile({"NVDA": 100})

    assert report.average_price == pytest.approx(100.10)
    assert report.commission == pytest.approx(1.05)
    assert report.slippage == pytest.approx(0.50)
    assert report.trading_costs == pytest.approx(1.55)
    assert snapshot.equity == pytest.approx(998.45)
