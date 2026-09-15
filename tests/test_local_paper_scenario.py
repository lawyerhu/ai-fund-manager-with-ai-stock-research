import json

import pytest

from src.data_provider import MockDataProvider
from src.execution.paper import LocalPaperExecutor
from src.models import OrderRequest, TradeIntent
from src.runner import StrategyRunner
from src.storage import SQLiteStore


def config(database):
    return {
        "portfolio": {"starting_equity": 1000.0, "database_path": str(database), "max_positions": 1},
        "agent": {"decision_horizon_days": 20},
        "execution": {"fractional_shares": True, "minimum_order_notional": 1.0},
        "risk": {
            "hard_drawdown_limit": 0.25, "target_annualized_vol": 0.25, "absolute_max_weight": 1.0,
            "min_confidence_to_open": 0.55, "max_data_age_minutes": 30, "max_gap_pct": 0.08,
            "max_annualized_volatility": 1.0, "min_avg_dollar_volume": 500000,
            "comparison_tolerance": 1e-9, "halt_recovery_drawdown": 0.20,
            "drawdown_tiers": [
                {"drawdown": 0.10, "max_weight": 0.70}, {"drawdown": 0.15, "max_weight": 0.50},
                {"drawdown": 0.20, "max_weight": 0.25}, {"drawdown": 0.25, "max_weight": 0.0},
            ],
            "event_risk": {"earnings_blackout_days": 3, "earnings_max_weight": 0.5, "macro_event_max_weight": 0.7},
        },
    }


def intent(action, symbol, weight, decision_id, current_symbol=None, new_symbol=None):
    return TradeIntent(
        action=action, symbol=symbol, target_weight=weight, confidence=0.8,
        holding_period_days=20, expected_alpha_vs_spy=0.02, expected_alpha_vs_qqq=0.01,
        thesis=["scenario"], risk_factors=["scenario"], invalidation_conditions=["scenario"],
        evidence_used=["scenario"], decision_id=decision_id,
        current_symbol=current_symbol, new_symbol=new_symbol,
    )


def test_complete_local_paper_scenario(tmp_path):
    database = tmp_path / "scenario.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    data.rows["META"].update({"price": 100.0, "ann_vol": 0.20})

    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        def verify(step, portfolio, risk=None):
            prices = {symbol: data.rows[symbol]["price"] for symbol in broker.position_symbols()}
            broker_state = broker.reconcile(prices)
            sqlite_state, sqlite_positions = store.latest_portfolio()
            dashboard_state = store.equity_history()[-1]
            position = broker_state.positions[0] if broker_state.positions else None
            expected_weight = position.market_value / broker_state.equity if position else 0.0

            assert portfolio.equity == pytest.approx(broker_state.equity)
            assert portfolio.cash == pytest.approx(broker_state.cash)
            assert portfolio.current_quantity == pytest.approx(position.quantity if position else 0.0)
            assert portfolio.current_weight == pytest.approx(expected_weight)
            assert sqlite_state == portfolio
            assert len(sqlite_positions) == len(broker_state.positions)
            assert dashboard_state["equity"] == pytest.approx(broker_state.equity)
            assert dashboard_state["cash"] == pytest.approx(broker_state.cash)
            assert dashboard_state["risk_state"] == portfolio.risk_state

            output = {
                "step": step, "equity": round(portfolio.equity, 2), "peak_equity": round(portfolio.peak_equity, 2),
                "cash": round(portfolio.cash, 2), "symbol": portfolio.current_symbol,
                "quantity": round(portfolio.current_quantity, 6), "weight": round(portfolio.current_weight, 6),
                "drawdown": round(portfolio.drawdown, 6), "risk_state": portfolio.risk_state,
                "open_orders": len(broker_state.open_orders),
                "risk_decision_state": risk.risk_state if risk else None,
                "portfolio_state": portfolio.risk_state,
                "sqlite_state": sqlite_state.risk_state,
            }
            print("LOCAL_PAPER_SCENARIO", json.dumps(output, sort_keys=True))

        cash = runner.reconcile_portfolio()
        verify("$1000 CASH", cash)

        bought = runner.run(intent=intent("BUY", "NVDA", 0.5, "scenario-buy-a"))["portfolio"]
        verify("BUY A", bought)

        held = runner.run(intent=intent("HOLD", "NVDA", 0.5, "scenario-hold-a"))["portfolio"]
        verify("HOLD A", held)

        reduced = runner.run(intent=intent("HOLD", "NVDA", 0.3, "scenario-reduce-a"))["portfolio"]
        verify("reduce A", reduced)

        switched = runner.run(intent=intent("SWITCH", "META", 0.4, "scenario-switch", current_symbol="NVDA", new_symbol="META"))["portfolio"]
        verify("SWITCH A -> B", switched)

        liquidated = runner.run(intent=intent("CASH", None, 0.0, "scenario-cash"))["portfolio"]
        verify("CASH", liquidated)

        bought_again = runner.run(intent=intent("BUY", "NVDA", 0.5, "scenario-buy-a-again"))["portfolio"]
        verify("BUY A again", bought_again)

        data.rows["NVDA"]["price"] = 60.0
        reduced_result = runner.run_daily_risk_check()
        verify("20% drawdown -> REDUCED", reduced_result["portfolio"], reduced_result["risk"])
        assert reduced_result["risk"].risk_state == "REDUCED"
        assert reduced_result["portfolio"].risk_state == "REDUCED"

        data.rows["NVDA"]["price"] = 45.0
        halted_result = runner.run_daily_risk_check()
        halted = halted_result["portfolio"]
        verify("25% drawdown -> RISK_HALTED", halted, halted_result["risk"])

        assert halted_result["state"] == "RISK_HALTED"
        assert halted.current_symbol is None
        assert halted.current_quantity == 0
        assert broker.reconcile({}).positions == []


def test_active_order_gate_resolution_scenario(tmp_path, capsys):
    database = tmp_path / "active-order-scenario.sqlite3"
    data = MockDataProvider()
    data.rows["NVDA"].update({"price": 100.0, "ann_vol": 0.20})
    with SQLiteStore(database) as store:
        broker = LocalPaperExecutor(starting_cash=1000.0, store=store)
        broker.connect()
        old = broker.submit_order(
            OrderRequest(decision_id="old", symbol="NVDA", action="BUY", quantity=5, reference_price=100),
            fill_ratio=0,
        )
        runner = StrategyRunner(config(database), data, executor=broker, store=store)

        blocked = runner.run(intent=intent("BUY", "NVDA", 0.5, "new-blocked"))
        broker.cancel_order(old.client_order_id)
        allowed = runner.run(intent=intent("BUY", "NVDA", 0.5, "new-allowed"))

        output = {
            "pending_old_order": old.status,
            "new_decision_blocked": str(blocked["state"]),
            "old_order_resolved": broker.order_status(old.client_order_id).status,
            "new_decision_allowed": str(allowed["state"]),
        }
        print("ACTIVE_ORDER_GATE_SCENARIO", json.dumps(output, sort_keys=True))
        assert output == {
            "pending_old_order": "PENDING",
            "new_decision_blocked": "ERROR",
            "old_order_resolved": "CANCELLED",
            "new_decision_allowed": "FILLED",
        }
        assert "ACTIVE_ORDER_GATE_SCENARIO" in capsys.readouterr().out
