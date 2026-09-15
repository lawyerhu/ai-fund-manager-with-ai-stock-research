from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import uuid4

from .analytics import compute_metrics
from .data_provider import DataProvider
from .entry_execution import EntryExecutionEngine
from .execution.base import Executor
from .execution.paper import LocalPaperExecutor
from .execution_quotes import ResearchSnapshotQuoteProvider
from .market_clock import TradingSessionProvider, USEquityMarketClock
from .market_data import MarketDataPolicy, MarketDataSnapshot
from .models import ExecutionQuote, ExecutionReport, ManagedPosition, OrderRequest, PortfolioState, PositionReview, RiskDecision, TradeIntent, make_client_order_id
from .operations import SwitchFrequencyGate, SwitchHysteresisPolicy
from .position_manager import PositionManager
from .risk_engine import RiskEngine
from .storage import SQLiteStore
from .trading_state import TradingState, TradingStateMachine


class BrokerReadinessError(RuntimeError):
    """A temporary broker connection/readiness failure before any order attempt."""


class StrategyRunner:
    """One decision cycle from research through risk-gated paper execution."""

    def __init__(self, cfg: dict[str, Any], data: DataProvider, executor: Executor | None = None, store: SQLiteStore | None = None, agent: Any | None = None, market_clock: TradingSessionProvider | None = None, quote_provider=None):
        self.cfg = cfg
        self.data = data
        self.executor = executor or LocalPaperExecutor(
            fractional_shares=bool(cfg.get("execution", {}).get("fractional_shares", True)),
            minimum_order_notional=float(cfg.get("execution", {}).get("minimum_order_notional", 1.0)),
            starting_cash=float(cfg.get("portfolio", {}).get("starting_equity", 1000.0)),
            commission_per_order=float(cfg.get("execution", {}).get("commission_per_order", 0.0)),
            commission_per_share=float(cfg.get("execution", {}).get("commission_per_share", 0.0)),
            minimum_commission=float(cfg.get("execution", {}).get("minimum_commission", 0.0)),
            slippage_bps=float(cfg.get("execution", {}).get("slippage_bps", 0.0)),
        )
        self.store = store or SQLiteStore(cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3"))
        self.agent = agent
        self.market_clock = market_clock or USEquityMarketClock()
        self.quote_provider = quote_provider or ResearchSnapshotQuoteProvider(data)
        self.entry_engine = EntryExecutionEngine(cfg, store=self.store, market_clock=self.market_clock)
        self.position_manager = PositionManager(cfg, self.store)
        self.broker_snapshot = None
        self._last_execution: ExecutionReport | None = None
        self._active_run_id: str | None = None
        self._active_decision_id: str | None = None

    def initial_portfolio(self) -> PortfolioState:
        return self.reconcile_portfolio()

    def probe_runtime_llm_exact_path(self) -> dict[str, Any]:
        probe = getattr(self.agent, "probe_runtime_llm_exact_path", None)
        if not callable(probe):
            return {"status": "FAIL", "ok": False, "error": "runtime LLM agent does not support exact-path probing"}
        return probe([{"symbol": "SPY"}])

    def _persisted_sol_resume_context(self, source_run_id: str) -> tuple[list[str], list[dict[str, Any]]]:
        events = list(reversed(self.store.runtime_events(limit=1000, run_id=source_run_id)))
        if not events:
            raise ValueError(f"Unknown Sol resume source run: {source_run_id}")
        event_types = {str(row.get("event_type") or "") for row in events}
        if "AI_RUN_FAILED" not in event_types or "SOL_RESEARCH_STARTED" not in event_types:
            raise ValueError("Sol resume source must be a failed run that reached Sol research")
        if {"SOL_DECISION_COMPLETED", "SOL_IC_DECISION_COMPLETED"} & event_types:
            raise ValueError("Sol resume source already has a final decision")
        sol_failure = False
        candidates: list[str] = []
        evidence_rows: list[dict[str, Any]] = []
        for row in events:
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            event_type = str(row.get("event_type") or "")
            if event_type == "LLM_ERROR" and str(metadata.get("stage") or "").startswith("SOL"):
                sol_failure = True
            if event_type == "LUNA_FINAL_SCREENING_COMPLETED" and metadata.get("final_screening_stage") in {"TIE_BREAK", "GLOBAL_TIE_BREAK"}:
                candidates = [str(symbol).upper() for symbol in metadata.get("candidate_symbols", []) if symbol]
            if event_type == "SOL_TOOL_RESULT" and metadata.get("status") in {"SUCCESS", "DATA_UNAVAILABLE"}:
                evidence_rows.append({
                    "symbol": row.get("symbol"),
                    "tool": metadata.get("tool"),
                    "arguments": {"symbol": row.get("symbol")} if row.get("symbol") else {},
                    "result": metadata.get("result"),
                })
        if not sol_failure:
            raise ValueError("Sol resume source has no recorded Sol failure")
        if not candidates:
            raise ValueError("Sol resume source has no completed Luna Final Candidates")
        return candidates, evidence_rows

    def reconcile_portfolio(self) -> PortfolioState:
        was_connected = bool(getattr(self.executor, "connected", False))
        if not was_connected:
            self.executor.connect()
            self.store.set_runtime("tws_status", "CONNECTED")
            self._emit_event("TWS_CONNECTED", "BROKER", "Broker connection established")
        symbols = self.executor.position_symbols()
        reconciliation_quotes = {symbol: self._reconciliation_quote(symbol) for symbol in symbols}
        prices = {symbol: quote.price for symbol, quote in reconciliation_quotes.items()}
        if reconciliation_quotes:
            quote = next(iter(reconciliation_quotes.values()))
            self.store.set_runtime("market_data_status", "MARKET_CLOSED" if quote.market_status == "CLOSED" else quote.data_type)
            self.store.set_runtime("market_data_source", quote.source)
            self.store.set_runtime("last_market_data_timestamp", quote.timestamp)
            self.store.set_runtime("execution_quote_ready", quote.market_status == "OPEN" and quote.data_type == "REALTIME")
        snapshot = self.executor.reconcile(prices)
        self.broker_snapshot = snapshot
        if len(snapshot.positions) > int(self.cfg.get("portfolio", {}).get("max_positions", 1)):
            snapshot = self._enforce_max_positions(snapshot, prices)
        self.broker_snapshot = snapshot

        persisted = self.store.latest_portfolio()
        previous_state = persisted[0] if persisted else None
        peak = max(snapshot.equity, previous_state.peak_equity if previous_state else snapshot.equity)
        history = self.store.equity_history()
        historical_max_drawdown = max(
            [
                previous_state.historical_max_drawdown if previous_state else 0.0,
                *(max(0.0, 1.0 - row["equity"] / row["peak_equity"]) for row in history if row["peak_equity"] > 0),
                max(0.0, 1.0 - snapshot.equity / peak) if peak > 0 else 0.0,
            ]
        )
        risk_state = self._risk_state_for_drawdown(
            max(0.0, 1.0 - snapshot.equity / peak) if peak > 0 else 0.0,
            previous_state.risk_state if previous_state else "NORMAL",
        )
        position = snapshot.positions[0] if snapshot.positions else None
        state = PortfolioState(
            equity=snapshot.equity,
            peak_equity=peak,
            cash=snapshot.cash,
            current_symbol=position.symbol if position else None,
            current_quantity=position.quantity if position else 0.0,
            current_weight=position.market_value / snapshot.equity if position and snapshot.equity > 0 else 0.0,
            invested_value=snapshot.invested_value,
            unrealized_pnl=snapshot.unrealized_pnl,
            realized_pnl=snapshot.realized_pnl,
            historical_max_drawdown=historical_max_drawdown,
            risk_state=risk_state,
            as_of=snapshot.as_of,
        )
        if previous_state and previous_state.risk_state != state.risk_state:
            self.store.save_risk_state_event(
                previous_state.risk_state,
                state.risk_state,
                f"Reconciled drawdown changed risk state to {state.risk_state}",
            )
        positions = [{**item.model_dump(mode="json"), "weight": item.market_value / snapshot.equity if snapshot.equity > 0 else 0.0} for item in snapshot.positions]
        self.store.save_portfolio(state, positions)
        if state.current_symbol:
            self._managed_position_for(state)
        for managed in self.store.active_managed_positions():
            if state.current_symbol is None or managed.symbol.upper() != state.current_symbol.upper():
                self.store.save_managed_position(managed.model_copy(update={
                    "monitoring_status": "CLOSED",
                    "updated_at": state.as_of,
                }))
        self.data.set_portfolio_context(state, positions)
        self.store.set_runtime("last_reconciliation", state.as_of)
        self.store.set_runtime("risk_state", state.risk_state)
        self._emit_event(
            "BROKER_RECONCILED",
            "BROKER",
            f"Reconciled {len(snapshot.positions)} position(s)",
            metadata={"equity": snapshot.equity, "cash": snapshot.cash, "position_count": len(snapshot.positions), "source": snapshot.source},
        )
        return state

    def run(self, intent: TradeIntent | None = None, use_llm: bool = False, resume_source_run_id: str | None = None) -> dict[str, Any]:
        self._active_run_id = str(uuid4())
        self._active_decision_id = None
        self.store.set_runtime("ai.current_run_id", self._active_run_id)
        self.store.set_runtime("ai_cycle", {"run_id": self._active_run_id, "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()})
        self._set_ai_progress("PREPARING", 0, status="RUNNING")
        self._emit_event(
            "AI_RUN_STARTED",
            "PIPELINE",
            "AI cycle started",
            metadata=self._pipeline_runtime_metadata(),
        )
        try:
            portfolio = self.initial_portfolio()
        except Exception as exc:
            self._emit_event("AI_RUN_FAILED", "PIPELINE", str(exc), metadata={"error_type": exc.__class__.__name__})
            raise
        machine = TradingStateMachine()
        self._last_execution = None
        research_snapshot: dict[str, Any] | None = None
        self._record_state(machine, TradingState.RESEARCHING, "start decision cycle")
        try:
            if intent is None:
                if use_llm:
                    if self.agent is None:
                        raise ValueError("LLM mode requires an LLMPortfolioManager")
                    if hasattr(self.agent, "set_event_sink"):
                        self.agent.set_event_sink(self._agent_event_sink)
                    sol_for_context = getattr(self.agent, "sol", None)
                    if sol_for_context is not None:
                        sol_for_context.sizing_context = (
                            self.position_manager.review_context(self._managed_position_for(portfolio))
                            if portfolio.current_symbol else {"configured_risk_budget": self.cfg.get("risk", {})}
                        )
                    if resume_source_run_id:
                        resume = getattr(self.agent, "resume_sol_decision", None)
                        if not callable(resume):
                            raise ValueError("LLM agent does not support persisted Sol resume")
                        candidates, evidence_rows = self._persisted_sol_resume_context(resume_source_run_id)
                        intent = resume(
                            portfolio,
                            candidates,
                            evidence_rows,
                            source_run_id=resume_source_run_id,
                            horizon_days=self.cfg.get("agent", {}).get("decision_horizon_days", 20),
                        )
                    else:
                        intent = self.agent.decide(portfolio, self.cfg.get("agent", {}).get("decision_horizon_days", 20))
                    sol = getattr(self.agent, "sol", None)
                    if portfolio.current_symbol and sol is not None and hasattr(sol, "review_position"):
                        metadata = getattr(self.agent, "last_pipeline_metadata", {}) or {}
                        candidates = list(metadata.get("candidate_symbols") or metadata.get("luna_candidates") or [])
                        position = self._managed_position_for(portfolio)
                        review = sol.review_position(
                            position,
                            portfolio,
                            candidate_symbols=[symbol for symbol in candidates if str(symbol).upper() != position.symbol.upper()],
                            review_type="WEEKLY",
                            event_context=self.position_manager.review_context(
                                position, {"event_type": "WEEKLY_DISCOVERY"}, metadata.get("sol_final_decision")
                            ),
                            replacement_threshold=self.position_manager.replacement_threshold,
                        )
                        outcome = self.position_manager.apply_review(position, PositionReview.model_validate(review))
                        self.position_manager.record_reduction_shadow(position, outcome.review, portfolio.equity)
                        intent = outcome.trade_intent or self._hold_intent_from_review(position, outcome.review)
                else:
                    intent = self.mock_intent()
            self._active_decision_id = intent.decision_id
            if intent.model_name == "unknown" and self.agent is not None:
                intent = intent.model_copy(update={"model_name": getattr(self.agent, "model", "unknown")})
            self._record_state(machine, TradingState.DECISION_READY, "validated TradeIntent")
            self.store.set_runtime("last_ai_decision", intent.timestamp)
            pipeline_metadata = getattr(self.agent, "last_pipeline_metadata", {}) if self.agent is not None else {}
            if not isinstance(pipeline_metadata, dict):
                pipeline_metadata = {}
            if use_llm and pipeline_metadata:
                model_signature = {
                    "gateway": pipeline_metadata.get("gateway", "unknown"),
                    "api_protocol": pipeline_metadata.get("api_protocol"),
                    "luna_model": pipeline_metadata.get("luna_model"),
                    "sol_model": pipeline_metadata.get("sol_model", intent.model_name),
                    "luna_reasoning_effort": pipeline_metadata.get("luna_reasoning_effort"),
                    "sol_reasoning_effort": pipeline_metadata.get("sol_reasoning_effort"),
                }
                previous_signature = self.store.get_runtime("llm.model_signature")
                if previous_signature and previous_signature != model_signature:
                    self.store.save_state(
                        "MODEL_CHANGE",
                        f"LLM model signature changed from {previous_signature} to {model_signature}",
                        intent.decision_id,
                    )
                self.store.set_runtime("llm.model_signature", model_signature)
                self.store.set_runtime("llm_gateway", model_signature["gateway"])
                self.store.set_runtime("llm_gateway_status", "ONLINE")
                self.store.set_runtime("llm_api_protocol", model_signature.get("api_protocol"))
                self.store.set_runtime("llm_protocol_fallback", pipeline_metadata.get("protocol_fallback", "NO"))
                self.store.set_runtime("llm_luna_model", model_signature["luna_model"])
                self.store.set_runtime("llm_sol_model", model_signature["sol_model"])
                self.store.set_runtime("llm_luna_status", "ONLINE")
                self.store.set_runtime("llm_sol_status", "ONLINE")
                self.store.set_runtime("llm_tool_calling", pipeline_metadata.get("tool_calling_status", "PASS"))
                self.store.set_runtime("llm_structured_output", pipeline_metadata.get("structured_output_status", "PASS"))
            self.store.save_decision(
                intent,
                getattr(self.agent, "prompt_version", "v1"),
                pipeline_metadata=pipeline_metadata,
                **getattr(self.agent, "last_usage", {}),
            )
            self.store.save_evidence(intent.decision_id, getattr(self.agent, "last_research_evidence", []))

            benchmark = self.data.benchmark_data(self.cfg.get("agent", {}).get("decision_horizon_days", 20))
            for symbol, values in benchmark.get("series", {}).items():
                for value in values:
                    self.store.save_benchmark(symbol, float(value))

            research_snapshot = self.data.stock_snapshot(intent.symbol) if intent.symbol else None
            snapshot = self._execution_snapshot(intent.symbol, research_snapshot) if intent.symbol else None
            events = self.data.upcoming_events(intent.symbol) if intent.symbol else None
            if events is not None:
                if events.get("macro_event_supported") is True:
                    self.store.set_runtime("macro_data_status", "AVAILABLE")
                else:
                    policy = self.cfg.get("risk", {}).get("event_risk", {}).get("macro_unknown_policy", "cap")
                    label = "UNKNOWN — POSITION CAPPED" if policy == "cap" else "UNKNOWN — NEW EXPOSURE BLOCKED" if policy == "reject_new" else "UNKNOWN — ALLOWED"
                    self.store.set_runtime("macro_data_status", label)
            enforce_hours = bool(self.cfg.get("execution", {}).get("enforce_market_hours", False))
            market_open = self.market_clock.is_open() if enforce_hours else True
            if (
                bool(self.cfg.get("execution", {}).get("entry_engine_enabled", False))
                and intent.action == "BUY"
                and (portfolio.current_symbol is None or portfolio.current_symbol == intent.symbol)
            ):
                reference_price = float((snapshot or {}).get("price") or (research_snapshot or {}).get("price", 0))
                if reference_price <= 0:
                    raise RuntimeError("Cannot create pending entry without a decision reference price")
                pending = self.entry_engine.capture(
                    intent,
                    run_id=self._active_run_id,
                    decision_reference_price=reference_price,
                    market_regime_at_decision=str((research_snapshot or {}).get("market_regime", "UNKNOWN")),
                )
                self._emit_event(
                    "ENTRY_PENDING",
                    "ENTRY_EXECUTION",
                    f"{intent.symbol} BUY routed through durable entry execution",
                    decision_id=intent.decision_id,
                    symbol=intent.symbol,
                    metadata=pending.model_dump(mode="json"),
                )
                if market_open:
                    entry_result = next(
                        (item for item in self.process_pending_entries() if item.get("decision_id") == intent.decision_id),
                        {"decision_id": intent.decision_id, "status": "ENTRY_FAILED", "reason": "Entry monitor returned no result"},
                    )
                    risk = RiskDecision.model_validate(entry_result["risk"]) if entry_result.get("risk") else None
                    report = ExecutionReport.model_validate(entry_result["report"]) if entry_result.get("report") else None
                    if risk is not None and risk.approved:
                        self._record_state(machine, TradingState.RISK_APPROVED, risk.reason, intent.decision_id)
                    if report is not None:
                        self._record_state(machine, TradingState.ORDER_PENDING, f"execute BUY {intent.symbol}", intent.decision_id)
                    status = str(entry_result.get("status"))
                    if status == "ENTRY_FILLED":
                        self._record_state(machine, TradingState.FILLED, report.message if report else "Entry filled", intent.decision_id)
                    elif status == "ENTRY_PARTIALLY_FILLED":
                        self._record_state(machine, TradingState.PARTIALLY_FILLED, report.message if report else "Entry partially filled", intent.decision_id)
                    elif status in {"PENDING_ENTRY", "ENTRY_REVIEW_REQUIRED", "ENTRY_VALID", "ENTRY_SUBMITTED"}:
                        self._record_state(machine, TradingState.PENDING_ENTRY, entry_result.get("reason", status), intent.decision_id)
                    else:
                        self._record_state(machine, TradingState.ERROR, entry_result.get("reason", report.message if report else status), intent.decision_id)
                    portfolio = self.reconcile_portfolio()
                    self._last_execution = report
                    self._save_journal(intent, risk, snapshot)
                    return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio, "pending_entry": self.store.pending_entry(intent.decision_id)})
                self._record_state(machine, TradingState.PENDING_ENTRY, "BUY retained until regular trading hours", intent.decision_id)
                portfolio = self.reconcile_portfolio()
                self._save_journal(intent, None, snapshot)
                return self._finish_run({"intent": intent, "risk": None, "state": machine.state, "portfolio": portfolio, "pending_entry": pending})
            self._emit_event(
                "RISK_EVALUATION_STARTED",
                "RISK_ENGINE",
                f"Evaluating {intent.action} {intent.symbol or 'CASH'}",
                decision_id=intent.decision_id,
                symbol=intent.symbol,
                metadata={"requested_weight": intent.target_weight, "market_open": market_open, "quote": snapshot or {}},
            )
            risk = RiskEngine(self.cfg["risk"]).evaluate(intent, portfolio, (snapshot or {}).get("ann_vol"), (snapshot or {}).get("quote_as_of", (snapshot or {}).get("as_of")), stock_snapshot=snapshot, events=events, market_open=market_open)
            market_data_values = self.store.get_runtime("current_market_data") if intent.symbol else None
            market_data = MarketDataSnapshot.model_validate(market_data_values) if market_data_values else None
            strict_execution_data = not self._observe_mode() and market_data is not None and market_data.source != "RESEARCH_SIMULATION"
            if strict_execution_data and intent.action in {"BUY", "SWITCH"} and not MarketDataPolicy.allows(
                market_data,
                purpose="EXECUTION",
                execution_mode=getattr(self.executor, "execution_mode", "PAPER"),
                max_age_seconds=float(self.cfg.get("execution", {}).get("max_quote_age_seconds", 30)),
            ):
                stale = market_data.current_age_seconds() > float(self.cfg.get("execution", {}).get("max_quote_age_seconds", 30))
                reason = (
                    f"Execution market data is stale: {market_data.market_data_type}, age={market_data.current_age_seconds():.1f}s"
                    if stale else f"Execution market data is not usable: {market_data.market_data_type}"
                )
                risk = risk.model_copy(update={"approved": False, "approved_weight": 0.0, "reason": reason, "limit_reasons": [*risk.limit_reasons, reason]})
                self._emit_event("PRICE_DATA_STALE" if stale else "ORDER_BLOCKED_STALE_QUOTE", "MARKET_DATA", reason, decision_id=intent.decision_id, symbol=intent.symbol, metadata=market_data.model_dump(mode="json"))
            switch_gate = SwitchFrequencyGate(self.store, self.cfg.get("strategy", {}))
            if intent.action == "SWITCH" and risk.approved and not switch_gate.allow("SWITCH", category="STRATEGIC_REBALANCE"):
                risk = risk.model_copy(update={
                    "approved": False,
                    "approved_weight": 0.0,
                    "reason": "Strategic SWITCH blocked by weekly frequency limit",
                    "limit_reasons": [*risk.limit_reasons, "weekly strategic switch limit"],
                })
            hysteresis_cfg = self.cfg.get("strategy", {}).get("switch_hysteresis", {})
            if intent.action == "SWITCH" and intent.model_name != "sol-position-review" and risk.approved and not SwitchHysteresisPolicy(hysteresis_cfg).allows(intent, self._current_position_intent(portfolio.current_symbol)):
                risk = risk.model_copy(update={
                    "approved": False,
                    "approved_weight": 0.0,
                    "reason": "Strategic SWITCH blocked by hysteresis; advantage does not justify turnover",
                    "limit_reasons": [*risk.limit_reasons, "switch hysteresis"],
                })
            if "stale" in risk.reason.lower():
                self.store.set_runtime("market_data_status", "STALE")
                self._emit_event("PRICE_DATA_STALE", "MARKET_DATA", risk.reason, decision_id=intent.decision_id, symbol=intent.symbol)
            elif "price" in risk.reason.lower() and not risk.approved:
                self.store.set_runtime("market_data_status", "ERROR")
            self.store.save_risk(intent.decision_id, risk)
            self._emit_event(
                "RISK_EVALUATED",
                "RISK_ENGINE",
                risk.reason,
                decision_id=intent.decision_id,
                symbol=intent.symbol,
                metadata=risk.model_dump(mode="json"),
            )
            active_orders = [
                order for order in self.broker_snapshot.open_orders
                if order.get("client_order_id") and order.get("decision_id") != intent.decision_id
            ]
            if active_orders and intent.action != "CASH" and risk.risk_state != "RISK_HALTED":
                order_ids = ", ".join(order["client_order_id"] for order in active_orders)
                reason = f"Unresolved broker order blocks new execution: {order_ids}"
                self._record_state(machine, TradingState.ERROR, reason, intent.decision_id)
                self._emit_event("EXECUTION_BLOCKED", "EXECUTION", reason, decision_id=intent.decision_id, symbol=intent.symbol, metadata={"reason": reason})
                portfolio = self._synchronize_risk_state(portfolio, risk.risk_state)
                self._save_journal(intent, risk, snapshot)
                return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio})
            if risk.risk_state == "RISK_HALTED":
                reports, portfolio = self._liquidate_all(intent.decision_id, machine, "risk-halt")
                failed = next((item for item in reports if not self._execution_succeeded(item)), None)
                if failed or (not self._observe_mode() and self.broker_snapshot.positions):
                    self._record_execution_state(machine, failed, intent.decision_id)
                else:
                    portfolio = portfolio.model_copy(update={"risk_state": "RISK_HALTED"})
                    positions = [{**item.model_dump(mode="json"), "weight": item.market_value / portfolio.equity if portfolio.equity else 0.0} for item in self.broker_snapshot.positions]
                    self.store.save_portfolio(portfolio, positions)
                    if self._observe_mode() and not reports:
                        self._emit_event("WOULD_CASH", "EXECUTION", "WOULD_CASH: no broker order was sent", decision_id=intent.decision_id, metadata={"order_not_sent": True, "place_order_calls": 0, "cancel_order_calls": 0})
                    self._record_state(machine, TradingState.RISK_HALTED, "WOULD_CASH: risk halt enforced in OBSERVE" if self._observe_mode() else risk.reason, intent.decision_id)
                    self._emit_event("RISK_HALTED", "RISK_ENGINE", risk.reason, decision_id=intent.decision_id, metadata={"risk_state": risk.risk_state})
            elif not risk.approved:
                target = TradingState.WAITING_FOR_MARKET if "market is not open" in risk.reason.lower() else TradingState.REJECTED
                self._record_state(machine, target, risk.reason, intent.decision_id)
                self._emit_event("EXECUTION_BLOCKED", "EXECUTION", risk.reason, decision_id=intent.decision_id, symbol=intent.symbol, metadata={"status": str(target), "reason": risk.reason})
            elif intent.action == "CASH" or risk.approved_weight <= 0:
                self._record_state(machine, TradingState.RISK_APPROVED, "cash allocation approved", intent.decision_id)
                reports, portfolio = self._liquidate_all(intent.decision_id, machine, "cash")
                if self._observe_mode() and not reports:
                    self._emit_event("WOULD_CASH", "EXECUTION", "WOULD_CASH: no broker order was sent", decision_id=intent.decision_id, metadata={"order_not_sent": True, "place_order_calls": 0, "cancel_order_calls": 0})
                if all(self._execution_succeeded(report) for report in reports) and (self._observe_mode() or not self.broker_snapshot.positions):
                    self._record_state(machine, TradingState.OBSERVED if self._observe_mode() else TradingState.FILLED, "WOULD_CASH: broker liquidation was not sent" if self._observe_mode() else "cash target confirmed", intent.decision_id)
                else:
                    report = next((item for item in reports if not self._execution_succeeded(item)), None)
                    self._record_execution_state(machine, report, intent.decision_id)
            else:
                self._record_state(machine, TradingState.RISK_APPROVED, risk.reason, intent.decision_id)
                if portfolio.current_symbol and portfolio.current_symbol != intent.symbol:
                    if intent.action != "SWITCH":
                        raise RuntimeError("A different stock is already held; use SWITCH to preserve max_positions=1")
                    sell_reports, portfolio = self._liquidate_all(intent.decision_id, machine, "switch-sell")
                    failed_sell = next((item for item in sell_reports if not self._execution_succeeded(item)), None)
                    if failed_sell or (not self._observe_mode() and self.broker_snapshot.positions):
                        self._record_execution_state(machine, failed_sell, intent.decision_id)
                        portfolio = self._synchronize_risk_state(portfolio, risk.risk_state)
                        self._save_journal(intent, risk, snapshot)
                        return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio})
                    if self._observe_mode():
                        self._record_state(machine, TradingState.OBSERVED, "WOULD_SWITCH: sell leg was not sent", intent.decision_id)
                        self._record_state(machine, TradingState.RISK_APPROVED, "WOULD_SWITCH: buy leg remains hypothetical", intent.decision_id)
                    else:
                        self._record_state(machine, TradingState.FILLED, "Switch sell leg confirmed flat", intent.decision_id)
                        self._record_state(machine, TradingState.RISK_APPROVED, "Switch buy leg approved", intent.decision_id)
                    snapshot = self._execution_snapshot(intent.symbol, research_snapshot)
                elif intent.action == "SWITCH":
                    raise RuntimeError("SWITCH requires a different existing position")

                price = float((snapshot or {}).get("price", 0))
                target_value = portfolio.equity * risk.approved_weight
                current_value = portfolio.current_quantity * price if portfolio.current_symbol == intent.symbol else 0.0
                delta_value = target_value - current_value
                action = "BUY" if delta_value > 0 else "SELL"
                if action == "BUY":
                    cash_safety_factor = float(self.cfg.get("execution", {}).get("buy_cash_safety_factor", 0.98))
                    if not 0 < cash_safety_factor <= 1:
                        raise ValueError("execution.buy_cash_safety_factor must be in (0, 1]")
                    delta_value = min(delta_value, max(0.0, portfolio.cash * cash_safety_factor))
                execution_cfg = self.cfg.get("execution", {})
                minimum_notional = float(execution_cfg.get("minimum_order_notional", 1.0))
                safety_reduction = (
                    delta_value < 0
                    and (
                        risk.risk_state == "RISK_HALTED"
                        or (risk.drawdown_weight_limit is not None and risk.drawdown_weight_limit < 1.0)
                        or (risk.event_weight_limit is not None and risk.event_weight_limit < 1.0)
                    )
                )
                rebalance_minimum = float(execution_cfg.get("minimum_rebalance_notional", minimum_notional))
                effective_minimum = (
                    max(minimum_notional, rebalance_minimum)
                    if intent.action == "HOLD" and not safety_reduction
                    else minimum_notional
                )
                if abs(delta_value) < effective_minimum:
                    if effective_minimum > minimum_notional:
                        hold_message = (
                            f"Rebalance delta ${abs(delta_value):,.2f} is below the "
                            f"${effective_minimum:,.2f} minimum; position unchanged"
                        )
                    else:
                        hold_message = "Position already matches approved target"
                    self._record_state(machine, TradingState.OBSERVED if self._observe_mode() else TradingState.FILLED, f"WOULD_HOLD: {hold_message}" if self._observe_mode() else hold_message, intent.decision_id)
                    portfolio = self.reconcile_portfolio()
                    portfolio = self._synchronize_risk_state(portfolio, risk.risk_state)
                    self._save_journal(intent, risk, snapshot)
                    return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio})
                quantity = abs(delta_value) / price
                if not bool(self.cfg.get("execution", {}).get("fractional_shares", True)):
                    quantity = float(int(quantity))
                    if quantity <= 0 or quantity * price < minimum_notional:
                        self._record_state(machine, TradingState.OBSERVED if self._observe_mode() else TradingState.FILLED, "WOULD_HOLD: no broker order was sent" if self._observe_mode() else "Approved target cannot fund one whole share", intent.decision_id)
                        portfolio = self.reconcile_portfolio()
                        portfolio = self._synchronize_risk_state(portfolio, risk.risk_state)
                        self._save_journal(intent, risk, snapshot)
                        return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio})
                order = self._build_executable_order(intent.decision_id, intent.symbol, action, quantity, "target", reference_price=price)
                report = self._execute_order(order, machine, hypothetical_label=intent.action)
                if report and report.status == "FILLED":
                    self._record_state(machine, TradingState.FILLED, report.message, intent.decision_id)
                elif report and report.status == "PARTIALLY_FILLED":
                    self._record_state(machine, TradingState.PARTIALLY_FILLED, report.message, intent.decision_id)
                elif report and report.status == "REJECTED":
                    self._record_state(machine, TradingState.REJECTED, report.message, intent.decision_id)
                elif report and report.status == "SIMULATED":
                    self._record_state(machine, TradingState.OBSERVED, report.message, intent.decision_id)
                elif report and report.status in {"PENDING", "CANCELLED", "TIMEOUT"}:
                    self._record_execution_state(machine, report, intent.decision_id)
                else:
                    self._record_state(machine, TradingState.ERROR, report.message if report else "No broker report", intent.decision_id)

            portfolio = self.reconcile_portfolio()
            portfolio = self._synchronize_risk_state(portfolio, risk.risk_state)
            if portfolio.current_symbol and machine.state in {TradingState.FILLED, TradingState.PARTIALLY_FILLED}:
                self._managed_position_for(portfolio)
            self._save_journal(intent, risk, snapshot)
            if intent.action == "SWITCH" and risk.approved and machine.state in {TradingState.FILLED, TradingState.OBSERVED}:
                switch_gate.record("SWITCH", category="STRATEGIC_REBALANCE", decision_id=intent.decision_id)
            return self._finish_run({"intent": intent, "risk": risk, "state": machine.state, "portfolio": portfolio})
        except Exception as exc:
            if isinstance(exc, BrokerReadinessError) and intent is not None and machine.state == TradingState.DECISION_READY:
                reason = str(exc)
                self._record_state(machine, TradingState.WAITING_FOR_BROKER, reason, intent.decision_id)
                self._emit_event(
                    "EXECUTION_WAITING_FOR_BROKER",
                    "EXECUTION",
                    f"Execution deferred until broker recovery: {reason}",
                    decision_id=intent.decision_id,
                    symbol=intent.symbol,
                    metadata={"reason": reason, "resume_required": True},
                )
                self._save_journal(intent, None, research_snapshot)
                return self._finish_run({
                    "intent": intent,
                    "risk": None,
                    "state": machine.state,
                    "portfolio": portfolio,
                    "broker_waiting_reason": reason,
                })
            self.store.save_error("runner", str(exc), {"intent": intent.model_dump(mode="json") if intent else None})
            if machine.state not in {TradingState.ERROR, TradingState.RISK_HALTED}:
                self._record_state(machine, TradingState.ERROR, str(exc), intent.decision_id if intent else None)
            if use_llm:
                self._emit_event("LLM_ERROR", "LLM", str(exc), decision_id=getattr(intent, "decision_id", None), metadata={"error_type": exc.__class__.__name__})
            self._emit_event("AI_RUN_FAILED", "PIPELINE", str(exc), decision_id=getattr(intent, "decision_id", None), metadata={"error_type": exc.__class__.__name__})
            raise

    def mock_intent(self) -> TradeIntent:
        best = max(self.data.universe_snapshot(), key=lambda row: row["ret_20d"])
        return TradeIntent(action="BUY", symbol=best["symbol"], target_weight=1.0, confidence=0.70, holding_period_days=self.cfg.get("agent", {}).get("decision_horizon_days", 20), expected_alpha_vs_spy=0.03, expected_alpha_vs_qqq=0.02, thesis=["Offline pipeline test only"], risk_factors=["Mock data is not investable evidence"], invalidation_conditions=["Any real-data deployment invalidates this mock decision"], evidence_used=["mock_universe_snapshot"], model_name="mock")

    def run_daily_risk_check(self, record_performance: bool = True) -> dict[str, Any]:
        portfolio = self.reconcile_portfolio()
        if not portfolio.current_symbol:
            result = {"state": TradingState.RISK_HALTED if portfolio.risk_state == "RISK_HALTED" else TradingState.IDLE, "portfolio": portfolio, "risk": None, "intent": None}
            if record_performance:
                self._record_daily_performance(portfolio)
            return result
        self.record_journal_observations()
        daily_intent = TradeIntent(
            action="HOLD",
            symbol=portfolio.current_symbol,
            target_weight=portfolio.current_weight,
            confidence=1.0,
            holding_period_days=1,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=["Deterministic daily risk check"],
            risk_factors=["Account drawdown and current market risk"],
            invalidation_conditions=["Risk limits require reduction or liquidation"],
            evidence_used=["reconciled_broker_state", "latest_market_snapshot"],
            model_name="daily-risk-engine",
        )
        result = self.run(intent=daily_intent)
        if record_performance:
            self._record_daily_performance(result["portfolio"])
        return result

    def process_pending_entries(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Advance durable pending entries through fresh quote, risk, and execution gates."""
        if not bool(self.cfg.get("execution", {}).get("entry_engine_enabled", False)):
            return []
        results: list[dict[str, Any]] = []
        for entry in self.store.pending_entries(("PENDING_ENTRY", "ENTRY_REVIEW_REQUIRED")):
            if not self.market_clock.is_open(now):
                results.append({"decision_id": entry.decision_id, "status": entry.status, "reason": "Market is closed"})
                continue
            try:
                now_value = now or datetime.now(timezone.utc)
                session_open = getattr(self.market_clock, "session_open", None)
                opened_at = session_open(now_value) if callable(session_open) else None
                delay = float(self.cfg.get("execution", {}).get("entry_delay_after_open_seconds", 120))
                if opened_at is not None and (now_value - opened_at).total_seconds() < delay:
                    results.append({"decision_id": entry.decision_id, "status": entry.status, "reason": "Waiting for post-open stabilization"})
                    continue
                decision_limit = float(self.cfg.get("risk", {}).get("max_decision_age_minutes_for_execution", 60))
                decision_time = datetime.fromisoformat(entry.decision_time.replace("Z", "+00:00"))
                if decision_time.tzinfo is None:
                    decision_time = decision_time.replace(tzinfo=timezone.utc)
                decision_age_minutes = (now_value - decision_time.astimezone(timezone.utc)).total_seconds() / 60
                if decision_age_minutes < 0 or decision_age_minutes > decision_limit:
                    reason = "Pending entry decision expired; fresh AI research is required"
                    expired = entry.model_copy(update={"status": "ENTRY_CANCELLED", "message": reason})
                    self.store.update_pending_entry(expired)
                    self._emit_event(
                        "ENTRY_DECISION_EXPIRED",
                        "ENTRY_EXECUTION",
                        reason,
                        decision_id=entry.decision_id,
                        symbol=entry.symbol,
                        metadata={"decision_time": entry.decision_time, "max_age_minutes": decision_limit},
                    )
                    results.append({"decision_id": entry.decision_id, "status": "DECISION_EXPIRED", "reason": reason})
                    continue
                portfolio = self.reconcile_portfolio()
                quote = self._execution_quote(entry.symbol)
                new_information = self._entry_information_since_decision(entry)
                revalidation = self.entry_engine.revalidate(entry, quote, now=now_value, opened_at=opened_at, material_events=new_information["material_events"])
                if revalidation.entry.status == "ENTRY_REVIEW_REQUIRED":
                    sol = getattr(self.agent, "sol", None)
                    quick_review = getattr(sol, "quick_entry_review", None)
                    if not callable(quick_review):
                        results.append({"decision_id": entry.decision_id, "status": revalidation.entry.status, "reason": "Sol quick entry review is unavailable"})
                        continue
                    review = quick_review({
                        "symbol": entry.symbol,
                        "original_thesis": entry.thesis,
                        "decision_reference_price": entry.decision_reference_price,
                        "decision_time": entry.decision_time,
                        "current_quote": quote.model_dump(mode="json"),
                        "price_gap_pct": revalidation.checks.get("price_gap_pct"),
                        **new_information,
                    })
                    reviewed = self.entry_engine.apply_quick_review(revalidation.entry, review)
                    if reviewed.status != "ENTRY_VALID":
                        results.append({"decision_id": entry.decision_id, "status": reviewed.status, "reason": reviewed.message})
                        continue
                    revalidation = revalidation.__class__(reviewed, revalidation.checks, reviewed.message)
                if revalidation.entry.status != "ENTRY_VALID":
                    results.append({"decision_id": entry.decision_id, "status": revalidation.entry.status, "reason": revalidation.reason})
                    continue
                intent = TradeIntent(
                    action="BUY", symbol=entry.symbol, target_weight=entry.requested_weight,
                    confidence=entry.sol_confidence, holding_period_days=entry.thesis_horizon_days,
                    expected_alpha_vs_spy=entry.expected_alpha_vs_spy, expected_alpha_vs_qqq=entry.expected_alpha_vs_qqq,
                    thesis=entry.thesis, risk_factors=entry.risks or ["Pending-entry review"],
                    invalidation_conditions=["Entry revalidation invalidates thesis"], evidence_used=["pending_entry_decision"],
                    timestamp=datetime.now(timezone.utc).isoformat(), model_name="gpt-5.6-sol-entry-revalidation", decision_id=entry.decision_id,
                )
                snapshot = self._execution_snapshot(entry.symbol, self.data.stock_snapshot(entry.symbol))
                risk = RiskEngine(self.cfg.get("risk", {})).evaluate(intent, portfolio, snapshot.get("ann_vol"), snapshot.get("quote_as_of"), stock_snapshot=snapshot, events=self.data.upcoming_events(entry.symbol), market_open=True)
                self.store.save_risk(entry.decision_id, risk)
                self._emit_event("RISK_EVALUATED", "RISK_ENGINE", risk.reason, decision_id=entry.decision_id, symbol=entry.symbol, metadata=risk.model_dump(mode="json"))
                if not risk.approved or not self.broker_snapshot:
                    updated = entry.model_copy(update={"status": "ENTRY_FAILED", "message": risk.reason})
                    self.store.update_pending_entry(updated)
                    results.append({"decision_id": entry.decision_id, "status": updated.status, "reason": risk.reason})
                    continue
                max_requotes = int(self.cfg.get("execution", {}).get("max_entry_requotes", 2))
                tick_size = 0.01
                resolver = getattr(self.executor, "resolve_instrument", None)
                if callable(resolver):
                    tick_size = float(getattr(resolver(entry.symbol), "min_tick", 0.01) or 0.01)
                report = None
                order = None
                for requote_index in range(max_requotes + 1):
                    if requote_index:
                        quote = self._execution_quote(entry.symbol)
                        refreshed = self.entry_engine.revalidate(revalidation.entry, quote, now=now)
                        if refreshed.entry.status != "ENTRY_VALID":
                            report = ExecutionReport(client_order_id=make_client_order_id(entry.decision_id, f"entry-{requote_index}", entry.symbol, "BUY"), decision_id=entry.decision_id, status="ERROR", message=refreshed.reason)
                            break
                    order = self.entry_engine.build_order(
                        revalidation.entry, quote, self.broker_snapshot, risk,
                        broker_connected=bool(getattr(self.executor, "connected", False)),
                        broker_reconciled=bool(getattr(self.executor, "broker_state_known", False)),
                        execution_mode=str(getattr(self.executor, "execution_mode", "PAPER")),
                        fractional_shares=bool(self.cfg.get("execution", {}).get("fractional_shares", True)),
                        tick_size=tick_size,
                        requote_index=requote_index,
                    )
                    self.store.save_order(order)
                    if self._observe_mode() and hasattr(self.executor, "simulate_order"):
                        report = self.executor.simulate_order(order, label="BUY")
                    else:
                        report = self._pre_submit_position_check(order)
                        if report is None:
                            report = self.executor.submit_order(order)
                        else:
                            self.store.save_order(order, status=report.status)
                        if report.status == "PENDING":
                            self.entry_engine.apply_execution(revalidation.entry.model_copy(update={"limit_price": order.limit_price, "approved_weight": risk.approved_weight}), report, quote_mid=quote.mid)
                            report = self.executor.await_order(order.client_order_id, self.cfg.get("execution", {}).get("entry_order_timeout_seconds", 45)) or report
                        if report.status == "TIMEOUT":
                            cancelled = self.executor.cancel_order(order.client_order_id)
                            self.store.save_execution(cancelled)
                            if report.filled_quantity > 0 or cancelled.filled_quantity > 0:
                                report = cancelled
                                break
                            if requote_index < max_requotes:
                                continue
                            report = cancelled
                    break
                if report is None or order is None:
                    raise RuntimeError("Entry execution did not produce an order report")
                self.store.save_execution(report)
                updated = self.entry_engine.apply_execution(revalidation.entry.model_copy(update={"limit_price": order.limit_price, "approved_weight": risk.approved_weight}), report, quote_mid=quote.mid)
                if updated.filled_quantity > 0:
                    filled_portfolio = self.reconcile_portfolio()
                    if filled_portfolio.current_symbol:
                        self._managed_position_for(filled_portfolio)
                results.append({"decision_id": entry.decision_id, "status": updated.status, "report": report.model_dump(mode="json"), "risk": risk.model_dump(mode="json")})
            except Exception as exc:
                failed = entry.model_copy(update={"status": "ENTRY_FAILED", "message": str(exc)})
                if self.store.pending_entry(entry.decision_id):
                    self.store.update_pending_entry(failed)
                self.store.save_error("entry-execution", str(exc), {"decision_id": entry.decision_id, "symbol": entry.symbol}, decision_id=entry.decision_id, component="entry_execution")
                self._emit_event("ENTRY_FAILED", "ENTRY_EXECUTION", str(exc), decision_id=entry.decision_id, symbol=entry.symbol, metadata={"error_type": exc.__class__.__name__})
                results.append({"decision_id": entry.decision_id, "status": "ENTRY_FAILED", "reason": str(exc)})
        return results

    def _entry_information_since_decision(self, entry) -> dict[str, Any]:
        """Collect only provider data available after the original decision timestamp."""
        evidence = {
            "overnight_news": self.data.news(entry.symbol, 10),
            "sec_updates": self.data.sec_filings(entry.symbol, 10),
            "analyst_changes": self.data.analyst_revisions(entry.symbol),
            "earnings_or_guidance": self.data.earnings(entry.symbol),
            "market_regime": self.data.market_regime(),
        }
        material = {
            "major_news": bool(evidence["overnight_news"].get("major_change", False)),
            "sec_filing": bool(evidence["sec_updates"].get("material_change", False)),
            "analyst_revision": bool(evidence["analyst_changes"].get("material_change", False)),
            "guidance_change": bool(evidence["earnings_or_guidance"].get("guidance_change", False)),
            "market_regime_change": bool(evidence["market_regime"].get("material_change", False)),
            "sector_regime_change": False,
        }
        return {**evidence, "material_events": material}

    def run_position_review(self, event_context: dict[str, Any] | None = None) -> dict[str, Any]:
        portfolio = self.reconcile_portfolio()
        if not portfolio.current_symbol:
            return {"review_action": "HOLD", "reason": "No current position", "symbol": None, "luna_calls": 0, "sol_tool_calls": 0}
        position = self._managed_position_for(portfolio)
        sol = getattr(self.agent, "sol", None)
        if sol is None or not (hasattr(sol, "review_position") or hasattr(sol, "decide")):
            raise RuntimeError("Position review requires the configured Sol agent")
        if hasattr(self.agent, "set_event_sink"):
            self.agent.set_event_sink(self._agent_event_sink)
        review_type = "EMERGENCY" if event_context else self.position_manager.review_due(position) or "DAILY"
        if hasattr(sol, "review_position"):
            review = sol.review_position(
                position,
                portfolio,
                candidate_symbols=list((event_context or {}).get("candidate_symbols", [])),
                review_type=review_type,
                event_context=self.position_manager.review_context(position, event_context),
                replacement_threshold=self.position_manager.replacement_threshold,
            )
            review = PositionReview.model_validate(review)
        else:
            intent = sol.decide(portfolio, [portfolio.current_symbol], horizon_days=1)
            review = self._legacy_position_review(position, portfolio, intent, review_type, event_context)
        outcome = self.position_manager.apply_review(position, review)
        self.position_manager.record_reduction_shadow(position, outcome.review, portfolio.equity)
        execution_result = self.run(intent=outcome.trade_intent) if outcome.trade_intent is not None else None
        return {
            "review_action": outcome.review.action,
            "thesis_status": outcome.review.thesis_status,
            "replacement_gap": outcome.review.replacement_gap,
            "reason": "; ".join(outcome.review.reason),
            "symbol": portfolio.current_symbol,
            "event_context": event_context,
            "luna_calls": 0,
            "sol_tool_calls": len(getattr(sol, "last_tool_calls", [])),
            "review": outcome.review.model_dump(mode="json"),
            "decision": outcome.trade_intent.model_dump(mode="json") if outcome.trade_intent else None,
            "execution": execution_result,
        }

    def run_position_monitor(self) -> dict[str, Any]:
        portfolio = self.reconcile_portfolio()
        if not portfolio.current_symbol:
            return {"monitor_action": "NO_POSITION", "execution": None}
        position = self._managed_position_for(portfolio)
        return_pct = position.current_price / position.entry_price - 1.0
        monitoring = self.cfg.get("monitoring", {})
        review_drawdown_pct = float(monitoring.get("review_drawdown_pct", 0.10))
        hard_stop_loss_pct = float(monitoring.get("hard_stop_loss_pct", monitoring.get("stop_loss_pct", 0.15)))
        take_profit_value = monitoring.get("take_profit_pct")
        take_profit_pct = float(take_profit_value) if take_profit_value is not None else None
        reduce_fraction = float(monitoring.get("take_profit_reduce_fraction", 0.50))

        if return_pct <= -hard_stop_loss_pct:
            action = "STOP_LOSS"
            threshold = -hard_stop_loss_pct
            target_weight = 0.0
            intent_action = "CASH"
            symbol = None
        elif return_pct <= -review_drawdown_pct:
            event_id = f"{position.position_id}:PRICE_SHOCK:DRAWDOWN_REVIEW"
            self.store.set_runtime("last_position_monitor", datetime.now(timezone.utc).isoformat())
            return {
                "monitor_action": "REVIEW_REQUIRED",
                "event_id": event_id,
                "event_type": "PRICE_SHOCK",
                "symbol": position.symbol,
                "return_pct": return_pct,
                "severity": "HIGH",
                "metadata": {"return_pct": return_pct, "review_threshold": -review_drawdown_pct},
                "execution": None,
            }
        elif take_profit_pct is not None and return_pct >= take_profit_pct:
            action = "TAKE_PROFIT"
            threshold = take_profit_pct
            target_weight = max(0.0, portfolio.current_weight * (1.0 - reduce_fraction))
            intent_action = "HOLD"
            symbol = position.symbol
        else:
            self.store.set_runtime("last_position_monitor", datetime.now(timezone.utc).isoformat())
            return {
                "monitor_action": "HOLD",
                "symbol": position.symbol,
                "return_pct": return_pct,
                "execution": None,
            }

        event_id = f"{position.position_id}:{action}"
        completion_key = f"position_monitor.completed.{event_id}"
        if self.store.get_runtime(completion_key, False):
            return {"monitor_action": "ALREADY_HANDLED", "symbol": position.symbol, "return_pct": return_pct, "execution": None}
        if self.store.position_trigger(event_id) is None and not self.position_manager.register_trigger(
            position,
            event_id,
            action,
            source="POSITION_MONITOR",
            severity="CRITICAL" if action == "STOP_LOSS" else "HIGH",
            metadata={"return_pct": return_pct, "threshold": threshold},
        ):
            return {"monitor_action": "COOLDOWN", "symbol": position.symbol, "return_pct": return_pct, "execution": None}

        now = datetime.now(timezone.utc).isoformat()
        intent = TradeIntent(
            action=intent_action,
            symbol=symbol,
            target_weight=target_weight,
            confidence=1.0,
            holding_period_days=position.current_thesis_horizon_days,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=[f"Deterministic {action} threshold reached at {return_pct:.2%}"],
            risk_factors=["Position-level price threshold triggered; final action remains subject to Risk Engine"],
            invalidation_conditions=["Fresh execution quote or broker reconciliation is unavailable"],
            evidence_used=[f"position_monitor:{event_id}"],
            timestamp=now,
            model_name="deterministic-position-monitor",
            decision_id=f"position-monitor-{position.position_id}-{action.lower()}",
        )
        execution = self.run(intent=intent, use_llm=False)
        if str(execution.get("state")) not in {"WAITING_FOR_MARKET", "PENDING_ENTRY", "REJECTED"}:
            self.store.set_runtime(completion_key, True)
        self.store.set_runtime("last_position_monitor", now)
        return {
            "monitor_action": action,
            "symbol": position.symbol,
            "return_pct": return_pct,
            "target_weight": target_weight,
            "decision": intent.model_dump(mode="json"),
            "execution": execution,
        }

    def _managed_position_for(self, portfolio: PortfolioState) -> ManagedPosition:
        symbol = str(portfolio.current_symbol).upper()
        source_intent = self._current_position_intent(symbol)
        existing = self.store.active_managed_position(symbol)
        if existing:
            current_price = next((item.market_price for item in self.broker_snapshot.positions if item.symbol.upper() == symbol), existing.current_price)
            updated = existing.model_copy(update={
                "current_price": current_price,
                "current_weight": portfolio.current_weight,
                "updated_at": portfolio.as_of,
            })
            updated = self.position_manager.ensure_entry_thesis_consistency(updated, source_intent)
            return self.position_manager.save_position(updated)
        broker_position = next(item for item in self.broker_snapshot.positions if item.symbol.upper() == symbol)
        decision_id = source_intent.decision_id if source_intent else f"reconciled-{symbol}"
        entry_time = source_intent.timestamp if source_intent else portfolio.as_of
        thesis = source_intent.thesis if source_intent else ["Position recovered from reconciled broker state; full review required"]
        horizon = source_intent.holding_period_days if source_intent else int(self.cfg.get("agent", {}).get("decision_horizon_days", 20))
        position = ManagedPosition(
            position_id=f"position-{decision_id}",
            run_id=self._active_run_id or f"reconcile-{decision_id}",
            decision_id=decision_id,
            symbol=symbol,
            entry_time=entry_time,
            entry_price=broker_position.average_cost or broker_position.market_price,
            current_price=broker_position.market_price,
            current_weight=portfolio.current_weight,
            original_thesis_horizon_days=horizon,
            current_thesis_horizon_days=horizon,
            original_thesis=thesis,
            original_positive_factors=thesis,
            original_risks=source_intent.risk_factors if source_intent else ["Original model risk context unavailable after reconciliation"],
            latest_thesis=thesis,
            current_thesis_status="EXPIRED_REVIEW_REQUIRED" if source_intent is None else "INTACT",
        )
        position = self.position_manager.ensure_entry_thesis_consistency(position, source_intent)
        return self.position_manager.save_position(position)

    def _legacy_position_review(self, position, portfolio, intent, review_type, event_context):
        if intent.action == "CASH":
            action = "EXIT_TO_CASH"
        elif intent.action == "SWITCH":
            action = "REPLACE"
        elif intent.target_weight + 1e-9 < portfolio.current_weight:
            action = "REDUCE"
        elif intent.target_weight > portfolio.current_weight + 1e-9:
            action = "ADD"
        else:
            action = "HOLD"
        current_score = 100.0 * min(1.0, max(0.0, intent.confidence))
        alternative_score = current_score + self.position_manager.replacement_threshold if action == "REPLACE" else None
        return PositionReview(
            position_id=position.position_id,
            run_id=self._active_run_id or f"review-{intent.decision_id}",
            decision_id=intent.decision_id,
            review_type=review_type,
            trigger=(event_context or {}).get("event_type"),
            current_holding=position.symbol,
            action=action,
            thesis_status="BROKEN" if action == "EXIT_TO_CASH" else "INTACT",
            current_holding_score=current_score,
            best_alternative=intent.new_symbol if action == "REPLACE" else None,
            best_alternative_score=alternative_score,
            replacement_gap=self.position_manager.replacement_threshold if action == "REPLACE" else 0,
            replacement_threshold=self.position_manager.replacement_threshold,
            confidence=intent.confidence,
            days_held=position.days_held_at(),
            original_horizon_days=position.original_thesis_horizon_days,
            new_horizon_days=position.current_thesis_horizon_days,
            reason=intent.thesis,
            target_weight=intent.target_weight,
        )

    @staticmethod
    def _hold_intent_from_review(position: ManagedPosition, review: PositionReview) -> TradeIntent:
        return TradeIntent(
            action="HOLD",
            symbol=position.symbol,
            target_weight=position.current_weight,
            confidence=review.confidence,
            holding_period_days=review.new_horizon_days,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=review.reason,
            risk_factors=[f"Position review risk level: {review.risk_level}"],
            invalidation_conditions=["Latest position thesis no longer remains valid"],
            evidence_used=[f"position_review:{review.review_id}"],
            timestamp=review.reviewed_at,
            model_name="sol-position-review",
            decision_id=review.decision_id,
        )

    def pre_execution_revalidate(self, intent: TradeIntent | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if intent is None:
            return {"valid": False, "executable": False, "reason": "No pending TradeIntent", "checks": {}}
        portfolio = self.reconcile_portfolio()
        context = context or {}
        connected = bool(getattr(self.executor, "connected", False))
        reconciled = bool(getattr(self.executor, "broker_state_known", False))
        market_open = bool(self.market_clock.is_open())
        maximum_age = float(self.cfg.get("execution", {}).get("max_quote_age_seconds", 30))
        decision_limit = float(self.cfg.get("risk", {}).get("max_decision_age_minutes_for_execution", 60))
        decision_fresh = RiskEngine(self.cfg.get("risk", {}))._fresh_with_limit(intent.timestamp, decision_limit)
        quote_error = None
        try:
            research_snapshot = self.data.stock_snapshot(intent.symbol) if intent.symbol else None
            execution_snapshot = self._execution_snapshot(intent.symbol, research_snapshot) if intent.symbol else None
            stored_market_data = self.store.get_runtime("current_market_data")
            snapshot = MarketDataSnapshot.model_validate(stored_market_data) if stored_market_data else None
        except Exception as exc:
            research_snapshot = None
            execution_snapshot = None
            snapshot = None
            quote_error = str(exc)
        company_checks = {}
        company_data_error = None
        if intent.symbol:
            try:
                company_checks = {
                    "news": self.data.news(intent.symbol, 10),
                    "earnings": self.data.earnings(intent.symbol),
                    "sec_filings": self.data.sec_filings(intent.symbol, 10),
                    "analyst_revisions": self.data.analyst_revisions(intent.symbol),
                }
            except Exception as exc:
                company_data_error = str(exc)
        mode = str(getattr(self.executor, "execution_mode", "PAPER")).upper()
        data_allowed = bool(snapshot and MarketDataPolicy.allows(snapshot, purpose="EXECUTION", execution_mode=mode, max_age_seconds=maximum_age))
        position_unchanged = not intent.current_symbol or intent.current_symbol.upper() == str(portfolio.current_symbol or "").upper()
        material_change = bool(context.get("material_change") or context.get("thesis_invalidated"))
        risk = None
        if execution_snapshot is not None:
            risk = RiskEngine(self.cfg.get("risk", {})).evaluate(
                intent,
                portfolio,
                execution_snapshot.get("ann_vol"),
                execution_snapshot.get("quote_as_of"),
                stock_snapshot=execution_snapshot,
                events=self.data.upcoming_events(intent.symbol) if intent.symbol else None,
                market_open=market_open,
            )
        risk_approved = bool(risk and risk.approved and risk.risk_state != "RISK_HALTED")
        executable = connected and reconciled and market_open and decision_fresh and data_allowed and position_unchanged and not material_change and company_data_error is None and risk_approved
        valid = data_allowed and decision_fresh and position_unchanged and not material_change and company_data_error is None if mode == "OBSERVE" else executable
        checks = {
            "market_open": market_open,
            "broker_connected": connected,
            "reconciliation_complete": reconciled,
            "decision_fresh": decision_fresh,
            "market_data_allowed": data_allowed,
            "risk_halted": portfolio.risk_state == "RISK_HALTED",
            "risk_approved": risk_approved,
            "position_symbol": portfolio.current_symbol,
            "position_unchanged": position_unchanged,
            "market_data_type": snapshot.market_data_type if snapshot else None,
            "quote_age_seconds": snapshot.current_age_seconds() if snapshot else None,
            "new_material_information_checked": company_data_error is None,
            "material_change": material_change,
            "company_data_error": company_data_error,
            "quote_error": quote_error,
        }
        reason = "PRE_EXECUTION_REVALIDATION PASS" if executable else "BLOCK_EXECUTION" if mode != "OBSERVE" else "OBSERVE validation only; ORDER NOT SENT"
        return {"valid": valid, "executable": executable, "reason": reason, "checks": checks, "market_data": snapshot.model_dump(mode="json") if snapshot else None, "company_checks": company_checks, "risk": risk.model_dump(mode="json") if risk else None}

    def _record_daily_performance(self, portfolio: PortfolioState):
        benchmark = self.data.benchmark_data(self.cfg.get("agent", {}).get("decision_horizon_days", 20))
        spy_series = benchmark.get("series", {}).get("SPY", [])
        qqq_series = benchmark.get("series", {}).get("QQQ", [])
        dates = benchmark.get("dates", [])
        if not spy_series or not qqq_series or len(dates) != len(spy_series) or len(dates) != len(qqq_series):
            raise RuntimeError("Daily performance observation requires synchronized trading dates, SPY, and QQQ values")
        observation_date = dates[-1]
        self.store.save_daily_performance(observation_date, portfolio.equity, spy_series[-1], qqq_series[-1])
        # Counterfactual analytics cannot change a risk decision or broker state.
        try:
            from .decision_audit import record_shadow_observations
            record_shadow_observations(self.store, self.data, benchmark)
        except Exception as exc:
            self.store.save_error("shadow-observation", str(exc))

    def record_journal_observations(self):
        benchmark = self.data.benchmark_data(self.cfg.get("agent", {}).get("decision_horizon_days", 20))
        spy = (benchmark.get("series", {}).get("SPY") or [None])[-1]
        qqq = (benchmark.get("series", {}).get("QQQ") or [None])[-1]
        for entry in self.store.journal_history():
            symbol = entry.get("symbol")
            if not symbol:
                continue
            try:
                price = self.data.stock_snapshot(symbol).get("price")
                self.store.record_journal_observation(entry["decision_id"], price, spy, qqq)
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue

    def emergency_stop(self) -> dict[str, Any]:
        portfolio = self.reconcile_portfolio()
        intent = TradeIntent(
            action="CASH",
            symbol=None,
            target_weight=0.0,
            confidence=1.0,
            holding_period_days=1,
            expected_alpha_vs_spy=0.0,
            expected_alpha_vs_qqq=0.0,
            thesis=["Operator emergency stop"],
            risk_factors=["Manual halt requires all positions closed"],
            invalidation_conditions=["Operator explicitly resumes the system"],
            evidence_used=["reconciled_broker_state"],
            model_name="operator-safety-control",
            decision_id=f"emergency-stop-{portfolio.as_of}",
        )
        return self.run(intent=intent)

    def _record_state(self, machine: TradingStateMachine, target: TradingState, reason: str, decision_id: str | None = None):
        if machine.state != target:
            machine.transition(target)
        self.store.save_state(machine.state, reason, decision_id)

    def _liquidate_all(self, decision_id: str, machine: TradingStateMachine, leg_prefix: str):
        portfolio = self.reconcile_portfolio()
        for active in list(self.broker_snapshot.open_orders):
            if active.get("action") == "BUY":
                if self._observe_mode() and hasattr(self.executor, "simulate_cancel"):
                    report = self.executor.simulate_cancel(active["client_order_id"])
                else:
                    report = self.executor.cancel_order(active["client_order_id"])
                self.store.save_execution(report)
                portfolio = self.reconcile_portfolio()
                if report.status not in {"CANCELLED", "FILLED", "SIMULATED"}:
                    return [report], portfolio

        reports: list[ExecutionReport] = []
        for position in list(self.broker_snapshot.positions):
            request = self._build_executable_order(
                decision_id, position.symbol, "SELL", position.quantity,
                f"{leg_prefix}-liquidate", reference_price=position.market_price,
            )
            hypothetical_label = "CASH" if leg_prefix in {"cash", "risk-halt"} else "SWITCH" if leg_prefix.startswith("switch") else None
            report = self._execute_order(request, machine, hypothetical_label=hypothetical_label)
            reports.append(report)
            portfolio = self.reconcile_portfolio()
            if not self._execution_succeeded(report):
                break
        return reports, portfolio

    def _enforce_max_positions(self, snapshot, prices):
        maximum = int(self.cfg.get("portfolio", {}).get("max_positions", 1))
        keep = sorted(snapshot.positions, key=lambda position: (-position.market_value, position.symbol))[:maximum]
        keep_symbols = {position.symbol for position in keep}
        machine = TradingStateMachine()
        decision_id = "mandate-reduction"
        self._record_state(machine, TradingState.RESEARCHING, "enforce maximum position mandate", decision_id)
        self._record_state(machine, TradingState.DECISION_READY, "broker reconciliation found excess positions", decision_id)
        for position in snapshot.positions:
            if position.symbol in keep_symbols:
                continue
            self._record_state(machine, TradingState.RISK_APPROVED, "automatic excess-position reduction", decision_id)
            request = self._build_executable_order(
                decision_id, position.symbol, "SELL", position.quantity,
                f"excess-{position.symbol}", reference_price=position.market_price,
            )
            report = self._execute_order(request, machine)
            if not self._execution_succeeded(report):
                raise RuntimeError(f"Mandate reduction for {position.symbol} did not fully fill")
            self._record_state(machine, TradingState.FILLED, report.message, decision_id)
        return self.executor.reconcile(prices)

    def _build_executable_order(
        self,
        decision_id: str,
        symbol: str,
        action: str,
        quantity: float,
        leg: str,
        *,
        reference_price: float | None,
    ) -> OrderRequest:
        client_order_id = self._order_id(decision_id, leg, symbol, action)
        expected_position_quantity = sum(
            position.quantity
            for position in (self.broker_snapshot.positions if self.broker_snapshot else [])
            if position.symbol.upper() == symbol.upper()
        )
        execution_cfg = self.cfg.get("execution", {})
        if not bool(execution_cfg.get("entry_engine_enabled", False)):
            return OrderRequest(
                decision_id=decision_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                reference_price=reference_price,
                client_order_id=client_order_id,
                expected_position_quantity=expected_position_quantity,
            )
        quote = self._execution_quote(symbol)
        tick_size = 0.01
        resolver = getattr(self.executor, "resolve_instrument", None)
        if callable(resolver):
            tick_size = float(getattr(resolver(symbol), "min_tick", 0.01) or 0.01)
        limit_price = self.entry_engine.marketable_limit_price(quote, action=action, tick_size=tick_size)
        comparison_price = reference_price or quote.mid or quote.price
        max_slippage = float(execution_cfg.get("max_total_slippage_bps", 30))
        if comparison_price and abs(limit_price - comparison_price) / comparison_price * 10000 > max_slippage:
            raise RuntimeError(f"Maximum execution slippage exceeded for {symbol}")
        return OrderRequest(
            decision_id=decision_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            reference_price=quote.mid or quote.price,
            order_type="LMT",
            limit_price=limit_price,
            client_order_id=client_order_id,
            expected_position_quantity=expected_position_quantity,
        )

    def _pre_submit_position_check(self, order: OrderRequest) -> ExecutionReport | None:
        """Fail closed if broker holdings changed after the order was sized."""
        try:
            symbols = self.executor.position_symbols()
            prices = {symbol: self._reconciliation_quote(symbol).price for symbol in symbols}
            snapshot = self.executor.reconcile(prices)
            self.broker_snapshot = snapshot
        except Exception as exc:
            return ExecutionReport(
                client_order_id=order.client_order_id,
                decision_id=order.decision_id,
                status="REJECTED",
                message=f"{order.action} blocked: broker holdings could not be refreshed ({exc})",
            )

        symbol = order.symbol.upper()
        actual_quantity = sum(
            position.quantity for position in snapshot.positions if position.symbol.upper() == symbol
        )
        expected_quantity = order.expected_position_quantity
        if order.action == "SELL":
            if actual_quantity <= 1e-9:
                reason = "SELL blocked: no broker position is held"
            elif order.quantity > actual_quantity + 1e-9:
                reason = f"SELL blocked: requested {order.quantity:g}, broker holds only {actual_quantity:g}"
            elif expected_quantity is not None and abs(actual_quantity - expected_quantity) > 1e-9:
                reason = (
                    f"SELL blocked: broker position changed from {expected_quantity:g} "
                    f"to {actual_quantity:g}; target must be recalculated"
                )
            else:
                return None
        else:
            other_symbols = sorted(
                position.symbol for position in snapshot.positions
                if position.symbol.upper() != symbol and position.quantity > 1e-9
            )
            if other_symbols:
                reason = f"BUY blocked: broker already holds another symbol ({', '.join(other_symbols)})"
            elif expected_quantity is not None and abs(actual_quantity - expected_quantity) > 1e-9:
                reason = (
                    f"BUY blocked: broker position changed from {expected_quantity:g} "
                    f"to {actual_quantity:g}; target must be recalculated"
                )
            else:
                return None
        return ExecutionReport(
            client_order_id=order.client_order_id,
            decision_id=order.decision_id,
            status="REJECTED",
            message=reason,
        )

    def _execute_order(self, order: OrderRequest, machine: TradingStateMachine, hypothetical_label: str | None = None) -> ExecutionReport:
        self._record_state(machine, TradingState.ORDER_PENDING, f"execute {order.action} {order.symbol}", order.decision_id)
        broker_report = self.executor.order_status(order.client_order_id)
        stored = self.store.order_by_client_id(order.client_order_id)
        if broker_report is not None:
            report = broker_report
        elif stored and stored["status"] in {"PENDING", "PARTIALLY_FILLED"}:
            report = ExecutionReport(client_order_id=order.client_order_id, decision_id=order.decision_id, status="ERROR", message="Stored active order is missing from broker; blocked to prevent duplicate")
        else:
            self.store.save_order(order)
            if self._observe_mode() and hasattr(self.executor, "simulate_order"):
                report = self.executor.simulate_order(order, label=hypothetical_label)
            else:
                report = self._pre_submit_position_check(order)
                if report is None:
                    report = self.executor.submit_order(order)
                else:
                    self.store.save_order(order, status=report.status)
            if report.status == "PENDING" and not self._observe_mode():
                report = self.executor.await_order(order.client_order_id, self.cfg.get("execution", {}).get("order_timeout_seconds")) or report
        if broker_report is not None and stored is None:
            self.store.save_order(order, status=broker_report.status)
        self.store.save_execution(report)
        self._last_execution = report
        if report.status == "SIMULATED":
            label = hypothetical_label or order.action
            self._emit_event(
                f"WOULD_{label}",
                "EXECUTION",
                report.message,
                decision_id=order.decision_id,
                symbol=order.symbol,
                metadata={
                    "order": order.model_dump(mode="json"),
                    "status": report.status,
                    "order_not_sent": True,
                    "place_order_calls": int(getattr(self.executor, "place_order_calls", 0) or 0),
                    "cancel_order_calls": int(getattr(self.executor, "cancel_order_calls", 0) or 0),
                },
            )
        elif report.status in {"REJECTED", "ERROR"}:
            self._emit_event("EXECUTION_BLOCKED", "EXECUTION", report.message, decision_id=order.decision_id, symbol=order.symbol, metadata={"status": report.status, "error_code": report.error_code, "error_string": report.error_string})
        self._emit_event(
            "EXECUTION_COMPLETED",
            "EXECUTION",
            report.message,
            decision_id=order.decision_id,
            symbol=order.symbol,
            metadata={
                "status": report.status,
                "filled_quantity": report.filled_quantity,
                "average_price": report.average_price,
                "order_not_sent": report.status == "SIMULATED",
            },
        )
        return report

    def _pipeline_runtime_metadata(self) -> dict[str, Any]:
        runtime = getattr(self.agent, "runtime", None)
        return {
            "pipeline": getattr(runtime, "pipeline", None),
            "luna_model": getattr(runtime, "luna_model", None),
            "sol_model": getattr(runtime, "sol_model", getattr(self.agent, "model", None)),
            "api_protocol": getattr(getattr(self.agent, "provider", None), "protocol", getattr(runtime, "api_protocol", None)),
            "protocol_fallback": "YES" if getattr(getattr(self.agent, "provider", None), "protocol_fallback", getattr(runtime, "protocol_fallback", False)) else "NO",
        }

    def _agent_event_sink(self, event: dict[str, Any]):
        metadata = event.get("metadata") if isinstance(event, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        decision = metadata.get("decision") if isinstance(metadata.get("decision"), dict) else {}
        decision_id = self._active_decision_id or decision.get("decision_id") or metadata.get("decision_id")
        event_type = event.get("event_type", "LLM_EVENT")
        if event_type == "AI_PROGRESS":
            self._set_ai_progress(str(metadata.get("stage", "PREPARING")), float(metadata.get("progress_percent", 0)), **{key: value for key, value in metadata.items() if key not in {"stage", "progress_percent", "updated_at"}})
        elif event_type == "SOL_RESEARCH_STARTED":
            self._set_ai_progress("SOL_RESEARCH", 55, initial_candidates=len(metadata.get("candidate_symbols", [])), sol_candidates_reviewed=0, sol_tool_calls=0, sol_tool_results=0)
        elif event_type == "SOL_TOOL_CALL":
            current = self.store.get_runtime("ai_research_progress", {}) or {}
            reviewed = list(current.get("sol_reviewed_symbols", []))
            if event.get("symbol") and event.get("symbol") not in reviewed:
                reviewed.append(event.get("symbol"))
            self._set_ai_progress("SOL_RESEARCH", current.get("progress_percent", 55), sol_current_symbol=event.get("symbol"), sol_tool_calls=int(current.get("sol_tool_calls", 0)) + 1, current_tool=metadata.get("tool"), sol_reviewed_symbols=reviewed, sol_candidates_reviewed=len(reviewed))
        elif event_type == "SOL_TOOL_RESULT":
            current = self.store.get_runtime("ai_research_progress", {}) or {}
            self._set_ai_progress("SOL_RESEARCH", current.get("progress_percent", 55), sol_current_symbol=event.get("symbol"), sol_tool_results=int(current.get("sol_tool_results", 0)) + 1)
        elif event_type == "SOL_DECISION_COMPLETED":
            self._set_ai_progress("SOL_DECISION", 92, status="COMPLETE")
        elif event_type == "LUNA_BATCH_SPLIT":
            current = self.store.get_runtime("ai_research_progress", {}) or {}
            self._set_ai_progress("LUNA_BATCH_SCREENING", current.get("progress_percent", 5), adaptive_split=metadata)
        self._emit_event(
            event_type,
            event.get("component", "LLM"),
            event.get("message", ""),
            decision_id=decision_id,
            symbol=event.get("symbol") or decision.get("symbol"),
            metadata=metadata,
        )

    def _emit_event(self, event_type: str, component: str, message: str, *, decision_id: str | None = None, symbol: str | None = None, metadata: dict[str, Any] | None = None):
        self.store.save_runtime_event(
            event_type,
            component,
            message,
            run_id=self._active_run_id,
            decision_id=decision_id or self._active_decision_id,
            symbol=symbol,
            metadata=metadata,
        )
        if event_type == "RISK_EVALUATION_STARTED":
            self._set_ai_progress("RISK_ENGINE", 92, status="RUNNING")
        elif event_type == "RISK_EVALUATED":
            self._set_ai_progress("RISK_ENGINE", 97, status="COMPLETE")
        elif event_type.startswith("WOULD_") or event_type in {"EXECUTION_BLOCKED", "EXECUTION_COMPLETED"}:
            self._set_ai_progress("OBSERVE_EXECUTION", 99, status="RUNNING")
        elif event_type == "AI_RUN_COMPLETED":
            status = str(metadata.get("status", "COMPLETE"))
            if status == "WAITING_FOR_BROKER":
                current = self.store.get_runtime("ai_research_progress", {}) or {}
                self._set_ai_progress("WAITING_FOR_BROKER", current.get("progress_percent", 92), status=status, completed_at=datetime.now(timezone.utc).isoformat())
            else:
                self._set_ai_progress("COMPLETE", 100, status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat())
        elif event_type == "AI_RUN_FAILED":
            current = self.store.get_runtime("ai_research_progress", {}) or {}
            self._set_ai_progress(str(current.get("stage", "FAILED")), float(current.get("progress_percent", 0)), status="FAILED", error_stage=current.get("stage"), error_message=message, completed_at=datetime.now(timezone.utc).isoformat())

    def _set_ai_progress(self, stage: str, progress_percent: float, **values):
        now = datetime.now(timezone.utc).isoformat()
        current = self.store.get_runtime("ai_research_progress", {}) or {}
        same_run = current.get("run_id") == self._active_run_id
        previous = float(current.get("progress_percent", 0)) if same_run else 0.0
        state = dict(current) if same_run else {"run_id": self._active_run_id, "started_at": now, "universe_total": 0, "universe_processed": 0, "batch_index": 0, "batch_total": 0, "sol_tool_calls": 0, "sol_tool_results": 0}
        state.update(values)
        state.update({"run_id": self._active_run_id, "stage": stage, "progress_percent": max(previous, min(100.0, float(progress_percent))), "updated_at": now})
        self.store.set_runtime("ai_research_progress", state)

    def _finish_run(self, result: dict[str, Any]) -> dict[str, Any]:
        state = str(result.get("state"))
        if state == TradingState.WAITING_FOR_BROKER.value:
            status = "WAITING_FOR_BROKER"
        elif state in {TradingState.WAITING_FOR_MARKET.value, TradingState.PENDING_ENTRY.value}:
            status = "WAITING_FOR_MARKET"
        else:
            status = "COMPLETED"
        self._emit_event("AI_RUN_COMPLETED", "PIPELINE", f"AI cycle {status.lower()}", metadata={"state": state, "status": status})
        self.store.set_runtime("ai_cycle", {"run_id": self._active_run_id, "status": status, "ended_at": datetime.now(timezone.utc).isoformat(), "state": state})
        return result

    def _save_journal(self, intent: TradeIntent, risk, stock_snapshot):
        metadata = getattr(self.agent, "last_pipeline_metadata", {}) if self.agent is not None else {}
        candidates = list(metadata.get("candidate_symbols", [])) if isinstance(metadata, dict) else []
        if not candidates and isinstance(metadata, dict):
            candidates = list(metadata.get("luna_candidates", []))
        for evidence in getattr(self.agent, "last_research_evidence", []):
            result = evidence.get("result")
            if not candidates and isinstance(result, list):
                candidates.extend(item.get("symbol") for item in result if isinstance(item, dict) and item.get("symbol"))
        entry_price = (stock_snapshot or {}).get("price")
        filled_price = self._last_execution.average_price if self._last_execution and self._last_execution.status in {"FILLED", "PARTIALLY_FILLED", "SIMULATED"} else None
        self.store.save_decision_journal(intent, risk, entry_price=entry_price, filled_price=filled_price, candidates=list(dict.fromkeys(candidates)))

    def _record_execution_state(self, machine: TradingStateMachine, report: ExecutionReport | None, decision_id: str):
        if report is None:
            self._record_state(machine, TradingState.ERROR, "No broker report", decision_id)
        elif report.status == "PARTIALLY_FILLED":
            self._record_state(machine, TradingState.PARTIALLY_FILLED, report.message, decision_id)
        elif report.status == "REJECTED":
            self._record_state(machine, TradingState.REJECTED, report.message, decision_id)
        elif report.status == "SIMULATED":
            self._record_state(machine, TradingState.OBSERVED, report.message, decision_id)
        elif report.status == "PENDING":
            self.store.save_state(machine.state, report.message, decision_id)
        else:
            self._record_state(machine, TradingState.ERROR, report.message, decision_id)

    def _risk_state_for_drawdown(self, drawdown: float, previous_state: str) -> str:
        risk_cfg = self.cfg.get("risk", {})
        tolerance = float(risk_cfg.get("comparison_tolerance", 1e-9))
        hard_limit = float(risk_cfg.get("hard_drawdown_limit", 0.25))
        if previous_state == "RISK_HALTED" and drawdown > float(risk_cfg.get("halt_recovery_drawdown", 0.20)) + tolerance:
            return "RISK_HALTED"
        if drawdown >= hard_limit - tolerance:
            return "RISK_HALTED"
        for tier in sorted(risk_cfg.get("drawdown_tiers", []), key=lambda item: float(item["drawdown"]), reverse=True):
            if drawdown >= float(tier["drawdown"]) - tolerance and float(tier.get("max_weight", 1.0)) < 1.0:
                return "REDUCED"
        return "NORMAL"

    def _execution_quote(self, symbol: str) -> ExecutionQuote:
        try:
            return ExecutionQuote.model_validate(self.quote_provider.get_quote(symbol))
        except Exception as exc:
            if self._is_broker_readiness_error(exc):
                raise BrokerReadinessError(str(exc)) from exc
            raise

    def _is_broker_readiness_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if any(
            marker in message
            for marker in (
                "ibkr is disconnected",
                "ibkr reconciliation required",
                "execution quote timed out",
                "nextvalidid timeout",
            )
        ):
            return True
        if str(getattr(self.executor, "broker_source", "")).upper() != "IBKR_PAPER":
            return False
        return not bool(getattr(self.executor, "connected", False)) or not bool(getattr(self.executor, "broker_state_known", False))

    def _reconciliation_quote(self, symbol: str) -> ExecutionQuote:
        provider = getattr(self.quote_provider, "get_reconciliation_quote", None)
        if callable(provider):
            return ExecutionQuote.model_validate(provider(symbol))
        return self._execution_quote(symbol)

    def _observe_mode(self) -> bool:
        return getattr(self.executor, "execution_mode", "PAPER").upper() == "OBSERVE" or not getattr(self.executor, "mutations_allowed", True)

    @staticmethod
    def _execution_succeeded(report: ExecutionReport) -> bool:
        return report.status in {"FILLED", "SIMULATED"}

    def _execution_snapshot(self, symbol: str, research_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        quote = self._execution_quote(symbol)
        market_data = quote.to_market_data_snapshot()
        self.store.set_runtime("current_market_data", market_data.model_dump(mode="json"))
        self.store.set_runtime("market_session", market_data.market_session)
        self.store.set_runtime("market_data_status", "MARKET_CLOSED" if market_data.market_session == "CLOSED" else market_data.market_data_type)
        self.store.set_runtime("market_data_source", market_data.source)
        self.store.set_runtime("risk_data_age_seconds", market_data.current_age_seconds())
        return {
            **(research_snapshot or {}),
            "symbol": symbol.upper(),
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "mid": quote.mid,
            "quote_as_of": quote.timestamp,
            "market_status": quote.market_status,
            "quote_source": quote.source,
            "quote_data_type": quote.data_type,
            "market_data_type": market_data.market_data_type,
            "broker_timestamp": market_data.broker_timestamp,
            "received_at": market_data.received_at,
            "age_seconds": market_data.current_age_seconds(),
            "market_session": market_data.market_session,
            "price_available": True,
            "trading_halted": quote.market_status == "HALTED",
        }

    def _synchronize_risk_state(self, portfolio: PortfolioState, risk_state: str) -> PortfolioState:
        if portfolio.risk_state == risk_state:
            return portfolio
        self.store.save_risk_state_event(
            portfolio.risk_state,
            risk_state,
            f"RiskDecision changed risk state to {risk_state}",
        )
        synchronized = portfolio.model_copy(update={"risk_state": risk_state})
        positions = [
            {**position.model_dump(mode="json"), "weight": position.market_value / synchronized.equity if synchronized.equity else 0.0}
            for position in self.broker_snapshot.positions
        ]
        self.store.save_portfolio(synchronized, positions)
        self.store.set_runtime("risk_state", risk_state)
        self.data.set_portfolio_context(synchronized, positions)
        return synchronized

    def _current_position_intent(self, symbol: str | None) -> TradeIntent | None:
        if not symbol:
            return None
        for row in self.store.recent("llm_decisions", 50):
            try:
                intent = TradeIntent.model_validate_json(row["decision_json"])
            except (KeyError, TypeError, ValueError):
                continue
            if intent.symbol and intent.symbol.upper() == symbol.upper() and intent.action != "SWITCH":
                return intent
        return None

    @staticmethod
    def _order_id(decision_id: str, leg: str, symbol: str, action: str) -> str:
        return make_client_order_id(decision_id, leg, symbol, action)


def performance_from_store(store: SQLiteStore) -> dict[str, float | None]:
    values = [row["equity"] for row in store.equity_history()]
    return compute_metrics(values)
