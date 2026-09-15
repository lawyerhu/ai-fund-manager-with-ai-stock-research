from pathlib import Path

import pytest

from src import config as project_config
from src.dashboard_data import build_dashboard_payload, dashboard_refresh_seconds
from src.models import PortfolioState, TradeIntent
from src.models import ExecutionReport
from src.storage import SQLiteStore, redact_sensitive


def test_dashboard_payload_exposes_monitoring_status_and_account_data():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("trading_mode", "LOCAL_PAPER")
        store.set_runtime("broker_status", "CONNECTED")
        store.set_runtime("llm_status", "ONLINE")
        store.set_runtime("market_data_status", "OK")
        store.set_runtime("scheduler_status", "RUNNING")
        store.set_runtime("risk_state", "NORMAL")
        store.set_runtime("last_reconciliation", "2026-08-30T10:00:00+00:00")
        store.save_portfolio(PortfolioState(equity=1000, peak_equity=1000, cash=600, current_symbol="NVDA", current_quantity=4, current_weight=0.4, invested_value=400), [{"symbol": "NVDA", "quantity": 4, "market_price": 100, "market_value": 400, "weight": 0.4}])

        payload = build_dashboard_payload(store, {"execution": {"mode": "LOCAL_PAPER"}})

        assert payload["status"]["system"] == "ONLINE"
        assert payload["status"]["trading_mode"] == "LOCAL_PAPER"
        assert payload["account"]["equity"] == 1000
        assert payload["account"]["invested_value"] == 400
        assert payload["account"]["positions"][0]["symbol"] == "NVDA"


def test_dashboard_configuration_status_never_contains_secret_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    with SQLiteStore(":memory:") as store:
        payload = build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

    assert payload["configuration"]["OPENAI_API_KEY"] == "Configured"
    assert payload["configuration"]["OPENAI_MODEL"] == "Configured"
    assert "secret-value" not in str(payload)


def test_nested_redaction_reads_secret_environment_once_per_payload(monkeypatch):
    import src.storage as storage_module

    original_getenv = storage_module.os.getenv
    calls = []

    def counted_getenv(name, default=None):
        calls.append(name)
        return original_getenv(name, default)

    monkeypatch.setattr(storage_module.os, "getenv", counted_getenv)
    result = redact_sensitive({"rows": [{"message": "safe text"} for _ in range(500)]})

    assert result["rows"][0]["message"] == "safe text"
    assert len(calls) <= 10


def test_redaction_fast_path_preserves_token_and_bearer_protection():
    result = redact_sensitive({
        "message": "provider returned SK-1234567890abcdefghijkl and Bearer abc.def-123",
        "authorization": "must never be shown",
    })

    assert "SK-1234567890abcdefghijkl" not in str(result)
    assert "abc.def-123" not in str(result)
    assert result["authorization"] == "[REDACTED]"


def test_dashboard_configuration_reads_project_root_env(monkeypatch, tmp_path):
    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_LUNA_MODEL", "LLM_SOL_MODEL", "LLM_PIPELINE"):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=ccswitch\n"
        "LLM_BASE_URL=http://gateway.test\n"
        "LLM_API_KEY=offline-test\n"
        "LLM_LUNA_MODEL=luna-test\n"
        "LLM_SOL_MODEL=sol-test\n"
        "LLM_PIPELINE=LUNA_SOL\n",
        encoding="utf-8",
    )
    project_config.load_project_env(env_path, override=True)

    with SQLiteStore(":memory:") as store:
        payload = build_dashboard_payload(store, {})

    assert payload["configuration"]["LLM_PROVIDER"] == "Configured"
    assert payload["configuration"]["LLM_BASE_URL"] == "Configured"
    assert payload["configuration"]["LLM_API_KEY"] == "Configured"
    assert payload["configuration"]["LLM_LUNA_MODEL"] == "Configured"
    assert payload["configuration"]["LLM_SOL_MODEL"] == "Configured"
    assert payload["configuration"]["LLM_PIPELINE"] == "Configured"


def test_dashboard_preserves_closed_market_data_state_without_marking_it_failed():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("market_data_status", "MARKET_CLOSED")
        store.set_runtime("market_session", "CLOSED")
        store.set_runtime("market_data_source", "IBKR historical")
        store.set_runtime("market_data_last_bar_time", "2026-08-28T20:00:00+00:00")
        store.set_runtime("setup.market_data_type", "MARKET_CLOSED")
        store.set_runtime("setup_details", {"quote": {"data_type": "FROZEN", "market_status": "CLOSED"}})
        payload = build_dashboard_payload(store, {})

    assert payload["status"]["market_data"] == "MARKET_CLOSED"
    assert payload["status"]["market_session"] == "CLOSED"
    assert payload["status"]["market_data_source"] == "IBKR historical"
    assert payload["status"]["market_data_last_bar_time"] == "2026-08-28T20:00:00+00:00"
    assert payload["setup"]["checks"]["Market Data Type"] == "WARN"
    assert payload["setup"]["values"]["Market Data Type"] == "FROZEN"


def test_dashboard_performance_series_have_matching_dates_and_lengths():
    with SQLiteStore(":memory:") as store:
        store.save_daily_performance("2026-08-28", 1000, 640, 570)
        store.save_daily_performance("2026-08-29", 1020, 646.4, 581.4)

        payload = build_dashboard_payload(store, {"execution": {"mode": "LOCAL_PAPER"}})
        timeline = payload["benchmarks"]

        assert timeline["dates"] == ["2026-08-28", "2026-08-29"]
        assert timeline["series"]["AI Strategy"] == [100.0, 102.0]
        assert timeline["series"]["SPY Buy & Hold"] == [100.0, 101.0]
        assert timeline["series"]["QQQ Buy & Hold"] == [100.0, 102.0]
        assert len({len(timeline["dates"]), *(len(values) for values in timeline["series"].values())}) == 1


def test_dashboard_exposes_unknown_macro_position_cap():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("macro_data_status", "UNKNOWN — POSITION CAPPED")
        payload = build_dashboard_payload(store, {})

    assert payload["status"]["macro_data"] == "UNKNOWN — POSITION CAPPED"


def test_net_excess_return_uses_net_equity_after_trading_costs():
    with SQLiteStore(":memory:") as store:
        store.save_daily_performance("2026-08-27", 1000, 100, 100)
        store.save_daily_performance("2026-08-28", 1090, 105, 108)
        store.save_execution(ExecutionReport(
            client_order_id="cost-order", decision_id="cost-decision", status="FILLED",
            filled_quantity=1, average_price=100, commission=6, fees=1, slippage=3,
        ))

        performance = build_dashboard_payload(store, {})["benchmarks"]["returns"]

    assert performance["gross_strategy_return"] == pytest.approx(0.10)
    assert performance["trading_costs"] == 10.0
    assert performance["net_strategy_return"] == pytest.approx(0.09)
    assert performance["spy_return"] == pytest.approx(0.05)
    assert performance["qqq_return"] == pytest.approx(0.08)
    assert performance["net_excess_return_vs_spy"] == pytest.approx(0.04)
    assert performance["net_excess_return_vs_qqq"] == pytest.approx(0.01)


def test_dashboard_setup_health_exposes_first_run_status_without_secrets():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("service_status", "READY_FOR_OBSERVE")
        store.set_runtime("first_run_state", "READY_FOR_OBSERVE")
        store.set_runtime("first_run_ready", True)
        store.set_runtime("setup.tws", "PASS")
        store.set_runtime("setup.paper_account", "PASS")
        store.set_runtime("setup.socket", "PASS")
        store.set_runtime("setup.market_data", "WARN")
        store.set_runtime("execution_mode", "OBSERVE")
        store.set_runtime("broker_source", "IBKR_PAPER")

        payload = build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

    assert payload["setup"]["state"] == "READY_FOR_OBSERVE"
    assert payload["setup"]["checks"]["TWS Connected"] == "PASS"
    assert payload["setup"]["checks"]["Market Data Type"] == "WARN"
    assert payload["setup"]["execution_mode"] == "OBSERVE"
    assert payload["setup"]["broker_source"] == "IBKR_PAPER"


def test_runtime_events_are_persisted_and_redacted():
    with SQLiteStore(":memory:") as store:
        event_id = store.save_runtime_event(
            "SOL_TOOL_CALL",
            "SOL",
            "Calling get_fundamentals",
            run_id="run-1",
            decision_id="decision-1",
            symbol="NVDA",
            metadata={"tool": "get_fundamentals", "api_key": "do-not-store"},
            timestamp="2026-08-30T15:31:15+00:00",
        )

        rows = store.runtime_events()

    assert event_id == 1
    assert rows[0]["event_type"] == "SOL_TOOL_CALL"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["decision_id"] == "decision-1"
    assert rows[0]["symbol"] == "NVDA"
    assert "do-not-store" not in str(rows[0])


def test_error_and_setup_payloads_redact_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "historical-secret")
    with SQLiteStore(":memory:") as store:
        store.save_error("llm", "gateway rejected historical-secret", {"account_number": "U123"})
        store.set_runtime("setup_details", {"account_number": "U123", "quote": {"data_type": "DELAYED"}})
        payload = build_dashboard_payload(store, {})

    assert "historical-secret" not in str(payload)
    assert "U123" not in str(payload)
    assert payload["setup"]["details"]["quote"]["data_type"] == "DELAYED"


def test_dashboard_payload_reconstructs_current_ai_activity_and_timeline():
    with SQLiteStore(":memory:") as store:
        events = [
            ("AI_RUN_STARTED", "PIPELINE", "AI cycle started", {"pipeline": "LUNA_SOL", "luna_model": "gpt-5.6-luna", "sol_model": "gpt-5.6-sol"}),
            ("LUNA_SCREEN_STARTED", "LUNA", "Screening universe", {"universe_size": 487}),
            ("LUNA_SCREEN_COMPLETED", "LUNA", "Screening completed", {"candidate_count": 2, "candidates": [{"symbol": "NVDA", "rationale": "Trend", "positive_signals": ["growth"], "risks": ["volatility"]}]}),
            ("SOL_RESEARCH_STARTED", "SOL", "Research started", {"candidate_symbols": ["NVDA"]}),
            ("SOL_TOOL_CALL", "SOL", "Calling get_fundamentals", {"tool": "get_fundamentals", "arguments": {"symbol": "NVDA"}}),
            ("SOL_TOOL_RESULT", "SOL", "get_fundamentals succeeded", {"tool": "get_fundamentals", "status": "SUCCESS", "result": {"symbol": "NVDA", "forward_pe": 31}}),
            ("SOL_DECISION_COMPLETED", "SOL", "BUY NVDA", {"action": "BUY", "symbol": "NVDA", "target_weight": 0.7, "confidence": 0.85, "expected_excess_vs_spy": 0.03, "expected_excess_vs_qqq": 0.01, "thesis": ["Trend"], "risk_factors": ["Volatility"], "invalidation_conditions": ["Trend breaks"], "evidence_used": ["get_fundamentals"]}),
            ("RISK_EVALUATED", "RISK_ENGINE", "Approved at 50%", {"requested_weight": 0.7, "approved_weight": 0.5, "reason": "Volatility limit"}),
            ("WOULD_BUY", "EXECUTION", "WOULD BUY NVDA 50%", {"approved_weight": 0.5, "place_order_calls": 0, "cancel_order_calls": 0}),
            ("AI_RUN_COMPLETED", "PIPELINE", "AI cycle completed", {}),
        ]
        for index, (event_type, component, message, metadata) in enumerate(events, start=1):
            store.save_runtime_event(
                event_type,
                component,
                message,
                run_id="run-1",
                decision_id="decision-1" if index >= 7 else None,
                symbol="NVDA" if index >= 3 and index != 4 else None,
                metadata=metadata,
                timestamp=f"2026-08-30T15:31:{index:02d}+00:00",
            )

        payload = build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

    activity = payload["ai_activity"]
    assert activity["run_id"] == "run-1"
    assert activity["status"] == "COMPLETED"
    assert activity["stages"]["Luna"]["status"] == "COMPLETE"
    assert activity["stages"]["Luna"]["candidate_count"] == 2
    assert activity["stages"]["Sol"]["status"] == "COMPLETE"
    assert activity["stages"]["Sol"]["tool_calls"][0]["tool"] == "get_fundamentals"
    assert activity["stages"]["Risk Engine"]["approved_weight"] == 0.5
    assert activity["stages"]["Execution"]["status"] == "WOULD_EXECUTE"
    assert [row["event_type"] for row in payload["timeline"]] == [item[0] for item in events]


def test_dashboard_exposes_only_persisted_historical_candidate_comparison():
    candidates = ["VEEV", "CRM", "ABNB", "PANW", "MPC"]
    comparison = (
        "Comparative screen: VEEV and CRM had stronger recent returns but higher volatility; "
        "MPC had more favorable valuation support."
    )
    intent = TradeIntent(
        action="BUY", symbol="MPC", target_weight=0.5, confidence=0.72,
        holding_period_days=20, expected_alpha_vs_spy=0.06, expected_alpha_vs_qqq=0.08,
        thesis=["MPC has the best risk-adjusted opportunity in this screen."],
        risk_factors=["Cyclical margin risk"],
        invalidation_conditions=["Relative strength breaks"],
        evidence_used=[comparison], decision_id="decision-historical-comparison",
    )
    with SQLiteStore(":memory:") as store:
        store.save_decision(intent, pipeline_metadata={
            "pipeline": "LUNA_SOL",
            "candidate_symbols": candidates,
            "final_decision": intent.model_dump(mode="json"),
        })
        store.save_runtime_event(
            "SOL_DECISION_COMPLETED", "SOL", "BUY MPC", run_id="run-historical-comparison",
            decision_id=intent.decision_id, symbol="MPC", metadata={"decision": intent.model_dump(mode="json")},
        )

        activity = build_dashboard_payload(store, {})["ai_activity"]

    assert activity["comparative_context"] == {
        "candidate_symbols": candidates,
        "comparative_notes": [comparison],
        "structured_ranking_available": False,
    }
    assert "top_five" not in activity["final_decision"]


def test_dashboard_refresh_seconds_defaults_to_three_and_reads_only_configuration(monkeypatch):
    monkeypatch.delenv("DASHBOARD_REFRESH_SECONDS", raising=False)
    assert dashboard_refresh_seconds({}) == 3.0

    monkeypatch.setenv("DASHBOARD_REFRESH_SECONDS", "7")
    assert dashboard_refresh_seconds({}) == 7.0

    monkeypatch.setenv("DASHBOARD_REFRESH_SECONDS", "0")
    assert dashboard_refresh_seconds({}) == 3.0


def test_dashboard_payload_refresh_is_read_only_and_does_not_duplicate_commands():
    with SQLiteStore(":memory:") as store:
        store.enqueue_command("EMERGENCY_STOP")
        before = store.pending_commands()

        build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})
        build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

        assert store.pending_commands() == before


def test_dashboard_reports_command_worker_and_latest_command_lifecycle():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("command_worker_status", "RUNNING")
        command_id = store.enqueue_command("RUN_AI_RESEARCH")
        claimed = store.claim_next_command("worker-test")
        assert claimed["id"] == command_id
        store.mark_command_running(command_id, "worker-test")
        store.complete_command(command_id, error="denied", error_stage="RUN_AI_RESEARCH", worker_id="worker-test")

        payload = build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

        assert payload["status"]["command_worker"] == "RUNNING"
        assert payload["status"]["latest_command"]["status"] == "FAILED"
    assert payload["status"]["latest_command"]["error_stage"] == "RUN_AI_RESEARCH"


def test_dashboard_exposes_only_current_worker_ai_research_as_active():
    with SQLiteStore(":memory:") as store:
        stale_id = store.enqueue_command("RUN_AI_RESEARCH")
        stale = store.claim_next_command("worker-old")
        assert stale["id"] == stale_id
        store.mark_command_running(stale_id, "worker-old")
        store.set_runtime("command_worker_id", "worker-current")
        active_id = store.enqueue_command("RUN_AI_RESEARCH")

        payload = build_dashboard_payload(store, {})

    assert payload["status"]["active_ai_research_command"]["id"] == active_id
    assert payload["status"]["active_ai_research_command"]["status"] == "QUEUED"


def test_dashboard_payload_exposes_acknowledged_failed_command_id():
    with SQLiteStore(":memory:") as store:
        store.set_runtime("dashboard.acknowledged_failed_command_id", 131)

        payload = build_dashboard_payload(store, {"execution": {"mode": "OBSERVE"}})

    assert payload["status"]["acknowledged_failed_command_id"] == 131


def test_dashboard_keeps_reconciliation_result_visible_after_newer_command(tmp_path):
    database = tmp_path / "dashboard-reconciliation-status.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        reconciliation_id = store.enqueue_command("RUN_RECONCILIATION")
        store.claim_next_command("worker-test")
        store.mark_command_running(reconciliation_id, "worker-test")
        store.complete_command(reconciliation_id, worker_id="worker-test")
        store.enqueue_command("RUN_IMMEDIATE_RISK_CHECK")

        payload = build_dashboard_payload(store, {})

    assert payload["status"]["latest_command"]["command"] == "RUN_IMMEDIATE_RISK_CHECK"
    assert payload["status"]["last_reconciliation_command"]["id"] == reconciliation_id
    assert payload["status"]["last_reconciliation_command"]["status"] == "SUCCEEDED"


def test_dashboard_uses_connected_reconciled_monitor_state_over_stale_broker_status(tmp_path):
    database = tmp_path / "dashboard-broker-state.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("broker_status", "RECONCILIATION_REQUIRED")
        store.set_runtime("broker_monitor_connected", True)
        store.set_runtime("broker_reconciliation_ready", True)
        store.set_runtime("service_status", "ONLINE")

        payload = build_dashboard_payload(store, {})

    assert payload["status"]["broker"] == "CONNECTED"


def test_dashboard_progress_comes_from_backend_runtime_state_and_refresh_is_read_only():
    progress = {
        "run_id": "run-progress",
        "stage": "LUNA_BATCH_SCREENING",
        "progress_percent": 28.0,
        "universe_total": 518,
        "universe_processed": 180,
        "batch_index": 3,
        "batch_total": 9,
        "current_batch_size": 30,
    }
    with SQLiteStore(":memory:") as store:
        store.set_runtime("ai.current_run_id", "run-progress")
        store.set_runtime("ai_research_progress", progress)
        before = store.runtime_snapshot()

        payload = build_dashboard_payload(store, {})
        build_dashboard_payload(store, {})

        assert payload["ai_activity"]["progress"] == progress
        assert store.runtime_snapshot() == before


def test_dashboard_uses_local_fragments_without_browser_or_full_page_refresh():
    source = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text(encoding="utf-8")
    static_shell = source[source.index("def _render_static_shell"):source.index("cfg = load_config()")]
    live_fragment = source[source.index("def _dashboard_live_fragment"):source.rindex("_render_static_shell(cfg, db_path)")]

    assert "st.rerun(" not in source
    assert "st.experimental_rerun(" not in source
    assert "st.components.v1.html" not in source
    assert "window.location.reload" not in source
    assert "location.reload" not in source
    assert "removeChild" not in source
    assert "appendChild" not in source
    assert "def _streamlit_fragment" in source
    assert "@_streamlit_fragment(run_every=dashboard_refresh_seconds(cfg))" in source
    assert "def _dashboard_live_fragment" in source
    assert "def _dashboard_controls_fragment" in source
    assert "_dashboard_controls_fragment()" in static_shell
    assert "tabs = st.tabs(" in static_shell
    assert "_dashboard_live_fragment()" in static_shell
    assert "def _dashboard_activity_fragment" in source
    assert "for tab, renderer in" not in live_fragment
    assert "tabs = st.tabs(" not in live_fragment
    assert "_render_live_ai_activity" in source
    assert "_render_timeline" in source
    assert "_render_overview" in source
    assert "_render_benchmarks" in source
    assert "@_streamlit_fragment(run_every=2)" in source
    assert "@_streamlit_fragment(run_every=15)" in source


def test_dashboard_only_schedules_required_dynamic_fragments():
    source = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text(encoding="utf-8")

    assert source.count("run_every=") == 5
    assert "@_streamlit_fragment(run_every=2)" in source
    assert "@_streamlit_fragment(run_every=15)" in source
    for name in (
        "_dashboard_setup_fragment",
        "_dashboard_overview_fragment",
        "_dashboard_decisions_fragment",
        "_dashboard_journal_fragment",
        "_dashboard_orders_fragment",
        "_dashboard_errors_fragment",
    ):
        definition = source.index(f"def {name}")
        decorator = source.rfind("@_streamlit_fragment", 0, definition)
        assert "run_every=" not in source[decorator:definition]


def test_dashboard_keeps_live_regions_in_separate_fragments():
    source = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text(encoding="utf-8")
    live_definition = source.index("def _dashboard_live_fragment")
    live_end = source.index("@_streamlit_fragment(run_every=2)", live_definition)
    live_fragment = source[live_definition:live_end]

    assert "_render_status(payload, db_path)" in live_fragment
    assert "_render_executive_summary(payload)" in live_fragment
    assert "_render_live_ai_activity" not in live_fragment
    assert "_render_timeline" not in live_fragment
    assert "_render_risk" not in live_fragment
    assert "_render_benchmarks" not in live_fragment
    assert "def _dashboard_activity_fragment" in source
    assert "def _dashboard_timeline_fragment" in source
    assert "def _dashboard_risk_fragment" in source
    assert "def _dashboard_benchmarks_fragment" in source


def test_dashboard_controls_have_stable_keys_and_are_not_auto_commands():
    source = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text(encoding="utf-8")

    for key in ("refresh_now", "start_scheduler", "stop_scheduler", "run_reconciliation", "run_risk_check", "run_ai_research", "emergency_stop", "resume_manual_halt"):
        assert f'key="{key}"' in source
    assert "enqueue_command" in source
    assert "def _render_static_shell" in source
    assert "SQLiteStore(path, read_only=True)" in source
    assert "@st.cache_data(ttl=1.0, show_spinner=False)" in source
    assert "def _dashboard_fragment" not in source


def test_dashboard_read_only_store_accepts_relative_database_path(tmp_path, monkeypatch):
    database_name = "relative-dashboard.sqlite3"
    database = tmp_path / database_name
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")

    monkeypatch.chdir(tmp_path)
    with SQLiteStore(Path(database_name), read_only=True) as store:
        assert store.get_runtime("service_status") == "ONLINE"
