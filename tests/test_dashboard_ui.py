from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.models import TradeIntent
from src.storage import SQLiteStore


def test_shadow_dashboard_refresh_is_read_only(tmp_path, monkeypatch):
    from tests.test_decision_audit import proposal

    database = tmp_path / "shadow-dashboard.sqlite3"
    with SQLiteStore(database) as store:
        store.save_reduction_shadow("shadow-review", proposal())
        store.save_shadow_observation("shadow-review", {"date": "2026-09-08", "price": 100, "spy": 100, "qqq": 100})
        before = store.reduction_shadows()
    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert any("减仓影子验证" in item.label for item in app.expander)
    with SQLiteStore(database, read_only=True) as store:
        assert store.reduction_shadows() == before
        assert store.pending_commands() == []
        assert store.recent("order_records") == []
        assert store.recent("llm_decisions") == []


def test_refresh_button_only_reruns_view_and_does_not_queue_command(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-ui.sqlite3"
    with SQLiteStore(database) as store:
        store.enqueue_command("EMERGENCY_STOP")
        before = store.pending_commands()

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert "立即刷新" in [button.label for button in app.button]
    app.button[0].click().run(timeout=30)
    assert not app.exception

    with SQLiteStore(database) as store:
        assert store.pending_commands() == before


def test_reconciliation_button_enqueues_without_dashboard_component_error(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-reconciliation.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    reconciliation = next(button for button in app.button if button.label == "立即运行对账")
    reconciliation.click().run(timeout=30)

    assert not app.exception
    with SQLiteStore(database) as store:
        commands = store.pending_commands()
        assert len(commands) == 1
        assert commands[0]["command"] == "RUN_RECONCILIATION"


def test_ai_research_button_disables_and_does_not_enqueue_duplicate(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-research-dedup.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        store.set_runtime("command_worker_id", "worker-current")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    research = next(button for button in app.button if button.label == "立即运行 AI 研究")
    research.click().run(timeout=30)
    app.run(timeout=30)

    assert not app.exception
    active_button = next(button for button in app.button if button.label.startswith("AI 研究进行中"))
    assert active_button.disabled is True
    assert any("完成前无需再次点击" in item.value for item in app.sidebar.info)
    with SQLiteStore(database) as store:
        commands = store.pending_commands()
        assert len(commands) == 1
        command_id, created = store.enqueue_command_unless_active(
            "RUN_AI_RESEARCH",
            active_commands=("RUN_AI_RESEARCH", "RUN_FULL_AI_RESEARCH"),
        )
        assert command_id == commands[0]["id"]
        assert created is False
        assert len(store.pending_commands()) == 1


def test_failed_command_alert_can_be_acknowledged_without_changing_command(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-command-ack.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        command_id = store.enqueue_command("RUN_IMMEDIATE_RISK_CHECK")
        store.complete_command(command_id, error="quote timeout", error_stage="RISK_CHECK")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    acknowledge = next(button for button in app.button if button.label == "确认")
    acknowledge.click().run(timeout=30)
    app.run(timeout=30)

    assert not app.exception
    assert not any(button.label == "确认" for button in app.button)
    with SQLiteStore(database) as store:
        assert store.get_runtime("dashboard.acknowledged_failed_command_id") == command_id
        assert store.command(command_id)["status"] == "FAILED"


def test_dashboard_can_render_repeatedly_without_component_exception(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-repeat.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)
    assert not app.exception

    for _ in range(3):
        app.run(timeout=30)
        assert not app.exception


def test_running_scheduler_is_prominent_and_start_button_is_disabled(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-scheduler-running.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        store.set_runtime("scheduler_status", "RUNNING")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert any("自动调度已启动" in item.value for item in app.success)
    assert any("自动调度：已启动" in item.value for item in app.sidebar.success)
    start = next(button for button in app.button if button.label == "启动调度器")
    stop = next(button for button in app.button if button.label == "停止调度器")
    assert start.disabled is True
    assert stop.disabled is False
    with SQLiteStore(database) as store:
        assert store.pending_commands() == []


def test_stopped_scheduler_is_prominent_and_stop_button_is_disabled(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-scheduler-stopped.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        store.set_runtime("scheduler_status", "STOPPED")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert any("自动调度已停止" in item.value for item in app.warning)
    assert any("自动调度：已停止" in item.value for item in app.sidebar.warning)
    start = next(button for button in app.button if button.label == "启动调度器")
    stop = next(button for button in app.button if button.label == "停止调度器")
    assert start.disabled is False
    assert stop.disabled is True
    with SQLiteStore(database) as store:
        assert store.pending_commands() == []


def test_paper_mode_open_session_shows_automatic_trading_window(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-paper-open.sqlite3"
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "PAPER")
        store.set_runtime("market_session", "OPEN")
        store.set_runtime("scheduler_status", "RUNNING")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert any("Paper 自动交易时段已开启" in item.value for item in app.success)
    assert not any("观察模式｜禁止下单" in item.value for item in app.markdown)


def test_dashboard_keeps_static_tabs_when_backend_database_is_not_ready(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-not-ready.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert any("AI 自动交易运营中心" in element.value for element in app.markdown)
    assert [tab.label for tab in app.tabs] == [
        "设置与系统健康", "账户概览", "AI 实时活动", "系统时间线",
        "风险引擎", "AI 决策记录", "决策日志", "基准与绩效", "订单", "错误",
    ]


def test_dashboard_presents_sol_cross_sectional_decision_without_expanding_raw_evidence(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-comparative.sqlite3"
    dimensions = {
        "momentum_relative_strength": "强", "earnings_trend": "改善", "revenue_eps_quality": "高",
        "valuation": "有吸引力", "analyst_revisions": "上调", "catalyst": "存在", "event_risk": "低",
        "volatility": "中", "liquidity": "高", "market_sector_fit": "良好", "downside_risk": "中",
        "expected_alpha": "正向",
    }
    ranking = [{
        "rank": rank, "symbol": symbol, "alpha_score": 90 - rank * 5, "confidence": 0.75 - rank * 0.03,
        "expected_alpha": 0.09 - rank * 0.01, "key_advantage": f"{symbol} 优势", "key_weakness": f"{symbol} 弱点",
        "why_not_selected": [] if rank == 1 else ["风险调整后预期 Alpha 较低"], "comparison": dimensions,
    } for rank, symbol in enumerate(("MPC", "ANET", "VEEV", "CRM", "ABNB"), start=1)]
    decision = {
        "action": "BUY", "symbol": "MPC", "target_weight": 0.5, "confidence": 0.72,
        "holding_period_days": 20, "expected_excess_vs_spy": 0.06, "expected_excess_vs_qqq": 0.08,
        "thesis": ["MPC 在本次共同评估中拥有最佳风险调整后预期 Alpha。"], "risk_factors": ["周期风险"],
        "thesis_invalidation_conditions": ["相对强度显著跌破趋势"], "evidence_used": ["get_price_history"],
        "top_five": ranking, "selected_symbol": "MPC", "runner_up_symbol": "ANET",
        "selected_alpha_score": 85, "runner_up_alpha_score": 80, "alpha_gap": 5,
        "why_selected": ["估值调整后的动量更优"], "why_selected_over_runner_up": ["MPC 估值风险低于 ANET"],
        "bull_case": {"scenario": "利润率扩张", "expected_return_or_direction": "上涨", "key_assumptions": ["需求稳定"]},
        "base_case": {"scenario": "趋势延续", "expected_return_or_direction": "温和上涨", "key_assumptions": ["无盈利冲击"]},
        "bear_case": {"scenario": "周期反转", "expected_return_or_direction": "下跌", "key_assumptions": ["利润率收缩"]},
        "expected_alpha_basis": ["momentum", "earnings", "valuation"], "estimate_type": "SOL_MODEL_ESTIMATE",
        "confidence_basis": ["多项证据一致"], "confidence_reducers": ["周期不确定性"],
    }
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        store.save_runtime_event("AI_RUN_STARTED", "PIPELINE", "started", run_id="run-compare", metadata={"pipeline": "LUNA_SOL"})
        store.save_runtime_event("LUNA_SCREEN_COMPLETED", "LUNA", "completed", run_id="run-compare", metadata={"candidate_count": 18})
        store.save_runtime_event("SOL_RESEARCH_STARTED", "SOL", "started", run_id="run-compare", metadata={"candidate_symbols": [item["symbol"] for item in ranking]})
        store.save_runtime_event("SOL_DECISION_COMPLETED", "SOL", "BUY MPC", run_id="run-compare", decision_id="decision-compare", symbol="MPC", metadata={"decision": decision})

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    headings = [item.value for item in app.subheader]
    assert "最终选择" in headings
    assert "与主要竞争者比较" in headings
    assert "Top 5 横向排名" in headings
    assert "上涨逻辑" in headings
    assert "什么情况下卖出" in headings
    assert any("Sol 模型预期" in item.value for item in app.markdown)
    assert "查看完整研究依据" in [item.label for item in app.expander]


def test_dashboard_presents_persisted_historical_comparison_without_invented_ranking(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-historical-comparison.sqlite3"
    candidates = ["VEEV", "CRM", "ABNB", "PANW", "MPC"]
    comparison = "Comparative screen: VEEV and CRM had stronger recent returns; MPC had better valuation support."
    intent = TradeIntent(
        action="BUY", symbol="MPC", target_weight=0.5, confidence=0.72,
        holding_period_days=20, expected_alpha_vs_spy=0.06, expected_alpha_vs_qqq=0.08,
        thesis=["MPC has the best risk-adjusted opportunity in this screen."],
        risk_factors=["Cyclical margin risk"], invalidation_conditions=["Relative strength breaks"],
        evidence_used=[comparison], decision_id="decision-historical-ui",
    )
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "PAPER")
        store.save_decision(intent, pipeline_metadata={
            "pipeline": "LUNA_SOL", "candidate_symbols": candidates,
            "final_decision": intent.model_dump(mode="json"),
        })
        store.save_runtime_event(
            "SOL_DECISION_COMPLETED", "SOL", "BUY MPC", run_id="run-historical-ui",
            decision_id=intent.decision_id, symbol="MPC", metadata={"decision": intent.model_dump(mode="json")},
        )

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    headings = [item.value for item in app.subheader]
    assert "同场候选横向比较" in headings
    assert "Top 5 横向排名" not in headings
    assert any("未保存逐项评分" in item.value for item in app.caption)
    assert any(comparison.removeprefix("Comparative screen: ") in item.value for item in app.markdown)


def test_decision_history_keeps_comparison_visible_after_a_newer_failed_run(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-historical-decision.sqlite3"
    candidates = ["VEEV", "CRM", "MPC"]
    comparison = "Comparative screen: MPC had more favorable valuation support."
    intent = TradeIntent(
        action="BUY", symbol="MPC", target_weight=0.5, confidence=0.72,
        holding_period_days=20, expected_alpha_vs_spy=0.06, expected_alpha_vs_qqq=0.08,
        thesis=["MPC was selected."], risk_factors=["Cyclical risk"],
        invalidation_conditions=["Relative strength breaks"], evidence_used=[comparison],
        decision_id="decision-visible-after-failure",
    )
    with SQLiteStore(database) as store:
        store.save_decision(intent, pipeline_metadata={
            "pipeline": "LUNA_SOL", "candidate_symbols": candidates,
            "final_decision": intent.model_dump(mode="json"),
        })
        store.save_runtime_event("AI_RUN_STARTED", "PIPELINE", "started", run_id="newer-failed-run")
        store.save_runtime_event("AI_RUN_FAILED", "PIPELINE", "upstream unavailable", run_id="newer-failed-run")

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    assert any("decision-visible-after-failure" in item.label for item in app.expander)
    assert any(item.value == "**同场候选横向比较**" for item in app.markdown)
    assert any("数据库中真实保存" in item.value for item in app.caption)


def test_dashboard_presents_structured_sol_deep_research(tmp_path, monkeypatch):
    database = tmp_path / "dashboard-deep-research.sqlite3"
    ranking = {"ranking": [
        {"rank": rank, "symbol": symbol, "preliminary_alpha_score": 95 - rank, "confidence": 0.8,
         "key_strengths": ["盈利修正"], "key_weaknesses": ["估值风险"], "research_priority": "HIGH"}
        for rank, symbol in enumerate(("MPC", "ANET", "VEEV", "CRM", "ABNB"), start=1)
    ], "missing_data_sources": ["Crack spread data unavailable"]}
    top_three = [{
        "symbol": symbol,
        "business_driver_summary": f"{symbol} 因果研究",
        "causal_drivers": [{"driver": "盈利修正", "category": "COMPANY", "estimated_importance": "HIGH", "evidence_quality": "HIGH", "priced_in": "PARTIAL", "mechanism": "盈利预期上调"}],
        "return_attribution": {"company_specific_alpha": "MEDIUM", "industry_beta": "MEDIUM", "energy_beta": "UNKNOWN", "broad_market_beta": "LOW", "explanation": "部分公司 Alpha"},
        "pricing_assessment": "FAIRLY_PRICED", "why_now": ["新盈利修正"], "key_risks": ["预期反转"],
        "thesis_invalidation_conditions": ["盈利修正转负"], "missing_data": ["Crack spread data unavailable"] if symbol == "MPC" else [],
    } for symbol in ("MPC", "ANET", "VEEV")]
    final = {
        "action": "BUY", "selected_symbol": "MPC", "rank": 1, "runner_up": "ANET", "alpha_score": 92,
        "runner_up_score": 87, "alpha_gap": 5, "confidence": 0.72, "confidence_basis": ["多维证据一致"],
        "confidence_reducers": ["周期性"], "expected_alpha_vs_spy": None, "expected_alpha_vs_qqq": None,
        "expected_alpha_basis": ["相对表现与盈利修正"], "estimate_type": "SOL_MODEL_ESTIMATE", "thesis_horizon_days": 20,
        "target_weight": 0.5, "primary_alpha_drivers": ["盈利修正"], "company_specific_drivers": ["执行改善"],
        "industry_drivers": ["炼化周期"], "macro_drivers": ["需求"], "geopolitical_drivers": ["UNKNOWN"],
        "priced_in_assessment": "FAIRLY_PRICED",
        "bull_case": {"scenario": "修正持续", "expected_direction": "STRONG_UPSIDE", "key_assumptions": ["盈利上调"]},
        "base_case": {"scenario": "温和延续", "expected_direction": "MODERATE_UPSIDE", "key_assumptions": ["需求稳定"]},
        "bear_case": {"scenario": "周期反转", "expected_direction": "MODERATE_DOWNSIDE", "key_assumptions": ["利润收缩"]},
        "thesis_invalidation_conditions": ["盈利修正转负"], "why_selected": ["因果证据最强"],
        "why_not_runner_up": ["催化剂较弱"], "why_not_rank3": ["Alpha 空间较低"],
        "pairwise_comparisons": [
            {"selected_symbol": "MPC", "alternative_symbol": "ANET", "selected_advantages": ["估值"], "alternative_advantages": ["成长"], "decision_reason": "风险调整后更优"},
            {"selected_symbol": "MPC", "alternative_symbol": "VEEV", "selected_advantages": ["催化剂"], "alternative_advantages": ["稳定性"], "decision_reason": "近期驱动更明确"},
        ],
        "strongest_bear_argument": "利好可能已被计价", "what_would_make_me_change_my_mind": ["修正转负"],
        "missing_data": ["Crack spread data unavailable"], "tool_coverage": ["price", "fundamentals"],
        "evidence_used": ["SEC", "analyst revisions"], "risk_factors": ["周期反转"],
    }
    deep = {"ranking": ranking, "top_three": top_three, "adversarial_review": {"reviews": []}, "final_decision": final,
            "tool_coverage": {"available_sources": ["price", "fundamentals"], "missing_sources": ["Reliable current crack spread series"]}}
    with SQLiteStore(database) as store:
        store.set_runtime("service_status", "ONLINE")
        store.set_runtime("execution_mode", "OBSERVE")
        store.save_runtime_event("AI_RUN_STARTED", "PIPELINE", "started", run_id="run-deep", metadata={"pipeline": "LUNA_SOL"})
        store.save_runtime_event("SOL_RESEARCH_STARTED", "SOL", "started", run_id="run-deep", metadata={"candidate_symbols": ["MPC", "ANET", "VEEV", "CRM", "ABNB"]})
        store.save_runtime_event("SOL_IC_DECISION_COMPLETED", "SOL", "BUY MPC", run_id="run-deep", metadata={"decision": final, "deep_research": deep})

    monkeypatch.setenv("DATABASE_PATH", str(database))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run(timeout=30)

    assert not app.exception
    headings = [item.value for item in app.subheader]
    assert "Top 5 横向排名" in headings
    assert "Top 3 因果研究" in headings
    assert "市场是否已经 Price In" in headings
    assert "最强反方观点" in headings
    assert "缺失研究数据" in headings
    assert not any("chain-of-thought" in str(item.value).lower() for item in app.markdown)
