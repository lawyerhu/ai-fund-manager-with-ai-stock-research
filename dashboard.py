from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import env, load_config
from src.dashboard_data import build_dashboard_payload, dashboard_refresh_seconds
from src.storage import SQLiteStore


st.set_page_config(page_title="AI 自动交易运营中心", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .block-container { max-width: 1500px; padding-top: 1rem; }
    [data-testid="stMetric"] { background: #f7f8fa; border: 1px solid #d9dee7; padding: 0.75rem; min-height: 92px; }
    [data-testid="stMetricLabel"] { color: #52606d; }
    [data-testid="stMetricValue"] { color: #17202a; }
    .status-strip { border-bottom: 1px solid #d9dee7; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    .ops-title { letter-spacing: 0; margin-bottom: 0.1rem; }
    .observe-banner { border: 1px solid #dc2626; background: #fff1f2; color: #991b1b; padding: 0.8rem 1rem; font-weight: 700; }
    .decision-hero { border-left: 5px solid #16a34a; background: #f0fdf4; padding: 1rem 1.2rem; margin: 0.5rem 0 1rem; }
    .decision-hero h2 { margin: 0; color: #166534; }
    .muted { color: #52606d; }
    .stage-running { color: #fbbf24; font-weight: 700; }
    .stage-complete { color: #34d399; font-weight: 700; }
    .stage-error, .stage-blocked { color: #f87171; font-weight: 700; }
    .stage-waiting { color: #9ca3af; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _pct(value):
    return "暂无" if value is None else f"{float(value) * 100:.2f}%"


def _money(value):
    return "暂无" if value is None else f"${float(value):,.2f}"


def _local_time(value):
    if not value:
        return "暂无"
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def _list_text(values, empty="暂无"):
    values = [str(value) for value in (values or []) if value]
    return "；".join(values) if values else empty


def _render_executive_summary(payload):
    status = payload.get("status", {})
    activity = payload.get("ai_activity", {})
    decision = activity.get("final_decision") or {}
    risk = activity.get("stages", {}).get("Risk Engine", {}) or {}
    execution = activity.get("stages", {}).get("Execution", {}) or {}
    symbol = decision.get("selected_symbol") or decision.get("symbol") or "现金"
    action_map = {"BUY": "买入", "HOLD": "继续持有", "REDUCE": "减仓", "SWITCH": "换仓", "CASH": "保持现金", "HOLD_CASH": "保持现金", "EXIT_TO_CASH": "退出到现金", "REPLACE": "换仓", "SELL": "卖出"}
    action = action_map.get(decision.get("action"), decision.get("action") or "暂无决策")
    monitoring = status.get("broker_monitor") == "RUNNING" and bool(status.get("command_worker_heartbeat"))
    monitor_text = "持续监控中" if monitoring else "监控未启动"
    if status.get("safe_mode") or status.get("system") == "ERROR":
        monitor_text = "监控异常"
    st.subheader("当前状态")
    state_cols = st.columns(4)
    state_cols[0].metric("当前决策", f"{action} {symbol}")
    state_cols[1].metric("系统状态", monitor_text)
    state_cols[2].metric("最后更新", _local_time(status.get("last_reconciliation") or activity.get("started_at")))
    state_cols[3].metric("风险状态", status.get("risk_state", "暂无"))
    if status.get("execution_mode") == "OBSERVE":
        st.markdown('<div class="observe-banner">观察模式 · 禁止下单 · 仅展示 WOULD_* 结果</div>', unsafe_allow_html=True)
    if not decision:
        st.info("当前没有已完成的最终交易决策。")
        _render_position_monitor(payload, decision)
        return
    st.markdown(f'<div class="decision-hero"><h2>最终交易决策：{action} {symbol}</h2><div class="muted">模型：{decision.get("model_name") or activity.get("sol_model") or "暂无"} · 决策编号：{decision.get("decision_id") or "暂无"}</div></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    cols[0].metric("AI 建议仓位", _pct(decision.get("target_weight")))
    cols[1].metric("风控批准仓位", _pct(risk.get("approved_weight")))
    cols[2].metric("账户权益", _money(payload.get("account", {}).get("equity")))
    approved = risk.get("approved_weight")
    equity = payload.get("account", {}).get("equity")
    cols[3].metric("计划投入金额", _money(float(equity) * float(approved)) if equity is not None and approved is not None else "暂无")
    horizon = decision.get("thesis_horizon_days") or decision.get("holding_period_days")
    cols[4].metric("计划持有", f'{horizon} 天' if horizon else "暂无")
    cols[5].metric("置信度", _pct(decision.get("confidence")))
    alpha = st.columns(2)
    alpha[0].metric("预期超额收益（相对 SPY）", _pct(decision.get("expected_alpha_vs_spy", decision.get("expected_excess_vs_spy"))))
    alpha[1].metric("预期超额收益（相对 QQQ）", _pct(decision.get("expected_alpha_vs_qqq", decision.get("expected_excess_vs_qqq"))))
    _render_sol_summary(activity, decision)
    _render_position_monitor(payload, decision)
    _render_risk_summary(risk, decision, payload)
    _render_luna_summary(activity)


def _render_sol_summary(activity, decision):
    sol = activity.get("stages", {}).get("Sol", {}) or {}
    st.subheader("Sol 深度研究")
    deep_research = activity.get("deep_research") or {}
    if deep_research:
        _render_sol_deep_research(deep_research, decision, sol)
        return
    ranking = decision.get("top_five") or []
    if not ranking:
        st.markdown("**核心结论**")
        st.write(_list_text(decision.get("thesis"), "暂无研究结论"))
        st.markdown("**主要风险**")
        st.write(_list_text(decision.get("risk_factors"), "暂无"))
        comparison = activity.get("comparative_context") or {}
        candidates = comparison.get("candidate_symbols") or []
        notes = comparison.get("comparative_notes") or []
        if candidates or notes:
            st.subheader("同场候选横向比较")
            if candidates:
                st.dataframe(
                    pd.DataFrame([{"候选序号": index, "标的": symbol} for index, symbol in enumerate(candidates, start=1)]),
                    width="stretch", hide_index=True, key="sol_historical_candidate_comparison",
                )
            for note in notes:
                st.write(note.removeprefix("Comparative screen:").strip())
            st.caption("本次历史运行保存了候选范围和比较结论，但未保存逐项评分；下一次研究将显示完整 Top 5 结构化横向排名。")
        else:
            st.caption("这是历史决策，尚未包含同场候选横向比较数据。")
        return

    selected = decision.get("selected_symbol") or decision.get("symbol") or "暂无"
    candidate_count = (activity.get("stages", {}).get("Luna", {}) or {}).get("candidate_count") or len(ranking)
    st.subheader("最终选择")
    choice = st.columns(5)
    choice[0].metric("标的", selected)
    choice[1].metric("候选中排名", f"#1 / {candidate_count}")
    choice[2].metric("Sol Alpha 评分", decision.get("selected_alpha_score", "暂无"))
    choice[3].metric("第二名", decision.get("runner_up_symbol") or "暂无")
    choice[4].metric("Alpha 分差", decision.get("alpha_gap", "暂无"))
    st.markdown("**核心结论**")
    st.write(_list_text(decision.get("thesis"), "暂无研究结论"))
    st.markdown(f"**为什么选择 {selected}**")
    for reason in decision.get("why_selected") or []:
        st.write(f"- {reason}")

    st.subheader("与主要竞争者比较")
    selected_row = ranking[0]
    for competitor in ranking[1:4]:
        st.markdown(f"**{selected} vs #{competitor.get('rank')} {competitor.get('symbol')}**")
        comparison = st.columns(3)
        comparison[0].write(f"**{selected} 优势**\n\n{selected_row.get('key_advantage') or 'UNKNOWN'}")
        comparison[1].write(f"**{competitor.get('symbol')} 优势**\n\n{competitor.get('key_advantage') or 'UNKNOWN'}")
        reasons = decision.get("why_selected_over_runner_up") if competitor.get("rank") == 2 else competitor.get("why_not_selected")
        comparison[2].write(f"**最终取舍**\n\n{_list_text(reasons, 'UNKNOWN')}")

    st.subheader("Top 5 横向排名")
    factor_labels = {
        "momentum_relative_strength": "动量/相对强度", "earnings_trend": "盈利趋势",
        "revenue_eps_quality": "营收/EPS 质量", "valuation": "估值", "analyst_revisions": "分析师修正",
        "catalyst": "催化剂", "event_risk": "事件风险", "volatility": "波动率", "liquidity": "流动性",
        "market_sector_fit": "市场/行业匹配", "downside_risk": "下行风险", "expected_alpha": "预期 Alpha",
    }
    table_rows = []
    for item in ranking:
        row = {
            "排名": item.get("rank"), "标的": item.get("symbol"), "Alpha 评分": item.get("alpha_score"),
            "置信度": _pct(item.get("confidence")), "预期 Alpha": _pct(item.get("expected_alpha")),
            "关键优势": item.get("key_advantage"), "关键弱点": item.get("key_weakness"),
            "未被选中原因": _list_text(item.get("why_not_selected"), "—" if item.get("rank") == 1 else "UNKNOWN"),
        }
        dimensions = item.get("comparison") or {}
        row.update({label: dimensions.get(key) or "UNKNOWN" for key, label in factor_labels.items()})
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True, key="sol_comparative_top_five")

    st.subheader("上涨逻辑")
    scenarios = st.columns(2)
    for column, (label, scenario) in zip(scenarios, (("Bull Case", decision.get("bull_case") or {}), ("Base Case", decision.get("base_case") or {}))):
        column.markdown(f"**{label}**")
        column.write(scenario.get("scenario") or "UNKNOWN")
        column.caption(f"方向/收益：{scenario.get('expected_return_or_direction') or 'UNKNOWN'} · 关键假设：{_list_text(scenario.get('key_assumptions'), 'UNKNOWN')}")
    st.markdown("**主要风险 · Bear Case**")
    bear = decision.get("bear_case") or {}
    st.write(bear.get("scenario") or "UNKNOWN")
    st.caption(f"方向/收益：{bear.get('expected_return_or_direction') or 'UNKNOWN'} · 关键假设：{_list_text(bear.get('key_assumptions'), 'UNKNOWN')}")

    st.subheader("什么情况下卖出")
    invalidations = decision.get("thesis_invalidation_conditions") or decision.get("invalidation_conditions") or []
    for condition in invalidations:
        st.write(f"- {condition}")

    st.subheader("持有计划")
    st.write(f"计划持有约 {decision.get('holding_period_days', '暂无')} 个交易日，但不是机械到期卖出；上述失效条件或持续监控结果可触发提前复核。")
    confidence = st.columns(2)
    confidence[0].metric("Sol 置信度", _pct(decision.get("confidence")))
    confidence[0].write(f"**置信度依据**\n\n{_list_text(decision.get('confidence_basis'), 'UNKNOWN')}")
    confidence[1].write(f"**为什么不是更高置信度**\n\n{_list_text(decision.get('confidence_reducers'), 'UNKNOWN')}")

    st.subheader("预期 Alpha")
    st.markdown("**Sol 模型预期，不是统计回测 Alpha。**")
    alpha = st.columns(2)
    alpha[0].metric("相对 SPY", _pct(decision.get("expected_excess_vs_spy", decision.get("expected_alpha_vs_spy"))))
    alpha[1].metric("相对 QQQ", _pct(decision.get("expected_excess_vs_qqq", decision.get("expected_alpha_vs_qqq"))))
    st.write(f"**判断依据**：{_list_text(decision.get('expected_alpha_basis'), 'UNKNOWN')} · 类型：{decision.get('estimate_type') or 'SOL_MODEL_ESTIMATE'}")

    with st.expander("查看完整研究依据"):
        st.write({"模型证据标签": decision.get("evidence_used") or [], "主要风险": decision.get("risk_factors") or []})
        if sol.get("tool_calls"):
            st.write({"工具调用": sol.get("tool_calls")})
        if sol.get("tool_results"):
            st.write({"工具结果": sol.get("tool_results")})


def _render_sol_deep_research(research, decision, sol):
    ranking = (research.get("ranking") or {}).get("ranking") or []
    top_three = research.get("top_three") or []
    final = research.get("final_decision") or decision
    selected = final.get("selected_symbol") or "现金"

    st.subheader("最终选择")
    choice = st.columns(6)
    choice[0].metric("标的", selected)
    choice[1].metric("排名", f"#{final.get('rank')}" if final.get("rank") else "现金")
    choice[2].metric("Alpha 评分", final.get("alpha_score", "暂无"))
    choice[3].metric("第二名", final.get("runner_up") or "暂无")
    choice[4].metric("Alpha 分差", final.get("alpha_gap", "暂无"))
    choice[5].metric("置信度", _pct(final.get("confidence")))
    st.markdown(f"**为什么是 {selected}**")
    for reason in (final.get("why_selected") or [])[:6]:
        st.write(f"- {reason}")

    st.subheader("Top 5 横向排名")
    st.dataframe(pd.DataFrame([{
        "排名": item.get("rank"),
        "标的": item.get("symbol"),
        "Alpha 评分": item.get("preliminary_alpha_score"),
        "置信度": _pct(item.get("confidence")),
        "核心优势": _list_text(item.get("key_strengths"), "UNKNOWN"),
        "核心风险": _list_text(item.get("key_weaknesses"), "UNKNOWN"),
        "研究优先级": item.get("research_priority") or "UNKNOWN",
    } for item in ranking]), width="stretch", hide_index=True, key="sol_deep_top_five")

    st.subheader("Top 3 因果研究")
    for item in top_three:
        symbol = item.get("symbol") or "UNKNOWN"
        st.markdown(f"**{symbol} · {item.get('business_driver_summary') or 'UNKNOWN'}**")
        attribution = item.get("return_attribution") or {}
        st.caption(
            f"公司 Alpha：{attribution.get('company_specific_alpha', 'UNKNOWN')} · "
            f"行业 Beta：{attribution.get('industry_beta', 'UNKNOWN')} · "
            f"能源 Beta：{attribution.get('energy_beta', 'UNKNOWN')} · "
            f"大盘 Beta：{attribution.get('broad_market_beta', 'UNKNOWN')}"
        )
        for driver in (item.get("causal_drivers") or [])[:5]:
            st.write(
                f"- {driver.get('driver', 'UNKNOWN')}：{driver.get('category', 'UNKNOWN')} / "
                f"影响 {driver.get('estimated_importance', 'UNKNOWN')} / 证据 {driver.get('evidence_quality', 'UNKNOWN')} / "
                f"计价 {driver.get('priced_in', 'UNKNOWN')}。{driver.get('mechanism', '')}"
            )
        scenario_columns = st.columns(3)
        for column, (label, scenario) in zip(scenario_columns, (
            ("Bull", item.get("bull_case") or {}), ("Base", item.get("base_case") or {}), ("Bear", item.get("bear_case") or {}),
        )):
            column.caption(
                f"{label}: {scenario.get('scenario') or 'UNKNOWN'} · "
                f"{scenario.get('expected_direction') or 'UNKNOWN'} · "
                f"{scenario.get('probability_confidence') or 'UNKNOWN'}"
            )

    st.subheader("Alpha 来自哪里")
    driver_columns = st.columns(4)
    for column, (label, key) in zip(driver_columns, (
        ("公司因素", "company_specific_drivers"), ("行业因素", "industry_drivers"),
        ("宏观因素", "macro_drivers"), ("地缘政治", "geopolitical_drivers"),
    )):
        column.markdown(f"**{label}**")
        column.write(_list_text(final.get(key), "UNKNOWN"))

    st.subheader("Top 3 直接比较")
    for comparison in final.get("pairwise_comparisons") or []:
        alternative = comparison.get("alternative_symbol") or "UNKNOWN"
        st.markdown(f"**{comparison.get('selected_symbol') or selected} vs {alternative}**")
        columns = st.columns(3)
        columns[0].write(f"**{selected} 优势**\n\n{_list_text(comparison.get('selected_advantages'), 'UNKNOWN')}")
        columns[1].write(f"**{alternative} 优势**\n\n{_list_text(comparison.get('alternative_advantages'), 'UNKNOWN')}")
        columns[2].write(f"**最终取舍**\n\n{comparison.get('decision_reason') or 'UNKNOWN'}")

    st.subheader("市场是否已经 Price In")
    st.write(final.get("priced_in_assessment") or "UNCERTAIN")

    st.subheader("Bull / Base / Bear")
    scenario_columns = st.columns(3)
    for column, (label, scenario) in zip(scenario_columns, (
        ("Bull", final.get("bull_case") or {}), ("Base", final.get("base_case") or {}), ("Bear", final.get("bear_case") or {}),
    )):
        column.markdown(f"**{label}**")
        column.write(scenario.get("scenario") or "UNKNOWN")
        column.caption(f"方向：{scenario.get('expected_direction') or 'UNKNOWN'} · 假设：{_list_text(scenario.get('key_assumptions'), 'UNKNOWN')}")

    st.subheader("最强反方观点")
    st.warning(final.get("strongest_bear_argument") or "UNKNOWN")
    st.caption(f"改变判断的证据：{_list_text(final.get('what_would_make_me_change_my_mind'), 'UNKNOWN')}")

    st.subheader("什么情况下卖出")
    for condition in final.get("thesis_invalidation_conditions") or []:
        st.write(f"- {condition}")

    st.subheader("预期 Alpha")
    st.markdown("**Sol 模型估计，不是统计回测 Alpha；证据不足时允许为空。**")
    alpha = st.columns(2)
    alpha[0].metric("相对 SPY", _pct(final.get("expected_alpha_vs_spy")))
    alpha[1].metric("相对 QQQ", _pct(final.get("expected_alpha_vs_qqq")))
    st.caption(f"依据：{_list_text(final.get('expected_alpha_basis'), 'UNKNOWN')} · {final.get('estimate_type') or 'SOL_MODEL_ESTIMATE'}")

    coverage = research.get("tool_coverage") or {}
    missing = list(dict.fromkeys([
        *((research.get("ranking") or {}).get("missing_data_sources") or []),
        *(coverage.get("missing_sources") or []),
        *(final.get("missing_data") or []),
        *(value for item in top_three for value in (item.get("missing_data") or [])),
    ]))
    st.subheader("缺失研究数据")
    if missing:
        for value in missing:
            st.write(f"- {value}")
    else:
        st.write("无已知缺失项")

    with st.expander("查看完整研究依据"):
        st.write({"可用数据源": coverage.get("available_sources") or [], "证据标签": final.get("evidence_used") or [], "风险": final.get("risk_factors") or []})
        st.write({"Sol 工具调用数": len(sol.get("tool_calls") or []), "工具调用": sol.get("tool_calls") or []})


def _render_position_monitor(payload, decision):
    from src.decision_audit import shadow_summary
    shadows = (payload.get("position_manager") or {}).get("reduction_shadows", [])
    with st.expander("减仓影子验证（模拟，非实际成交）"):
        st.caption("比较固定股票/现金组合：建议后的首个观测收盘价起步，扣模拟手续费与滑点；不含股息、现金利息及共同LLM成本。回撤仅基于已观测收盘，不代表完整日内回撤。")
        if not shadows:
            st.info("暂无前瞻记录；历史决策不补造结果。")
        for row in shadows[:20]:
            proposal = row["proposal"]
            summary = shadow_summary(proposal, row["observations"])
            st.write({"复核编号": row["review_id"], "标的": proposal["symbol"],
                      "固定观察天数（日历日）": proposal["horizon_days"],
                      "状态": "观察结束" if row["status"] == "COMPLETE" else "等待首个有效收盘" if not row["observations"] else "观察中",
                      "开始日期": summary.get("start_date"), "最新日期": summary.get("last_date"),
                      "有效观测数": summary.get("observation_count", 0),
                      "保持原仓位收益": _pct(summary.get("hold_return")),
                      "减仓后净收益": _pct(summary.get("reduce_net_return")),
                      "模拟成本占比": _pct(summary.get("simulation_cost_fraction")),
                      "减仓收益差（负值代表错过上涨）": _pct(summary.get("reduction_benefit")),
                      "原仓位已观测最大回撤": _pct(summary.get("hold_observed_max_drawdown")),
                      "减仓后已观测最大回撤": _pct(summary.get("reduce_observed_max_drawdown")),
                      "原仓位超额 vs SPY": _pct(summary.get("hold_excess_vs_spy")),
                      "减仓后净超额 vs SPY": _pct(summary.get("reduce_net_excess_vs_spy")),
                      "原仓位超额 vs QQQ": _pct(summary.get("hold_excess_vs_qqq")),
                      "减仓后净超额 vs QQQ": _pct(summary.get("reduce_net_excess_vs_qqq"))})
    account = payload.get("account", {})
    positions = account.get("positions") or []
    st.subheader("持仓与持续监控")
    if not positions:
        st.info("当前空仓。系统会继续监控新的有效决策。")
        return
    position = positions[0]
    managed = (payload.get("position_manager") or {}).get("position") or {}
    review = (payload.get("position_manager") or {}).get("latest_review") or {}
    with st.expander("为什么调整仓位：新旧证据与调整依据"):
        audit = review.get("sizing_audit")
        if audit:
            context = review.get("comparison_context") or {}
            st.write({"当前仓位（复核时）": context.get("current_weight"),
                      "上次建议仓位": context.get("previous_target_weight"),
                      "本轮委员会建议": (context.get("committee_decision") or {}).get("target_weight"),
                      "持仓复核建议": review.get("target_weight"),
                      "新增负面证据": audit.get("new_negative_evidence"),
                      "未改变的数据缺失": audit.get("unchanged_missing_data"),
                      "投资逻辑变化": audit.get("thesis_changes"),
                      "仓位依据": audit.get("weight_basis"),
                      "额外调整理由": audit.get("incremental_reason"),
                      "风险预算依据": audit.get("risk_budget_basis"),
                      "下行情景": audit.get("downside_scenario"),
                      "证据来源": audit.get("evidence_refs")})
            st.caption("Sol模型判断，不是统计胜率或Risk Engine批准；来源变化本身不等于负面变化。")
        else:
            st.info("该历史复核没有结构化仓位依据；不会自动补造。")
    trigger = (payload.get("position_manager") or {}).get("latest_trigger") or {}
    cols = st.columns(5)
    cols[0].metric("持仓标的", position.get("symbol", "暂无"))
    cols[1].metric("当前价格", _money(position.get("market_price")))
    cols[2].metric("持仓市值", _money(position.get("market_value")))
    cols[3].metric("当前收益", _money(account.get("unrealized_pnl")))
    cols[4].metric("AI 当前观点", review.get("action") or {"BUY": "继续持有", "HOLD": "继续持有", "REDUCE": "减仓", "CASH": "卖出"}.get(decision.get("action"), "观察"))
    if managed:
        details = st.columns(5)
        details[0].metric("Thesis 状态", managed.get("current_thesis_status", "暂无"))
        details[1].metric("已持有天数", managed.get("days_held", "暂无"))
        details[2].metric("原计划天数", managed.get("original_thesis_horizon_days", "暂无"))
        details[3].metric("最新预计天数", managed.get("current_thesis_horizon_days", "暂无"))
        details[4].metric("Replacement Gap", review.get("replacement_gap", "暂无"))
        st.write({
            "最近复评": managed.get("last_sol_review_at"),
            "下次复评": managed.get("next_scheduled_review_at"),
            "最近触发": trigger.get("event_type") or managed.get("last_trigger"),
            "持续监控": managed.get("monitoring_status") == "ACTIVE",
            "最新 Sol 观点": review.get("reason") or managed.get("latest_thesis"),
        })
    st.caption("持仓监控数据来自后端账户状态；尚未成交的计划值不会显示为实际成交值。")


def _render_risk_summary(risk, decision, payload):
    st.subheader("风险审批")
    approved = risk.get("approved")
    st.success("风控通过" if approved else "风控拒绝" if approved is False else "等待风控结果")
    cols = st.columns(3)
    cols[0].metric("AI 建议仓位", _pct(decision.get("target_weight")))
    cols[1].metric("风控批准仓位", _pct(risk.get("approved_weight")))
    equity = payload.get("account", {}).get("equity")
    weight = risk.get("approved_weight")
    cols[2].metric("批准金额", _money(float(equity) * float(weight)) if equity is not None and weight is not None else "暂无")
    if risk.get("reason"):
        st.caption(f"调整原因：{risk['reason']}")


def _render_luna_summary(activity):
    luna = activity.get("stages", {}).get("Luna", {}) or {}
    st.subheader("Luna 初筛")
    cols = st.columns(4)
    cols[0].metric("股票池", luna.get("universe_size", "暂无"))
    cols[1].metric("候选股", luna.get("candidate_count", "暂无"))
    cols[2].metric("模型", luna.get("model") or activity.get("luna_model") or "暂无")
    cols[3].metric("耗时", f"{float(luna.get('usage', {}).get('latency_ms', 0)) / 1000:.1f} 秒" if luna.get("usage", {}).get("latency_ms") is not None else "暂无")
    candidates = luna.get("candidates") or []
    if candidates:
        with st.expander(f"查看 {len(candidates)} 只候选股"):
            st.dataframe(pd.DataFrame([{"标的": item.get("symbol"), "入选理由": item.get("rationale"), "利好信号": _list_text(item.get("positive_signals")), "主要风险": _list_text(item.get("risks"))} for item in candidates]), width="stretch", hide_index=True, key="summary_luna_candidates")


def _queue(store, command: str):
    if command == "RUN_AI_RESEARCH":
        command_id, created = store.enqueue_command_unless_active(
            command,
            active_commands=("RUN_AI_RESEARCH", "RUN_FULL_AI_RESEARCH"),
        )
        st.session_state["submitted_ai_research_command_id"] = command_id
        if created:
            st.success(f"AI 研究已提交（#{command_id}）。研究完成前无需再次点击。")
        else:
            st.warning(f"AI 研究任务 #{command_id} 已在排队或运行，本次没有重复提交。")
    else:
        command_id = store.enqueue_command(command)
        st.success(f"命令已排队：{command}（#{command_id}）")
    _cached_dashboard_payload.clear()


def _scheduler_display_status(value):
    normalized = str(value or "UNKNOWN").upper()
    if normalized == "RUNNING":
        return "已启动"
    if normalized == "STOPPED":
        return "已停止"
    return "异常/未知"


def _render_status(payload, database_path=None):
    status = payload["status"]
    scheduler_status = str(status.get("scheduler") or "UNKNOWN").upper()
    st.markdown('<div class="status-strip">', unsafe_allow_html=True)
    values = [
        ("系统", status.get("system", "未知")),
        ("IBKR Paper", status.get("broker_source", "未知")),
        ("执行模式", status.get("execution_mode", "未知")),
        ("TWS", status.get("tws", status.get("broker", "未知"))),
        ("CC Switch", status.get("cc_switch", status.get("llm", "未知"))),
        ("Luna", status.get("luna", "未知")),
        ("Sol", status.get("sol", "未知")),
        ("行情数据", status.get("market_data", "未知")),
        ("自动调度", _scheduler_display_status(scheduler_status)),
        ("命令执行器", status.get("command_worker", "未知")),
        ("经纪商监控", status.get("broker_monitor", "未知")),
        ("风险状态", status.get("risk_state", "未知")),
        ("当前持仓", status.get("current_position", "现金")),
        ("最后 AI 决策", _local_time(status.get("last_ai_decision"))),
    ]
    for offset in range(0, len(values), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, values[offset:offset + 4]):
            column.metric(label, value)
    operations = payload.get("operations", {}) or {}
    next_research = _local_time(operations.get("next_full_research"))
    risk_interval = operations.get("risk_interval_minutes", 15)
    if scheduler_status == "RUNNING":
        st.success(
            f"自动调度已启动｜下一次周度研究：{next_research}（本机时间）｜"
            f"定时风控：美股交易时段每 {risk_interval} 分钟"
        )
    elif scheduler_status == "STOPPED":
        st.warning("自动调度已停止｜不会自动创建周度研究或定时风控任务；手工命令仍可由后台执行。")
    else:
        st.error(f"自动调度状态异常：{scheduler_status}｜自动任务是否会运行尚未确认。")
    latest_command = status.get("latest_command") or {}
    if latest_command:
        command_text = f"#{latest_command.get('id')} {latest_command.get('command')} | {latest_command.get('status')}"
        if latest_command.get("error_message") or latest_command.get("error"):
            command_text += f" | {latest_command.get('error_message') or latest_command.get('error')}"
        command_status = str(latest_command.get("status") or "").upper()
        if command_status == "SUCCEEDED":
            st.success(f"命令已完成：{command_text}")
        elif command_status in {"QUEUED", "CLAIMED", "RUNNING"}:
            st.info(f"命令处理中：{command_text}")
        elif command_status == "FAILED":
            acknowledged_id = status.get("acknowledged_failed_command_id")
            command_id = latest_command.get("id")
            if command_id != acknowledged_id:
                alert_column, action_column = st.columns([8, 1])
                acknowledged = action_column.button(
                    "确认",
                    key=f"ack_failed_command_{command_id}",
                    help="隐藏这条历史失败提醒；不会重试或删除命令记录。",
                )
                if acknowledged and database_path is not None:
                    with SQLiteStore(database_path) as store:
                        store.set_runtime("dashboard.acknowledged_failed_command_id", command_id)
                    _cached_dashboard_payload.clear()
                else:
                    alert_column.error(f"命令执行失败：{command_text}")
        else:
            st.caption(f"最近命令：{command_text}")
    reconciliation = status.get("last_reconciliation_command") or {}
    if reconciliation:
        reconciliation_text = f"#{reconciliation.get('id')} RUN_RECONCILIATION | {reconciliation.get('status')}"
        if reconciliation.get("error_message") or reconciliation.get("error"):
            reconciliation_text += f" | {reconciliation.get('error_message') or reconciliation.get('error')}"
        st.caption(f"最近对账命令：{reconciliation_text}")
    st.markdown('</div>', unsafe_allow_html=True)
    if status.get("execution_mode") == "OBSERVE":
        st.markdown('<div class="observe-banner">观察模式｜禁止下单</div>', unsafe_allow_html=True)
    elif status.get("execution_mode") == "PAPER" and status.get("market_session") == "OPEN":
        st.success("IBKR Paper 自动交易时段已开启｜仅在实时行情、最新决策和风险审批全部通过后发送模拟订单")
    elif status.get("execution_mode") == "PAPER":
        st.info("IBKR Paper 自动交易已启用｜当前休市，开盘后将自动获取实时行情并检查待执行决策")
    if status["safe_mode"]:
        st.error("安全模式：对账失败，交易已禁用。")
    elif status["system"] == "WAITING_FOR_TWS_PAPER_LOGIN":
        st.warning("等待 TWS Paper 登录：请在 TWS 登录 Paper 账户后重新运行设置检查。")
    elif status["manual_halt"]:
        st.error("手动暂停：需要操作员明确恢复。")
    elif status["risk_state"] == "RISK_HALTED":
        st.error("风险暂停：在恢复条件满足前，持仓必须保持现金。")
    elif status["broker"] != "CONNECTED":
        st.error("经纪商已断开：完成重新连接和对账前，交易已禁用。")
    elif status["llm"] in {"ERROR", "OFFLINE"}:
        st.error("LLM 网关离线：新的 AI 决策已禁用，风险监控仍继续运行。")
    elif status["market_data"] in {"FROZEN", "DELAYED", "DELAYED_FROZEN", "MARKET_CLOSED"}:
        source = status.get("market_data_source", "UNKNOWN")
        last_bar = status.get("market_data_last_bar_time") or "UNKNOWN"
        st.warning(f"市场已休市 · 非实时行情 · 最近完整行情：{_local_time(last_bar)} · 来源：{source} · 未发送订单")
    elif status["market_data"] in {"STALE", "ERROR", "PERMISSION_DENIED", "UNAVAILABLE"}:
        st.warning(f"行情数据状态：{status['market_data']}，新订单需要可用行情。")
    elif status["risk_state"] == "REDUCED":
        st.warning("仓位受限：回撤或事件风险正在限制批准仓位。")
    elif status["system"] == "ERROR":
        st.error("系统异常：恢复交易前请查看“错误”页面。")
    if status.get("last_error"):
        st.warning(f"最近一次后端错误：{status['last_error']}")


def _render_controls(store, status):
    st.sidebar.subheader("控制")
    st.sidebar.button("立即刷新", key="refresh_now")
    waiting_for_tws = status["system"] == "WAITING_FOR_TWS_PAPER_LOGIN"
    scheduler_running = str(status.get("scheduler") or "STOPPED").upper() == "RUNNING"
    if scheduler_running:
        st.sidebar.success("自动调度：已启动")
    else:
        st.sidebar.warning("自动调度：已停止")
    if st.sidebar.button("启动调度器", key="start_scheduler", disabled=scheduler_running or status["safe_mode"] or status["manual_halt"] or waiting_for_tws or status.get("first_run_ready", False)):
        _queue(store, "START_SCHEDULER")
    if st.sidebar.button("停止调度器", key="stop_scheduler", disabled=not scheduler_running):
        _queue(store, "STOP_SCHEDULER")
    if st.sidebar.button("立即运行对账", key="run_reconciliation"):
        _queue(store, "RUN_RECONCILIATION")
    if st.sidebar.button("立即运行风控检查", key="run_risk_check"):
        _queue(store, "RUN_RISK_CHECK")
    active_research = status.get("active_ai_research_command")
    submitted_research_id = st.session_state.get("submitted_ai_research_command_id")
    if active_research:
        st.session_state.pop("submitted_ai_research_command_id", None)
    elif submitted_research_id:
        active_research = {"id": submitted_research_id, "status": "QUEUED"}
    if active_research:
        st.sidebar.info(
            f"AI 研究正在{('排队' if str(active_research.get('status')).upper() in {'QUEUED', 'PENDING'} else '运行')}"
            f"（#{active_research.get('id')}），完成前无需再次点击。"
        )
        research_label = f"AI 研究进行中（#{active_research.get('id')}）"
    else:
        research_label = "运行首次 AI 研究（不下单）" if status.get("first_run_ready", False) else "立即运行 AI 研究"
    research_disabled = bool(active_research) or status["safe_mode"] or status["manual_halt"] or waiting_for_tws or status.get("first_run_state") == "STARTING"
    if st.sidebar.button(research_label, key="run_ai_research", disabled=research_disabled):
        _queue(store, "RUN_AI_RESEARCH")
    st.sidebar.divider()
    if st.sidebar.button("紧急停止", key="emergency_stop", type="primary"):
        _queue(store, "EMERGENCY_STOP")
    if st.sidebar.button("从手动暂停恢复", key="resume_manual_halt", disabled=not status["manual_halt"]):
        _queue(store, "RESUME_MANUAL_HALT")
    st.sidebar.subheader("配置")
    for name, state in st.session_state.get("configuration_status", {}).items():
        if name.startswith("OPENAI_"):
            continue
        st.sidebar.caption(f"{name}: {state}")
    st.sidebar.caption(f"最近对账：{_local_time(status.get('last_reconciliation'))}")
    st.sidebar.caption(f"最近 AI 决策：{_local_time(status.get('last_ai_decision'))}")


def _render_disabled_controls():
    """Keep the static control layout safe while the backend database is starting."""
    st.sidebar.subheader("控制")
    st.sidebar.button("立即刷新", key="refresh_now")
    for label, key in (
        ("启动调度器", "start_scheduler"),
        ("停止调度器", "stop_scheduler"),
        ("立即运行对账", "run_reconciliation"),
        ("立即运行风控检查", "run_risk_check"),
        ("立即运行 AI 研究", "run_ai_research"),
        ("紧急停止", "emergency_stop"),
        ("从手动暂停恢复", "resume_manual_halt"),
    ):
        st.sidebar.button(label, key=key, disabled=True)
    st.sidebar.divider()
    st.sidebar.caption("后端数据库尚未就绪，命令控制暂不可用。")
    st.sidebar.subheader("配置")
    st.sidebar.caption("后端启动后才能读取配置。")
    st.sidebar.caption("最近对账：暂无")
    st.sidebar.caption("最近 AI 决策：暂无")


def _render_live_ai_activity(payload):
    activity = payload.get("ai_activity") or {}
    st.subheader("AI 运行周期")
    header = st.columns(4)
    header[0].metric("Run ID", activity.get("run_id") or "n/a")
    header[1].metric("状态", activity.get("status", "IDLE"))
    header[2].metric("开始时间", _local_time(activity.get("started_at")))
    header[3].metric("流程", activity.get("pipeline") or "暂无")
    st.caption("Luna 初筛 → Sol 深度研究 → Sol 决策 → 风险引擎 → 执行")

    progress = activity.get("progress") or {}
    percent = max(0.0, min(100.0, float(progress.get("progress_percent", 0) or 0)))
    st.subheader("AI 研究进度")
    st.progress(percent / 100.0, text=f"{percent:.0f}% | {progress.get('stage', 'IDLE')}")
    progress_metrics = st.columns(4)
    progress_metrics[0].metric("当前阶段", progress.get("stage", "IDLE"))
    progress_metrics[1].metric("股票池进度", f"{progress.get('universe_processed', 0)} / {progress.get('universe_total', 0)}")
    progress_metrics[2].metric("批次进度", f"{progress.get('batch_index', 0)} / {progress.get('batch_total', 0)}")
    progress_metrics[3].metric("当前批次", progress.get("current_batch_size") or "暂无")
    if progress.get("stage") == "LUNA_BATCH_SCREENING":
        st.caption(f"本批候选股：{progress.get('batch_candidates', 0)} · 合并候选股：{progress.get('merged_candidates', 0)}")
        split = progress.get("adaptive_split") or {}
        if split:
            st.warning(f"自适应拆分：{split.get('original_batch_size')} -> {' + '.join(str(value) for value in split.get('split_sizes', []))} | {split.get('reason')}")
    if progress.get("stage") in {"SOL_RESEARCH", "SOL_DECISION"}:
        st.caption(f"已研究候选：{progress.get('sol_candidates_reviewed', 0)} / {progress.get('initial_candidates', 0)} | 工具调用：{progress.get('sol_tool_calls', 0)} | 工具结果：{progress.get('sol_tool_results', 0)} | 当前：{progress.get('current_tool') or progress.get('sol_current_symbol') or '暂无'}")
    if progress.get("status") == "FAILED":
        st.error(f"失败于 {percent:.0f}% | {progress.get('error_stage')}: {progress.get('error_message')}")
    luna_usage = progress.get("luna_usage") or {}
    usage_columns = st.columns(4)
    usage_columns[0].metric("Luna 批次调用", progress.get("luna_batch_calls", 0))
    usage_columns[1].metric("Luna 重试调用", progress.get("luna_retry_calls", 0))
    usage_columns[2].metric("Luna 最终调用", progress.get("luna_final_screening_calls", 0))
    usage_columns[3].metric("Luna 成本", _money(luna_usage.get("estimated_cost")) if luna_usage.get("available") else "暂无")
    st.caption(f"Luna tokens：输入={luna_usage.get('input_tokens') if luna_usage.get('available') else '暂无'} · 输出={luna_usage.get('output_tokens') if luna_usage.get('available') else '暂无'}")
    stage_order = ("PREPARING", "LUNA_BATCH_SCREENING", "LUNA_FINAL_SCREENING", "SOL_RESEARCH", "SOL_DECISION", "RISK_ENGINE", "OBSERVE_EXECUTION", "COMPLETE")
    current_index = stage_order.index(progress.get("stage")) if progress.get("stage") in stage_order else -1
    st.caption("  ".join(("✓" if index < current_index or progress.get("stage") == "COMPLETE" else "→" if index == current_index else "○") + " " + stage.replace("_", " ").title() for index, stage in enumerate(stage_order)))

    stage_rows = []
    for name in ("Luna", "Sol", "Risk Engine", "Execution"):
        stage = activity.get("stages", {}).get(name, {})
        stage_rows.append({"Stage": name, "Status": stage.get("status", "WAITING"), "Started": stage.get("started_at"), "Completed": stage.get("completed_at")})
    st.dataframe(pd.DataFrame([{"阶段": row["Stage"], "状态": row["Status"], "开始时间": _local_time(row["Started"]), "完成时间": _local_time(row["Completed"])} for row in stage_rows]), width="stretch", hide_index=True, key="ai_stage_status")

    luna = activity.get("stages", {}).get("Luna", {})
    st.subheader("Luna 初筛详情")
    luna_header = st.columns(4)
    luna_header[0].metric("股票池规模", luna.get("universe_size", "暂无"))
    luna_header[1].metric("候选数量", luna.get("candidate_count", "暂无"))
    luna_header[2].metric("模型", luna.get("model") or activity.get("luna_model") or "暂无")
    luna_header[3].metric("耗时", f"{float(luna.get('usage', {}).get('latency_ms', 0)):.0f} 毫秒" if luna.get("usage", {}).get("latency_ms") is not None else "暂无")
    candidates = luna.get("candidates") or []
    if candidates:
        st.dataframe(pd.DataFrame([
            {
                "#": index,
                "标的": item.get("symbol"),
                "筛选理由": item.get("rationale"),
                "积极信号": " | ".join(item.get("positive_signals", [])),
                "风险": " | ".join(item.get("risks", [])),
            }
            for index, item in enumerate(candidates, start=1)
        ]), width="stretch", hide_index=True, key="luna_candidates")
    elif luna.get("status") == "WAITING":
        st.info("本轮 Luna 尚未开始。")

    sol = activity.get("stages", {}).get("Sol", {})
    st.subheader("Sol 深度研究过程")
    sol_header = st.columns(3)
    sol_header[0].metric("模型", sol.get("model") or activity.get("sol_model") or "暂无")
    sol_header[1].metric("工具调用次数", len(sol.get("tool_calls", [])))
    sol_header[2].metric("工具结果次数", len(sol.get("tool_results", [])))
    if sol.get("tool_calls"):
        st.dataframe(pd.DataFrame([
            {"时间": _local_time(item.get("timestamp")), "工具": item.get("tool"), "标的": item.get("symbol"), "参数": json.dumps(item.get("arguments", {}), ensure_ascii=False)}
            for item in sol["tool_calls"]
        ]), width="stretch", hide_index=True, key="sol_tool_calls")
    if sol.get("tool_results"):
        st.dataframe(pd.DataFrame([
            {"时间": _local_time(item.get("timestamp")), "工具": item.get("tool"), "标的": item.get("symbol"), "状态": item.get("status"), "结构化证据": json.dumps(item.get("result", {}), ensure_ascii=False)}
            for item in sol["tool_results"]
        ]), width="stretch", hide_index=True, key="sol_tool_results")

    decision = activity.get("final_decision") or {}
    if decision:
        st.subheader("最终交易决策")
        decision_header = st.columns(6)
        decision_header[0].metric("决策", {"BUY": "买入", "HOLD": "继续持有", "SWITCH": "换仓", "CASH": "保持现金"}.get(decision.get("action"), decision.get("action", "暂无")))
        decision_header[1].metric("标的", decision.get("symbol") or "现金")
        decision_header[2].metric("AI 建议仓位", _pct(decision.get("target_weight")))
        decision_header[3].metric("决策置信度", _pct(decision.get("confidence")))
        decision_header[4].metric("预期超额收益（相对 SPY）", _pct(decision.get("expected_alpha_vs_spy", decision.get("expected_excess_vs_spy"))))
        decision_header[5].metric("预期超额收益（相对 QQQ）", _pct(decision.get("expected_alpha_vs_qqq", decision.get("expected_excess_vs_qqq"))))
        with st.expander("查看完整决策依据"):
            st.write({
                "决策编号": decision.get("decision_id"), "模型": decision.get("model_name") or sol.get("model") or activity.get("sol_model"),
                "计划持有天数": decision.get("holding_period_days"), "核心判断": decision.get("thesis", []),
                "主要风险": decision.get("risk_factors", []), "失效条件": decision.get("invalidation_conditions", []), "使用的证据": decision.get("evidence_used", []),
            })

    risk = activity.get("stages", {}).get("Risk Engine", {})
    with st.container(border=True):
        st.subheader("风险引擎审批")
        risk_header = st.columns(5)
        risk_header[0].metric("AI 请求仓位", _pct(risk.get("requested_weight")))
        risk_header[1].metric("批准仓位", _pct(risk.get("approved_weight")))
        risk_header[2].metric("回撤上限", _pct(risk.get("drawdown_weight_limit")))
        risk_header[3].metric("波动率上限", _pct(risk.get("volatility_weight_limit")))
        risk_header[4].metric("事件上限", _pct(risk.get("event_weight_limit")))
        st.write({
            "Risk State": risk.get("risk_state", "n/a"),
            "Liquidity Limit": _money(risk.get("liquidity_min_dollar_volume")),
            "Confidence Limit": _pct(risk.get("confidence_minimum")),
            "Approved": risk.get("approved"),
            "Why the weight changed": risk.get("reason", "n/a"),
            "Limit Reasons": risk.get("limit_reasons", []),
        })

    execution = activity.get("stages", {}).get("Execution", {})
    with st.container(border=True):
        st.subheader("执行结果")
        if payload["status"].get("execution_mode") == "OBSERVE":
            st.error("观察模式｜禁止下单")
            if execution.get("event_type", "").startswith("WOULD_"):
                st.warning(execution.get("message", "WOULD_EXECUTE"))
            st.write({
                "状态": execution.get("status", "WAITING"),
                "订单": "未发送订单",
                "placeOrder 调用次数": execution.get("place_order_calls", 0),
                "cancelOrder 调用次数": execution.get("cancel_order_calls", 0),
            })
        else:
            st.write({"状态": execution.get("status", "WAITING"), "说明": execution.get("message", "暂无")})


def _render_timeline(payload):
    st.subheader("系统时间线")
    rows = payload.get("timeline", [])
    if not rows:
        st.info("暂无运行事件。")
        return
    st.dataframe(pd.DataFrame([
        {
            "时间": row.get("timestamp"),
            "运行编号": row.get("run_id"),
            "决策编号": row.get("decision_id"),
            "组件": row.get("component"),
            "事件类型": row.get("event_type"),
            "说明": row.get("message"),
            "标的": row.get("symbol"),
        }
        for row in rows[-300:]
    ]), width="stretch", hide_index=True, key="runtime_timeline")


def _render_overview(payload):
    account = payload["account"]
    columns = st.columns(7)
    for column, (label, value) in zip(columns, [
        ("账户权益", _money(account["equity"])),
        ("现金", _money(account["cash"])),
        ("投资市值", _money(account["invested_value"])),
        ("当前回撤", _pct(account["current_drawdown"])),
        ("历史最大回撤", _pct(account["historical_max_drawdown"])),
        ("权益峰值", _money(account["peak_equity"])),
        ("未实现盈亏", _money(account["unrealized_pnl"])),
    ]):
        column.metric(label, value)
    st.subheader("当前持仓")
    if account["positions"]:
        st.dataframe(pd.DataFrame(account["positions"]), width="stretch", hide_index=True, key="portfolio_positions")
    else:
        st.info("当前没有持仓")
    st.subheader("待执行交易")
    pending_entries = payload.get("pending_entries") or []
    active_entries = [entry for entry in pending_entries if entry.get("status") not in {"ENTRY_FILLED", "ENTRY_CANCELLED", "ENTRY_FAILED"}]
    if active_entries:
        rows = [{
            "标的": entry.get("symbol"),
            "决策时间": _local_time(entry.get("decision_time")),
            "决策参考价": _money(entry.get("decision_reference_price")),
            "当前状态": entry.get("status"),
            "目标仓位": _pct(entry.get("requested_weight")),
            "当前价格": _money(entry.get("current_price")),
            "相对决策价": _pct(entry.get("price_gap_pct")),
            "订单类型": "可成交限价单",
            "说明": entry.get("message"),
        } for entry in active_entries]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, key="pending_entries_table")
    else:
        st.info("暂无待执行入场决策。")
    st.metric("已实现盈亏", _money(account["realized_pnl"]))
    operations = payload.get("operations", {})
    st.subheader("调度与风控运行")
    st.dataframe(pd.DataFrame([
        {"运行项": "完整 Luna → Sol 研究", "频率": operations.get("full_research_frequency"), "下次": operations.get("next_full_research"), "最近": operations.get("last_full_research")},
        {"运行项": "每日持仓复核", "频率": operations.get("daily_review_time"), "下次": "下一个 NYSE 交易时段", "最近": operations.get("last_daily_review")},
        {"运行项": "完整风控检查", "频率": f"美股交易时段每 {operations.get('risk_interval_minutes', 15)} 分钟", "下次": "下一个有效交易时段", "最近": operations.get("last_full_risk_check")},
        {"运行项": "经纪商监控", "频率": "持续运行", "下次": "运行中", "最近": operations.get("broker_monitor_heartbeat")},
    ]), hide_index=True, width="stretch", key="operations_schedule")
    st.write({
        "经纪商监控": operations.get("broker_monitor"),
        "行情数据": operations.get("market_data_type"),
        "风控数据新鲜度（秒）": operations.get("risk_data_freshness_seconds"),
        "当前持仓复核": operations.get("current_position_review"),
        "最近即时风控检查": operations.get("last_immediate_risk_check"),
    })


def _render_risk(payload):
    rows = payload["risk"]
    if not rows:
        st.info("暂无风险决策记录。")
        return
    decision = json.loads(rows[0]["risk_json"])
    current_drawdown = payload["account"].get("current_drawdown")
    values = [
        ("当前风险状态", decision.get("risk_state")),
        ("当前回撤", _pct(current_drawdown)),
        ("AI 请求仓位", _pct(decision.get("requested_weight"))),
        ("回撤上限", _pct(decision.get("drawdown_weight_limit"))),
        ("波动率上限", _pct(decision.get("volatility_weight_limit"))),
        ("事件上限", _pct(decision.get("event_weight_limit"))),
        ("流动性最低要求", _money(decision.get("liquidity_min_dollar_volume"))),
        ("置信度下限", _pct(decision.get("confidence_minimum"))),
        ("批准仓位", _pct(decision.get("approved_weight"))),
    ]
    for column, (label, value) in zip(st.columns(len(values)), values):
        column.metric(label, value)
    st.write({
        "是否批准": decision.get("approved"),
        "原因": decision.get("reason"),
        "触发的限制": decision.get("limit_reasons", []),
    })


def _render_decisions(payload):
    llm = payload.get("llm", {})
    st.subheader("LLM 网关与流水线")
    st.write({
        "网关": llm.get("gateway"),
        "流水线": llm.get("pipeline_label", llm.get("pipeline")),
        "初筛模型": llm.get("screening_model"),
        "研究与 CIO": llm.get("research_cio_model"),
        "CC Switch": llm.get("gateway_status"),
        "CC Switch 网关": llm.get("CC Switch Gateway", llm.get("gateway_status")),
        "模型发现": llm.get("model_discovery"),
        "Luna": llm.get("luna_status"),
        "Luna 基础调用": llm.get("luna_basic_call"),
        "Luna 结构化输出": llm.get("luna_structured_output"),
        "Sol": llm.get("sol_status"),
        "Sol 基础调用": llm.get("sol_basic_call"),
        "Sol 结构化 JSON": llm.get("sol_structured_json"),
        "Sol 工具调用": llm.get("sol_tool_calling"),
        "Sol 决策": llm.get("sol_decision"),
        "工具调用": llm.get("tool_calling"),
        "结构化输出": llm.get("structured_output"),
        "API 协议": llm.get("api_protocol"),
        "协议回退": llm.get("protocol_fallback"),
        "推理元数据": llm.get("reasoning_metadata"),
        "Token 使用量": llm.get("token_usage"),
    })
    latest_run = llm.get("latest_run") or {}
    if latest_run:
        st.subheader("最近一次决策用量")
        usage_rows = [
            {"阶段": "Luna", "输入 Token": latest_run.get("luna_input_tokens"), "输出 Token": latest_run.get("luna_output_tokens"), "耗时（毫秒）": latest_run.get("luna_latency_ms"), "API 成本": latest_run.get("luna_api_cost")},
            {"阶段": "Sol", "输入 Token": latest_run.get("sol_input_tokens"), "输出 Token": latest_run.get("sol_output_tokens"), "推理 Token": latest_run.get("sol_reasoning_tokens"), "耗时（毫秒）": latest_run.get("sol_latency_ms"), "API 成本": latest_run.get("sol_api_cost")},
            {"阶段": "合计", "输入 Token": latest_run.get("total_input_tokens"), "输出 Token": latest_run.get("total_output_tokens"), "推理 Token": latest_run.get("total_reasoning_tokens"), "耗时（毫秒）": latest_run.get("total_latency_ms"), "API 成本": latest_run.get("total_decision_cost")},
        ]
        st.dataframe(pd.DataFrame(usage_rows), width="stretch", hide_index=True, key="decision_usage")
        if latest_run.get("fallback_events_json") not in (None, "[]"):
            st.warning("已记录流水线协议回退事件")
    for row in payload["decisions"]:
        decision = json.loads(row["decision_json"])
        with st.expander(f"{row['recorded_at']} | {decision.get('action')} | {decision.get('symbol') or 'CASH'} | {row['decision_id']}"):
            st.write({
                "决策编号": row["decision_id"],
                "模型": row.get("model_name"),
                "提示词版本": row.get("prompt_version"),
                "耗时（毫秒）": row.get("latency_ms"),
                "输入 Token": row.get("input_tokens"),
                "输出 Token": row.get("output_tokens"),
                "预计 API 成本": row.get("estimated_cost"),
            })
            pipeline_run = next(
                (item for item in payload.get("pipeline_history", []) if item.get("decision_id") == row["decision_id"]),
                None,
            )
            if pipeline_run:
                try:
                    candidates = json.loads(pipeline_run.get("candidate_symbols_json") or "[]")
                except (TypeError, ValueError, json.JSONDecodeError):
                    candidates = []
                notes = [
                    str(item).removeprefix("Comparative screen:").strip()
                    for item in decision.get("evidence_used", [])
                    if str(item).strip().lower().startswith("comparative screen:")
                ]
                if candidates or notes:
                    st.markdown("**同场候选横向比较**")
                    if candidates:
                        st.dataframe(
                            pd.DataFrame([{"候选序号": index, "标的": symbol} for index, symbol in enumerate(candidates, start=1)]),
                            width="stretch", hide_index=True, key=f"decision_candidates_{row['decision_id']}",
                        )
                    for note in notes:
                        st.write(note)
                    if not decision.get("top_five"):
                        st.caption("该历史运行未保存逐项评分；这里只展示数据库中真实保存的候选范围和比较结论。")
            with st.expander("查看原始决策 JSON"):
                st.json(decision)
            evidence = [item for item in payload["evidence"] if item["decision_id"] == row["decision_id"]]
            if evidence:
                with st.expander("查看研究证据与工具调用"):
                    for item in evidence:
                        st.write({"工具": item["tool_name"], "参数": json.loads(item["arguments_json"]), "结果": json.loads(item["result_json"])})


def _render_journal(payload):
    if payload["journal"]:
        rows = []
        for row in payload["journal"]:
            journal = json.loads(row["journal_json"])
            decision = journal.get("decision", {})
            risk = journal.get("risk", {})
            rows.append({
                "decision_id": row["decision_id"],
                "recorded_at": row["recorded_at"],
                "action": row["action"],
                "symbol": row["symbol"] or "CASH",
                "thesis": " | ".join(decision.get("thesis", [])),
                "risk_factors": " | ".join(decision.get("risk_factors", [])),
                "candidates": ", ".join(journal.get("candidates", [])),
                "requested_weight": row["requested_weight"],
                "approved_weight": risk.get("approved_weight", row["approved_weight"]),
                "entry_price": row["entry_price"],
                "filled_price": row["filled_price"],
                "return_1d": row["return_1d"],
                "return_5d": row["return_5d"],
                "return_20d": row["return_20d"],
                "alpha_vs_spy": row["alpha_vs_spy"],
                "alpha_vs_qqq": row["alpha_vs_qqq"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, key="decision_journal")
    else:
        st.info("暂无决策记录。")


def _render_benchmarks(payload):
    series = payload["benchmarks"]["series"]
    frame = pd.DataFrame({name: pd.Series(values) for name, values in series.items() if values})
    if not frame.empty:
        st.line_chart(frame)
    metrics = payload["benchmarks"]["metrics"]
    series_metrics = payload["benchmarks"].get("series_metrics", {})
    if series_metrics:
        st.dataframe(pd.DataFrame(series_metrics).T, width="stretch", key="performance_series_metrics")
    st.subheader("净绩效")
    returns = payload["benchmarks"].get("returns", {})
    st.metric("相对 QQQ 的净超额收益", _pct(returns.get("net_excess_return_vs_qqq")))
    columns = st.columns(7)
    values = [
        ("策略毛收益", _pct(returns.get("gross_strategy_return"))),
        ("交易成本", _money(returns.get("trading_costs"))),
        ("LLM 成本", _money(returns.get("llm_costs"))),
        ("策略净收益", _pct(returns.get("net_strategy_return"))),
        ("SPY 收益", _pct(returns.get("spy_return"))),
        ("QQQ 收益", _pct(returns.get("qqq_return"))),
        ("相对 SPY 的净超额收益", _pct(returns.get("net_excess_return_vs_spy"))),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    st.dataframe(pd.DataFrame([metrics]), width="stretch", hide_index=True, key="performance_metrics")


def _render_orders(payload):
    rows = []
    latest_executions = {}
    for execution_row in payload["executions"]:
        execution = json.loads(execution_row["execution_json"])
        latest_executions.setdefault(execution_row["client_order_id"], execution)
    for row in payload["orders"]:
        order = json.loads(row["order_json"])
        execution = latest_executions.get(row["client_order_id"], {})
        rows.append({
            "order_id": row["client_order_id"],
            "decision_id": row["decision_id"],
            "symbol": order.get("symbol"),
            "side": order.get("action"),
            "requested_quantity": order.get("quantity"),
            "filled_quantity": execution.get("filled_quantity", 0),
            "average_fill_price": execution.get("average_price"),
            "status": row["status"],
            "submitted_time": row["recorded_at"],
            "filled_time": execution.get("timestamp"),
            "broker_order_id": row.get("broker_order_id"),
            "perm_id": row.get("perm_id"),
            "rejected_reason": execution.get("message") if row["status"] == "REJECTED" else None,
        })
    st.dataframe(pd.DataFrame([{
        "订单编号": row["order_id"], "决策编号": row["decision_id"], "标的": row["symbol"], "方向": row["side"],
        "请求数量": row["requested_quantity"], "成交数量": row["filled_quantity"], "平均成交价": row["average_fill_price"],
        "状态": row["status"], "提交时间": row["submitted_time"], "成交时间": row["filled_time"],
        "经纪商订单号": row["broker_order_id"], "拒绝原因": row["rejected_reason"],
    } for row in rows]), width="stretch", hide_index=True, key="orders_table")
    if payload["executions"]:
        st.subheader("成交回报")
        st.dataframe(pd.DataFrame([json.loads(row["execution_json"]) for row in payload["executions"]]), width="stretch", hide_index=True, key="execution_reports")


def _render_errors(payload):
    if payload["errors"]:
        st.dataframe(pd.DataFrame(payload["errors"]), width="stretch", hide_index=True, key="errors_table")
    else:
        st.success("暂无错误记录。")


def _render_setup(payload):
    setup = payload["setup"]
    st.subheader("设置与系统健康")
    rows = []
    for name, status in setup["checks"].items():
        rows.append({"检查项": name, "状态": status, "值": setup["values"].get(name, "")})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, key="setup_health")
    if setup["state"] == "READY_FOR_OBSERVE":
        st.success("已准备好进入观察模式")
    elif setup["state"] == "WAITING_FOR_TWS_PAPER_LOGIN":
        st.warning("请在 TWS 登录 Paper Trading，并保持 Read-Only API = ON。")
    elif setup["state"] == "SAFE_MODE":
        st.error("安全模式：设置检查未通过，新的 AI 执行已禁用。")
    if setup.get("details"):
        with st.expander("查看详细技术信息"):
            st.json(setup["details"])


@st.cache_data(ttl=1.0, show_spinner=False)
def _cached_dashboard_payload(database_path: str, config: dict):
    path = Path(database_path)
    if not path.exists():
        return None
    with SQLiteStore(path, read_only=True) as store:
        return build_dashboard_payload(store, config)


def _read_dashboard_payload(cfg, db_path):
    return _cached_dashboard_payload(str(db_path.resolve()), cfg)


def _render_read_only_fragment(cfg, db_path, renderer):
    payload = _read_dashboard_payload(cfg, db_path)
    if payload is None:
        st.info("后端数据库尚未创建，请先启动后端服务。")
        return
    renderer(payload)


def _render_static_shell(cfg, db_path):
    st.markdown('<h1 class="ops-title">AI 自动交易运营中心</h1>', unsafe_allow_html=True)
    st.caption("Paper 账户 · 风控约束下的自动化研究与执行监控")
    _dashboard_controls_fragment()
    _dashboard_live_fragment()
    tabs = st.tabs([
        "设置与系统健康", "账户概览", "AI 实时活动", "系统时间线",
        "风险引擎", "AI 决策记录", "决策日志", "基准与绩效", "订单", "错误",
    ])
    tab_renderers = (
        (_dashboard_setup_fragment, 0), (_dashboard_overview_fragment, 1),
        (_dashboard_activity_fragment, 2), (_dashboard_timeline_fragment, 3),
        (_dashboard_risk_fragment, 4), (_dashboard_decisions_fragment, 5),
        (_dashboard_journal_fragment, 6), (_dashboard_benchmarks_fragment, 7),
        (_dashboard_orders_fragment, 8), (_dashboard_errors_fragment, 9),
    )
    for renderer, index in tab_renderers:
        tab = tabs[index]
        with tab:
            renderer()


cfg = load_config()
db_path = Path(env("DATABASE_PATH", cfg.get("dashboard", {}).get("database_path", "data/ai_fund_manager.sqlite3")))

def _streamlit_fragment(**kwargs):
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        return lambda function: function
    return fragment(**kwargs)


@_streamlit_fragment(run_every=dashboard_refresh_seconds(cfg))
def _dashboard_live_fragment():
    """Refresh status without rendering into containers owned by another scope."""
    payload = _read_dashboard_payload(cfg, db_path)
    if payload is None:
        st.info("后端数据库尚未创建，请先启动后端服务。")
        return
    _render_status(payload, db_path)
    _render_executive_summary(payload)


@_streamlit_fragment(run_every=2)
def _dashboard_activity_fragment():
    _render_read_only_fragment(cfg, db_path, _render_live_ai_activity)


@_streamlit_fragment(run_every=2)
def _dashboard_timeline_fragment():
    _render_read_only_fragment(cfg, db_path, _render_timeline)


@_streamlit_fragment(run_every=dashboard_refresh_seconds(cfg))
def _dashboard_risk_fragment():
    _render_read_only_fragment(cfg, db_path, _render_risk)


@_streamlit_fragment(run_every=15)
def _dashboard_benchmarks_fragment():
    _render_read_only_fragment(cfg, db_path, _render_benchmarks)


@_streamlit_fragment()
def _dashboard_controls_fragment():
    """Keep user-triggered controls local to this fragment."""
    payload = _read_dashboard_payload(cfg, db_path)
    if payload is None:
        _render_disabled_controls()
        return
    st.session_state["configuration_status"] = payload["configuration"]
    with SQLiteStore(db_path) as store:
        _render_controls(store, payload["status"])


@_streamlit_fragment()
def _dashboard_setup_fragment():
    _render_read_only_fragment(cfg, db_path, _render_setup)


@_streamlit_fragment()
def _dashboard_overview_fragment():
    _render_read_only_fragment(cfg, db_path, _render_overview)


@_streamlit_fragment()
def _dashboard_decisions_fragment():
    _render_read_only_fragment(cfg, db_path, _render_decisions)


@_streamlit_fragment()
def _dashboard_journal_fragment():
    _render_read_only_fragment(cfg, db_path, _render_journal)


@_streamlit_fragment()
def _dashboard_orders_fragment():
    _render_read_only_fragment(cfg, db_path, _render_orders)


@_streamlit_fragment()
def _dashboard_errors_fragment():
    _render_read_only_fragment(cfg, db_path, _render_errors)


_render_static_shell(cfg, db_path)
