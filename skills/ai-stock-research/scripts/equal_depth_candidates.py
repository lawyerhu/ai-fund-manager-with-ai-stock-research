"""Research every saved finalist with the same Sol deep-research evidence set."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from evidence_discipline import ResearchPending, install_evidence_discipline
from research import source_universe

ASTRA_MODEL = "gpt-6-astra"


class NoTradingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            "src.main", "src.runner", "src.service", "src.scheduler", "src.risk_engine",
            "src.execution",
        }
        if fullname in blocked or fullname.startswith("src.execution."):
            raise ImportError(f"Research-only skill blocks {fullname}")
        return None


def load_source(store, decision_id: str | None):
    snapshot = store.latest_portfolio()
    if snapshot is None:
        raise ValueError("No persisted portfolio snapshot; do not invent a portfolio")
    portfolio = snapshot[0]
    query = "SELECT decision_id,recorded_at,candidate_symbols_json FROM pipeline_runs"
    row = store.connection.execute(
        query + (" WHERE decision_id=?" if decision_id else " ORDER BY recorded_at DESC LIMIT 1"),
        (decision_id,) if decision_id else (),
    ).fetchone()
    if row is None:
        raise ValueError("No saved candidate decision")
    source = dict(row)
    candidates = json.loads(source.pop("candidate_symbols_json", "[]"))
    if not isinstance(candidates, list) or any(not isinstance(s, str) or not s for s in candidates):
        raise ValueError("Invalid persisted candidates")
    candidates = [s.upper() for s in candidates]
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate persisted candidates")
    return portfolio, candidates, source


def read_event_state(path: Path | None):
    cache: dict[tuple[str | None, str], object] = {}
    completed: dict[str, dict] = {}
    awaiting_audit: dict[str, dict] = {}
    if not path:
        return cache, completed
    if not path.is_file():
        raise ValueError(f"Resume event file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") == "SKILL_EVIDENCE_AUDIT_PASSED":
            symbol = str(event.get("symbol") or "").upper()
            if (event.get("metadata") or {}).get("stage") == "EQUAL_DEPTH_DEEP_DIVE" and symbol in awaiting_audit:
                completed[symbol] = awaiting_audit.pop(symbol)
            continue
        if event.get("event_type") == "EQUAL_DEPTH_DEEP_DIVE_COMPLETED":
            symbol = event.get("symbol")
            result = (event.get("metadata") or {}).get("result")
            if symbol and isinstance(result, dict):
                assessments = result.get("evidence_assessments", [])
                if any(item.get("research_requests") for item in assessments):
                    completed.pop(str(symbol).upper(), None)
                    continue
                cleaned = {key: value for key, value in result.items() if key not in {"evidence_assessments", "selection_rationale"}}
                if "evidence_assessments" in result:
                    completed.pop(str(symbol).upper(), None)
                    awaiting_audit[str(symbol).upper()] = cleaned
                else:
                    completed[str(symbol).upper()] = cleaned
            continue
        if event.get("event_type") != "SOL_TOOL_RESULT":
            continue
        metadata = event.get("metadata") or {}
        if metadata.get("status") != "SUCCESS":
            continue
        tool = metadata.get("tool")
        result = metadata.get("result")
        if not tool or not isinstance(result, dict):
            continue
        symbol = event.get("symbol")
        if tool == "get_price_history":
            if result.get("days") != 252 or not symbol:
                continue
            key = (str(symbol).upper(), "price_history_252")
        elif tool in {"get_stock_snapshot", "get_fundamentals", "get_earnings", "get_analyst_revisions", "get_news", "get_sec_filings", "get_upcoming_events"}:
            if not symbol:
                continue
            key = (str(symbol).upper(), tool)
        elif tool in {"get_market_regime", "get_benchmark_data"}:
            key = (None, tool)
        else:
            continue
        cache[key] = result
    return cache, completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--decision-id")
    parser.add_argument("--resume-events", type=Path,
                        help="Prior redacted events.jsonl; successful deep data is reused")
    parser.add_argument("--verification-packet", type=Path,
                        help="Dated evidence and gap audit from Codex's actual source searches")
    parser.add_argument("--run", action="store_true", help="Actually send research requests; never orders")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Equal-depth research requires --run")
    project = args.project.resolve()
    if not (project / "src/llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")
    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())

    from src.config import env, load_config, load_project_env
    load_project_env(project / ".env", override=True)
    os.environ["EXECUTION_MODE"] = "OBSERVE"
    from src.llm_agent import (
        CCSwitchProvider,
        CrossSectionalRanking,
        DeepDiveResearch,
        LLMRuntimeConfig,
        SolResearchCIOAgent,
        _strict_json_schema,
    )
    from src.storage import SQLiteStore, redact_sensitive
    from src.data_provider import YahooFinanceDataProvider

    cfg = load_config(project / "config.yaml")
    runtime = replace(
        LLMRuntimeConfig.from_mapping(cfg),
        sol_model=ASTRA_MODEL,
        sol_reasoning_effort="medium",
        pipeline="LUNA_SOL",
        fallback_to_sol_only=False,
    )
    db = Path(env("DATABASE_PATH") or cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3"))
    if not db.is_absolute():
        db = project / db
    with SQLiteStore(db, read_only=True) as store:
        portfolio, candidates, source = load_source(store, args.decision_id)
        source["universe"] = source_universe(store, source.get("decision_id"))
    if len(candidates) < 5:
        raise ValueError("Equal-depth comparison requires at least five candidates")
    cache, resumed_deep = read_event_state(args.resume_events.resolve() if args.resume_events else None)

    output = project / "outputs" / "skill-research" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-equal-depth-" + str(uuid4())
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)

    def write(name, value):
        (output / name).write_text(
            json.dumps(redact_sensitive(value), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def sink(event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact_sensitive(event), ensure_ascii=False, default=str) + "\n")

    manifest = {
        "mode": "equal_depth_candidates",
        "research_model": runtime.sol_model,
        "research_effort_requested": runtime.sol_reasoning_effort,
        "source": source,
        "candidate_count": len(candidates),
        "safety": "RESEARCH_ONLY; production database read-only; no broker/Risk Engine instance",
        "placeOrder_calls": 0,
        "cancelOrder_calls": 0,
        "risk_approval": "NOT_RUN",
        "resume_events": str(args.resume_events.resolve()) if args.resume_events else None,
        "resumed_deep_research": sorted(set(candidates) & set(resumed_deep)),
        "same_depth_evidence": [
            "stock_snapshot", "price_history_252", "fundamentals", "earnings",
            "analyst_revisions", "news", "sec_filings", "upcoming_events",
            "market_regime", "benchmarks",
        ],
    }
    write("manifest.json", manifest)

    provider = CCSwitchProvider(runtime=runtime, event_sink=sink)
    rows = provider.client.models.list()
    model_ids = {row.get("id") if isinstance(row, dict) else row.id for row in (rows.get("data", []) if isinstance(rows, dict) else rows.data)}
    if ASTRA_MODEL not in model_ids:
        raise ValueError(f"Required model unavailable: {ASTRA_MODEL}")
    data = YahooFinanceDataProvider()
    agent = SolResearchCIOAgent(
        runtime.sol_model,
        data,
        provider=provider,
        max_tool_rounds=int(cfg.get("agent", {}).get("max_tool_rounds", 12)),
        deep_research_enabled=True,
        event_sink=sink,
    )
    agent._candidate_symbols = candidates
    packet = json.loads(args.verification_packet.read_text(encoding="utf-8")) if args.verification_packet else {}
    agent.external_verification = packet
    install_evidence_discipline(agent, write, capture_selection_path=True)
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    reused_count = 0
    new_call_count = 0

    def cached_or_call(symbol, tool, callback, *, cache_key=None):
        nonlocal reused_count, new_call_count
        key = cache_key or (symbol, tool)
        if key in cache:
            reused_count += 1
            value = cache[key]
            agent.last_tool_calls.append({"name": tool, "arguments": {"symbol": symbol} if symbol else {}})
            agent.last_research_evidence.append({
                "stage": agent.stage, "tool": tool,
                "arguments": {"symbol": symbol} if symbol else {},
                "result": value, "reused": True,
            })
            sink({
                "event_type": "SOL_TOOL_RESULT_REUSED",
                "component": agent.stage,
                "message": f"{tool} reused from prior run",
                "symbol": symbol,
                "metadata": {"tool": tool, "status": "REUSED", "source_events": str(args.resume_events) if args.resume_events else None},
            })
            return value
        new_call_count += 1
        return agent._record_research_call(tool, symbol, callback)

    common = {
        "market_regime": cached_or_call(None, "get_market_regime", data.market_regime),
        "benchmarks": cached_or_call(None, "get_benchmark_data", lambda: data.benchmark_data(252)),
    }

    deep_results: dict[str, DeepDiveResearch] = {}
    failures: dict[str, str] = {}
    research_started = time.perf_counter()
    for index, symbol in enumerate(candidates, start=1):
        if symbol in resumed_deep:
            try:
                deep_results[symbol] = DeepDiveResearch.model_validate(resumed_deep[symbol])
                sink({
                    "event_type": "EQUAL_DEPTH_DEEP_DIVE_REUSED",
                    "component": "SOL",
                    "message": "Equal-depth research reused from prior run",
                    "symbol": symbol,
                    "metadata": {"index": index, "total": len(candidates), "same_template": True},
                })
                continue
            except Exception:
                # A malformed prior result must be recomputed, never trusted.
                pass
        sink({
            "event_type": "EQUAL_DEPTH_DEEP_DIVE_STARTED",
            "component": "SOL",
            "message": f"Equal-depth research {index}/{len(candidates)}",
            "symbol": symbol,
            "metadata": {"index": index, "total": len(candidates), "model": runtime.sol_model},
        })
        evidence = {
            "symbol": symbol,
            "stock_snapshot": cached_or_call(symbol, "get_stock_snapshot", lambda symbol=symbol: data.stock_snapshot(symbol)),
            "price_history": cached_or_call(symbol, "get_price_history", lambda symbol=symbol: data.price_history(symbol, 252), cache_key=(symbol, "price_history_252")),
            "fundamentals": cached_or_call(symbol, "get_fundamentals", lambda symbol=symbol: data.fundamentals(symbol)),
            "earnings": cached_or_call(symbol, "get_earnings", lambda symbol=symbol: data.earnings(symbol)),
            "analyst_revisions": cached_or_call(symbol, "get_analyst_revisions", lambda symbol=symbol: data.analyst_revisions(symbol)),
            "news": cached_or_call(symbol, "get_news", lambda symbol=symbol: data.news(symbol, 10)),
            "sec_filings": cached_or_call(symbol, "get_sec_filings", lambda symbol=symbol: data.sec_filings(symbol, 10)),
            "upcoming_events": cached_or_call(symbol, "get_upcoming_events", lambda symbol=symbol: data.upcoming_events(symbol)),
            "market_regime": common["market_regime"],
            "benchmarks": common["benchmarks"],
        }
        sink({
            "event_type": "EQUAL_DEPTH_EVIDENCE_READY",
            "component": "SOL",
            "message": "Same-depth evidence ready",
            "symbol": symbol,
            "metadata": {"fields": list(evidence), "index": index, "total": len(candidates)},
        })
        prompt = (
            "Apply equal evidence standards to every finalist; choose research questions for this company's thesis. "
            "The candidate set is supplied only for context; do not favor a symbol because it appears in an old ranking. "
            "Explain company-specific alpha versus industry, sector, market, and macro beta; test at least three alternative "
            "hypotheses; explain why now, expectations, priced-in level, remaining alpha, and 20-day persistence. "
            "Return compact structured evidence only, with bull/base/bear scenarios and observable invalidation conditions. "
            "Use DATA_UNAVAILABLE for missing facts. Never invent prices, estimates, dates, outages, citations, or returns. "
            "Do not make an order or use hidden chain-of-thought.\n"
            f"FINALIST_SET: {json.dumps(candidates, ensure_ascii=False)}\n"
            f"SYMBOL: {symbol}\n"
            f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{json.dumps(packet, ensure_ascii=False, default=str)}\n"
            f"RESEARCH_EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        try:
            research = agent._structured_deep_stage(
                DeepDiveResearch,
                "sol_equal_depth_deep_dive",
                "EQUAL_DEPTH_DEEP_DIVE",
                prompt,
                totals,
                symbol=symbol,
                metadata={"index": index, "total": len(candidates), "same_template": True},
            )
            if research.symbol.upper() != symbol:
                raise ValueError(f"Deep Dive returned {research.symbol}, expected {symbol}")
            deep_results[symbol] = research
        except ResearchPending:
            write("result.json", {**manifest, **agent.pending_research,
                                 "deep_research": {s: r.model_dump(mode="json") for s, r in deep_results.items()},
                                 "usage": totals})
            return 2
        except Exception as exc:
            failures[symbol] = str(exc)
            sink({
                "event_type": "EQUAL_DEPTH_DEEP_DIVE_FAILED",
                "component": "SOL",
                "message": "Equal-depth research failed",
                "symbol": symbol,
                "metadata": {"error": str(exc), "index": index, "total": len(candidates)},
            })

    ranking = None
    if not failures:
        compact = []
        for symbol in candidates:
            item = deep_results[symbol].model_dump(mode="json")
            compact.append({
                "symbol": symbol,
                "business_driver_summary": item["business_driver_summary"],
                "causal_drivers": item["causal_drivers"][:5],
                "return_attribution": item["return_attribution"],
                "why_now": item["why_now"][:4],
                "market_expectations": item["market_expectations"][:4],
                "pricing_assessment": item["pricing_assessment"],
                "priced_in_drivers": item["priced_in_drivers"][:4],
                "primary_hypothesis": item["primary_hypothesis"],
                "near_term_catalyst": item["near_term_catalyst"],
                "twenty_day_persistence": item["twenty_day_persistence"],
                "bull_case": item["bull_case"],
                "base_case": item["base_case"],
                "bear_case": item["bear_case"],
                "key_risks": item["key_risks"][:6],
                "thesis_invalidation_conditions": item["thesis_invalidation_conditions"][:6],
                "missing_data": item["missing_data"][:12],
                "required_data": item["required_data"],
                "evidence": item["evidence"],
            })
        try:
            ranking = agent._structured_deep_stage(
                CrossSectionalRanking,
                "sol_equal_depth_cross_sectional_ranking",
                "EQUAL_DEPTH_RANKING",
                (
                    "Rank all supplied finalists after applying equal evidence standards. Return exactly Top 5. "
                    "Scores must be relative to this single evaluation. Choose decision-relevant comparisons autonomously. "
                    "Use UNKNOWN when evidence is unavailable; do not treat missing data as automatic "
                    "negative evidence. Explain why rank 1 beats rank 2 and provide specific weaknesses for ranks 2 through 5.\n"
                    f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{json.dumps(packet, ensure_ascii=False, default=str)}\n"
                    f"ALL_EQUAL_DEPTH_RESEARCH_JSON:\n{json.dumps(compact, ensure_ascii=False, default=str)}"
                ),
                totals,
                metadata={"candidate_count": len(candidates), "all_candidates_same_depth": True},
            )
        except ResearchPending:
            write("result.json", {**manifest, **agent.pending_research,
                                 "deep_research": {s: r.model_dump(mode="json") for s, r in deep_results.items()},
                                 "usage": totals})
            return 2

    elapsed_ms = (time.perf_counter() - research_started) * 1000
    result = {
        **manifest,
        "status": "COMPLETE" if not failures else "INCOMPLETE",
        "evidence_assessments": agent.evidence_assessments,
        "selection_rationale": agent.selection_rationale.get("EQUAL_DEPTH_RANKING"),
        "candidate_symbols": candidates,
        "universe": source.get("universe"),
        "completed_deep_research": len(deep_results),
        "missing_deep_research": sorted(failures),
        "errors": failures,
        "new_tool_calls": new_call_count,
        "reused_tool_results": reused_count,
        "llm_usage": totals,
        "latency_ms": elapsed_ms,
        "provider_protocol": getattr(provider, "protocol", runtime.api_protocol),
        "reasoning_parameter_status": getattr(provider, "reasoning_parameter_status", "UNAVAILABLE"),
        "reasoning_metadata_status": getattr(provider, "reasoning_metadata_status", "UNAVAILABLE"),
        "ranking": ranking.model_dump(mode="json") if ranking else None,
        "deep_research": {symbol: item.model_dump(mode="json") for symbol, item in deep_results.items()},
    }
    write("result.json", result)
    write("portfolio_context.json", {"as_of": portfolio.as_of, "mode": "RESEARCH_ONLY"})
    if failures:
        print("Equal-depth research INCOMPLETE; see redacted result.json", flush=True)
        return 1
    print("Equal-depth research COMPLETE; no orders submitted.", flush=True)
    for item in ranking.ranking:
        print(f"RANK {item.rank}: {item.symbol} score={item.preliminary_alpha_score} confidence={item.confidence}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Equal-depth research startup FAILED ({type(exc).__name__}); no orders submitted", file=sys.stderr)
        raise SystemExit(1)
