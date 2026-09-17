"""Legacy isolated research entry point; production database is always read-only.

The skill's research decision-maker is the current conversation model, which runs
through `session_handoff.py` and never calls a model API. This CCSwitch API path
is retained only as an explicitly requested fallback and no longer pins the
research model or a `medium` reasoning effort.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.abc
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from evidence_discipline import ResearchPending, install_evidence_discipline
from research_model import api_effort, api_model

ASTRA_MODEL = api_model()


class NoTradingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"src.main", "src.runner", "src.service", "src.scheduler", "src.risk_engine"} or fullname == "src.execution" or fullname.startswith("src.execution."):
            raise ImportError(f"Research-only skill blocks {fullname}")
        return None


def research_runtime(runtime):
    return replace(runtime, sol_model=ASTRA_MODEL, sol_reasoning_effort=api_effort(),
                   pipeline="LUNA_SOL", fallback_to_sol_only=False)


def require_models(provider, runtime, full):
    result = provider.client.models.list()
    rows = result.get("data", []) if isinstance(result, dict) else result.data
    names = {row.get("id") if isinstance(row, dict) else row.id for row in rows}
    required = {ASTRA_MODEL, runtime.luna_model} if full else {ASTRA_MODEL}
    if not required.issubset(names):
        raise ValueError("Required model unavailable: " + ", ".join(sorted(required - names)))


def source_context(store, decision_id=None):
    snapshot = store.latest_portfolio()
    if snapshot is None:
        raise ValueError("No persisted portfolio snapshot; do not invent a portfolio")
    portfolio = snapshot[0]
    query = "SELECT decision_id,recorded_at,candidate_symbols_json FROM pipeline_runs"
    if decision_id:
        row = store.connection.execute(query + " WHERE decision_id=?", (decision_id,)).fetchone()
    else:
        row = store.connection.execute(query + " ORDER BY recorded_at DESC LIMIT 1").fetchone()
    source = dict(row) if row else {}
    candidates = json.loads(source.pop("candidate_symbols_json", "[]"))
    if not isinstance(candidates, list) or any(not isinstance(s, str) or not s for s in candidates):
        raise ValueError("Invalid persisted candidates")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate persisted candidates")
    position = store.active_managed_position(portfolio.current_symbol) if portfolio.current_symbol else None
    reviews = store.position_reviews(position.position_id, 1) if position else []
    context = {"previous_review": reviews[0].model_dump(mode="json", exclude={"comparison_context"}) if reviews else None}
    return portfolio, position, candidates, source, context


def research_only_context(store, decision_id=None):
    """Read candidate provenance only; never load an account or managed position."""
    from src.models import PortfolioState

    query = "SELECT decision_id,recorded_at,candidate_symbols_json FROM pipeline_runs"
    if decision_id:
        row = store.connection.execute(query + " WHERE decision_id=?", (decision_id,)).fetchone()
    else:
        row = store.connection.execute(query + " ORDER BY recorded_at DESC LIMIT 1").fetchone()
    source = dict(row) if row else {}
    candidates = json.loads(source.pop("candidate_symbols_json", "[]"))
    if not isinstance(candidates, list) or any(not isinstance(s, str) or not s for s in candidates):
        raise ValueError("Invalid persisted candidates")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate persisted candidates")
    # The core agents require a PortfolioState-shaped value, but this is deliberately
    # not an account snapshot and must never be populated from SQLite or a broker.
    research_context = PortfolioState(equity=0.0, peak_equity=0.0, cash=0.0, as_of="RESEARCH_ONLY")
    return research_context, None, candidates, source, {}


def source_universe(store, decision_id):
    """Recover the source screen's recorded pool size, never today's replacement universe."""
    if not decision_id:
        return None
    rows = store.connection.execute(
        "SELECT timestamp,event_type,metadata_json FROM runtime_events WHERE decision_id=? "
        "AND event_type IN ('AI_PROGRESS','LUNA_SCREEN_STARTED') ORDER BY id", (decision_id,)
    ).fetchall()
    for row in reversed(rows):
        metadata = json.loads(row["metadata_json"])
        count = metadata.get("universe_total") if row["event_type"] == "AI_PROGRESS" else metadata.get("universe_size")
        if isinstance(count, int) and count > 0:
            return {"count": count, "source_decision_id": decision_id, "recorded_at": row["timestamp"],
                    "source_event": row["event_type"]}
    return None


def execute_research(mode, pipeline, portfolio, position, candidates, context, cfg):
    pipeline.sol.sizing_context = context
    pipeline.sol.external_verification = context.get("verified_external_evidence") or {}
    horizon = int(cfg.get("agent", {}).get("decision_horizon_days", 20))
    if mode == "full":
        try:
            decision = pipeline.decide(portfolio, horizon)
        except Exception as exc:
            # The project wraps CIO exceptions in PipelineError; keep this as a
            # research continuation, not a failed screen or a fallback decision.
            if getattr(pipeline.sol, "pending_research", None):
                raise ResearchPending("Decision-critical evidence requires actual research") from exc
            raise
        return {"decision": decision.model_dump(mode="json"), "pipeline": pipeline.last_pipeline_metadata,
                "usage": pipeline.last_usage, "luna_batches": pipeline.last_luna_batches}
    if mode == "candidates":
        if not candidates:
            raise ValueError("No saved candidates available")
        decision = pipeline.sol.decide(portfolio, candidates, horizon_days=horizon)
        deep = getattr(pipeline.sol, "last_deep_research", None)
        return {"decision": decision.model_dump(mode="json"),
                "deep_research": deep.model_dump(mode="json") if deep else None,
                "usage": pipeline.sol.last_usage, "evidence": pipeline.sol.last_research_evidence,
                "tool_calls": pipeline.sol.last_tool_calls}
    if position is None:
        raise ValueError("No stored managed position to review")
    review = pipeline.sol.review_position(position, portfolio, candidate_symbols=candidates,
        review_type="DAILY", event_context={**context, "event_type": "SKILL_RESEARCH_ONLY"},
        replacement_threshold=float(cfg.get("portfolio_review", {}).get("replacement_threshold", 10)))
    return {"review": review.model_dump(mode="json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "full", "candidates", "review"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--decision-id")
    parser.add_argument("--verification-packet", type=Path,
                        help="Independent, redacted evidence packet produced after active source verification")
    parser.add_argument("--run", action="store_true", help="Actually send research API requests (never orders)")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if not (project / "src/llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")
    if args.mode != "inspect" and not args.run:
        parser.error("Research requires --run; use inspect for a non-network check")
    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import load_config, load_project_env, env
    load_project_env(project / ".env", override=True)
    # Subprocess overrides only. No writes to .env or runtime_state.
    os.environ["EXECUTION_MODE"] = "OBSERVE"
    from src.llm_agent import CCSwitchProvider, LLMRuntimeConfig, LunaSolPipeline
    from src.storage import SQLiteStore, redact_sensitive
    from src.data_provider import YahooFinanceDataProvider

    cfg = load_config(project / "config.yaml")
    runtime = research_runtime(LLMRuntimeConfig.from_mapping(cfg))
    db = Path(env("DATABASE_PATH") or cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3"))
    if not db.is_absolute():
        db = project / db
    if args.mode == "review":
        raise ValueError("The skill no longer performs account or position review; compare against --previous-result instead")
    with SQLiteStore(db, read_only=True) as store:
        portfolio, position, candidates, source, context = research_only_context(store, args.decision_id)
        source["universe"] = source_universe(store, source.get("decision_id"))
    verification_packet = {}
    if args.verification_packet:
        verification_packet = json.loads(args.verification_packet.resolve().read_text(encoding="utf-8"))
        if not isinstance(verification_packet, dict) or not isinstance(verification_packet.get("evidence"), list):
            raise ValueError("Verification packet must contain an evidence list")
        context["verified_external_evidence"] = verification_packet
    manifest = {
        "mode": args.mode, "luna_model": runtime.luna_model, "luna_effort": runtime.luna_reasoning_effort,
        "research_model": runtime.sol_model, "research_effort_requested": runtime.sol_reasoning_effort,
        "reasoning_effective": "UNAVAILABLE", "returned_model": "UNAVAILABLE",
        "portfolio_snapshot_at": "NOT_READ", "source": source, "candidate_count": len(candidates),
        "safety": "RESEARCH_ONLY; production DB read-only; no broker/Risk Engine instance",
        "placeOrder_calls": 0, "cancelOrder_calls": 0, "risk_approval": "NOT_RUN",
        "account_review": "NOT_RUN; previous-first comparison is handled separately and no holdings are loaded",
        "verification_packet": str(args.verification_packet.resolve()) if args.verification_packet else None,
    }
    if args.mode == "inspect":
        print(json.dumps(redact_sensitive(manifest), ensure_ascii=False, indent=2))
        return 0
    if cfg.get("data", {}).get("provider", "mock").lower() not in {"yahoo", "yahoo_finance"}:
        raise ValueError("Real research requires configured Yahoo provider; refusing mock data")
    if not runtime.deep_research_enabled and args.mode != "review":
        raise ValueError("Project deep research must be enabled for the complete CIO workflow")
    if args.mode == "candidates" and not candidates:
        raise ValueError("No persisted candidates for requested decision")
    output = project / "outputs" / "skill-research" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid4()))
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    def write(name, value):
        (output / name).write_text(json.dumps(redact_sensitive(value), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    def event_sink(event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact_sensitive(event), ensure_ascii=False, default=str) + "\n")
    write("manifest.json", manifest)
    try:
        provider = CCSwitchProvider(runtime=runtime, event_sink=event_sink)
        require_models(provider, runtime, args.mode == "full")
        pipeline = LunaSolPipeline(YahooFinanceDataProvider(), provider, runtime,
            max_tool_rounds=int(cfg.get("agent", {}).get("max_tool_rounds", 12)),
            max_candidates=int(cfg.get("agent", {}).get("max_candidates", 25)), event_sink=event_sink)
        install_evidence_discipline(pipeline.sol, write, capture_selection_path=True)
        result = execute_research(args.mode, pipeline, portfolio, position, candidates, context, cfg)
        write("result.json", {**manifest, "status": "COMPLETE", "result": result,
                              "evidence_assessments": pipeline.sol.evidence_assessments,
                              "selection_rationale": pipeline.sol.selection_rationale.get("SOL_STAGE_A_RANKING"),
                              "candidate_symbols": list(pipeline.sol._candidate_symbols),
                              "candidate_count": len(pipeline.sol._candidate_symbols),
                              "universe": next(({"count": len(row["result"]),
                                                 "symbols": [item["symbol"] for item in row["result"]]}
                                                for row in pipeline.last_research_evidence
                                                if row.get("tool") == "universe_snapshot"), source.get("universe")),
                              "reasoning_parameter_status": provider.reasoning_parameter_status,
                              "reasoning_metadata_status": provider.reasoning_metadata_status})
    except ResearchPending:
        write("result.json", {**manifest, **pipeline.sol.pending_research})
        print("Research NEEDS_RESEARCH; retrieve the requested evidence before continuing.", flush=True)
        return 2
    except Exception as exc:
        write("result.json", {**manifest, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
        print("Research FAILED; see redacted result.json", flush=True)
        return 1
    print("Research COMPLETE; research suggestions only, no orders.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not echo arbitrary provider/config strings before redaction is ready.
        print(f"Research startup FAILED ({type(exc).__name__}); no orders submitted", file=sys.stderr)
        raise SystemExit(1)
