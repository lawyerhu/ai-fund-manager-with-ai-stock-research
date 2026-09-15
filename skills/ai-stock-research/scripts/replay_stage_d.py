"""Replay a saved skill run's completed A/B/C evidence; only Stage D calls the LLM."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from uuid import uuid4

from research import NoTradingImports, require_models, research_runtime
from evidence_discipline import ResearchPending, install_evidence_discipline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import env, load_config, load_project_env
    from src.llm_agent import (CCSwitchProvider, LLMRuntimeConfig, SolResearchCIOAgent,
                               CrossSectionalRanking, DeepDiveResearch, AdversarialReview)
    from src.models import PortfolioState
    from src.storage import SQLiteStore, redact_sensitive

    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (args.source / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    research_starts = [e for e in events if e["event_type"] == "SOL_RESEARCH_STARTED"]
    if len(research_starts) != 1:
        raise ValueError("Source must contain exactly one research run")
    candidates = research_starts[0]["metadata"]["candidate_symbols"]
    horizon = research_starts[0]["metadata"]["horizon_days"]
    stages = [e for e in events if e["event_type"] in {
        "SOL_STAGE_A_RANKING_COMPLETED", "SOL_DEEP_DIVE_COMPLETED", "SOL_STAGE_C_ADVERSARIAL_COMPLETED"}]
    expected = [CrossSectionalRanking, DeepDiveResearch, DeepDiveResearch, DeepDiveResearch, AdversarialReview]
    if len(stages) != len(expected):
        raise ValueError("Missing completed A/B/C research; refusing Stage D replay")
    for event in stages:
        payload = event["metadata"]["result"]
        if any(item.get("research_requests") for item in payload.get("evidence_assessments", [])):
            raise ValueError("Saved A/B/C stage still requires evidence research; cannot replay final decision")
        if "evidence_assessments" in payload and not any(
            audit.get("event_type") == "SKILL_EVIDENCE_AUDIT_PASSED"
            and audit.get("symbol") == event.get("symbol")
            and audit.get("metadata", {}).get("stage") == event["event_type"].removesuffix("_COMPLETED")
            for audit in events[events.index(event) + 1:]
        ):
            raise ValueError("Saved A/B/C stage has not passed its evidence audit")
        # New skill audit fields are persisted separately from the legacy result schema.
        event["metadata"]["result"] = {key: value for key, value in payload.items() if key not in {"evidence_assessments", "selection_rationale"}}
    # Validate every saved result before any paid request.
    validated = [model.model_validate(event["metadata"]["result"]) for model, event in zip(expected, stages)]
    top = [row.symbol for row in validated[0].ranking[:3]]
    if [row.symbol for row in validated[1:4]] != top or {row.symbol for row in validated[4].reviews} != set(top):
        raise ValueError("Saved stages disagree on Top3")
    tool_events = [e for e in events if e["event_type"] == "SOL_TOOL_RESULT"]
    load_project_env(args.project / ".env", override=True)
    cfg = load_config(args.project / "config.yaml")
    runtime = research_runtime(LLMRuntimeConfig.from_mapping(cfg))
    database = Path(env("DATABASE_PATH") or cfg.get("portfolio", {}).get("database_path", "data/ai_fund_manager.sqlite3"))
    if not database.is_absolute():
        database = args.project / database
    with SQLiteStore(database, read_only=True) as store:
        row = store.connection.execute(
            "SELECT state_json FROM portfolio_snapshots WHERE json_extract(state_json,'$.as_of')=? ORDER BY id DESC LIMIT 1",
            (manifest["portfolio_snapshot_at"],)).fetchone()
    if row is None:
        raise ValueError("Original portfolio snapshot unavailable; cannot substitute current account")
    portfolio = PortfolioState.model_validate_json(row["state_json"])
    output = args.project / "outputs/skill-research" / ("stage-d-" + str(uuid4())) if args.run else None
    def sink(event):
        if output and output.exists():
            with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact_sensitive({"timestamp": datetime.now(timezone.utc).isoformat(), **event}), ensure_ascii=False, default=str) + "\n")

    class ReplayAgent(SolResearchCIOAgent):
        def _record_research_call(self, name, symbol, callback):
            if not self.saved_tools:
                raise ValueError("Missing saved tool result")
            event = self.saved_tools.popleft()
            if event["metadata"]["tool"] != name or event.get("symbol") != symbol:
                raise ValueError("Saved tool order/contract mismatch")
            return event["metadata"]["result"]

        def _structured_deep_stage(self, model_type, schema_name, event_prefix, prompt, totals, **kwargs):
            if event_prefix == "SOL_STAGE_D_IC_DECISION":
                if self.saved_stages or self.saved_tools:
                    raise ValueError("Incomplete replay context")
                if self.preflight:
                    raise PreflightComplete()
                return super()._structured_deep_stage(model_type, schema_name, event_prefix, prompt, totals, **kwargs)
            if not self.saved_stages:
                raise ValueError("Unexpected research stage")
            event = self.saved_stages.popleft()
            if event["event_type"] != event_prefix + "_COMPLETED" or event.get("symbol") != kwargs.get("symbol"):
                raise ValueError("Saved stage mismatch")
            return model_type.model_validate(event["metadata"]["result"])

    class PreflightComplete(Exception):
        pass

    def make_agent(provider, preflight=False):
        def no_data_calls():
            raise RuntimeError("Replay cannot fetch new research data")
        agent = ReplayAgent(runtime.sol_model, SimpleNamespace(market_regime=no_data_calls),
                           provider=provider, deep_research_enabled=True, event_sink=sink)
        agent.preflight = preflight
        agent.saved_stages = deque(stages)
        agent.saved_tools = deque(tool_events)
        # The original sizing context was not persisted. Disclose it instead of copying newer local history.
        agent.sizing_context = {"previous_review": "UNKNOWN: not persisted in source run",
                                "configured_risk_budget": "UNKNOWN: original configuration not persisted"}
        return agent

    try:
        make_agent(object(), preflight=True)._decide_deep(portfolio, candidates, horizon)
    except PreflightComplete:
        print("PREFLIGHT_PASS: saved A/B/C and tool sequence complete; Stage D only", flush=True)
    if not args.run:
        return
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    report = {"source_run": str(args.source), "source_evidence_time": research_starts[0]["timestamp"],
              "model_requested": runtime.sol_model, "reasoning_requested": "medium", "reused_tools": len(tool_events),
              "candidate_count": len(candidates), "sizing_context": "UNKNOWN: source context not persisted",
              "portfolio_source": "original local historical snapshot, not connector account",
              "policy_projection": "NOT_APPLIED: raw committee research", "placeOrder_calls": 0,
              "cancelOrder_calls": 0, "risk_approval": "NOT_RUN"}
    try:
        provider = CCSwitchProvider(runtime=runtime, event_sink=sink)
        require_models(provider, runtime, False)
        agent = make_agent(provider)
        def write(name, value):
            (output / name).write_text(json.dumps(redact_sensitive(value), ensure_ascii=False, indent=2), encoding="utf-8")
        install_evidence_discipline(agent, write, stages={"SOL_STAGE_D_IC_DECISION"})
        intent = agent._decide_deep(portfolio, candidates, horizon)
        report.update(status="COMPLETE", completed_at=datetime.now(timezone.utc).isoformat(),
                      evidence_assessments=agent.evidence_assessments,
                      decision=intent.model_dump(mode="json"), deep_research=agent.last_deep_research.model_dump(mode="json"),
                      usage=agent.last_usage)
    except ResearchPending:
        report.update(agent.pending_research)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc))
    (output / "result.json").write_text(json.dumps(redact_sensitive(report), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(report["status"], flush=True)
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
