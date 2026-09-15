"""Ask the same GPT-6 Astra research path to turn observed gaps into a verification plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.abc
import json
from pathlib import Path
import sys
from uuid import uuid4

from pydantic import BaseModel, Field

ASTRA_MODEL = "gpt-6-astra"


class NoTradingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"src.main", "src.runner", "src.service", "src.scheduler", "src.risk_engine"} or fullname == "src.execution" or fullname.startswith("src.execution."):
            raise ImportError(f"Research-only skill blocks {fullname}")
        return None


class VerificationItem(BaseModel):
    symbol: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    query: str = Field(min_length=1)
    decision_impact: str = Field(pattern="^(HIGH|MEDIUM|LOW)$")
    evidence_needed: list[str] = Field(min_length=1, max_length=6)


class VerificationPlan(BaseModel):
    items: list[VerificationItem] = Field(min_length=1, max_length=20)
    stopping_rule: str = Field(min_length=1)


def _schema(model_type):
    schema = model_type.model_json_schema()
    schema["additionalProperties"] = False
    for value in schema.get("$defs", {}).values():
        if isinstance(value, dict):
            value["additionalProperties"] = False
    return {"type": "json_schema", "name": "gpt6_information_verification_plan", "strict": True, "schema": schema}


def _response_text(response):
    value = getattr(response, "output_text", None)
    if value:
        return value
    if isinstance(response, dict):
        value = response.get("output_text")
        if value:
            return value
    raise ValueError("Provider returned no structured text")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    baseline_path = args.baseline.resolve()
    if not (project / "src/llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") not in {"COMPLETE", "NEEDS_RESEARCH"}:
        raise ValueError("Verification planning requires a saved baseline or pending research result")
    deep = baseline.get("result", {}).get("deep_research") or {}
    final = deep.get("final_decision") or {}
    ranking = deep.get("ranking", {}).get("ranking") or baseline.get("ranking", {}).get("ranking") or []
    missing = final.get("missing_data") or []
    coverage = deep.get("tool_coverage", {}).get("missing_sources") or []
    ranked_candidates = [
        {
            "rank": item.get("rank"),
            "symbol": item.get("symbol"),
            "score": item.get("preliminary_alpha_score"),
            "unknown_fields": item.get("unknown_fields", []),
        }
        for item in ranking
    ]
    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import load_config, load_project_env
    from src.llm_agent import CCSwitchProvider, LLMRuntimeConfig
    from research import research_runtime

    load_project_env(project / ".env", override=True)
    cfg = load_config(project / "config.yaml")
    runtime = research_runtime(LLMRuntimeConfig.from_mapping(cfg))
    events: list[dict] = []

    def event_sink(event):
        events.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})

    provider = CCSwitchProvider(runtime=runtime, event_sink=event_sink)
    prompt = (
        "You are GPT-6 Astra planning a bounded information-completion pass for an existing stock research run. "
        "Do not invent facts and do not make a trade decision. Independently identify decision-critical unknowns "
        "from the supplied company research and observed gaps; do not limit attention to the prior Top 5 or to a "
        "fixed financial checklist. Return at most 20 prioritized verification items for this batch; disclose any "
        "remaining critical questions in stopping_rule instead of silently dropping them. Prefer original company "
        "announcements, financial reports, IR and regulatory/SEC filings; choose suitable sources autonomously. Each query must be concrete "
        "enough for a researcher to search, and each item must say what evidence would resolve it. Prioritize questions "
        "that could materially change rank, the winner or the new-versus-previous selection. Prioritize any unresolved "
        "fact on which the winner depends. Choose sources and depth autonomously within the existing two supplemental "
        "rounds; a failed search does not imply an adverse fact or justify a mechanical investment-score penalty. "
        "A plan item is not evidence; do not state that the fact is true. Use UNKNOWN when a source cannot be identified. "
        "Return only the exact JSON schema.\n"
        f"RANKED_CANDIDATES_JSON:\n{json.dumps(ranked_candidates, ensure_ascii=False)}\n"
        f"EVIDENCE_ASSESSMENTS:\n{json.dumps(baseline.get('evidence_assessments', []), ensure_ascii=False)}\n"
        f"PENDING_REQUESTS:\n{json.dumps(baseline.get('research_requests', []), ensure_ascii=False)}\n"
        f"FINAL_MISSING_DATA:\n{json.dumps(missing, ensure_ascii=False)}\n"
        f"TOOL_COVERAGE_MISSING:\n{json.dumps(coverage, ensure_ascii=False)}\n"
        f"BASELINE_SELECTED_SYMBOL: {final.get('selected_symbol') or baseline.get('result', {}).get('decision', {}).get('symbol')}\n"
    )
    response = provider.create_response(
        model=ASTRA_MODEL,
        input=prompt,
        text={"format": _schema(VerificationPlan)},
    )
    plan = VerificationPlan.model_validate_json(_response_text(response))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": str(uuid4()),
        "baseline": str(baseline_path),
        "model": ASTRA_MODEL,
        "reasoning_effort": "medium",
        "candidate_count": baseline.get("candidate_count"),
        "selected_symbol": final.get("selected_symbol") or baseline.get("result", {}).get("decision", {}).get("symbol"),
        "plan": plan.model_dump(mode="json"),
        "provider_telemetry": {
            "reasoning_metadata_status": provider.reasoning_metadata_status,
            "token_usage_status": provider.token_usage_status,
            "protocol": provider.protocol,
        },
        "events": events,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"VERIFICATION_PLAN={output}")
    print(f"ITEMS={len(plan.items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
