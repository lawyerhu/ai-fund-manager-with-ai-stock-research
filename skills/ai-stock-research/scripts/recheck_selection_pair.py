"""Re-score CRM with Astra and compare it with the saved MPC selection.

This is a research-only replay. It reads saved evidence, uses the project's
CCSwitchProvider, and never imports the broker, scheduler, worker, or Risk
Engine. It deliberately does not inspect account holdings or write the
production database.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evidence_discipline import ResearchPending, install_evidence_discipline
from selection_state import resolve_active_selection, transition_fields


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from top_five_deep_research import (  # noqa: E402
    ASTRA_MODEL,
    EVIDENCE_PACKET_INSTRUCTIONS,
    NoTradingImports,
    PreviousWinnerComparison,
    validate_evidence_packet,
)


class CRMReScore(BaseModel):
    """A compact, auditable single-stock score from the fresh Astra call."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    alpha_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    score_basis: list[str] = Field(min_length=1, max_length=6)
    score_interpretation: str = Field(min_length=1)
    evidence_strengths: list[str] = Field(min_length=1, max_length=6)
    evidence_limitations: list[str] = Field(default_factory=list, max_length=8)
    expected_alpha_basis: list[str] = Field(min_length=1, max_length=8)
    estimate_type: str = "SOL_MODEL_ESTIMATE"
    bull_case: str = Field(min_length=1)
    base_case: str = Field(min_length=1)
    bear_case: str = Field(min_length=1)
    thesis_invalidation_conditions: list[str] = Field(min_length=1, max_length=5)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def extract_pair_context(source: dict) -> dict:
    """Extract CRM and the saved incumbent without consulting live state."""
    if source.get("status") != "COMPLETE":
        raise ValueError("Source result is not COMPLETE")
    ranking = source.get("final_ranking")
    rows = ranking.get("ranking") if isinstance(ranking, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Source result has no final ranking")
    crm_row = next(
        (row for row in rows if isinstance(row, dict) and str(row.get("symbol", "")).upper() == "CRM"),
        None,
    )
    if crm_row is None:
        raise ValueError("Source result has no CRM final-ranking record")
    supplemental = source.get("supplemental_research")
    crm_research = supplemental.get("CRM") if isinstance(supplemental, dict) else None
    if not isinstance(crm_research, dict):
        raise ValueError("Source result has no CRM supplemental research")
    incumbent = resolve_active_selection(source)
    if incumbent != "MPC":
        raise ValueError(f"Expected saved previous first-place symbol MPC, got {incumbent or 'missing'}")
    mpc_research = source.get("previous_first_supplemental_research")
    if not isinstance(mpc_research, dict) or str(mpc_research.get("symbol", "")).upper() != "MPC":
        raise ValueError("Source result has no saved MPC supplemental research")
    return {
        "crm_row": crm_row,
        "crm_research": crm_research,
        "mpc_research": mpc_research,
        "mpc_context": source.get("previous_winner_record") or {"winner_record": {"symbol": "MPC"}},
        "previous_score": crm_row.get("preliminary_alpha_score"),
        "incumbent_score": (source.get("previous_winner_record") or {}).get("winner_record", {}).get("preliminary_alpha_score"),
    }


def _model_ids(rows) -> set[str]:
    data = rows.get("data", []) if isinstance(rows, dict) else getattr(rows, "data", [])
    result = set()
    for row in data or []:
        value = row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
        if value:
            result.add(str(value))
    return result


def _safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _write(path: Path, name: str, value, redact):
    (path / name).write_text(
        json.dumps(redact(value), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def merge_evidence_addenda(packet: dict, addendum_paths: list[Path]) -> dict:
    """Attach newly verified, source-labeled evidence without changing the saved base packet."""
    merged = dict(packet)
    attached = []
    evidence = list(packet.get("evidence", []))
    for path in addendum_paths:
        addendum = load_json(path)
        attached.append({"path": str(path), "packet": addendum})
        rows = addendum.get("evidence")
        if isinstance(rows, list):
            evidence.extend(rows)
    merged["evidence"] = evidence
    merged["supplemental_evidence_addenda"] = attached
    return merged


def _crm_prompt(context: dict, packet: dict) -> str:
    return (
        "You are GPT-6 Astra. Re-score CRM independently for a concentrated-alpha research horizon of 20 "
        "trading days. This is a fresh single-stock assessment, not an order, account review, Risk Engine decision, "
        "or broker action. Return exactly one compact CRMReScore JSON object. alpha_score is a judgmental 0-100 "
        "relative opportunity score for this call, not a probability of profit, backtest result, or statistical alpha. "
        "confidence is subjective conviction in this judgment, not calibrated win probability. "
        f"{EVIDENCE_PACKET_INSTRUCTIONS} "
        "Decide autonomously which evidence is decision-relevant and how much it matters. Differences in dates, "
        "fiscal periods, lookback windows, vendor definitions, or quote timing are not automatically contradictions; "
        "determine whether each difference is timing, methodology, business change, or a material conflict. Do not "
        "mechanically lower confidence because a field is UNKNOWN or because two non-identical windows differ. Do "
        "not invent missing figures. If a gap remains, state its material effect and still give the best-available "
        "score and confidence. Use only the supplied evidence and distinguish observed facts from assumptions and "
        "SOL_MODEL_ESTIMATE. Do not return REVIEW_REQUIRED, CASH, or an abstention. Return schema JSON only; no "
        "private chain-of-thought.\n"
        f"CRM_PRIOR_FINAL_RANKING_RECORD:\n{_safe_json(context['crm_row'])}\n"
        f"CRM_PRIOR_SUPPLEMENTAL_RESEARCH:\n{_safe_json(context['crm_research'])}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{_safe_json(packet)}\n"
    )


def _comparison_prompt(context: dict, crm_rescore: dict, packet: dict) -> str:
    return (
        "Compare CRM, the newly rescored candidate, with MPC, the saved previous first-place research selection, "
        "in one common evidence context. This is a security-selection rotation recommendation only. Do not inspect "
        "or infer account holdings, cash, shares, target weights, costs, broker state, or orders. Return exactly "
        "new_first_symbol=CRM and previous_first_symbol=MPC in the PreviousWinnerComparison schema. "
        f"{EVIDENCE_PACKET_INSTRUCTIONS} "
        "You must choose exactly one rebalance_decision: SWITCH_TO_NEW_FIRST or KEEP_PREVIOUS. After the supplied "
        "evidence, make a best-available-evidence choice even if fields remain UNKNOWN. Do not abstain, return "
        "REVIEW_REQUIRED, default to KEEP_PREVIOUS merely because data are incomplete, or choose CASH. Explain why "
        "the selected action is preferable to its alternative, including material assumptions and what evidence would "
        "reverse it. Re-evaluate both securities in this request; do not subtract CRM's new score from MPC's old "
        "score. Set alpha_gap_status=COMPARABLE with a numeric gap only if a same-context comparable gap is explicitly "
        "supported; otherwise use alpha_gap_status=UNKNOWN and alpha_gap=null. Confidence is subjective conviction, "
        "not a profit probability. Return only schema JSON and no private chain-of-thought or orders.\n"
        f"NEW_FIRST_SYMBOL: CRM\nCRM_RESCORING:\n{_safe_json(crm_rescore)}\n"
        f"CRM_PRIOR_CONTEXT:\n{_safe_json({'ranking_record': context['crm_row'], 'research': context['crm_research']})}\n"
        f"PREVIOUS_FIRST_SYMBOL: MPC\nMPC_PRIOR_CONTEXT:\n{_safe_json(context['mpc_context'])}\n"
        f"MPC_SUPPLEMENTAL_RESEARCH:\n{_safe_json(context['mpc_research'])}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{_safe_json(packet)}\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--verification-packet", type=Path, required=True)
    parser.add_argument("--evidence-addendum", type=Path, action="append", default=[],
                        help="Additional source-labeled evidence JSON; may be supplied more than once")
    parser.add_argument("--run", action="store_true", help="Actually send Astra requests; never orders")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("CRM/MPC re-score requires --run")

    project = args.project.resolve()
    source_path = args.source_result.resolve()
    packet_path = args.verification_packet.resolve()
    source = load_json(source_path)
    packet = load_json(packet_path)
    validate_evidence_packet(packet)
    addendum_paths = [path.resolve() for path in args.evidence_addendum]
    packet = merge_evidence_addenda(packet, addendum_paths)
    context = extract_pair_context(source)
    if not (project / "src" / "llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")

    output = project / "outputs" / "skill-research" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-crm-mpc-recheck-" + str(uuid4())
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)

    # Import the project only after the research-only import guard is installed.
    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import load_config, load_project_env  # noqa: E402
    from src.llm_agent import (  # noqa: E402
        CCSwitchProvider,
        LLMRuntimeConfig,
        SolResearchCIOAgent,
        _accumulate_usage,
    )
    from src.storage import redact_sensitive  # noqa: E402

    def write(name, value):
        _write(output, name, value, redact_sensitive)

    def sink(event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact_sensitive(event), ensure_ascii=False, default=str) + "\n")

    manifest = {
        "mode": "crm_rescore_and_mpc_comparison",
        "research_model": ASTRA_MODEL,
        "research_effort_requested": "medium",
        "source_result": str(source_path),
        "verification_packet": str(packet_path),
        "verification_addenda": [str(path) for path in addendum_paths],
        "verification_packet_type": packet.get("packet_type"),
        "verification_basis": packet.get("as_of_basis"),
        "new_first_symbol": "CRM",
        "previous_first_symbol": "MPC",
        "prior_crm_score_for_audit_only": context.get("previous_score"),
        "prior_mpc_score_for_audit_only": context.get("incumbent_score"),
        "safety": "RESEARCH_ONLY; no account review; no broker/Risk Engine instance",
        "placeOrder_calls": 0,
        "cancelOrder_calls": 0,
        "risk_approval": "NOT_RUN",
        "account_review": "NOT_RUN",
    }
    write("manifest.json", manifest)

    started = time.perf_counter()
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    try:
        load_project_env(project / ".env", override=True)
        # A research replay never inherits a production execution mode.
        os.environ["EXECUTION_MODE"] = "OBSERVE"
        cfg = load_config(project / "config.yaml")
        runtime = replace(
            LLMRuntimeConfig.from_mapping(cfg),
            sol_model=ASTRA_MODEL,
            sol_reasoning_effort="medium",
            pipeline="LUNA_SOL",
            fallback_to_sol_only=False,
        )
        provider = CCSwitchProvider(runtime=runtime, event_sink=sink)
        model_ids = _model_ids(provider.client.models.list())
        sink({
            "event_type": "MODEL_DISCOVERY_COMPLETED",
            "component": "LLM_GATEWAY",
            "message": "Model discovery completed for CRM/MPC re-score",
            "metadata": {"required_model": ASTRA_MODEL, "required_model_available": ASTRA_MODEL in model_ids},
        })
        if ASTRA_MODEL not in model_ids:
            raise RuntimeError(f"Required model unavailable: {ASTRA_MODEL}")

        # The agent method supplies the exact project structured-output/provider path;
        # it does not call data tools for this evidence-replay request.
        agent = SolResearchCIOAgent(
            ASTRA_MODEL,
            None,
            provider=provider,
            max_tool_rounds=1,
            max_repair_retries=1,
            deep_research_enabled=False,
            event_sink=sink,
        )
        agent.external_verification = packet
        install_evidence_discipline(agent, write)
        crm = agent._structured_deep_stage(
            CRMReScore,
            "crm_rescore",
            "CRM_RESCORE",
            _crm_prompt(context, packet),
            totals,
            symbol="CRM",
            metadata={"prior_score_for_audit_only": context.get("previous_score")},
        )
        if crm.symbol.upper() != "CRM":
            raise ValueError(f"Astra returned {crm.symbol}, expected CRM")
        crm_json = crm.model_dump(mode="json")

        comparison = agent._structured_deep_stage(
            PreviousWinnerComparison,
            "crm_vs_mpc_comparison",
            "CRM_VS_MPC",
            _comparison_prompt(context, crm_json, packet),
            totals,
            symbol="CRM",
            metadata={"new_first_symbol": "CRM", "previous_first_symbol": "MPC"},
        )
        comparison_json = comparison.model_dump(mode="json")
        if comparison.new_first_symbol.upper() != "CRM":
            raise ValueError("Comparison returned an unexpected new-first symbol")
        if comparison.previous_first_symbol.upper() != "MPC":
            raise ValueError("Comparison returned an unexpected previous-first symbol")
        if comparison.alpha_gap_status == "COMPARABLE" and comparison.alpha_gap is None:
            raise ValueError("Comparable alpha gap cannot be null")
        if comparison.alpha_gap_status == "UNKNOWN" and comparison.alpha_gap is not None:
            raise ValueError("Unknown alpha gap must be null")

        result = {
            **manifest,
            "status": "COMPLETE",
            "evidence_assessments": agent.evidence_assessments,
            "crm_rescore": crm_json,
            "comparison": comparison_json,
            "rebalance_decision": comparison.rebalance_decision,
            **transition_fields("MPC", "CRM", comparison.rebalance_decision),
            "active_selection_research": context["mpc_research"] if comparison.rebalance_decision == "KEEP_PREVIOUS" else {"symbol": "CRM", **crm_json},
            "provider_protocol": f"{provider.protocol} via CCSwitchProvider",
            "protocol_events": provider.protocol_events,
            "reasoning_parameter_status": provider.reasoning_parameter_status,
            "reasoning_metadata_status": provider.reasoning_metadata_status,
            "token_usage_status": provider.token_usage_status,
            "usage": totals,
            "stage_metrics": agent.last_stage_metrics,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        write("active_selection.json", {"status": "COMPLETE", "result_path": str(output / "result.json"),
                                       **transition_fields("MPC", "CRM", comparison.rebalance_decision)})
        print(
            f"CRM/MPC RECHECK COMPLETE; CRM score={crm.alpha_score}; confidence={crm.confidence}; "
            f"decision={comparison.rebalance_decision}; protocol={provider.protocol}; no orders.",
            flush=True,
        )
        return 0
    except ResearchPending:
        write("result.json", {**manifest, **agent.pending_research, "usage": totals})
        print("CRM/MPC recheck NEEDS_RESEARCH; see research_pending.json", flush=True)
        return 2
    except Exception as exc:
        result = {
            **manifest,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "usage": totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        print(f"CRM/MPC RECHECK FAILED ({type(exc).__name__}); no orders submitted", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
