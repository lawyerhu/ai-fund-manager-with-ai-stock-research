"""Re-score any challenger and compare it with the saved active selection.

This is an explicit legacy-API research-only replay. It reads saved evidence,
uses the project's ``CCSwitchProvider``, and never imports the broker,
scheduler, worker, or Risk Engine. It deliberately does not inspect account
holdings or write the production database. The pair is supplied by symbols or
derived from the saved research result; no historical symbol is special-cased.
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
from research_model import api_effort
from selection_state import resolve_active_selection, transition_fields


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from top_five_deep_research import (  # noqa: E402
    ASTRA_MODEL,
    EVIDENCE_PACKET_INSTRUCTIONS,
    NoTradingImports,
    PreviousWinnerComparison,
    pair_evidence_audit,
    validate_evidence_packet,
)


class CandidateReScore(BaseModel):
    """A compact, auditable score for the explicitly supplied challenger."""

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


def _ranking_rows(source: dict) -> list[dict]:
    """Read current and older ranking wrappers without changing their shape."""
    containers = [source]
    nested = source.get("result")
    if isinstance(nested, dict):
        containers.append(nested)
    for container in containers:
        ranking = container.get("final_ranking") or container.get("ranking")
        if isinstance(ranking, dict) and isinstance(ranking.get("ranking"), list):
            return [row for row in ranking["ranking"] if isinstance(row, dict)]
        if isinstance(ranking, list):
            return [row for row in ranking if isinstance(row, dict)]
    raise ValueError("Source result has no final ranking")


def _symbol_from_record(record) -> str:
    return str(record.get("symbol", "")).strip().upper() if isinstance(record, dict) else ""


def _research_record(source: dict, wanted: str) -> dict | None:
    """Find a saved research record for a symbol across current/legacy shapes."""
    containers = [source]
    nested = source.get("result")
    if isinstance(nested, dict):
        containers.append(nested)
    for container in containers:
        for key in ("supplemental_research", "active_selection_research",
                    "previous_first_supplemental_research", "challenger_research",
                    "incumbent_research"):
            value = container.get(key)
            if isinstance(value, dict):
                direct = value.get(wanted) or value.get(wanted.upper())
                if isinstance(direct, dict):
                    return direct
                if _symbol_from_record(value) == wanted:
                    return value
            elif isinstance(value, list):
                for row in value:
                    if _symbol_from_record(row) == wanted:
                        return row
        # Some older outputs store the map under a symbol key in a generic
        # research object. Only a matching symbol key is accepted.
        for value in container.values():
            if isinstance(value, dict):
                direct = value.get(wanted) or value.get(wanted.upper())
                if isinstance(direct, dict):
                    return direct
    return None


def _resolve_challenger(source: dict, rows: list[dict], requested: str | None, incumbent: str) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    for container in (source, source.get("result") if isinstance(source.get("result"), dict) else {}):
        for key in ("new_first_symbol", "selected_symbol"):
            if container.get(key):
                candidates.append(container[key])
    candidates.extend(row.get("symbol") for row in rows)
    for value in candidates:
        normalized = str(value or "").strip().upper()
        if normalized and normalized != incumbent:
            if not normalized[0].isalpha() or not all(char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in normalized):
                raise ValueError(f"Invalid challenger symbol: {value}")
            return normalized
    raise ValueError("A challenger symbol different from the incoming active selection is required")


def extract_pair_context(source: dict, challenger_symbol: str | None = None,
                         incumbent_symbol: str | None = None) -> dict:
    """Extract an arbitrary challenger and the saved research incumbent."""
    if source.get("status") != "COMPLETE":
        raise ValueError("Source result is not COMPLETE")
    rows = _ranking_rows(source)
    incumbent = resolve_active_selection(source)
    if incumbent_symbol and str(incumbent_symbol).strip().upper() != incumbent:
        raise ValueError("Requested incumbent symbol disagrees with saved active selection")
    challenger = _resolve_challenger(source, rows, challenger_symbol, incumbent)
    challenger_row = next((row for row in rows if _symbol_from_record(row) == challenger), None)
    if challenger_row is None:
        raise ValueError(f"Source result has no {challenger} final-ranking record")
    challenger_research = _research_record(source, challenger)
    if not isinstance(challenger_research, dict):
        raise ValueError(f"Source result has no {challenger} supplemental research")
    incumbent_research = _research_record(source, incumbent)
    if not isinstance(incumbent_research, dict):
        raise ValueError(f"Source result has no {incumbent} supplemental research")
    previous_context = source.get("previous_winner_record")
    if not isinstance(previous_context, dict):
        previous_context = {"winner_record": {"symbol": incumbent, **incumbent_research}}
    winner_record = previous_context.get("winner_record") if isinstance(previous_context.get("winner_record"), dict) else {}
    return {
        "challenger_symbol": challenger,
        "incumbent_symbol": incumbent,
        "challenger_row": challenger_row,
        "challenger_research": challenger_research,
        "incumbent_research": incumbent_research,
        "incumbent_context": previous_context,
        "previous_score": challenger_row.get("preliminary_alpha_score"),
        "incumbent_score": winner_record.get("preliminary_alpha_score"),
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
    """Attach source-labeled evidence without changing the saved base packet."""
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


def _candidate_prompt(context: dict, packet: dict) -> str:
    challenger = context["challenger_symbol"]
    incumbent = context["incumbent_symbol"]
    return (
        "You are the research decision model for the current run. Re-score the explicitly supplied challenger "
        f"{challenger} independently against the incoming research-layer incumbent {incumbent} for a concentrated-alpha "
        "research comparison. This is a fresh single-stock assessment, not an order, account review, Risk Engine "
        "decision, or broker action. Return exactly one compact CandidateReScore JSON object. alpha_score is a "
        "judgmental 0-100 relative opportunity score for this call, not a probability of profit, backtest result, "
        "or statistical alpha. confidence is subjective conviction in this judgment, not calibrated win probability. "
        f"{EVIDENCE_PACKET_INSTRUCTIONS} "
        "Decide autonomously which evidence is decision-relevant and how much it matters. Differences in dates, "
        "fiscal periods, lookback windows, vendor definitions, or quote timing are not automatically contradictions; "
        "determine whether each difference is timing, methodology, business change, or a material conflict. Do not "
        "mechanically lower confidence because a field is UNKNOWN or because two non-identical windows differ. Do "
        "not invent missing figures. If a gap remains, state its material effect and still give the best-available "
        "score and confidence. Use only supplied evidence and distinguish observed facts from assumptions and "
        "SOL_MODEL_ESTIMATE. Do not return REVIEW_REQUIRED, CASH, or an abstention. Return schema JSON only; no "
        "private chain-of-thought.\n"
        f"CHALLENGER_SYMBOL: {challenger}\n"
        f"INCUMBENT_SYMBOL_FOR_CONTEXT_ONLY: {incumbent}\n"
        f"CHALLENGER_PRIOR_FINAL_RANKING_RECORD:\n{_safe_json(context['challenger_row'])}\n"
        f"CHALLENGER_PRIOR_SUPPLEMENTAL_RESEARCH:\n{_safe_json(context['challenger_research'])}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{_safe_json(packet)}\n"
    )


def _comparison_prompt(context: dict, candidate_rescore: dict, packet: dict, pair_audit: dict) -> str:
    challenger = context["challenger_symbol"]
    incumbent = context["incumbent_symbol"]
    return (
        f"Compare {challenger}, the newly rescored challenger, with {incumbent}, the saved incoming active research "
        "selection, in one common evidence context. This is a security-selection recommendation only. Do not inspect "
        "or infer account holdings, cash, shares, target weights, costs, broker state, or orders. Return exactly "
        f"new_first_symbol={challenger} and previous_first_symbol={incumbent} in the PreviousWinnerComparison schema. "
        f"{EVIDENCE_PACKET_INSTRUCTIONS} "
        "Answer the sole forward-looking question: from current executable prices, is continuing with the incumbent "
        "or switching to the challenger more likely to deliver higher relative return after a reasonable research-layer "
        "friction assumption? There is no fixed score, confidence, alpha-gap or replacement hurdle. A small advantage "
        "inside the model's own error may support KEEP_PREVIOUS; a small, highly credible net advantage that covers "
        "research friction may support SWITCH_TO_NEW_FIRST. Neither choice requires a broken incumbent thesis, a "
        "minimum holding period, a fixed elapsed time, a catalyst within 20 trading days, or a particular style or "
        "industry. This is model judgment, not a program threshold. "
        "After supplied evidence searches, make a best-available-evidence choice even if fields remain unresolved. Do "
        "not abstain, default to KEEP_PREVIOUS because data are incomplete, or choose CASH. Do not treat the "
        "challenger's newer or more complete information as an investment advantage merely because the incumbent has "
        "a RETRIEVAL_FAILED or STALE field; first resolve that asymmetry. NOT_PUBLIC, PAID_DATA_REQUIRED and "
        "NOT_YET_OCCURRED may remain after reasonable search and do not by themselves block the final model decision. "
        "Explain the remaining expectation gap: what is priced, what verified evidence implies, how earnings/revenue/FCF "
        "revisions support or fail to support price, whether multiple expansion did the work, catalyst pricing status, "
        "the key downside path and why the chosen security is better from now. Event presence is evidence, not Alpha, "
        "and no near-term event is required. Ignore entry cost, floating P&L, historical ranking and holding time as "
        "sunk-cost reasons. If an energy/refiner thesis depends on the cycle, consider relevant operating evidence "
        "without applying a fixed factor model. Return pair_comparison_complete, decision_basis_sufficient and "
        "material_asymmetry_resolved as true only after the substantive pair audit. The structural pair audit below is "
        "context, not an investment score or automatic action. Re-evaluate both securities; do not subtract a new "
        "score from an old score. Set alpha_gap_status=COMPARABLE with a numeric gap only when same-context comparable "
        "evidence supports it; otherwise use UNKNOWN and alpha_gap=null. Confidence is subjective conviction, not a "
        "profit probability. Return schema JSON only and no private chain-of-thought or orders.\n"
        f"NEW_FIRST_SYMBOL: {challenger}\nCHALLENGER_RESCORING:\n{_safe_json(candidate_rescore)}\n"
        f"CHALLENGER_PRIOR_CONTEXT:\n{_safe_json({'ranking_record': context['challenger_row'], 'research': context['challenger_research']})}\n"
        f"PREVIOUS_FIRST_SYMBOL: {incumbent}\nINCUMBENT_PRIOR_CONTEXT:\n{_safe_json(context['incumbent_context'])}\n"
        f"INCUMBENT_SUPPLEMENTAL_RESEARCH:\n{_safe_json(context['incumbent_research'])}\n"
        f"PAIR_EVIDENCE_AUDIT:\n{_safe_json(pair_audit)}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{_safe_json(packet)}\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--verification-packet", type=Path, required=True)
    parser.add_argument("--challenger-symbol", help="New candidate symbol; defaults to saved new-first/first ranking record")
    parser.add_argument("--incumbent-symbol", help="Incoming research-layer active selection; defaults to saved active_selection")
    parser.add_argument("--evidence-addendum", type=Path, action="append", default=[],
                        help="Additional source-labeled evidence JSON; may be supplied more than once")
    parser.add_argument("--run", action="store_true", help="Actually send the explicitly requested legacy API requests; never orders")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Generic pair re-score requires --run")

    project = args.project.resolve()
    source_path = args.source_result.resolve()
    packet_path = args.verification_packet.resolve()
    source = load_json(source_path)
    packet = load_json(packet_path)
    packet = validate_evidence_packet(packet) or packet
    addendum_paths = [path.resolve() for path in args.evidence_addendum]
    packet = merge_evidence_addenda(packet, addendum_paths)
    packet = validate_evidence_packet(packet) or packet
    context = extract_pair_context(source, args.challenger_symbol, args.incumbent_symbol)
    challenger = context["challenger_symbol"]
    incumbent = context["incumbent_symbol"]
    if not (project / "src" / "llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")

    output = project / "outputs" / "skill-research" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-pair-recheck-" + str(uuid4())
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
    )
    from src.storage import redact_sensitive  # noqa: E402

    def write(name, value):
        _write(output, name, value, redact_sensitive)

    def sink(event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact_sensitive(event), ensure_ascii=False, default=str) + "\n")

    manifest = {
        "mode": "candidate_rescore_and_incumbent_comparison",
        "research_model": ASTRA_MODEL,
        "research_effort_requested": "medium",
        "source_result": str(source_path),
        "verification_packet": str(packet_path),
        "verification_addenda": [str(path) for path in addendum_paths],
        "verification_packet_type": packet.get("packet_type"),
        "verification_basis": packet.get("as_of_basis"),
        "challenger_symbol": challenger,
        "incumbent_symbol": incumbent,
        "new_first_symbol": challenger,
        "previous_first_symbol": incumbent,
        "prior_challenger_score_for_audit_only": context.get("previous_score"),
        "prior_incumbent_score_for_audit_only": context.get("incumbent_score"),
        "research_layer_friction_assumption": "STANDARDIZED_OR_CONFIGURED_RESEARCH_LAYER_ASSUMPTION_ONLY; NOT_ACCOUNT_COMMISSION_OR_SLIPPAGE",
        "active_selection_meaning": "RESEARCH_LAYER_CURRENT_AI_PREFERENCE; NOT_BROKER_POSITION",
        "safety": "RESEARCH_ONLY; no account review; no broker/Risk Engine instance",
        "placeOrder_calls": 0,
        "cancelOrder_calls": 0,
        "risk_approval": "NOT_RUN",
        "account_review": "NOT_RUN",
    }
    write("manifest.json", manifest)

    started = time.perf_counter()
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    agent = None
    try:
        load_project_env(project / ".env", override=True)
        os.environ["EXECUTION_MODE"] = "OBSERVE"
        cfg = load_config(project / "config.yaml")
        runtime = replace(
            LLMRuntimeConfig.from_mapping(cfg),
            sol_model=ASTRA_MODEL,
            sol_reasoning_effort=api_effort(),
            pipeline="LUNA_SOL",
            fallback_to_sol_only=False,
        )
        provider = CCSwitchProvider(runtime=runtime, event_sink=sink)
        model_ids = _model_ids(provider.client.models.list())
        sink({
            "event_type": "MODEL_DISCOVERY_COMPLETED",
            "component": "LLM_GATEWAY",
            "message": "Model discovery completed for generic pair re-score",
            "metadata": {"required_model": ASTRA_MODEL, "required_model_available": ASTRA_MODEL in model_ids},
        })
        if ASTRA_MODEL not in model_ids:
            raise RuntimeError(f"Required model unavailable: {ASTRA_MODEL}")

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
        candidate = agent._structured_deep_stage(
            CandidateReScore,
            "candidate_rescore",
            "CANDIDATE_RESCORE",
            _candidate_prompt(context, packet),
            totals,
            symbol=challenger,
            metadata={"prior_score_for_audit_only": context.get("previous_score")},
        )
        if candidate.symbol.upper() != challenger:
            raise ValueError(f"Model returned {candidate.symbol}, expected {challenger}")
        candidate_json = candidate.model_dump(mode="json")

        historical_assessments = source.get("evidence_assessments", [])
        pair_audit = pair_evidence_audit(
            packet,
            [*(historical_assessments if isinstance(historical_assessments, list) else []),
              *agent.evidence_assessments],
            challenger,
            incumbent,
            additional_research=[context["incumbent_research"]],
        )
        write("pair_evidence_audit.json", pair_audit)
        if pair_audit["requires_targeted_research"]:
            pending = {
                **manifest,
                "status": "NEEDS_RESEARCH",
                "phase": "CHALLENGER_VS_INCUMBENT_COMPARISON",
                "pair_evidence_audit": pair_audit,
                "research_requests": pair_audit["research_requests"],
                "candidate_rescore": candidate_json,
                "challenger_rescore": candidate_json,
                "incumbent_research": context["incumbent_research"],
                "research_rounds_remaining": pair_audit["rounds_remaining"],
                "next_step": "只补查指定 challenger/incumbent 成对比较中受影响的变量；不重跑 Luna 或整个 Top 5。",
                "usage": totals,
            }
            write("research_pending.json", pending)
            write("result.json", pending)
            print("Pair evidence audit NEEDS_RESEARCH; active selection was not advanced.", flush=True)
            return 2

        comparison = agent._structured_deep_stage(
            PreviousWinnerComparison,
            "candidate_vs_incumbent_comparison",
            "CANDIDATE_VS_INCUMBENT",
            _comparison_prompt(context, candidate_json, packet, pair_audit),
            totals,
            symbol=challenger,
            metadata={"new_first_symbol": challenger, "previous_first_symbol": incumbent},
        )
        comparison_json = comparison.model_dump(mode="json")
        if comparison.new_first_symbol.upper() != challenger:
            raise ValueError("Comparison returned an unexpected challenger symbol")
        if comparison.previous_first_symbol.upper() != incumbent:
            raise ValueError("Comparison returned an unexpected incumbent symbol")
        if comparison.alpha_gap_status == "COMPARABLE" and comparison.alpha_gap is None:
            raise ValueError("Comparable alpha gap cannot be null")
        if comparison.alpha_gap_status == "UNKNOWN" and comparison.alpha_gap is not None:
            raise ValueError("Unknown alpha gap must be null")
        pair_flags = ("pair_comparison_complete", "decision_basis_sufficient", "material_asymmetry_resolved")
        if any(comparison_json.get(key) is not True for key in pair_flags):
            pending = {
                **manifest,
                "status": "NEEDS_RESEARCH",
                "phase": "CHALLENGER_VS_INCUMBENT_COMPARISON",
                "pair_evidence_audit": {**pair_audit, "model_flags": {key: comparison_json.get(key) for key in pair_flags}},
                "research_requests": pair_audit["research_requests"],
                "candidate_rescore": candidate_json,
                "challenger_rescore": candidate_json,
                "incumbent_research": context["incumbent_research"],
                "comparison_provisional": comparison_json,
                "research_rounds_remaining": pair_audit["rounds_remaining"],
                "usage": totals,
            }
            write("research_pending.json", pending)
            write("result.json", pending)
            return 2

        result = {
            **manifest,
            "status": "COMPLETE",
            "evidence_assessments": agent.evidence_assessments,
            "candidate_rescore": candidate_json,
            "challenger_rescore": candidate_json,
            "incumbent_research": context["incumbent_research"],
            "comparison": comparison_json,
            "pair_evidence_audit": pair_audit,
            "pair_comparison_complete": comparison_json["pair_comparison_complete"],
            "decision_basis_sufficient": comparison_json["decision_basis_sufficient"],
            "material_asymmetry_resolved": comparison_json["material_asymmetry_resolved"],
            "rebalance_decision": comparison.rebalance_decision,
            **transition_fields(incumbent, challenger, comparison.rebalance_decision),
            "active_selection_research": (context["incumbent_research"]
                                           if comparison.rebalance_decision == "KEEP_PREVIOUS"
                                           else {"symbol": challenger, **candidate_json}),
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
                                       **transition_fields(incumbent, challenger, comparison.rebalance_decision)})
        print(
            f"PAIR RECHECK COMPLETE; challenger={challenger} score={candidate.alpha_score}; "
            f"confidence={candidate.confidence}; incumbent={incumbent}; "
            f"decision={comparison.rebalance_decision}; protocol={provider.protocol}; no orders.",
            flush=True,
        )
        return 0
    except ResearchPending:
        write("result.json", {**manifest, **agent.pending_research, "usage": totals})
        print("Pair recheck NEEDS_RESEARCH; see research_pending.json", flush=True)
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
        print(f"Pair recheck FAILED ({type(exc).__name__}); no orders submitted", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
