"""Offline handoff for interactive-session research; never a model API or trader.

prepare exports an immutable historical baseline and creates a new run.
finalize accepts the current session's saved evidence and decisions, validates
coverage/state, and records a new selection. It does not choose an investment.
This is an interactive workflow, not unattended access to a chat subscription.
Only the Python standard library is used; .env and production DB are not read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import uuid4


NEW_GAP_STATUS_VALUES = {
    "VERIFIED", "NOT_PUBLIC", "NOT_YET_OCCURRED", "RETRIEVAL_FAILED",
    "PAID_DATA_REQUIRED", "DERIVATION_REQUIRED", "INSUFFICIENT_SPECIFICITY",
    "CONFLICTING_EVIDENCE", "STALE", "NOT_APPLICABLE",
}
LEGACY_GAP_STATUS_VALUES = {"UNAVAILABLE", "CONFLICT"}
GAP_STATUS_VALUES = NEW_GAP_STATUS_VALUES | LEGACY_GAP_STATUS_VALUES
MISSING_GAP_STATUS = "NOT_RECORDED"
BLOCKING_GAP_STATUS_VALUES = {"RETRIEVAL_FAILED", "STALE"}
CRITICALITY_VALUES = {"CRITICAL", "IMPORTANT", "NON_CRITICAL", "NOT_RECORDED"}
SELECTION_RATIONALE_TEXT_FIELDS = (
    "why_final_first", "why_first_over_second", "remaining_alpha",
    "incumbent_comparison", "remaining_alpha_comparison", "core_reason", "biggest_risk",
)
SELECTION_RATIONALE_LIST_FIELDS = (
    "core_catalysts", "main_risks", "thesis_invalidation_conditions", "why_keep", "why_switch",
)


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", value):
        raise ValueError("Invalid stock symbol")
    if value in {"CASH", "WAIT", "NONE", "UNKNOWN"}:
        raise ValueError("Active selection must be a stock")
    return value


def resolve_selection(raw):
    if raw.get("status") != "COMPLETE":
        raise ValueError("Previous selection is not COMPLETE")
    known = []
    for record in (raw, raw.get("result", {})):
        if not isinstance(record, dict):
            continue
        comparison = record.get("rebalance_comparison") or record.get("comparison") or {}
        action = record.get("rebalance_decision") or comparison.get("rebalance_decision")
        if record.get("rebalance_decision") and comparison.get("rebalance_decision") and record["rebalance_decision"] != comparison["rebalance_decision"]:
            raise ValueError("Conflicting selection actions")
        if action:
            incoming = symbol(record.get("incoming_active_selection") or record.get("previous_first_symbol") or comparison.get("previous_first_symbol"))
            new = symbol(record.get("new_first_symbol") or comparison.get("new_first_symbol") or record.get("selected_symbol"))
            if action not in {"KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"}:
                raise ValueError("Unsupported selection action")
            if incoming == new and action != "KEEP_PREVIOUS":
                raise ValueError("Identical symbols require KEEP_PREVIOUS")
            for key, expected in (("previous_first_symbol", incoming), ("new_first_symbol", new)):
                if comparison.get(key) and comparison[key] != expected:
                    raise ValueError("Conflicting comparison symbols")
            if record.get("previous_first_symbol") and record["previous_first_symbol"] != incoming:
                raise ValueError("Conflicting incoming selection")
            known.append(incoming if action == "KEEP_PREVIOUS" else new)
        elif "incoming_active_selection" in record or "outgoing_active_selection" in record:
            raise ValueError("Transition missing final action")
        for key in ("outgoing_active_selection", "active_selection"):
            if key in record:
                known.append(symbol(record[key]))
    if not known or len(set(known)) != 1:
        raise ValueError("Missing or conflicting active selection; ranking is not selection")
    return known[0]


def compact(value):
    """Retain every fact and source except full time-series arrays and secrets."""
    omit = {"prices", "volumes", "dates", "series", "api_key", "token", "password",
            "account_id", "account_number", "portfolio", "positions", "holdings"}
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k.lower() not in omit}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def require_exact(rows, expected):
    actual = [symbol(row.get("symbol")) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Missing, duplicate, or unexpected symbol coverage")


def _gap_records(row):
    if not isinstance(row, dict):
        return []
    result = []
    for key in ("evidence_gaps", "important_evidence_gaps", "unresolved_information_gaps"):
        values = row.get(key) or []
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        result.extend(item for item in values if isinstance(item, dict))
    return result


def _validate_gap_metadata(rows):
    for row in rows:
        row_symbol = str(row.get("symbol", "")).upper()
        for gap in _gap_records(row):
            status = gap.get("gap_status")
            if status is None:
                continue
            status = str(status).upper()
            if status not in GAP_STATUS_VALUES and status != MISSING_GAP_STATUS:
                raise ValueError("Unknown evidence gap status")
            criticality = str(gap.get("criticality", "NOT_RECORDED")).upper()
            if criticality not in CRITICALITY_VALUES:
                raise ValueError("Unknown evidence gap criticality")
            gap_symbol = str(gap.get("symbol", row_symbol)).upper()
            if gap_symbol not in {row_symbol, "NOT_RECORDED", ""}:
                raise ValueError("Evidence gap symbol disagrees with assessment")
            required = ("field_name", "symbol", "criticality", "reason", "source_required",
                        "last_checked_at", "retrieval_attempts", "evidence_refs",
                        "decision_impact", "blocking_research")
            if status in NEW_GAP_STATUS_VALUES:
                if any(key not in gap for key in required):
                    raise ValueError("New evidence gaps require structured metadata")
                if criticality not in {"CRITICAL", "IMPORTANT", "NON_CRITICAL"}:
                    raise ValueError("New evidence gaps require model-selected criticality")
                if any(gap.get(key) in {None, "", "NOT_RECORDED"}
                       for key in ("field_name", "reason", "decision_impact")):
                    raise ValueError("New evidence gaps require field, reason and decision impact")
            if not isinstance(gap.get("blocking_research", False), bool):
                raise ValueError("Evidence gap blocking_research must be boolean")


def _packet_evidence(packet):
    evidence = [item for item in packet.get("evidence", []) if isinstance(item, dict)]
    calculations = packet.get("deterministic_calculations", [])
    if isinstance(calculations, list):
        evidence.extend(item for item in calculations if isinstance(item, dict) and item.get("result") is not None)
    return evidence


def _date_text(value):
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else None


def pair_evidence_audit(packet, decision, challenger, incumbent):
    """Standalone counterpart of the skill-local pre-comparison audit."""
    rows = []
    for key in ("supplemental_research", "final_ranking"):
        value = decision.get(key, [])
        if isinstance(value, list):
            rows.extend(value)
    if isinstance(decision.get("incumbent_research"), dict):
        rows.append(decision["incumbent_research"])
    _validate_gap_metadata(rows)
    evidence = _packet_evidence(packet)
    verified = {str(item.get("symbol", "")).upper() for item in evidence
                if str(item.get("gap_status") or item.get("status") or "").upper() in {"VERIFIED", "DERIVED"}}
    pending = []
    for row in rows:
        symbol_value = str(row.get("symbol", "")).upper()
        if symbol_value not in {str(challenger).upper(), str(incumbent).upper()}:
            continue
        for gap in _gap_records(row):
            status = str(gap.get("gap_status", "")).upper()
            if status not in BLOCKING_GAP_STATUS_VALUES:
                continue
            if gap.get("criticality") != "CRITICAL" and gap.get("blocking_research") is not True:
                continue
            other = str(incumbent if symbol_value == str(challenger).upper() else challenger).upper()
            if other in verified:
                pending.append({"symbol": symbol_value, "other_symbol": other,
                                "field_name": gap.get("field_name", "NOT_RECORDED"),
                                "gap_status": status, "could_change_direction": True})
    evidence_dates = {}
    for symbol_value in (str(challenger).upper(), str(incumbent).upper()):
        dates = []
        for item in evidence:
            if str(item.get("symbol", "")).upper() == symbol_value:
                for key in ("as_of", "as_of_date", "published_at", "retrieved_at", "observed_at"):
                    parsed = _date_text(item.get(key))
                    if parsed:
                        dates.append(parsed)
        evidence_dates[symbol_value] = sorted(set(dates))
    basis_dates = [dates[-1] for dates in evidence_dates.values() if dates]
    market_status = "NOT_RECORDED"
    if len(basis_dates) == 2:
        from datetime import date as _date
        market_status = "ALIGNED" if abs((_date.fromisoformat(basis_dates[0]) - _date.fromisoformat(basis_dates[1])).days) <= 3 else "MISALIGNED"
    rounds = packet.get("targeted_research_rounds_used", packet.get("research_rounds_used", 0))
    try:
        rounds = max(0, int(rounds or 0))
    except (TypeError, ValueError):
        rounds = 0
    requests = [{"symbol": item["symbol"], "question": f"核验 {item['symbol']} 的 {item['field_name']}",
                 "decision_impact": "可能实质改变新旧首选方向", "query": f"{item['symbol']} {item['field_name']} 最新权威来源",
                 "sources": ["发行人、SEC、交易所或监管机构"],
                 "research_depth": "打开原文并记录时点、口径及失败原因", "field_name": item["field_name"]}
                for item in pending]
    requires = bool(pending) and rounds < 2
    return {"status": "NEEDS_RESEARCH" if requires else "READY_FOR_MODEL_DECISION",
            "challenger_symbol": str(challenger).upper(), "incumbent_symbol": str(incumbent).upper(),
            "market_as_of_basis": {"status": market_status, "dates": evidence_dates},
            "company_evidence_freshness": {symbol_value: {"dates": dates, "latest_as_of": dates[-1] if dates else None}
                                            for symbol_value, dates in evidence_dates.items()},
            "material_asymmetries": pending, "requires_targeted_research": requires,
            "research_requests": requests if requires else [], "rounds_used": rounds,
            "rounds_remaining": max(0, 2 - rounds), "material_asymmetry_resolved": not requires}


def _chinese_rationale_text(value, field_name):
    if not isinstance(value, str) or not value.strip() or value == MISSING_GAP_STATUS:
        raise ValueError(f"Decision rationale field {field_name} must be recorded")
    # Keep the same decision-time language boundary as the report schema,
    # while allowing stock symbols and uppercase abbreviations in prose.
    if not re.search(r"[\u3400-\u9fff]", value) or re.search(r"[a-z]", value):
        raise ValueError(f"Decision rationale field {field_name} must be Chinese prose")


def _validate_selection_rationale(decision, ranking, incumbent):
    """Validate decision-time explanations without interpreting their content."""
    rationale = decision.get("selection_rationale")
    if not isinstance(rationale, dict):
        raise ValueError("Decision-time selection_rationale is required")
    for field_name in SELECTION_RATIONALE_TEXT_FIELDS:
        _chinese_rationale_text(rationale.get(field_name), field_name)
    for field_name in SELECTION_RATIONALE_LIST_FIELDS:
        values = rationale.get(field_name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Decision rationale field {field_name} must be a non-empty list")
        for value in values:
            _chinese_rationale_text(value, field_name)
    finalists = rationale.get("why_not_finalists")
    if (
        not isinstance(finalists, list)
        or any(not isinstance(row, dict) for row in finalists)
        or {str(row.get("symbol", "")).upper() for row in finalists}
        != {str(row.get("symbol", "")).upper() for row in ranking[1:]}
        or len(finalists) != 4
    ):
        raise ValueError("Decision rationale must cover the four non-winning finalists")
    for row in finalists:
        if not isinstance(row, dict):
            raise ValueError("Finalist rationale rows must be objects")
        _chinese_rationale_text(row.get("reason"), "why_not_finalists")
    comparison = rationale.get("incumbent_comparison")
    if not comparison or str(incumbent).upper() not in str(comparison).upper() and "原有效首选" not in str(comparison):
        # This is deliberately a light identity sanity check, not a quality
        # test: the model remains responsible for the actual comparison.
        raise ValueError("Decision rationale must explain the incumbent comparison")


def prepare(args):
    source_path, previous_path, packet_path = [p.resolve() for p in (args.source_result, args.previous_result, args.verification_packet)]
    source, previous, packet = [load(p) for p in (source_path, previous_path, packet_path)]
    if source.get("status") != "COMPLETE":
        raise ValueError("Initial research is not COMPLETE")
    candidates = source.get("candidate_symbols", [])
    if len(candidates) < 5 or len(set(candidates)) != len(candidates) or source.get("candidate_count") != len(candidates):
        raise ValueError("Invalid initial candidate set")
    for item in candidates:
        symbol(item)
    ranking = source.get("ranking", {}).get("ranking", [])
    require_exact(ranking, candidates)
    if [r.get("rank") for r in ranking] != list(range(1, len(candidates) + 1)):
        raise ValueError("Initial ranking order is invalid")
    incumbent = resolve_selection(previous)
    top5 = [r["symbol"] for r in ranking[:5]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root.resolve() / (stamp + "-session-" + str(uuid4()))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "mode": "INTERACTIVE_SESSION_RESEARCH", "created_at": now(),
        "status": "WAITING_SESSION_RESEARCH", "research_provider": "CURRENT_CONVERSATION",
        "research_model": args.session_model, "model_provenance": "SESSION_DECLARED_NOT_API_VERIFIED",
        "reasoning_effort_requested": None, "reasoning_effort_effective": "NOT_EXPOSED_BY_SESSION",
        "standalone_model_api": False, "api_calls": 0, "unattended": False,
        "source_result": str(source_path), "source_sha256": digest(source_path),
        "previous_result": str(previous_path), "previous_sha256": digest(previous_path),
        "verification_packet": str(packet_path), "packet_sha256": digest(packet_path),
        "candidate_symbols": candidates, "candidate_count": len(candidates), "initial_top_five": top5,
        "incoming_active_selection": incumbent, "initial_stage": "REUSED_HISTORICAL_NOT_NEW_SESSION_RESULT",
        "initial_model": source.get("research_model"), "initial_source": source.get("source"),
        "safety": {"production_database": "NOT_OPENED", "environment_file": "NOT_READ",
                   "risk_engine": "NOT_RUN", "broker": "NOT_CONNECTED", "orders": "NOT_SENT"},
        "objective": "From each stock's current executable price, choose the eligible stock with the highest expected forward relative return over a reasonable model-chosen research horizon after a reasonable standardized or configured research-layer friction assumption; compare it directly with the incoming research-layer incumbent. No style prior, incumbent privilege, fixed factor weights, mechanical investment veto, deliberate cash allocation, account sizing or orders; unknown is not negative evidence.",
        "research_layer_friction_assumption": "STANDARDIZED_OR_CONFIGURED_RESEARCH_LAYER_ASSUMPTION_ONLY; NOT_ACCOUNT_COMMISSION_OR_SLIPPAGE",
        "active_selection_meaning": "RESEARCH_LAYER_CURRENT_AI_PREFERENCE; NOT_BROKER_POSITION",
        "required_stages": ["TOP5_SUPPLEMENTAL", "FINAL_TOP5_RANKING", "INCUMBENT_COMPARISON"],
    }
    write(output / "manifest.json", manifest)
    write(output / "baseline.json", compact({"ranking": source.get("ranking"), "selection_rationale": source.get("selection_rationale"), "deep_research": source.get("deep_research"), "evidence_assessments": source.get("evidence_assessments")}))
    write(output / "historical_evidence.json", compact(packet))
    write(output / "events.json", [{"timestamp": now(), "event_type": "SESSION_HANDOFF_PREPARED", "status": "WAITING_SESSION_RESEARCH"}])
    print(json.dumps({"output_directory": str(output), "candidate_count": len(candidates), "top_five": top5, "incoming_active_selection": incumbent, "status": manifest["status"]}, ensure_ascii=False, indent=2))
    return output


def validate_decision(manifest, decision, packet):
    initial = manifest["initial_top_five"]
    incumbent = manifest["incoming_active_selection"]
    if decision.get("research_provider") != "CURRENT_CONVERSATION":
        raise ValueError("Decision must disclose current-conversation provenance")
    if decision.get("research_model") != manifest["research_model"]:
        raise ValueError("Session model declaration mismatch")
    if decision.get("status") != "COMPLETE":
        raise ValueError("Do not finalize incomplete research")
    supplemental = decision.get("supplemental_research", [])
    require_exact(supplemental, initial)
    ranking = decision.get("final_ranking", [])
    require_exact(ranking, initial)
    if [r.get("rank") for r in ranking] != list(range(1, 6)):
        raise ValueError("Final ranks must be ordered 1 through 5")
    evidence = packet.get("evidence", [])
    refs = {e.get("id"): e for e in evidence}
    calculations = packet.get("deterministic_calculations", [])
    if isinstance(calculations, list):
        refs.update({e.get("id"): e for e in calculations if isinstance(e, dict) and e.get("id")})
    calculation_count = sum(1 for e in calculations if isinstance(e, dict) and e.get("id")) if isinstance(calculations, list) else 0
    if not evidence or None in refs or len(refs) != len(evidence) + calculation_count:
        raise ValueError("Evidence requires unique ids")
    valid_refs = set()
    for key, item in refs.items():
        item_status = str(item.get("status") or item.get("gap_status") or "").upper()
        if item_status not in GAP_STATUS_VALUES and item_status not in {"DERIVED", MISSING_GAP_STATUS}:
            raise ValueError("Unknown evidence status")
        if item_status == "VERIFIED":
            if not item.get("source_url", "").startswith(("http://", "https://")) or not item.get("retrieved_at") or not item.get("facts") or item.get("source_opened") is not True:
                raise ValueError("Verified evidence needs opened source, time and facts")
            valid_refs.add(key)
        elif item_status == "DERIVED" and item.get("result") is not None and item.get("evidence_refs"):
            valid_refs.add(key)
    rows = supplemental + ranking + [decision.get("incumbent_research", {})]
    if rows[-1].get("symbol") != incumbent:
        raise ValueError("Missing incumbent research")
    for row in rows:
        for key in ("investment_confidence", "evidence_completeness", "confidence_basis", "evidence_refs"):
            if key not in row:
                raise ValueError("Missing evidence assessment")
        confidence = row["investment_confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Invalid investment confidence")
        if row.get("research_requests"):
            raise ValueError("Decision-critical research requests remain open")
        if not row["evidence_refs"] or not set(row["evidence_refs"]) <= refs.keys():
            raise ValueError("Invalid evidence references")
        if not set(row["evidence_refs"]) & valid_refs:
            raise ValueError("Each stock requires actually verified evidence")
        if row.get("confidence", confidence) != confidence:
            raise ValueError("Confidence fields disagree")
    new = ranking[0]["symbol"]
    _validate_selection_rationale(decision, ranking, incumbent)
    comparison = decision.get("comparison", {})
    if comparison.get("new_first_symbol") != new or comparison.get("previous_first_symbol") != incumbent:
        raise ValueError("Comparison identities disagree")
    action = comparison.get("rebalance_decision")
    if action not in {"KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"} or (new == incumbent and action != "KEEP_PREVIOUS"):
        raise ValueError("Invalid final selection action")
    if not comparison.get("why_keep") or not comparison.get("why_switch") or not comparison.get("decision_reason"):
        raise ValueError("Both alternatives and final reason must be recorded")
    if not comparison.get("evidence_refs") or not set(comparison["evidence_refs"]) <= refs.keys():
        raise ValueError("Comparison evidence is missing")
    for key in ("pair_comparison_complete", "decision_basis_sufficient", "material_asymmetry_resolved"):
        if key in comparison and not isinstance(comparison[key], bool):
            raise ValueError("Pair comparison flags must be boolean")
        if key in comparison and comparison[key] is False:
            raise ValueError("Pair comparison is not COMPLETE")
    pair_audit = pair_evidence_audit(packet, decision, new, incumbent)
    if pair_audit["requires_targeted_research"]:
        raise ValueError("Material retrieval/stale asymmetry requires targeted research")
    flags = {key: comparison[key] for key in
             ("pair_comparison_complete", "decision_basis_sufficient", "material_asymmetry_resolved")
             if key in comparison}
    return {"incoming_active_selection": incumbent, "new_first_symbol": new,
            "rebalance_decision": action, "outgoing_active_selection": incumbent if action == "KEEP_PREVIOUS" else new,
            "active_selection": incumbent if action == "KEEP_PREVIOUS" else new,
            "pair_evidence_audit": pair_audit, **flags}


def finalize(args):
    output = args.run_directory.resolve()
    manifest = load(output / "manifest.json")
    for path_key, hash_key in (("source_result", "source_sha256"), ("previous_result", "previous_sha256"), ("verification_packet", "packet_sha256")):
        if digest(manifest[path_key]) != manifest[hash_key]:
            raise ValueError("Historical input changed after prepare")
    decision, packet = load(args.decision), load(args.evidence)
    transition = validate_decision(manifest, decision, packet)
    for name in ("result.json", "active_selection.json", "evidence_assessments.json", "verification_packet.json"):
        if (output / name).exists():
            raise ValueError("Refusing to overwrite existing final result")
    write(output / "verification_packet.json", packet)
    write(output / "evidence_assessments.json", decision["supplemental_research"] + decision["final_ranking"] + [decision["incumbent_research"]])
    write(output / "result.json", {**manifest, **decision, **transition, "completed_at": now(), "selection_scope": "RESEARCH_ONLY_NOT_ACCOUNT_HOLDING"})
    write(output / "active_selection.json", {"status": "COMPLETE", **transition, "result_file": str(output / "result.json"), "research_provider": "CURRENT_CONVERSATION"})
    print(json.dumps({"status": "COMPLETE", **transition, "output_directory": str(output)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for key in ("source-result", "previous-result", "verification-packet", "output-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--session-model", required=True, help="Declared by the active session, not an API model-id verification")
    p = sub.add_parser("finalize")
    for key in ("run-directory", "decision", "evidence"):
        p.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else finalize(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Session research stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
