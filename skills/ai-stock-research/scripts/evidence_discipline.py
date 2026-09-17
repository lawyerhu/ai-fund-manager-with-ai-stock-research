"""Skill-local evidence audit; leave the project's ranking and decision models intact."""
from __future__ import annotations

from datetime import date, datetime
import re
from types import MethodType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


# ``UNAVAILABLE`` and ``CONFLICT`` remain readable for historical results.
# New model output should use one of the more specific gap statuses below.
NEW_GAP_STATUS_VALUES = frozenset({
    "VERIFIED", "NOT_PUBLIC", "NOT_YET_OCCURRED", "RETRIEVAL_FAILED",
    "PAID_DATA_REQUIRED", "DERIVATION_REQUIRED", "INSUFFICIENT_SPECIFICITY",
    "CONFLICTING_EVIDENCE", "STALE", "NOT_APPLICABLE",
})
LEGACY_GAP_STATUS_VALUES = frozenset({"UNAVAILABLE", "CONFLICT"})
GAP_STATUS_VALUES = NEW_GAP_STATUS_VALUES | LEGACY_GAP_STATUS_VALUES
MISSING_GAP_STATUS = "NOT_RECORDED"
BLOCKING_GAP_STATUS_VALUES = frozenset({"RETRIEVAL_FAILED", "STALE"})
CRITICALITY_VALUES = frozenset({"CRITICAL", "IMPORTANT", "NON_CRITICAL", "NOT_RECORDED"})

GapStatus = Literal[
    "VERIFIED", "NOT_PUBLIC", "NOT_YET_OCCURRED", "RETRIEVAL_FAILED",
    "PAID_DATA_REQUIRED", "DERIVATION_REQUIRED", "INSUFFICIENT_SPECIFICITY",
    "CONFLICTING_EVIDENCE", "STALE", "NOT_APPLICABLE", "UNAVAILABLE", "CONFLICT",
]
Criticality = Literal["CRITICAL", "IMPORTANT", "NON_CRITICAL", "NOT_RECORDED"]


EVIDENCE_INSTRUCTIONS = (
    "EVIDENCE DISCIPLINE (governs evidence handling and fixed-field interpretation in the legacy prompt): "
    "You are the research decision model for the current run. Choose decision-relevant questions from each company's industry, business model, "
    "current market, catalysts and risks. Existing financial fields, data tools and named sources are optional "
    "context/output slots, not a required factor model, research template or buy checklist. Preserve the requested "
    "candidate set, stages, ranking size and decision authority. Unknown or inapplicable legacy slots may say "
    "UNKNOWN or NOT_APPLICABLE; their count is not a score. A failed search does not prove a fact does not exist. "
    "Never interpret unknown as adverse fundamentals, penalize investment scores mechanically, invent missing "
    "data, or guess to increase confidence. Prefer original company announcements, regulatory/SEC filings, "
    "financial reports and IR materials, choosing other suitable sources autonomously. Assess conflicting sources "
    "for reliability, timing and definitions; do not merely count sources or discard a conflict without examining it. "
    "Return evidence_assessments for every stock evaluated, including candidates outside the returned Top 5. "
    "investment_confidence is conviction in the investment judgment, not a calibrated profit probability; "
    "evidence_completeness describes coverage of the key facts YOU consider material, not a fixed-field percentage. "
    "They are distinct; low completeness does not mechanically lower confidence or rank. Where the legacy output "
    "also has confidence for this judgment, return the same value as investment_confidence. Explain material "
    "uncertainty in confidence_basis; low confidence is allowed and need not rise after research. "
    "If a missing or conflicting fact could materially change rank, the winner or rotation, put a concrete question, "
    "decision impact, query, preferred sources and research depth in research_requests. Do this BEFORE treating "
    "any score/rank/action as final, especially when the winner hinges on that fact. Do not claim this API browsed: "
    "Codex executes the search and returns dated evidence using the same provider. Retain the existing maximum "
    "of two supplemental rounds; choose scope and depth within those rounds yourself. Do not repeat an exhausted "
    "endpoint blindly. Use a precise gap_status for important evidence gaps: VERIFIED, NOT_PUBLIC, "
    "NOT_YET_OCCURRED, RETRIEVAL_FAILED, PAID_DATA_REQUIRED, DERIVATION_REQUIRED, "
    "INSUFFICIENT_SPECIFICITY, CONFLICTING_EVIDENCE, STALE or NOT_APPLICABLE. UNAVAILABLE and CONFLICT "
    "are legacy input values only. For every important gap, preserve field_name, symbol, gap_status, "
    "criticality (CRITICAL, IMPORTANT or NON_CRITICAL), reason, source_required, last_checked_at, "
    "retrieval_attempts, evidence_refs, decision_impact and blocking_research. Choose criticality from "
    "the company, industry, business model and current thesis; never use a fixed financial checklist or "
    "gap count as a score. If public raw inputs support a deterministic calculation, use the skill's "
    "deterministic calculation helper and remove the resolved item from unresolved gaps; do not let the "
    "language model invent the result. Only put a material question in unresolved_information_gaps after documented reasonable "
    "supplemental attempts; cite supplied gap_audit records using JSON pointers (e.g. /gap_audit/searches/0) "
    "in search_record_refs and explain why further research is unlikely to "
    "resolve it. A round limit alone does not prove a reasonable search occurred. Unsearched gaps require requests. "
    "Schema-required ranks/actions during an open request are provisional, not an abstention or a final choice. "
    "After reasonable research, make the required best-evidence decision despite genuine unresolved uncertainty. "
    "Do not use evidence completeness to default to KEEP_PREVIOUS or trigger a switch. "
    "SELECTION FREEDOM: The only investment question is which eligible stock, from its current executable price, "
    "has the highest expected relative return over a reasonable model-chosen research horizon after a reasonable "
    "standardized or configured research-layer friction assumption. There is no style prior, incumbent privilege, "
    "or mechanical investment veto. Twenty/sixty-day, year-to-date or 52-week price moves, highs, RSI, technical "
    "position, volatility, beta, valuation, FCF yield, crowding, drawdown, momentum, price extension, entry risk, "
    "industry, event status and catalyst timing are research evidence only; do not apply fixed weights, penalties, "
    "score hurdles or champion exclusions. Past gains do not imply poor future opportunity and past lagging does not "
    "imply upside. An event is not alpha by itself and a near-term catalyst is not required. Compare expected future "
    "outcomes and expectation gaps, not sunk entry cost, P&L, elapsed holding time or historical rank/score. Do not "
    "require a minimum holding period, a mechanical sell date, a broken incumbent thesis, or a fixed replacement gap. "
    "The model may KEEP when a small advantage is inside its own error, or SWITCH when a small but well-supported "
    "net advantage covers the research-layer friction; this is model judgment, not a program threshold. Luna and old "
    "scores are retrieval context only, never a current-round prior. Normal research chooses the best eligible stock, "
    "not cash; stop only for the existing technical, identity, evidence-integrity or parsing safety blocks. Account-level "
    "commission, spread, slippage, quantity and broker state are outside this research layer and must not be invented."
)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    decision_impact: str = Field(min_length=1)
    query: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    research_depth: str = Field(min_length=1)
    field_name: str | None = None
    gap_status: GapStatus | None = None


class EvidenceGap(BaseModel):
    """Structured metadata for a material gap.

    Defaults make an older assessment readable without pretending that the
    historical run recorded the new concepts.
    """

    model_config = ConfigDict(extra="forbid")
    field_name: str = Field(default="NOT_RECORDED", min_length=1)
    symbol: str = Field(default="NOT_RECORDED", min_length=1)
    # ``None`` is the compatibility shape for an old result that did not
    # record the new field; normalization persists it as NOT_RECORDED.
    gap_status: GapStatus | None = None
    criticality: Criticality = "NOT_RECORDED"
    reason: str = Field(default="NOT_RECORDED", min_length=1)
    source_required: list[str] | str = Field(default_factory=list)
    last_checked_at: str | None = None
    retrieval_attempts: int | list[str] | list[dict[str, Any]] = 0
    evidence_refs: list[str] = Field(default_factory=list)
    decision_impact: str = Field(default="NOT_RECORDED", min_length=1)
    blocking_research: bool = False
    question: str = Field(default="NOT_RECORDED", min_length=1)
    search_record_refs: list[str] = Field(default_factory=list)
    stopping_reason: str = Field(default="NOT_RECORDED", min_length=1)


class UnresolvedGap(EvidenceGap):
    """Legacy unresolved-gap shape plus the new optional metadata."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    search_record_refs: list[str] = Field(min_length=1)
    stopping_reason: str = Field(min_length=1)
    decision_impact: str = Field(min_length=1)


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1)
    investment_confidence: float = Field(ge=0, le=1)
    evidence_completeness: str = Field(min_length=1)
    confidence_basis: str = Field(min_length=1)
    research_requests: list[ResearchRequest] = Field(default_factory=list)
    unresolved_information_gaps: list[UnresolvedGap] = Field(default_factory=list)
    # New records may use this clearer name; the old field remains accepted.
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    important_evidence_gaps: list[EvidenceGap] = Field(default_factory=list)


def normalize_gap_record(gap: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    """Fill compatibility metadata without classifying an old gap as adverse."""
    if not isinstance(gap, dict):
        raise ValueError("Evidence gap must be an object")
    record = dict(gap)
    record.setdefault("field_name", "NOT_RECORDED")
    record.setdefault("symbol", symbol or "NOT_RECORDED")
    if record.get("gap_status") is None:
        legacy_status = str(record.get("status") or "").upper()
        record["gap_status"] = (legacy_status if legacy_status in GAP_STATUS_VALUES
                                 else MISSING_GAP_STATUS)
    record.setdefault("criticality", "NOT_RECORDED")
    record.setdefault("reason", record.get("stopping_reason") or "NOT_RECORDED")
    record.setdefault("source_required", [])
    record.setdefault("last_checked_at", None)
    record.setdefault("retrieval_attempts", 0)
    record.setdefault("evidence_refs", [])
    record.setdefault("decision_impact", "NOT_RECORDED")
    record.setdefault("blocking_research", False)
    if record.get("symbol") in {None, "", "NOT_RECORDED"} and symbol:
        record["symbol"] = symbol
    record["symbol"] = str(record["symbol"]).upper()
    status = str(record["gap_status"]).upper()
    if status not in GAP_STATUS_VALUES and status != MISSING_GAP_STATUS:
        raise ValueError(f"Unknown evidence gap status: {status}")
    record["gap_status"] = status
    criticality = str(record["criticality"]).upper()
    if criticality not in CRITICALITY_VALUES:
        raise ValueError(f"Unknown evidence gap criticality: {criticality}")
    record["criticality"] = criticality
    if status in NEW_GAP_STATUS_VALUES:
        if criticality not in {"CRITICAL", "IMPORTANT", "NON_CRITICAL"}:
            raise ValueError("New evidence gaps require model-selected criticality")
        if any(record.get(key) in {None, "", "NOT_RECORDED"}
               for key in ("field_name", "reason", "decision_impact")):
            raise ValueError("New evidence gaps require field, reason and decision impact")
    attempts = record["retrieval_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, (int, list)) or (isinstance(attempts, int) and attempts < 0):
        raise ValueError("Evidence gap retrieval_attempts must be a non-negative count or attempt list")
    if not isinstance(record["blocking_research"], bool):
        raise ValueError("Evidence gap blocking_research must be boolean")
    return record


def iter_gap_records(assessment: dict[str, Any], *, symbol: str | None = None) -> list[dict[str, Any]]:
    """Return all new and legacy gap records in one assessment."""
    if not isinstance(assessment, dict):
        return []
    effective_symbol = symbol or assessment.get("symbol")
    rows: list[dict[str, Any]] = []
    for key in ("evidence_gaps", "important_evidence_gaps", "unresolved_information_gaps"):
        values = assessment.get(key) or []
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        rows.extend(normalize_gap_record(value, effective_symbol) for value in values)
    return rows


def normalize_assessments(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist explicit metadata for newly produced assessments.

    The function does not infer criticality or investment impact.  Missing
    metadata in an old-shaped gap is marked ``NOT_RECORDED``/``UNAVAILABLE``
    so it remains distinguishable from a newly researched status.
    """
    for assessment in assessments:
        symbol = assessment.get("symbol")
        for key in ("evidence_gaps", "important_evidence_gaps", "unresolved_information_gaps"):
            values = assessment.get(key) or []
            assessment[key] = [normalize_gap_record(value, symbol) for value in values]
    return assessments


def _date_value(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _record_dates(record: dict[str, Any]) -> list[date]:
    values = []
    for key in ("as_of", "as_of_date", "as_of_basis", "snapshot_date", "quote_as_of",
                "last_bar_at", "observed_at", "published_at", "retrieved_at", "date", "timestamp",
                "last_checked_at"):
        value = record.get(key)
        if isinstance(value, dict):
            values.extend(_record_dates(value))
        else:
            parsed = _date_value(value)
            if parsed:
                values.append(parsed)
    return values


def _packet_records(packet: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("evidence", "evidence_gaps", "important_evidence_gaps", "unresolved_information_gaps",
                "comparison_market_snapshot", "same_date_market_snapshot", "project_same_basis_market_data",
                "market_snapshot", "market_data", "deterministic_calculations"):
        value = packet.get(key)
        if isinstance(value, list):
            records.extend(row for row in value if isinstance(row, dict))
    audit = packet.get("gap_audit")
    if isinstance(audit, dict):
        for key in ("gaps", "items", "searches", "records"):
            value = audit.get(key)
            if isinstance(value, list):
                records.extend(row for row in value if isinstance(row, dict))
    return records


def _assessment_records(assessments: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in assessments or []:
        values = stage.get("evidence_assessments") if isinstance(stage, dict) else None
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and str(value.get("symbol", "")).upper() == symbol:
                    rows.extend(iter_gap_records(value, symbol=symbol))
        elif isinstance(stage, dict) and str(stage.get("symbol", "")).upper() == symbol:
            rows.extend(iter_gap_records(stage, symbol=symbol))
    return rows


def _status(record: dict[str, Any]) -> str:
    status = str(record.get("gap_status") or record.get("status") or "").upper()
    return "VERIFIED" if status == "DERIVED" else status


def _symbol_records(packet: dict[str, Any], assessments: list[dict[str, Any]], symbol: str,
                    additional_research: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    gaps = _assessment_records(assessments, symbol)
    for record in _packet_records(packet):
        if str(record.get("symbol", "")).upper() != symbol:
            continue
        evidence.append(record)
        if record.get("gap_status") or record.get("field_name") and _status(record) in GAP_STATUS_VALUES - {"VERIFIED"}:
            gaps.append(normalize_gap_record(record, symbol))
    for record in additional_research or []:
        if isinstance(record, dict) and str(record.get("symbol", "")).upper() == symbol:
            gaps.extend(iter_gap_records(record, symbol=symbol))
    return evidence, gaps


def _basis_dates(packet: dict[str, Any], evidence: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[date]:
    dates = []
    for row in evidence + gaps:
        dates.extend(_record_dates(row))
    if not dates:
        dates = _record_dates(packet.get("as_of_basis", {}))
    return sorted(set(dates))


def _request_for_gap(gap: dict[str, Any], role: str) -> dict[str, Any]:
    sources = gap.get("source_required") or ["issuer, SEC, exchange or applicable regulator"]
    if isinstance(sources, str):
        sources = [sources]
    field = gap.get("field_name") or gap.get("question") or "decision-critical variable"
    question = gap.get("question") or f"核验 {gap.get('symbol')} 的 {field}"
    return {
        "symbol": str(gap.get("symbol") or "").upper(),
        "question": question,
        "decision_impact": gap.get("decision_impact") or "May materially change the incumbent/challenger direction",
        "query": f"{gap.get('symbol')} {field} latest primary source and as-of basis",
        "sources": sources,
        "research_depth": "Open the current primary sources, reconcile dates and definitions, and record the result or the concrete retrieval failure",
        "field_name": field,
        "gap_status": gap.get("gap_status"),
        "pair_role": role,
    }


def pair_evidence_audit(packet: dict[str, Any], assessments: list[dict[str, Any]],
                        challenger_symbol: str, incumbent_symbol: str, *,
                        additional_research: list[dict[str, Any]] | None = None,
                        rounds_used: int | None = None, max_rounds: int = 2) -> dict[str, Any]:
    """Audit evidence symmetry before the model is allowed to choose an action.

    This is a structural audit, not an investment score.  It only blocks when
    the model/packet explicitly marks a critical or blocking retrieval failure
    or stale variable on one leg while the other leg has verified evidence.
    Ambiguous legacy ``UNAVAILABLE`` values do not trigger the gate.
    """
    packet = packet if isinstance(packet, dict) else {}
    challenger = str(challenger_symbol).upper()
    incumbent = str(incumbent_symbol).upper()
    extra = additional_research or []
    legs = {}
    for symbol in dict.fromkeys((challenger, incumbent)):
        evidence, gaps = _symbol_records(packet, assessments, symbol, extra)
        dates = _basis_dates(packet, evidence, gaps)
        legs[symbol] = {"evidence": evidence, "gaps": gaps, "dates": dates}

    asymmetries = []
    for symbol, other in ((challenger, incumbent), (incumbent, challenger)):
        for gap in legs[symbol]["gaps"]:
            status = str(gap.get("gap_status", "")).upper()
            critical = gap.get("criticality") == "CRITICAL" or gap.get("blocking_research") is True
            if status not in BLOCKING_GAP_STATUS_VALUES or not critical:
                continue
            field = str(gap.get("field_name") or "").lower()
            other_pending_same = any(
                str(item.get("field_name") or "").lower() == field
                and str(item.get("gap_status") or "").upper() in BLOCKING_GAP_STATUS_VALUES
                for item in legs[other]["gaps"]
            )
            other_verified = any(_status(item) == "VERIFIED" for item in (legs[other]["evidence"] + legs[other]["gaps"]))
            other_verified_same = any(
                _status(item) == "VERIFIED"
                and (not field or str(item.get("field_name") or "").lower() == field)
                for item in (legs[other]["evidence"] + legs[other]["gaps"])
            )
            if other_pending_same or not other_verified:
                continue
            asymmetries.append({
                "symbol": symbol,
                "other_symbol": other,
                "field_name": gap.get("field_name", "NOT_RECORDED"),
                "gap_status": status,
                "criticality": gap.get("criticality", "NOT_RECORDED"),
                "reason": gap.get("reason", "NOT_RECORDED"),
                "decision_impact": gap.get("decision_impact", "NOT_RECORDED"),
                "other_evidence_status": "VERIFIED",
                "other_evidence_same_field": other_verified_same,
                "could_change_direction": True,
            })

    if rounds_used is None:
        value = packet.get("targeted_research_rounds_used", packet.get("research_rounds_used", 0))
        try:
            rounds_used = max(0, int(value or 0))
        except (TypeError, ValueError):
            rounds_used = 0
    rounds_remaining = max(0, max_rounds - rounds_used)
    requires_research = bool(asymmetries) and rounds_remaining > 0
    requests = []
    seen = set()
    for item in asymmetries:
        key = (item["symbol"], item["field_name"])
        if key in seen:
            continue
        seen.add(key)
        gap = next(
            gap for gap in legs[item["symbol"]]["gaps"]
            if str(gap.get("field_name") or "") == str(item["field_name"] or "")
            and str(gap.get("gap_status") or "").upper() == item["gap_status"]
        )
        requests.append(_request_for_gap(gap, "challenger" if item["symbol"] == challenger else "incumbent"))

    leg_dates = {symbol: [value.isoformat() for value in data["dates"]] for symbol, data in legs.items()}
    if all(leg_dates.get(symbol) for symbol in (challenger, incumbent)):
        latest = [date.fromisoformat(leg_dates[symbol][-1]) for symbol in (challenger, incumbent)]
        market_basis_status = "ALIGNED" if abs((latest[0] - latest[1]).days) <= 3 else "MISALIGNED"
    else:
        market_basis_status = "NOT_RECORDED"
    return {
        "status": "NEEDS_RESEARCH" if requires_research else "READY_FOR_MODEL_DECISION",
        "challenger_symbol": challenger,
        "incumbent_symbol": incumbent,
        "market_as_of_basis": {"status": market_basis_status, "dates": leg_dates},
        "company_evidence_freshness": {
            symbol: {"latest_as_of": values[-1] if values else None, "dates": values}
            for symbol, values in leg_dates.items()
        },
        "material_asymmetries": asymmetries,
        "requires_targeted_research": requires_research,
        "research_requests": requests,
        "rounds_used": rounds_used,
        "rounds_remaining": rounds_remaining,
        "stop_reason": ("Critical retrieval/stale asymmetry may change the rotation direction"
                         if requires_research else
                         "No blocking asymmetry detected, or the reasonable research limit was reached"),
        "material_asymmetry_resolved": not requires_research,
    }


class ResearchPending(ValueError):
    """The caller must retrieve evidence, not repair JSON or announce a winner."""


def install_evidence_discipline(agent, write, *, stages=None, capture_selection_path=False):
    """Audit each existing Astra stage without extra LLM calls or production edits.

    ``write`` persists redacted artifacts in the existing isolated output directory.
    The original typed result is returned unchanged after the audit passes.
    """
    original = agent._structured_deep_stage
    agent.evidence_assessments = []
    agent.pending_research = None
    agent.completed_research_stages = []
    agent.selection_rationale = {}
    agent.resolved_derivations = []
    agent.pending_derivations = []

    def audited(self, model_type, schema_name, event_prefix, prompt, totals, **kwargs):
        if stages is not None and event_prefix not in stages:
            return original(model_type, schema_name, event_prefix, prompt, totals, **kwargs)
        self.pending_research = None
        extra_fields = {}
        if capture_selection_path:
            from selection_rationale import RATIONALE_STAGES, RATIONALE_INSTRUCTIONS
            if event_prefix in RATIONALE_STAGES:
                extra_fields["selection_rationale"] = (RATIONALE_STAGES[event_prefix], ...)
            prompt += RATIONALE_INSTRUCTIONS
        audited_type = create_model(
            model_type.__name__ + "WithEvidenceAudit", __base__=model_type,
            evidence_assessments=(list[EvidenceAssessment], Field(min_length=1)),
            **extra_fields,
        )
        context = prompt + "\n" + EVIDENCE_INSTRUCTIONS
        # The project's repair path special-cases its original committee class.
        # Preserve evidence context for repairs of the skill's augmented schemas too.
        create_response = self.provider.create_response

        def with_repair_context(**request):
            if isinstance(request.get("input"), str) and request["input"].startswith("Repair the prior response."):
                request["input"] += "\nORIGINAL_RESEARCH_CONTEXT:\n" + context
            return create_response(**request)

        self.provider.create_response = with_repair_context
        try:
            result = original(audited_type, schema_name, event_prefix, context, totals, **kwargs)
        finally:
            self.provider.create_response = create_response
        payload = result.model_dump(mode="json")
        assessments = payload.pop("evidence_assessments")
        normalize_assessments(assessments)
        stage_resolved = []
        stage_pending_derivations = []
        try:
            from deterministic_calculations import (matching_derivation, resolve_packet_derivations,
                                                    resolve_assessment_derivations)
            packet = getattr(self, "external_verification", {})
            resolve_packet_derivations(packet)
            stage_resolved = resolve_assessment_derivations(assessments, packet)
            self.resolved_derivations.extend(stage_resolved)
            # A derivation is not an unresolved information gap.  Keep a
            # pending calculation in the structured evidence section and ask
            # Codex to supply explicit raw inputs/formula instead of allowing
            # the model to invent a number.
            for item in assessments:
                moved = []
                for gap in item.get("unresolved_information_gaps", []) or []:
                    if (gap.get("gap_status") == "DERIVATION_REQUIRED"
                            and matching_derivation(packet, gap, item.get("symbol", "")) is None):
                        moved.append(gap)
                if moved:
                    item["unresolved_information_gaps"] = [
                        gap for gap in item.get("unresolved_information_gaps", []) if gap not in moved
                    ]
                    item.setdefault("evidence_gaps", []).extend(moved)
                    stage_pending_derivations.extend(moved)
            for item in assessments:
                for gap in iter_gap_records(item):
                    if (gap.get("gap_status") == "DERIVATION_REQUIRED"
                            and matching_derivation(packet, gap, item.get("symbol", "")) is None
                            and gap not in stage_pending_derivations):
                        stage_pending_derivations.append(gap)
            self.pending_derivations.extend(stage_pending_derivations)
        except ImportError:
            # The helper is skill-local; keep the existing audit usable when a
            # caller imports this module in a minimal test harness.
            pass
        if capture_selection_path:
            from selection_rationale import chinese_text
            for item in assessments:
                chinese_text(item["evidence_completeness"])
                chinese_text(item["confidence_basis"])
        rationale = payload.pop("selection_rationale", None)
        if rationale is not None:
            if capture_selection_path and event_prefix in {"SOL_TOP5_FINAL_RANKING", "SOL_NEW_VS_PREVIOUS"}:
                from selection_rationale import chinese_text
                required_key = ("remaining_alpha" if event_prefix == "SOL_TOP5_FINAL_RANKING"
                                else "remaining_alpha_comparison")
                remaining_alpha = rationale.get(required_key)
                if not isinstance(remaining_alpha, str) or not remaining_alpha.strip() or remaining_alpha == "NOT_RECORDED":
                    raise ValueError(f"{event_prefix} requires a decision-time remaining-alpha rationale")
                chinese_text(remaining_alpha)
            from selection_rationale import check_coverage
            check_coverage(rationale, payload, getattr(self, "_candidate_symbols", []))
        symbols = [item["symbol"].upper() for item in assessments]
        expected = set()
        if "ranking" in payload:
            expected = {s.upper() for s in self._candidate_symbols}
        elif "new_first_symbol" in payload:
            expected = {payload["new_first_symbol"].upper(), payload["previous_first_symbol"].upper()}
        elif kwargs.get("symbol"):
            expected = {kwargs["symbol"].upper()}
        elif "reviews" in payload:
            expected = {row["symbol"].upper() for row in payload["reviews"]}
        elif "pairwise_comparisons" in payload:
            expected = {row[key].upper() for row in payload["pairwise_comparisons"]
                        for key in ("selected_symbol", "alternative_symbol")}
        if len(set(symbols)) != len(symbols) or (expected and set(symbols) != expected):
            raise ValueError("Evidence assessment must cover every evaluated stock exactly once")
        by_symbol = {item["symbol"].upper(): item for item in assessments}
        for item in assessments:
            for gap in iter_gap_records(item):
                for ref in gap["search_record_refs"]:
                    parts = ref.split("/")[1:]
                    if not ref.startswith("/") or "gap_audit" not in parts:
                        raise ValueError("Unresolved gaps must cite actual gap_audit search records")
                    node = getattr(self, "external_verification", {})
                    try:
                        for part in parts:
                            part = part.replace("~1", "/").replace("~0", "~")
                            node = node[int(part)] if isinstance(node, list) else node[part]
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise ValueError("Unresolved gap cites an unavailable search record") from exc
                    if not isinstance(node, dict) or not node:
                        raise ValueError("Unresolved gap must reference a non-empty search record")
        for row in payload.get("ranking", []):
            if row["confidence"] != by_symbol[row["symbol"].upper()]["investment_confidence"]:
                raise ValueError("confidence and investment_confidence disagree for the same ranking judgment")
        record = {"stage": event_prefix, "symbol": kwargs.get("symbol"),
                  "evidence_assessments": assessments}
        if stage_resolved:
            record["resolved_derivations"] = list(stage_resolved)
        self.evidence_assessments.append(record)
        write("evidence_assessments.json", self.evidence_assessments)
        requests = [{"symbol": item["symbol"], **request}
                    for item in assessments for request in item["research_requests"]]
        for gap in stage_pending_derivations:
            requests.append({
                "symbol": gap["symbol"],
                "question": gap.get("question") or f"为 {gap.get('field_name')} 提供公开原始输入并执行确定性计算",
                "decision_impact": gap.get("decision_impact", "需要得到可追溯计算结果后再比较"),
                "query": f"{gap['symbol']} {gap.get('field_name', 'derived field')} raw public inputs for deterministic calculation",
                "sources": gap.get("source_required") or ["issuer, SEC or regulator"],
                "research_depth": "只使用已核验的原始数值，保存输入、公式、结果和 as_of；不得由模型估算",
                "field_name": gap.get("field_name"),
                "gap_status": "DERIVATION_REQUIRED",
            })
        if requests:
            self.pending_research = {
                "status": "NEEDS_RESEARCH", **record, "research_requests": requests,
                "provisional_result": payload,
                "stage_context": {"schema_name": schema_name, "result_model": model_type.__name__,
                                  "prompt": prompt, "arguments": kwargs,
                                  "candidate_symbols": list(getattr(self, "_candidate_symbols", []))},
                "completed_stages": self.completed_research_stages,
                "next_step": "Codex must execute targeted read-only searches, update gap_audit, then reassess the affected stage; do not rerun Luna.",
            }
            write("research_pending.json", self.pending_research)
            raise ResearchPending("Decision-critical evidence requires actual research; see research_pending.json")
        if rationale is not None:
            key = kwargs.get("symbol") if event_prefix == "SOL_TOP5_SUPPLEMENTAL" else event_prefix
            self.selection_rationale[key] = rationale
            write("selection_rationale.json", self.selection_rationale)
        self.completed_research_stages.append({**record, "result": payload})
        if self.event_sink:
            self.event_sink({"event_type": "SKILL_EVIDENCE_AUDIT_PASSED", "symbol": kwargs.get("symbol"),
                             "metadata": {"stage": event_prefix}})
        return model_type.model_validate(payload)

    agent._structured_deep_stage = MethodType(audited, agent)
