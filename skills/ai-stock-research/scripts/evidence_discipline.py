"""Skill-local evidence audit; leave the project's ranking and decision models intact."""
from types import MethodType

from pydantic import BaseModel, ConfigDict, Field, create_model


EVIDENCE_INSTRUCTIONS = (
    "EVIDENCE DISCIPLINE (governs evidence handling and fixed-field interpretation in the legacy prompt): "
    "You are GPT-6 Astra. Choose decision-relevant questions from each company's industry, business model, "
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
    "endpoint blindly. Only put a material question in unresolved_information_gaps after documented reasonable "
    "supplemental attempts; cite supplied gap_audit records using JSON pointers (e.g. /gap_audit/searches/0) "
    "in search_record_refs and explain why further research is unlikely to "
    "resolve it. A round limit alone does not prove a reasonable search occurred. Unsearched gaps require requests. "
    "Schema-required ranks/actions during an open request are provisional, not an abstention or a final choice. "
    "After reasonable research, make the required best-evidence decision despite genuine unresolved uncertainty. "
    "Do not use evidence completeness to default to KEEP_PREVIOUS or trigger a switch."
)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    decision_impact: str = Field(min_length=1)
    query: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    research_depth: str = Field(min_length=1)


class UnresolvedGap(BaseModel):
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
    research_requests: list[ResearchRequest]
    unresolved_information_gaps: list[UnresolvedGap]


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
        if capture_selection_path:
            from selection_rationale import chinese_text
            for item in assessments:
                chinese_text(item["evidence_completeness"])
                chinese_text(item["confidence_basis"])
        rationale = payload.pop("selection_rationale", None)
        if rationale is not None:
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
            for gap in item["unresolved_information_gaps"]:
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
        self.evidence_assessments.append(record)
        write("evidence_assessments.json", self.evidence_assessments)
        requests = [{"symbol": item["symbol"], **request}
                    for item in assessments for request in item["research_requests"]]
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
