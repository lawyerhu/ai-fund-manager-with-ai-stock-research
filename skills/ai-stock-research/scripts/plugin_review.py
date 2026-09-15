"""Review a snapshot captured by Codex's IBKR connector; never opens a broker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from research import NoTradingImports, require_models, research_runtime


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    contract_id: int = Field(gt=0)
    contract_description: str
    position: float
    currency: str
    asset_class: str
    market_price: float | None = None
    market_value: float | None = None
    average_price: float | None = None
    unrealized_pnl: float | None = None
    daily_pnl: float | None = None


class Summary(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    currency: str
    net_liquidation: float = Field(gt=0)
    total_cash_value: float


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1)
    contract_id: int = Field(gt=0)
    currency: str
    asset_class: str
    research_result_path: str = Field(min_length=1)
    researched_at: datetime
    # Structured result from this run, not an invented summary or local account history.
    research: dict


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["IBKR_CONNECTOR"]
    fetched_at: datetime
    account_type: Literal["UNKNOWN"] = "UNKNOWN"
    positions: list[Position]
    summary: Summary
    # Only structured public evidence keyed by the actual broker conId.
    evidence: dict[str, dict] = Field(default_factory=dict)
    selection: Selection | None = None
    # Only constraints explicitly confirmed for this connector account, never local Paper defaults.
    risk_constraints: dict = Field(default_factory=dict)

    def context(self):
        return {
            **self.model_dump(mode="json"),
            "original_thesis": "UNKNOWN", "entry_dates": "UNKNOWN",
            "quote_timestamp": "UNKNOWN", "market_data_type": "UNKNOWN",
            "positions": [{**p.model_dump(), "weight": p.market_value / self.summary.net_liquidation
                           if p.market_value is not None and p.currency == self.summary.currency else None}
                          for p in self.positions],
        }


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_id: int
    action: Literal["HOLD", "REDUCE", "EXIT", "REVIEW_REQUIRED"]
    target_weight: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str] = Field(min_length=1, max_length=5)
    new_negative_evidence: list[str] = Field(default_factory=list, max_length=5)
    missing_data: list[str] = Field(default_factory=list, max_length=8)
    weight_basis: str
    incremental_reason: str
    thesis_invalidation: list[str] = Field(default_factory=list, max_length=5)
    evidence_refs: list[str] = Field(min_length=1)


class CandidateSizing(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target_weight: float | None = Field(default=None, ge=0, le=1)
    portfolio_loss_budget: float | None = Field(default=None, gt=0, le=1)
    downside_fraction: float | None = Field(default=None, gt=0, le=1)
    basis: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_budget(self):
        if self.target_weight is not None:
            if self.portfolio_loss_budget is None or self.downside_fraction is None:
                raise ValueError("Numeric sizing requires a sourced loss budget and downside scenario")
            if self.target_weight * self.downside_fraction > self.portfolio_loss_budget + 1e-9:
                raise ValueError("Target exceeds the stated scenario loss budget")
        return self


class CandidateAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_id: int
    action: Literal["BUY", "KEEP_EXISTING", "WAIT", "REVIEW_REQUIRED"]
    reasons: list[str] = Field(min_length=1, max_length=5)
    comparison_with_holdings: list[str] = Field(min_length=1)
    funding_basis: str = Field(min_length=1)
    prerequisites: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    sizing: CandidateSizing | None = None


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio_findings: list[str] = Field(min_length=1, max_length=8)
    holdings: list[Assessment]
    candidate_advice: CandidateAdvice | None = None
    estimate_type: Literal["SOL_MODEL_ESTIMATE"] = "SOL_MODEL_ESTIMATE"


def validate_snapshot(raw, now=None):
    snapshot = Snapshot.model_validate(raw)
    now = now or datetime.now(timezone.utc)
    if snapshot.fetched_at.tzinfo is None or not 0 <= (now - snapshot.fetched_at).total_seconds() <= 900:
        raise ValueError("Refresh IBKR connector snapshot: must be timezone-aware and no older than 15 minutes")
    ids = [p.contract_id for p in snapshot.positions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate contract IDs; do not merge accounts or positions implicitly")
    if not set(snapshot.evidence).issubset({str(i) for i in ids}):
        raise ValueError("Evidence must refer to held contracts")
    if snapshot.selection:
        at = snapshot.selection.researched_at
        if at.tzinfo is None or at > now or not snapshot.selection.research:
            raise ValueError("Selection requires dated, nonempty actual research evidence")
    return snapshot


def review_snapshot(provider, runtime, snapshot):
    from src.llm_agent import _strict_json_schema, _response_text
    context = snapshot.context()
    prompt = (
        "Perform a Chinese, research-only multi-position review using the supplied IBKR connector evidence. "
        "You (GPT-6 Astra) are the decision maker for the recommendation: directly choose the buy/hold/reduce/exit "
        "action and, only when justified by supplied same-account constraints, the target weight; do not defer that "
        "decision to the host program or derive it from a fixed rule. "
        "Review every contract exactly once by conId; contract_description is a label, not a verified Yahoo ticker. "
        "This account is of UNKNOWN type and is NOT matched to local Paper portfolio history. "
        "Entry dates and original thesis are UNKNOWN: do not invent days held, thesis changes, or compare prior scores. "
        "Retrieval time does not establish live quote freshness. Distinguish missing data from negative facts; "
        "missing information alone is not grounds for repeated mechanical reductions. "
        "Account weights are computed only for positions with matching account currency. "
        "For currency mismatch, non-STK assets, short positions, missing valuation or insufficient research evidence, "
        "use REVIEW_REQUIRED and null target_weight. Do not infer company fundamentals from a ticker or price. "
        "Explain every weight recommendation and its incremental basis. Scores/confidence are model estimates, not "
        "calibrated probabilities. Do not invent a 20-day holding commitment or quantitative risk budget. "
        "evidence_refs must be positions, summary, or evidence.<actual conId> supplied below. "
        "If selection is present, evaluate that research alongside ALL current holdings in this same context. "
        "Return candidate_advice for exactly the selected conId, citing selection as an additional evidence source. "
        "Compare keeping existing holdings versus introducing the candidate: opportunity cost, overlapping exposures, "
        "research dates, evidence quality, cash funding, and trading costs (UNKNOWN if absent). "
        "Being ranked first does not require BUY or selling existing holdings. Do not force replacement. "
        "BUY is only for a not-yet-held supported same-currency STK; for an already held selection use KEEP_EXISTING, "
        "WAIT or REVIEW_REQUIRED, and put its holding recommendation only in holdings. "
        "Do not use WAIT/REVIEW_REQUIRED as a shortcut for an unperformed data search: the orchestrator must actively "
        "verify decision-critical gaps before this call. If the supplied packet documents exhausted verification, use "
        "WAIT/REVIEW_REQUIRED only for unresolved safety-critical gaps or conflicts; for ordinary unknown fields, make "
        "the best-supported HOLD/REDUCE/EXIT or BUY decision from verified evidence and list the unknowns. "
        "External verification is performed by the skill orchestrator, not by this API call: "
        "do not claim to browse or verify sources absent from the supplied packet. "
        "List actionable missing sources for the next verification round. "
        "Discard historical local portfolio weights, cash and incumbent-HOLD rationales. "
        "For candidate sizing, provide target_weight, portfolio_loss_budget, downside_fraction, basis and evidence_refs. "
        "Use only explicitly supplied same-account loss constraints, never infer a budget from confidence or past weights. "
        "When supplied, cite risk_constraints and match its portfolio_loss_budget exactly. "
        "If budget or downside evidence is unavailable, leave numeric sizing null and explain what is missing. "
        "Treat a scenario loss as an estimate, not a guaranteed stop loss. "
        "For every holding explain target versus current weight and whether a change is justified. "
        "Explain whether a purchase needs future sales; cash proceeds are not yet available and no orders are approved. "
        "For an empty portfolio, explicitly compare the candidate with retaining cash. "
        "Without selection return null candidate_advice. Never infer selection from local portfolio history. "
        "No orders, order instructions, risk approvals, or private chain-of-thought. Return only schema JSON.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    response = provider.create_response(model=runtime.sol_model, reasoning={"effort": "medium"}, input=prompt,
        text={"format": {"type": "json_schema", "name": "ibkr_connector_review", "strict": True,
                         "schema": _strict_json_schema(Review.model_json_schema())}})
    result = Review.model_validate_json(_response_text(response))
    expected = {p.contract_id for p in snapshot.positions}
    actual = [r.contract_id for r in result.holdings]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Review must cover each broker contract exactly once")
    allowed = {"positions", "summary", *(f"evidence.{k}" for k in snapshot.evidence)}
    if snapshot.risk_constraints:
        allowed.add("risk_constraints")
    if snapshot.selection:
        allowed.add("selection")
        advice = result.candidate_advice
        selection = snapshot.selection
        if advice is None or advice.contract_id != selection.contract_id:
            raise ValueError("Joint review must assess the actual selected contract")
        if not set(advice.evidence_refs).issubset(allowed) or "selection" not in advice.evidence_refs:
            raise ValueError("Candidate advice must cite available selection evidence")
        if advice.action == "BUY" and (selection.contract_id in expected or
                selection.currency != snapshot.summary.currency or selection.asset_class != "STK"):
            raise ValueError("Unsupported or already-held contract cannot receive a new BUY recommendation")
        if advice.action == "KEEP_EXISTING" and selection.contract_id not in expected:
            raise ValueError("KEEP_EXISTING requires an actual held contract")
        if advice.sizing:
            if not set(advice.sizing.evidence_refs).issubset(allowed):
                raise ValueError("Sizing cited an unavailable evidence source")
            if advice.action in {"WAIT", "REVIEW_REQUIRED"} and advice.sizing.target_weight is not None:
                raise ValueError("Waiting or unresolved advice cannot carry a numeric purchase target")
            if advice.sizing.target_weight is not None:
                constraints = snapshot.risk_constraints
                if (constraints.get("source") != "USER_CONFIRMED_SAME_ACCOUNT"
                        or constraints.get("portfolio_loss_budget") != advice.sizing.portfolio_loss_budget
                        or "risk_constraints" not in advice.sizing.evidence_refs):
                    raise ValueError("Sizing budget must match explicit same-account constraints")
    elif result.candidate_advice is not None:
        raise ValueError("Cannot invent a candidate without selection research")
    for item in result.holdings:
        if not set(item.evidence_refs).issubset(allowed):
            raise ValueError("Review cited an unavailable evidence source")
        p = next(p for p in snapshot.positions if p.contract_id == item.contract_id)
        limited = (p.currency != snapshot.summary.currency or p.asset_class != "STK" or p.position <= 0
                   or p.market_value is None or p.market_value <= 0 or p.market_price is None or p.market_price <= 0 or str(p.contract_id) not in snapshot.evidence
                   or not snapshot.evidence[str(p.contract_id)])
        if limited and (item.action != "REVIEW_REQUIRED" or item.target_weight is not None):
            raise ValueError("Insufficient valuation/research context requires REVIEW_REQUIRED, not a fabricated target")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    snapshot = validate_snapshot(json.loads(args.snapshot.read_text(encoding="utf-8")))
    if not args.run:
        print(json.dumps({"status": "VALIDATED", "source": snapshot.source, "account_type": "UNKNOWN",
                          "position_count": len(snapshot.positions), "fetched_at": snapshot.fetched_at.isoformat(),
                          "production_db_access": False, "orders": "DISABLED"}, ensure_ascii=False))
        return
    if not snapshot.positions and snapshot.selection is None:
        print("No positions returned by IBKR connector; no LLM review needed")
        return
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import load_config, load_project_env
    from src.llm_agent import LLMRuntimeConfig, CCSwitchProvider
    from src.storage import redact_sensitive
    load_project_env(project / ".env", override=True)
    runtime = research_runtime(LLMRuntimeConfig.from_mapping(load_config(project / "config.yaml")))
    output = project / "outputs/skill-research" / ("plugin-" + str(uuid4()))
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIRECTORY={output}", flush=True)
    report = {"source": "IBKR_CONNECTOR", "account_type": "UNKNOWN", "model_requested": runtime.sol_model,
              "effort_requested": "medium", "risk_approval": "NOT_RUN", "orders": "DISABLED",
              "snapshot": snapshot.model_dump(mode="json")}
    try:
        provider = CCSwitchProvider(runtime=runtime)
        require_models(provider, runtime, False)
        result = review_snapshot(provider, runtime, snapshot)
        report.update(status="COMPLETE", review=result.model_dump(mode="json"),
                      reasoning_metadata_status=provider.reasoning_metadata_status)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc))
    (output / "result.json").write_text(json.dumps(redact_sensitive(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"], flush=True)
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Connector review startup failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
