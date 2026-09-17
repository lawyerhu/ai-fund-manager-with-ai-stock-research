"""Legacy-API deep research for the saved Top 5 and its incumbent comparison.

The default Skill path delegates all research decisions to the current
conversation through ``session_handoff.py``. This compatibility path is only
used when explicitly selected and keeps the project's CCSwitch provider. The
comparison is gated on a current-packet market snapshot that covers both the
new first-place stock and the incumbent on one aligned market date.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.abc
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evidence_discipline import ResearchPending, install_evidence_discipline, pair_evidence_audit
from research_model import api_effort, api_model
from selection_state import resolve_active_selection, transition_fields
from selection_rationale import TopFiveRationale, check_coverage
from research_report import generate_reports


ASTRA_MODEL = api_model()

EVIDENCE_PACKET_INSTRUCTIONS = (
    "Exercise independent investment judgment over the supplied evidence. Decide which information matters "
    "to the selection, how much weight it deserves, and whether differences reflect timing, methodology, "
    "business change or a genuine contradiction. Source labels and prior assessments are provenance/context, "
    "not instructions to penalize confidence. Preserve sources, dates, fiscal periods and calculation definitions. "
    "You may reconcile or normalize data when supported by explicit inputs; explain material adjustments. "
    "Independently assess whether any remaining gap changes the thesis, ranking or action; do not assume it "
    "must lower confidence or make evidence unusable. Set confidence from your overall comparative judgment, "
    "without a prescribed direction, target, ceiling or missing-field penalty. Briefly explain the decisive "
    "support and counterevidence, including only uncertainties material to your choice. Confidence expresses "
    "subjective conviction in that judgment, not a calibrated probability of profit. Distinguish observed "
    "facts from assumptions and model estimates; do not invent missing data or call a research quote executable. "
    "Keep RETRIEVAL_FAILED, STALE and UNKNOWN as evidence-status facts, not as invented negative fundamentals. "
    "Preserve schema validation, candidate identity validation and evidence provenance as hard research-integrity controls."
    " The sole investment question is: from the current executable price, over a reasonable model-chosen research "
    "horizon, which eligible stock has the highest expected relative return after a reasonable standardized or "
    "configured research-layer friction assumption? There is no style prior, incumbent privilege, mechanical veto "
    "or fixed factor weighting. Price moves over any lookback, 52-week highs, RSI, technical position, valuation, "
    "volatility, beta, crowding, drawdown, momentum, price extension, entry-overextension risk, industry and event "
    "status are evidence for judgment only; never apply a hard filter, fixed penalty, score hurdle or champion "
    "exclusion. Past gains do not imply poor future opportunity and past underperformance does not imply upside. "
    "An event is not alpha by itself and a near-term catalyst is not required. Assess the remaining expectation gap, "
    "earnings/revenue/FCF revisions, valuation expansion versus operating improvement, catalyst pricing and downside "
    "as relevant to the thesis. Ignore sunk entry cost, P&L, historical rank/score and elapsed holding time as "
    "investment reasons. Do not require a minimum holding period, mechanical sell date, broken incumbent thesis or "
    "fixed replacement hurdle. You may KEEP when a small apparent advantage is inside your own error, or SWITCH "
    "when a small but highly credible net advantage covers research friction; that is model judgment, not a program "
    "threshold. Luna ranking is retrieval context only, not a final prior. Normal successful research selects among "
    "eligible stocks rather than cash. Account-level commissions, spread, slippage, quantity and broker state are "
    "outside this research layer and must not be invented."
)


MARKET_SNAPSHOT_SECTIONS = (
    "comparison_market_snapshot",
    "same_date_market_snapshot",
    "project_same_basis_market_data",
)
UNUSABLE_MARKET_SNAPSHOT_STATUSES = {"UNKNOWN", "UNAVAILABLE", "NOT_APPLICABLE", "MISSING", "RETRIEVAL_FAILED", "STALE"}
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


class PairMarketSnapshotRequired(ValueError):
    """The current evidence packet cannot support a same-date pair comparison."""


def _date_part(value):
    if isinstance(value, str):
        match = DATE_PATTERN.search(value)
        return match.group(1) if match else None
    return None


def _basis_date(value):
    if isinstance(value, dict):
        for key in ("market_snapshot_date", "snapshot_date", "as_of_date", "as_of", "date", "retrieved_at"):
            date = _date_part(value.get(key))
            if date:
                return date
    return _date_part(value)


def _snapshot_records(snapshot):
    """Return ``(implied_symbol, record)`` pairs from legacy packet shapes."""
    values = []
    if isinstance(snapshot, list):
        values = [(None, row) for row in snapshot if isinstance(row, dict)]
    elif isinstance(snapshot, dict):
        for key in ("records", "symbols", "items", "data"):
            nested = snapshot.get(key)
            if isinstance(nested, list):
                values = [(None, row) for row in nested if isinstance(row, dict)]
                break
            if isinstance(nested, dict):
                values = [(name, row) for name, row in nested.items() if isinstance(row, dict)]
                break
        if not values:
            values = [
                (name, row) for name, row in snapshot.items()
                if isinstance(row, dict) and re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", str(name).upper())
            ]
    records = []
    for implied, value in values:
        record = dict(value)
        symbol = str(record.get("symbol") or implied or "").upper()
        if symbol:
            record["symbol"] = symbol
            records.append(record)
    return records


def _snapshot_record_date(record):
    for key in ("snapshot_date", "quote_as_of", "last_bar_at", "observed_at", "as_of", "date", "timestamp"):
        date = _date_part(record.get(key))
        if date:
            return date
    return None


def _usable_market_snapshot_record(record):
    status = str(record.get("status", "")).upper()
    if status in UNUSABLE_MARKET_SNAPSHOT_STATUSES:
        return False
    facts = record.get("facts")
    if isinstance(facts, dict):
        return any(value is not None for value in facts.values())
    metadata = {"symbol", "status", "source", "source_url", "observed_at", "quote_as_of", "last_bar_at", "as_of", "date", "timestamp", "limitations"}
    return any(key not in metadata and value is not None for key, value in record.items())


def require_pair_market_snapshot(packet: dict, new_first_symbol: str, previous_first_symbol: str) -> dict:
    """Return one aligned snapshot covering both comparison legs or fail closed.

    The helper accepts the skill's existing list and symbol-keyed packet forms.
    It deliberately reads only the current verification packet, never a cached
    snapshot embedded in a prior research result.
    """
    new_symbol = str(new_first_symbol).upper()
    previous_symbol = str(previous_first_symbol).upper()
    basis = packet.get("as_of_basis")
    for section in MARKET_SNAPSHOT_SECTIONS:
        snapshot = packet.get(section)
        records = _snapshot_records(snapshot)
        if not records:
            continue
        selected = {}
        usable = True
        for symbol in dict.fromkeys((new_symbol, previous_symbol)):
            matches = [record for record in records if record["symbol"] == symbol]
            if len(matches) != 1 or not _usable_market_snapshot_record(matches[0]):
                usable = False
                break
            selected[symbol] = matches[0]
        if not usable:
            continue
        row_dates = {_snapshot_record_date(record) for record in selected.values()}
        row_dates.discard(None)
        section_date = _basis_date(snapshot)
        basis_date = _basis_date(basis)
        if len(row_dates) > 1:
            continue
        if len(row_dates) == 1:
            as_of_date = next(iter(row_dates))
            if any(_snapshot_record_date(record) is None for record in selected.values()) and section_date != as_of_date:
                continue
        else:
            as_of_date = section_date or basis_date
        if not as_of_date:
            continue
        sources = {
            str(record.get("source") or (snapshot.get("source") if isinstance(snapshot, dict) else "")).strip()
            for record in selected.values()
        }
        sources.discard("")
        if len(sources) > 1:
            continue
        return {
            "status": "VERIFIED_PAIR_SAME_DATE",
            "section": section,
            "as_of_date": as_of_date,
            "source": next(iter(sources), None),
            "basis": basis,
            "symbols": selected,
        }
    raise PairMarketSnapshotRequired(
        "Current verification packet lacks one usable, same-date market snapshot covering "
        f"both {new_symbol} and {previous_symbol}"
    )


def validate_evidence_packet(packet: dict) -> dict:
    """Require an explicit audit while allowing precise gaps and old statuses."""
    if not isinstance(packet.get("evidence"), list) or not packet["evidence"]:
        raise ValueError("Verification packet must contain a non-empty evidence list")
    required_sections = ("as_of_basis", "gap_audit")
    missing = [name for name in required_sections if name not in packet]
    if missing:
        raise ValueError("Verification packet is missing evidence audit sections: " + ", ".join(missing))
    if not isinstance(packet.get("gap_audit"), dict):
        raise ValueError("Verification packet gap_audit must be an object")
    from evidence_discipline import GAP_STATUS_VALUES
    for item in packet["evidence"]:
        if not isinstance(item, dict):
            raise ValueError("Each verification packet evidence item must be an object")
        for key in ("status", "gap_status"):
            value = item.get(key)
            # UNKNOWN/MISSING are legacy packet markers, not new gap
            # classifications; keep them readable without using them to gate
            # the pair decision.
            if value is not None and str(value).upper() not in GAP_STATUS_VALUES | {"UNKNOWN", "MISSING", "NOT_RECORDED"}:
                raise ValueError(f"Unknown verification evidence status: {value}")
    from deterministic_calculations import resolve_packet_derivations
    resolve_packet_derivations(packet)
    return packet


class PreviousWinnerComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_first_symbol: str = Field(min_length=1)
    previous_first_symbol: str = Field(min_length=1)
    rebalance_decision: Literal["SWITCH_TO_NEW_FIRST", "KEEP_PREVIOUS"]
    alpha_gap: float | None = None
    alpha_gap_status: Literal["COMPARABLE", "UNKNOWN"]
    # Backward-compatible narrative field; the decision-time rationale carries
    # the same comparison in the final report when it was recorded.
    remaining_alpha_comparison: str = "NOT_RECORDED"
    why_new_beats_previous: list[str] = Field(min_length=1, max_length=5)
    why_keep_previous: list[str] = Field(min_length=1, max_length=5)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0, le=1)
    confidence_reducers: list[str] = Field(default_factory=list, max_length=8)
    thesis_invalidation_conditions: list[str] = Field(min_length=1, max_length=5)
    # ``None`` keeps an older comparison readable without silently asserting
    # that its new completeness fields were recorded.  The final flow accepts
    # only explicit ``True`` values for a newly completed comparison.
    pair_comparison_complete: bool | None = None
    decision_basis_sufficient: bool | None = None
    material_asymmetry_resolved: bool | None = None


class NoTradingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {
            "src.main", "src.runner", "src.service", "src.scheduler", "src.risk_engine",
            "src.execution",
        }
        if fullname in blocked or fullname.startswith("src.execution."):
            raise ImportError(f"Research-only skill blocks {fullname}")
        return None


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def extract_initial_result(raw: dict) -> tuple[dict, dict[str, dict], list[dict]]:
    """Accept the current equal-depth result shape and the older nested shape."""
    ranking = raw.get("ranking")
    deep = raw.get("deep_research")
    if isinstance(ranking, dict) and isinstance(ranking.get("ranking"), list):
        initial_ranking = ranking
    else:
        nested = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        nested_deep = nested.get("deep_research") if isinstance(nested.get("deep_research"), dict) else {}
        initial_ranking = nested_deep.get("ranking") if isinstance(nested_deep.get("ranking"), dict) else {}
        deep = nested_deep.get("deep_research") or nested.get("deep_research")
    if not isinstance(initial_ranking, dict) or not isinstance(initial_ranking.get("ranking"), list):
        raise ValueError("Source result has no unified ranking")
    rows = initial_ranking["ranking"]
    if len(rows) < 5:
        raise ValueError("Source result has fewer than five ranked candidates")
    top5 = rows[:5]
    symbols = [str(row.get("symbol", "")).upper() for row in top5]
    if any(not symbol for symbol in symbols) or len(set(symbols)) != 5:
        raise ValueError("Unified Top 5 is invalid or contains duplicates")
    if not isinstance(deep, dict):
        deep = {}
    deep_by_symbol = {str(key).upper(): value for key, value in deep.items() if isinstance(value, dict)}
    return initial_ranking, deep_by_symbol, top5


def extract_previous_winner(raw: dict) -> tuple[str, dict]:
    """Resolve the retained AI selection; the function name is kept for existing callers."""
    symbol = resolve_active_selection(raw)
    nested = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    containers = [raw, nested]
    winner_record = None

    if winner_record is None:
        for container in containers:
            for key in ("final_ranking", "ranking"):
                ranking = container.get(key)
                rows = ranking.get("ranking") if isinstance(ranking, dict) else None
                if isinstance(rows, list):
                    winner_record = next(
                        (row for row in rows if isinstance(row, dict) and str(row.get("symbol", "")).upper() == symbol),
                        None,
                    )
                    if winner_record:
                        break
            if winner_record:
                break

    research_record = None
    for container in containers:
        for key in ("active_selection_research", "previous_first_supplemental_research"):
            record = container.get(key)
            if isinstance(record, dict) and str(record.get("symbol", "")).upper() == symbol:
                research_record = record
                break
        if research_record:
            break
        for key in ("supplemental_research", "deep_research"):
            value = container.get(key)
            if isinstance(value, dict):
                candidate = value.get(symbol) or value.get(symbol.upper())
                if isinstance(candidate, dict):
                    research_record = candidate
                    break
        if research_record:
            break

    return symbol, {
        "winner_record": winner_record or {"symbol": symbol},
        "research_record": research_record or {},
        "source_status": raw.get("status"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--verification-packet", type=Path, required=True)
    parser.add_argument("--previous-result", type=Path, required=True,
                        help="Completed prior result carrying the retained active selection after KEEP/SWITCH")
    parser.add_argument("--run", action="store_true", help="Actually send Astra requests; never orders")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Top-five deep research requires --run")

    project = args.project.resolve()
    source_path = args.source_result.resolve()
    packet_path = args.verification_packet.resolve()
    previous_path = args.previous_result.resolve()
    if not (project / "src/llm_agent.py").is_file():
        parser.error("Project does not contain src/llm_agent.py")
    source = load_json(source_path)
    packet = load_json(packet_path)
    previous = load_json(previous_path)
    packet = validate_evidence_packet(packet) or packet
    initial_ranking, deep_by_symbol, top5_rows = extract_initial_result(source)
    if source.get("status") != "COMPLETE":
        raise ValueError("Unified candidate research is not COMPLETE")
    initial_rationale = TopFiveRationale.model_validate(source.get("selection_rationale")).model_dump(mode="json")
    candidate_symbols = source.get("candidate_symbols") or list(deep_by_symbol)
    if len(set(candidate_symbols)) != len(candidate_symbols) or source.get("candidate_count", len(candidate_symbols)) != len(candidate_symbols):
        raise ValueError("Source candidate coverage is incomplete or inconsistent")
    check_coverage(initial_rationale, initial_ranking, candidate_symbols)
    previous_symbol, previous_context = extract_previous_winner(previous)
    top5 = [str(row["symbol"]).upper() for row in top5_rows]
    missing = [symbol for symbol in top5 if symbol not in deep_by_symbol]
    if missing:
        raise ValueError("Source result is missing unified research for: " + ", ".join(missing))

    sys.path.insert(0, str(project))
    sys.meta_path.insert(0, NoTradingImports())
    from src.config import load_config, load_project_env, env
    load_project_env(project / ".env", override=True)
    os.environ["EXECUTION_MODE"] = "OBSERVE"
    from src.data_provider import YahooFinanceDataProvider
    from src.llm_agent import (
        CCSwitchProvider,
        CrossSectionalRanking,
        DeepDiveResearch,
        LLMRuntimeConfig,
        SolResearchCIOAgent,
        _accumulate_usage,
    )
    from src.storage import redact_sensitive

    cfg = load_config(project / "config.yaml")
    runtime = replace(
        LLMRuntimeConfig.from_mapping(cfg),
        sol_model=ASTRA_MODEL,
        sol_reasoning_effort=api_effort(),
        pipeline="LUNA_SOL",
        fallback_to_sol_only=False,
    )
    if cfg.get("data", {}).get("provider", "mock").lower() not in {"yahoo", "yahoo_finance"}:
        raise ValueError("Real research requires configured Yahoo provider; refusing mock data")

    output = project / "outputs" / "skill-research" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-top5-deep-" + str(uuid4())
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
        "mode": "top_five_supplemental_deep_research_and_previous_first_comparison",
        "research_model": runtime.sol_model,
        "research_effort_requested": runtime.sol_reasoning_effort,
        "source_result": str(source_path),
        "verification_packet": str(packet_path),
        "verification_packet_type": packet.get("packet_type"),
        "verification_basis": packet.get("as_of_basis"),
        "evidence_sections_required": ["evidence", "as_of_basis", "gap_audit"],
        "previous_result": str(previous_path),
        "previous_first_symbol": previous_symbol,
        "incoming_active_selection": previous_symbol,
        "candidate_count": len(candidate_symbols),
        "candidate_symbols": candidate_symbols,
        "universe": source.get("universe"),
        "source_decision_id": (source.get("source") or {}).get("decision_id"),
        "top_five_count": 5,
        "top_five": top5,
        "objective": "From current executable prices, select the eligible stock with the highest expected forward relative return over a reasonable model-chosen horizon after a reasonable research-layer friction assumption; no style prior, incumbent privilege or mechanical investment veto.",
        "research_layer_friction_assumption": "STANDARDIZED_OR_CONFIGURED_RESEARCH_LAYER_ASSUMPTION_ONLY; NOT_ACCOUNT_COMMISSION_OR_SLIPPAGE",
        "active_selection_meaning": "RESEARCH_LAYER_CURRENT_AI_PREFERENCE; NOT_BROKER_POSITION",
        "safety": "RESEARCH_ONLY; source result read-only; no broker/Risk Engine instance",
        "placeOrder_calls": 0,
        "cancelOrder_calls": 0,
        "risk_approval": "NOT_RUN",
        "final_score_field": "preliminary_alpha_score in the final Top 5 ranking call; never mixed with prior scores",
        "account_review": "NOT_RUN; IBKR connector and plugin_review.py are not used",
    }
    write("manifest.json", manifest)

    provider = CCSwitchProvider(runtime=runtime, event_sink=sink)
    rows = provider.client.models.list()
    model_ids = {
        row.get("id") if isinstance(row, dict) else row.id
        for row in (rows.get("data", []) if isinstance(rows, dict) else rows.data)
    }
    if ASTRA_MODEL not in model_ids:
        raise ValueError(f"Required model unavailable: {ASTRA_MODEL}")

    agent = SolResearchCIOAgent(
        runtime.sol_model,
        YahooFinanceDataProvider(),
        provider=provider,
        max_tool_rounds=int(cfg.get("agent", {}).get("max_tool_rounds", 12)),
        deep_research_enabled=True,
        event_sink=sink,
    )
    agent._candidate_symbols = top5
    agent.external_verification = packet
    install_evidence_discipline(agent, write, capture_selection_path=True)
    external = json.dumps(packet, ensure_ascii=False, default=str)
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    supplemental: dict[str, dict] = {}
    failures: dict[str, str] = {}
    started = time.perf_counter()

    for index, symbol in enumerate(top5, start=1):
        prompt = (
            "You are the research decision model for the current run, performing supplemental verification for one candidate that survived a unified "
            "cross-sectional screen. Reassess the supplied initial research using the independently retrieved evidence. "
            "Choose the material questions for this company, its business model and current investment thesis. "
            f"{EVIDENCE_PACKET_INSTRUCTIONS} "
            "Use the applicable packet sections. If relevant, for the same-date market snapshot, "
            "assess valuation, volatility and liquidity proxies against the other candidates using your judgment "
            "of comparability and materiality; explain necessary adjustments. For earnings revisions, distinguish consensus "
            "drift/revision observations from issuer guidance and from price-target actions. For cash flow, preserve "
            "the reported period and GAAP/non-GAAP basis. For catalysts and peer data, name the source date or say "
            "UNKNOWN. "
            "Use 20 trading days as the default review/comparison/forecasting reference horizon only; it is not a "
            "minimum or mandatory holding period, mechanical sell date, or deadline for a catalyst. The model may "
            "choose days, weeks, months or quarters when the thesis warrants it. Do not invent an exact return, "
            "consensus revision, outage, spread, quote or probability. Explicitly say what new evidence confirms, "
            "weakens or leaves unresolved. "
            "Confidence may rise or fall; do not force it upward. Return compact structured DeepDiveResearch JSON only.\n"
            f"SYMBOL: {symbol}\n"
            f"INITIAL_UNIFIED_RANKING_ITEM:\n{json.dumps(next(row for row in top5_rows if str(row.get('symbol')).upper() == symbol), ensure_ascii=False, default=str)}\n"
            f"INITIAL_DEEP_RESEARCH:\n{json.dumps(deep_by_symbol[symbol], ensure_ascii=False, default=str)}\n"
            f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{external}\n"
        )
        try:
            research = agent._structured_deep_stage(
                DeepDiveResearch,
                "sol_top_five_supplemental_deep_dive",
                "SOL_TOP5_SUPPLEMENTAL",
                prompt,
                totals,
                symbol=symbol,
                metadata={"index": index, "total": len(top5), "source_result": str(source_path)},
            )
            if research.symbol.upper() != symbol:
                raise ValueError(f"Astra returned {research.symbol}, expected {symbol}")
            supplemental[symbol] = research.model_dump(mode="json")
            sink({
                "event_type": "SOL_TOP5_SUPPLEMENTAL_COMPLETED",
                "component": "SOL",
                "message": f"Supplemental deep research completed for {symbol}",
                "symbol": symbol,
                "metadata": {"index": index, "total": len(top5)},
            })
        except ResearchPending:
            write("result.json", {**manifest, **agent.pending_research,
                                 "supplemental_research": supplemental, "usage": totals})
            return 2
        except Exception as exc:
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            sink({
                "event_type": "SOL_TOP5_SUPPLEMENTAL_FAILED",
                "component": "SOL",
                "message": f"Supplemental deep research failed for {symbol}",
                "symbol": symbol,
                "metadata": {"index": index, "total": len(top5), "error": str(exc)},
            })

    if failures:
        result = {
            **manifest,
            "status": "FAILED",
            "completed_supplemental_research": sorted(supplemental),
            "missing_supplemental_research": sorted(failures),
            "errors": failures,
            "usage": totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        print("Top-five supplemental research FAILED; see redacted result.json", flush=True)
        return 1

    final_prompt = (
        "Rank exactly the five supplied Top 5 candidates after their supplemental verification. This is a new "
        "cross-sectional evaluation in one common context. Return all five symbols exactly once, ranks 1 through 5, "
        "and a judgmental 0-100 score in preliminary_alpha_score that is comparable only within this final call. "
        f"{EVIDENCE_PACKET_INSTRUCTIONS} Do not copy initial scores mechanically. Weigh the same-date common "
        "evidence you judge material alongside company-specific research. Disclose material gaps. "
        "UNKNOWN is unresolved evidence, not a reason to invent a negative fact. Include concrete strengths and "
        "weaknesses, and explain why #1 beats #2 from the current price forward. The narrative must explicitly "
        "address the remaining expectation gap: what the market appears to price, what the verified evidence implies, "
        "whether past price movement has consumed the gap, whether earnings/revenue/FCF revisions support the price, "
        "whether valuation change came from operating growth or multiple expansion, catalyst pricing status, the "
        "most important downside path and why #1 is better than the other candidates. Do not turn these topics into "
        "fixed factor weights or vetoes. "
        "This ranking is not a trade order, expected return, probability or statistical alpha. Return only schema JSON.\n"
        f"TOP5_SYMBOLS: {json.dumps(top5, ensure_ascii=False)}\n"
        f"INITIAL_TOP5_RANKING:\n{json.dumps(top5_rows, ensure_ascii=False, default=str)}\n"
        f"SUPPLEMENTAL_RESEARCH:\n{json.dumps(supplemental, ensure_ascii=False, default=str)}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{external}\n"
    )
    try:
        final_ranking = agent._structured_deep_stage(
            CrossSectionalRanking,
            "sol_top_five_final_ranking",
            "SOL_TOP5_FINAL_RANKING",
            final_prompt,
            totals,
            metadata={"candidate_count": len(top5), "source_result": str(source_path)},
        )
        final_symbols = [item.symbol.upper() for item in final_ranking.ranking]
        if len(set(final_symbols)) != len(final_symbols):
            raise ValueError("Final ranking contains duplicate symbols")
        if set(final_symbols) != set(top5) or len(final_symbols) != 5:
            raise ValueError("Final ranking does not cover exactly the saved Top 5")
    except ResearchPending:
        write("result.json", {**manifest, **agent.pending_research,
                             "supplemental_research": supplemental, "usage": totals})
        return 2
    except Exception as exc:
        result = {
            **manifest,
            "status": "FAILED",
            "completed_supplemental_research": top5,
            "errors": {"FINAL_RANKING": f"{type(exc).__name__}: {exc}"},
            "supplemental_research": supplemental,
            "usage": totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        print("Top-five final ranking FAILED; see redacted result.json", flush=True)
        return 1

    final_rows = [item.model_dump(mode="json") for item in final_ranking.ranking]
    selected = final_rows[0]
    runner_up = final_rows[1]

    previous_supplemental = supplemental.get(previous_symbol)
    previous_failure = None
    if previous_supplemental is None:
        previous_prompt = (
            "You are the research decision model for the current run, performing the same supplemental verification for the incoming active selection "
            "stock. This stock is an incumbent research selection, not an account holding. Reassess the supplied "
            "prior research using the labeled evidence layers from VERIFIED_EXTERNAL_EVIDENCE_JSON. Choose the material "
            f"questions for this company's investment thesis. {EVIDENCE_PACKET_INSTRUCTIONS} "
            "Use the same-date market and revision sections when available, while preserving UNKNOWN for fields not "
            "covered for the incumbent. Use 20 trading days as a review/comparison/forecasting reference only, not a "
            "minimum holding period, mechanical sell date or catalyst deadline. The model may choose a longer or "
            "shorter thesis horizon. Do not invent an exact return, consensus revision, event, spread, quote or probability. "
            "Return compact structured DeepDiveResearch JSON only.\n"
            f"PREVIOUS_SYMBOL: {previous_symbol}\n"
            f"PREVIOUS_RESULT_RECORD:\n{json.dumps(previous_context, ensure_ascii=False, default=str)}\n"
            f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{external}\n"
        )
        try:
            previous_research = agent._structured_deep_stage(
                DeepDiveResearch,
                "sol_previous_first_supplemental_deep_dive",
                "SOL_PREVIOUS_FIRST_SUPPLEMENTAL",
                previous_prompt,
                totals,
                symbol=previous_symbol,
                metadata={"source_result": str(previous_path), "previous_first_symbol": previous_symbol},
            )
            if previous_research.symbol.upper() != previous_symbol:
                raise ValueError(f"Astra returned {previous_research.symbol}, expected {previous_symbol}")
            previous_supplemental = previous_research.model_dump(mode="json")
            sink({
                "event_type": "SOL_PREVIOUS_FIRST_SUPPLEMENTAL_COMPLETED",
                "component": "SOL",
                "message": f"Supplemental research completed for previous first-place stock {previous_symbol}",
                "symbol": previous_symbol,
            })
        except ResearchPending:
            write("result.json", {**manifest, **agent.pending_research,
                                 "supplemental_research": supplemental,
                                 "provisional_ranking": final_ranking.model_dump(mode="json"), "usage": totals})
            return 2
        except Exception as exc:
            previous_failure = f"{type(exc).__name__}: {exc}"
            sink({
                "event_type": "SOL_PREVIOUS_FIRST_SUPPLEMENTAL_FAILED",
                "component": "SOL",
                "message": f"Supplemental research failed for previous first-place stock {previous_symbol}",
                "symbol": previous_symbol,
                "metadata": {"error": str(exc)},
            })

    if previous_failure:
        result = {
            **manifest,
            "status": "FAILED",
            "completed_supplemental_research": sorted(supplemental),
            "missing_supplemental_research": [previous_symbol],
            "errors": {"PREVIOUS_FIRST_SUPPLEMENTAL": previous_failure},
            "final_ranking": final_ranking.model_dump(mode="json"),
            "supplemental_research": supplemental,
            "previous_winner_record": previous_context,
            "usage": totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        print("Previous-first comparison blocked because incumbent verification failed; see result.json", flush=True)
        return 1

    pair_audit = pair_evidence_audit(
        packet,
        [*(previous.get("evidence_assessments", []) if isinstance(previous.get("evidence_assessments"), list) else []),
          *agent.evidence_assessments],
        selected["symbol"],
        previous_symbol,
        additional_research=[previous_supplemental],
    )
    write("pair_evidence_audit.json", pair_audit)
    if pair_audit["requires_targeted_research"]:
        pending = {
            **manifest,
            "status": "NEEDS_RESEARCH",
            "phase": "INCUMBENT_COMPARISON",
            "pair_evidence_audit": pair_audit,
            "research_requests": pair_audit["research_requests"],
            "supplemental_research": supplemental,
            "previous_first_supplemental_research": previous_supplemental,
            "provisional_ranking": final_ranking.model_dump(mode="json"),
            "completed_supplemental_research": sorted(set(supplemental) | {previous_symbol}),
            "research_rounds_remaining": pair_audit["rounds_remaining"],
            "next_step": "补查成对比较中受影响的 incumbent/challenger 变量后，继续比较；不重跑 Luna 或整个 Top 5。",
            "usage": totals,
        }
        write("research_pending.json", pending)
        write("result.json", pending)
        sink({
            "event_type": "PAIR_EVIDENCE_AUDIT_NEEDS_RESEARCH",
            "component": "SOL",
            "message": "Pair evidence audit found a direction-changing retrieval/stale asymmetry",
            "metadata": {"research_requests": pair_audit["research_requests"], "rounds_remaining": pair_audit["rounds_remaining"]},
        })
        print("Pair evidence audit NEEDS_RESEARCH; active selection was not advanced.", flush=True)
        return 2

    comparison_prompt = (
        "Compare the new first-place stock with incoming_active_selection, the AI selection retained after the "
        "previous final KEEP/SWITCH decision, in one common research "
        "context. This is a security-selection rotation recommendation only, not an account review. Do not read or "
        "infer holdings, cash, target weights, shares, trade costs or broker state. Return exactly the two symbols "
        f"provided. {EVIDENCE_PACKET_INSTRUCTIONS} Choose exactly one rebalance_decision based on the direct forward-looking comparison: "
        "from current executable prices, is continuing with the incumbent or switching to the challenger more likely "
        "to deliver the higher relative return after a reasonable research-layer friction assumption? There is no "
        "fixed score, confidence, alpha-gap or replacement hurdle. The model may KEEP when a small challenger edge "
        "is within its own estimation error, or SWITCH when a small but highly credible net edge covers friction; "
        "neither is incumbent protection or a program rule. Do not require the incumbent thesis to be broken, a minimum "
        "holding period, a fixed elapsed time, a catalyst within 20 trading days, or a particular style/industry. "
        "After documented evidence searches, make a best-available-evidence choice even when important fields remain "
        "unresolved. Do not abstain, request review, or automatically keep the incumbent because data are missing. "
        "A research_requests entry suspends finality until Codex has performed the requested search. "
        "Do not treat the challenger's newer or more complete public information as an investment advantage merely "
        "because the incumbent's key variable is RETRIEVAL_FAILED or STALE. First resolve that asymmetry; only then "
        "may a switch rely on evidence that the incumbent variable actually weakened, or that it is objectively not "
        "publicly/affordably obtainable and the remaining evidence still supports the challenger. "
        "NOT_PUBLIC, PAID_DATA_REQUIRED and NOT_YET_OCCURRED may remain after a reasonable search and do not by "
        "themselves block the model's final choice. If the thesis materially depends on a refiner/energy cycle, "
        "consider crack spread, inventories, utilization, throughput, capture rate, maintenance/outage, earnings "
        "revisions and mid-cycle earnings, while distinguishing current peak profit from expectations about its "
        "duration; do not apply a fixed factor formula or mechanically penalize a risen price or peak profit. "
        "Explain the remaining Alpha / expectation gap, including what is already priced, how earnings/revenue/FCF "
        "revisions relate to price, whether multiple expansion did the work, whether catalysts are unpriced/partly "
        "priced/fully priced, and the key downside path. Event presence is evidence, not Alpha, and no near-term event "
        "is required. Ignore entry cost, floating P&L and holding time as sunk-cost reasons. "
        "Return pair_comparison_complete, decision_basis_sufficient and material_asymmetry_resolved as true only "
        "after the substantive pair comparison and reasonable handling of decision-critical retrieval failures. "
        "The structural pair audit below is context, not an investment score or an automatic action. "
        "Explain why the chosen action is preferable to its alternative, material assumptions, "
        "remaining gaps and the evidence that would reverse it. This is a small-capital concentrated-alpha "
        "experiment: seek prospective excess return, exclude cash as a selection, and do not let lower volatility "
        "alone determine the choice. Confidence need not be high to make a decision. If symbols are "
        "identical, return KEEP_PREVIOUS. Do not subtract scores from different runs. alpha_gap must be null with "
        "alpha_gap_status UNKNOWN unless a same-date, same-basis comparable gap is explicitly supported. Give "
        "concrete reasons why the new first beats the previous first and why keeping the previous first could still "
        "be correct. Include short evidence_refs that point to the supplied research/evidence records. "
        "Return only schema JSON; no private chain-of-thought and no orders.\n"
        f"NEW_FIRST_SYMBOL: {selected['symbol']}\n"
        f"NEW_FIRST_FINAL_RANKING_ITEM:\n{json.dumps(selected, ensure_ascii=False, default=str)}\n"
        f"NEW_TOP5_FINAL_RANKING:\n{json.dumps(final_rows, ensure_ascii=False, default=str)}\n"
        f"NEW_FIRST_SUPPLEMENTAL_RESEARCH:\n{json.dumps(supplemental.get(selected['symbol'].upper(), {}), ensure_ascii=False, default=str)}\n"
        f"PREVIOUS_FIRST_SYMBOL: {previous_symbol}\n"
        f"INCOMING_ACTIVE_SELECTION: {previous_symbol}\n"
        f"PREVIOUS_FIRST_RESULT:\n{json.dumps(previous_context, ensure_ascii=False, default=str)}\n"
        f"PREVIOUS_FIRST_SUPPLEMENTAL_RESEARCH:\n{json.dumps(previous_supplemental, ensure_ascii=False, default=str)}\n"
        f"PAIR_EVIDENCE_AUDIT:\n{json.dumps(pair_audit, ensure_ascii=False, default=str)}\n"
        f"VERIFIED_EXTERNAL_EVIDENCE_JSON:\n{external}\n"
    )
    try:
        comparison = agent._structured_deep_stage(
            PreviousWinnerComparison,
            "sol_new_first_vs_previous_first_comparison",
            "SOL_NEW_VS_PREVIOUS",
            comparison_prompt,
            totals,
            symbol=selected["symbol"],
            metadata={
                "new_first_symbol": selected["symbol"],
                "previous_first_symbol": previous_symbol,
                "source_result": str(source_path),
                "previous_result": str(previous_path),
            },
        )
        comparison_json = comparison.model_dump(mode="json")
        if comparison.new_first_symbol.upper() != selected["symbol"].upper():
            raise ValueError("Comparison returned an unexpected new first-place symbol")
        if comparison.previous_first_symbol.upper() != previous_symbol:
            raise ValueError("Comparison returned an unexpected previous first-place symbol")
        if comparison.new_first_symbol.upper() == comparison.previous_first_symbol.upper() and comparison.rebalance_decision != "KEEP_PREVIOUS":
            raise ValueError("Identical first-place symbols must produce KEEP_PREVIOUS")
        if comparison.alpha_gap_status == "COMPARABLE" and comparison.alpha_gap is None:
            raise ValueError("Comparable alpha gap cannot be null")
        if comparison.alpha_gap_status == "UNKNOWN" and comparison.alpha_gap is not None:
            raise ValueError("Unknown alpha gap must be null")
        required_pair_flags = (
            "pair_comparison_complete", "decision_basis_sufficient", "material_asymmetry_resolved",
        )
        if any(comparison_json.get(key) is not True for key in required_pair_flags):
            pending = {
                **manifest,
                "status": "NEEDS_RESEARCH",
                "phase": "INCUMBENT_COMPARISON",
                "pair_evidence_audit": {
                    **pair_audit,
                    "model_flags": {key: comparison_json.get(key) for key in required_pair_flags},
                },
                "research_requests": pair_audit["research_requests"],
                "supplemental_research": supplemental,
                "previous_first_supplemental_research": previous_supplemental,
                "provisional_ranking": final_ranking.model_dump(mode="json"),
                "comparison_provisional": comparison_json,
                "research_rounds_remaining": pair_audit["rounds_remaining"],
                "next_step": "成对比较尚未达到 COMPLETE；仅补查受影响的 incumbent/challenger，随后重新提交最终比较。",
                "usage": totals,
            }
            write("research_pending.json", pending)
            write("result.json", pending)
            return 2
    except ResearchPending:
        write("result.json", {**manifest, **agent.pending_research,
                             "supplemental_research": supplemental,
                             "previous_first_supplemental_research": previous_supplemental,
                             "provisional_ranking": final_ranking.model_dump(mode="json"), "usage": totals})
        return 2
    except Exception as exc:
        result = {
            **manifest,
            "status": "FAILED",
            "completed_supplemental_research": sorted(set(supplemental) | {previous_symbol}),
            "missing_supplemental_research": [],
            "errors": {"NEW_VS_PREVIOUS_COMPARISON": f"{type(exc).__name__}: {exc}"},
            "initial_ranking": initial_ranking,
            "initial_top5": top5_rows,
            "supplemental_research": supplemental,
            "previous_first_supplemental_research": previous_supplemental,
            "final_ranking": final_ranking.model_dump(mode="json"),
            "previous_winner_record": previous_context,
            "usage": totals,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        write("result.json", result)
        print("New-vs-previous comparison FAILED; see redacted result.json", flush=True)
        return 1

    result = {
        **manifest,
        "status": "COMPLETE",
        "evidence_assessments": agent.evidence_assessments,
        "completed_supplemental_research": sorted(set(supplemental) | {previous_symbol}),
        "missing_supplemental_research": [],
        "initial_ranking": initial_ranking,
        "initial_top5": top5_rows,
        "supplemental_research": supplemental,
        "previous_first_supplemental_research": previous_supplemental,
        "previous_winner_record": previous_context,
        "final_ranking": final_ranking.model_dump(mode="json"),
        "pair_evidence_audit": pair_audit,
        "selected_symbol": selected["symbol"],
        "selected_final_score": selected["preliminary_alpha_score"],
        "runner_up_symbol": runner_up["symbol"],
        "runner_up_final_score": runner_up["preliminary_alpha_score"],
        "final_alpha_gap": selected["preliminary_alpha_score"] - runner_up["preliminary_alpha_score"],
        "new_first_symbol": selected["symbol"],
        "previous_first_symbol": previous_symbol,
        "rebalance_comparison": comparison_json,
        "rebalance_decision": comparison_json["rebalance_decision"],
        "pair_comparison_complete": comparison_json["pair_comparison_complete"],
        "decision_basis_sufficient": comparison_json["decision_basis_sufficient"],
        "material_asymmetry_resolved": comparison_json["material_asymmetry_resolved"],
        **transition_fields(previous_symbol, selected["symbol"], comparison_json["rebalance_decision"]),
        "selection_rationale": {**initial_rationale,
                                **agent.selection_rationale["SOL_TOP5_FINAL_RANKING"],
                                **agent.selection_rationale["SOL_NEW_VS_PREVIOUS"]},
        "supplemental_findings": {symbol: agent.selection_rationale[symbol] for symbol in top5},
        "usage": totals,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "provider_protocol": "CHAT_COMPLETIONS via CCSwitchProvider",
        "reasoning_parameter_status": provider.reasoning_parameter_status,
        "reasoning_metadata_status": provider.reasoning_metadata_status,
        "placeOrder_calls": 0,
        "cancelOrder_calls": 0,
        "risk_approval": "NOT_RUN",
    }
    result["active_selection_research"] = (previous_supplemental if result["rebalance_decision"] == "KEEP_PREVIOUS"
                                           else supplemental[selected["symbol"].upper()])
    write("result.json", result)
    write("active_selection.json", {"status": "COMPLETE", "result_path": str(output / "result.json"),
                                   **transition_fields(previous_symbol, selected["symbol"], comparison_json["rebalance_decision"])})
    try:
        result["reports"] = generate_reports(output / "result.json")
        result["report_status"] = "COMPLETE"
        sink({"event_type": "CHINESE_REPORT_COMPLETED", "metadata": result["reports"]})
    except Exception as exc:
        result["report_status"] = "FAILED"
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        write("result.json", result)
        print("Research decision saved; required Chinese DOCX report failed. Retry report generation only.", flush=True)
        return 1
    write("result.json", result)
    print(
        f"TOP5 FINAL COMPLETE; selected={selected['symbol']} score={selected['preliminary_alpha_score']}; "
        f"runner_up={runner_up['symbol']} score={runner_up['preliminary_alpha_score']}; "
        f"previous_first={previous_symbol}; rebalance={comparison_json['rebalance_decision']}; no orders.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Top-five research startup FAILED ({type(exc).__name__}); no orders submitted", file=sys.stderr)
        raise SystemExit(1)
