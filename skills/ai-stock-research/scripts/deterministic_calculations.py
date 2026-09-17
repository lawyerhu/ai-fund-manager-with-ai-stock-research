"""Small, auditable calculations for public raw financial inputs.

The research model may identify that a value is derivable, but this module is
the only place in the skill that turns supplied numeric inputs into a derived
value.  It never fetches data and it never supplies missing inputs.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


DERIVATION_STATUS = "DERIVATION_REQUIRED"


def _field_key(value: Any) -> str:
    name = str(value or "").upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    return {
        "FREE_CASH_FLOW": "FCF",
        "FREE_CASH_FLOW_YIELD": "FCF_YIELD",
        "SHARES_CHANGE": "SHARE_COUNT_CHANGE",
        "BUYBACK_ADJUSTED_SHARES": "BUYBACK_ADJUSTED_SHARE_COUNT",
        "P_E": "PE",
        "EV_EBITDA_MULTIPLE": "EV_EBITDA",
        "ORGANIC_GROWTH_RATE": "ORGANIC_GROWTH",
    }.get(name, name)


def _number(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Deterministic calculation input {name} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Deterministic calculation input {name} must be numeric") from exc
    if not number.is_finite():
        raise ValueError(f"Deterministic calculation input {name} must be finite")
    return number


def _percent(value: Decimal) -> float:
    return float(value * Decimal("100"))


def calculate(calculation: str, inputs: dict[str, Any]) -> tuple[float, str]:
    """Calculate one supported value and return ``(result, formula)``.

    Inputs are deliberately explicit.  In particular, ``capex`` is the
    positive capital-spending amount for FCF; the caller must normalize the
    sign from the source statement before calling this function.
    """
    if not isinstance(inputs, dict):
        raise ValueError("Deterministic calculation inputs must be an object")
    name = _field_key(calculation)
    if name in {"FCF", "FREE_CASH_FLOW"}:
        result = _number(inputs.get("cfo"), "cfo") - _number(inputs.get("capex"), "capex")
        return float(result), "FCF = CFO - Capex"
    if name == "NET_DEBT":
        result = _number(inputs.get("debt"), "debt") - _number(inputs.get("cash"), "cash")
        return float(result), "Net Debt = Debt - Cash"
    if name in {"SHARE_COUNT_CHANGE", "SHARES_CHANGE"}:
        result = _number(inputs.get("current_shares"), "current_shares") - _number(inputs.get("prior_shares"), "prior_shares")
        return float(result), "Share Count Change = Current Shares - Prior Shares"
    if name in {"BUYBACK_ADJUSTED_SHARE_COUNT", "BUYBACK_ADJUSTED_SHARES"}:
        result = (_number(inputs.get("prior_shares"), "prior_shares")
                  + _number(inputs.get("issued_shares", 0), "issued_shares")
                  - _number(inputs.get("repurchased_shares", 0), "repurchased_shares"))
        return float(result), "Buyback-adjusted Shares = Prior Shares + Issued Shares - Repurchased Shares"
    if name in {"FCF_YIELD", "FREE_CASH_FLOW_YIELD"}:
        market_cap = _number(inputs.get("market_cap"), "market_cap")
        if market_cap == 0:
            raise ValueError("FCF yield requires non-zero market_cap")
        return _percent(_number(inputs.get("fcf"), "fcf") / market_cap), "FCF Yield = FCF / Market Cap"
    if name in {"PE", "P_E"}:
        eps = _number(inputs.get("eps"), "eps")
        if eps == 0:
            raise ValueError("PE requires non-zero eps")
        return float(_number(inputs.get("price"), "price") / eps), "PE = Price / EPS"
    if name in {"EV_EBITDA", "EV_EBITDA_MULTIPLE"}:
        ebitda = _number(inputs.get("ebitda"), "ebitda")
        if ebitda == 0:
            raise ValueError("EV/EBITDA requires non-zero ebitda")
        enterprise_value = _number(inputs.get("market_cap"), "market_cap") + _number(inputs.get("net_debt"), "net_debt")
        return float(enterprise_value / ebitda), "EV/EBITDA = (Market Cap + Net Debt) / EBITDA"
    if name in {"ORGANIC_GROWTH", "ORGANIC_GROWTH_RATE"}:
        prior = _number(inputs.get("prior_organic_revenue"), "prior_organic_revenue")
        if prior == 0:
            raise ValueError("Organic growth requires non-zero prior_organic_revenue")
        current = _number(inputs.get("current_organic_revenue"), "current_organic_revenue")
        return _percent(current / prior - Decimal("1")), "Organic Growth = Current Organic Revenue / Prior Organic Revenue - 1"
    raise ValueError(f"Unsupported deterministic calculation: {calculation}")


def derive(record: dict[str, Any]) -> dict[str, Any]:
    """Derive one packet record from explicit raw inputs."""
    if not isinstance(record, dict):
        raise ValueError("Derivation record must be an object")
    calculation = record.get("calculation") or record.get("field_name")
    inputs = record.get("inputs")
    if not calculation or not isinstance(inputs, dict):
        raise ValueError("Derivation record requires calculation and inputs")
    value, formula = calculate(str(calculation), inputs)
    as_of = record.get("as_of") or record.get("as_of_basis")
    if not as_of:
        raise ValueError("Deterministic calculation requires as_of")
    refs = record.get("evidence_refs") or record.get("source_refs")
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref for ref in refs):
        raise ValueError("Deterministic calculation requires evidence_refs")
    symbol = str(record.get("symbol") or "").upper()
    if not symbol:
        raise ValueError("Deterministic calculation requires symbol")
    field_name = str(record.get("field_name") or calculation).lower()
    return {
        "id": str(record.get("id") or f"derived:{symbol}:{field_name}"),
        "symbol": symbol,
        "field_name": field_name,
        "status": "DERIVED",
        "gap_status": "VERIFIED",
        "calculation": str(calculation).upper(),
        "inputs": dict(inputs),
        "formula": formula,
        "result": value,
        "as_of": as_of,
        "evidence_refs": list(refs),
        "estimate_type": "DETERMINISTIC",
    }


def resolve_packet_derivations(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve explicit packet derivations in place and return their results.

    ``derivation_inputs`` is the preferred input key.  A pre-existing
    ``deterministic_calculations`` list is accepted for continuation runs and
    is never recomputed when it already contains a result.
    """
    if not isinstance(packet, dict):
        return []
    requests = packet.get("derivation_inputs")
    if requests is None:
        requests = packet.get("deterministic_calculations", [])
    if not isinstance(requests, list):
        raise ValueError("derivation_inputs must be a list")
    existing = packet.get("deterministic_calculations")
    results = [row for row in existing if isinstance(row, dict) and row.get("result") is not None] if isinstance(existing, list) else []
    existing_ids = {str(row.get("id")) for row in results}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Each derivation input must be an object")
        if request.get("result") is not None and request.get("status") in {"DERIVED", "VERIFIED"}:
            value = dict(request)
            value.setdefault("estimate_type", "DETERMINISTIC")
            results.append(value)
            existing_ids.add(str(value.get("id")))
            continue
        value = derive(request)
        if value["id"] not in existing_ids:
            results.append(value)
            existing_ids.add(value["id"])
    if results:
        packet["deterministic_calculations"] = results
    return results


def matching_derivation(packet: dict[str, Any], gap: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    """Find a resolved calculation for an assessment gap."""
    if not isinstance(packet, dict):
        return None
    field = _field_key(gap.get("field_name"))
    for row in packet.get("deterministic_calculations", []) if isinstance(packet.get("deterministic_calculations"), list) else []:
        if not isinstance(row, dict) or row.get("result") is None:
            continue
        if str(row.get("symbol") or "").upper() != str(symbol).upper():
            continue
        row_field = _field_key(row.get("field_name") or row.get("calculation"))
        calculation = _field_key(row.get("calculation"))
        if field and field not in {row_field, calculation}:
            continue
        return row
    return None


def resolve_assessment_derivations(assessments: list[dict[str, Any]], packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove resolved DERIVATION_REQUIRED gaps from saved assessments."""
    resolved: list[dict[str, Any]] = []
    for assessment in assessments:
        symbol = assessment.get("symbol", "")
        item_results = []
        for key in ("unresolved_information_gaps", "evidence_gaps", "important_evidence_gaps"):
            kept = []
            for gap in assessment.get(key, []) or []:
                if isinstance(gap, dict) and gap.get("gap_status") == DERIVATION_STATUS:
                    calculation = matching_derivation(packet, gap, symbol)
                    if calculation is not None:
                        item_results.append({"gap": dict(gap), "calculation": calculation})
                        continue
                kept.append(gap)
            if key in assessment:
                assessment[key] = kept
        if item_results:
            assessment["resolved_derivations"] = item_results
            resolved.extend(item_results)
    return resolved
