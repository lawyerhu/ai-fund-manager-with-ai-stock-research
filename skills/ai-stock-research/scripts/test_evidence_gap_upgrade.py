"""Offline regression tests for precise gaps and pair-evidence gating."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deterministic_calculations import resolve_assessment_derivations  # noqa: E402
from evidence_discipline import (  # noqa: E402
    normalize_assessments,
    pair_evidence_audit,
)
from selection_state import resolve_active_selection, transition_fields  # noqa: E402
from top_five_deep_research import PreviousWinnerComparison, validate_evidence_packet  # noqa: E402


def assessment(symbol, *, gaps=None, evidence_gaps=None):
    return {
        "stage": "SOL_NEW_VS_PREVIOUS",
        "evidence_assessments": [{
            "symbol": symbol,
            "investment_confidence": 0.6,
            "evidence_completeness": "关键证据已核验，仍有明确限制",
            "confidence_basis": "比较依据来自离线测试证据",
            "research_requests": [],
            "unresolved_information_gaps": gaps or [],
            "evidence_gaps": evidence_gaps or [],
        }],
    }


def gap(symbol, field_name, status, *, criticality="CRITICAL", blocking=True):
    return {
        "field_name": field_name,
        "symbol": symbol,
        "gap_status": status,
        "criticality": criticality,
        "reason": f"{field_name} 的离线测试状态",
        "source_required": ["issuer or regulator"],
        "last_checked_at": "2026-09-15T00:00:00Z",
        "retrieval_attempts": 1,
        "evidence_refs": ["gap-audit-1"],
        "decision_impact": "可能实质改变新旧首选方向",
        "blocking_research": blocking,
        "question": f"核验 {symbol} 的 {field_name}",
        "search_record_refs": ["/gap_audit/searches/0"],
        "stopping_reason": "离线测试仍需定向补查",
    }


def packet_for(symbol="CRM", status="VERIFIED"):
    return {
        "evidence": [{"id": "ev-1", "symbol": symbol, "field_name": "operating_fact", "status": status,
                      "source_url": "https://example.test/source", "source_opened": True,
                      "retrieved_at": "2026-09-15T00:00:00Z", "facts": {"value": 1}}],
        "as_of_basis": {"market_snapshot_date": "2026-09-15"},
        "gap_audit": {"searches": [{"id": "gap-audit-1", "retrieved_at": "2026-09-15T00:00:00Z"}]},
    }


class EvidenceGapUpgradeTests(unittest.TestCase):
    def test_a_latest_challenger_does_not_switch_past_mpc_retrieval_failure(self):
        packet = packet_for("CRM")
        audit = pair_evidence_audit(packet, [assessment("MPC", gaps=[gap("MPC", "crack_spread", "RETRIEVAL_FAILED")])], "CRM", "MPC")
        self.assertEqual(audit["status"], "NEEDS_RESEARCH")
        self.assertEqual([row["symbol"] for row in audit["research_requests"]], ["MPC"])
        self.assertTrue(audit["material_asymmetries"][0]["could_change_direction"])

    def test_b_verified_mpc_can_reach_model_keep_and_preserve_mpc(self):
        packet = packet_for("MPC")
        audit = pair_evidence_audit(packet, [assessment("MPC", evidence_gaps=[gap("MPC", "crack_spread", "VERIFIED", blocking=False)])], "CRM", "MPC")
        self.assertFalse(audit["requires_targeted_research"])
        comparison = PreviousWinnerComparison.model_validate({
            "new_first_symbol": "CRM", "previous_first_symbol": "MPC", "rebalance_decision": "KEEP_PREVIOUS",
            "alpha_gap": None, "alpha_gap_status": "UNKNOWN", "why_new_beats_previous": ["新候选仍有催化剂"],
            "why_keep_previous": ["MPC 的 thesis 仍强"], "evidence_refs": ["ev-1"], "confidence": 0.6,
            "thesis_invalidation_conditions": ["MPC 变量转弱"], "pair_comparison_complete": True,
            "decision_basis_sufficient": True, "material_asymmetry_resolved": True,
        })
        self.assertEqual(transition_fields("MPC", comparison.new_first_symbol, comparison.rebalance_decision)["active_selection"], "MPC")

    def test_c_not_public_does_not_permanently_block_final_choice(self):
        audit = pair_evidence_audit(packet_for("CRM"), [assessment("MPC", gaps=[gap("MPC", "contract_terms", "NOT_PUBLIC")])], "CRM", "MPC")
        self.assertFalse(audit["requires_targeted_research"])
        self.assertEqual(audit["status"], "READY_FOR_MODEL_DECISION")

    def test_d_paid_data_is_recorded_without_repeated_research_request(self):
        audit = pair_evidence_audit(packet_for("CRM"), [assessment("MPC", gaps=[gap("MPC", "consensus_revision", "PAID_DATA_REQUIRED")])], "CRM", "MPC")
        self.assertFalse(audit["requires_targeted_research"])
        self.assertEqual(audit["research_requests"], [])

    def test_e_derivation_required_is_resolved_by_deterministic_calculation(self):
        packet = {
            "evidence": [{"id": "cfo", "symbol": "MPC", "status": "VERIFIED", "source_url": "https://example.test/10q",
                          "source_opened": True, "retrieved_at": "2026-09-15", "facts": {"cfo": 100}},
                         {"id": "capex", "symbol": "MPC", "status": "VERIFIED", "source_url": "https://example.test/10q",
                          "source_opened": True, "retrieved_at": "2026-09-15", "facts": {"capex": 30}}],
            "as_of_basis": {"financial_period": "2026-Q2"}, "gap_audit": {},
            "derivation_inputs": [{"id": "fcf-mpc", "symbol": "MPC", "field_name": "fcf", "calculation": "FCF",
                                    "inputs": {"cfo": 100, "capex": 30}, "as_of": "2026-Q2",
                                    "evidence_refs": ["cfo", "capex"]}],
        }
        validate_evidence_packet(packet)
        self.assertEqual(packet["deterministic_calculations"][0]["result"], 70.0)
        rows = normalize_assessments([assessment("MPC", gaps=[gap("MPC", "fcf", "DERIVATION_REQUIRED")]) ["evidence_assessments"][0]])
        resolved = resolve_assessment_derivations(rows, packet)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(rows[0]["unresolved_information_gaps"], [])

    def test_f_old_result_without_new_fields_still_resolves_active_selection(self):
        old = {"status": "COMPLETE", "selected_symbol": "CRM", "previous_first_symbol": "MPC",
               "rebalance_decision": "KEEP_PREVIOUS"}
        self.assertEqual(resolve_active_selection(old), "MPC")

    def test_g_material_asymmetry_keeps_active_selection_unadvanced(self):
        audit = pair_evidence_audit(packet_for("CRM"), [assessment("MPC", gaps=[gap("MPC", "crack_spread", "STALE")])], "CRM", "MPC")
        pending = {"status": audit["status"], "pair_evidence_audit": audit}
        self.assertEqual(pending["status"], "NEEDS_RESEARCH")
        with self.assertRaises(ValueError):
            resolve_active_selection(pending)

    def test_h_selection_state_executes_model_action_without_rejudging_score(self):
        payload = {
            "new_first_symbol": "CRM", "previous_first_symbol": "MPC", "rebalance_decision": "SWITCH_TO_NEW_FIRST",
            "alpha_gap": None, "alpha_gap_status": "UNKNOWN", "why_new_beats_previous": ["有相对优势"],
            "why_keep_previous": ["MPC 仍有支持因素"], "evidence_refs": ["ev-1"], "confidence": 0.4,
            "thesis_invalidation_conditions": ["催化剂失效"], "pair_comparison_complete": True,
            "decision_basis_sufficient": True, "material_asymmetry_resolved": True,
        }
        comparison = PreviousWinnerComparison.model_validate(payload)
        self.assertEqual(transition_fields("MPC", comparison.new_first_symbol, comparison.rebalance_decision)["active_selection"], "CRM")


if __name__ == "__main__":
    unittest.main()
