"""Offline tests for the research-only Top 5 and previous-first comparison helper."""

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from top_five_deep_research import (
    PairMarketSnapshotRequired,
    PreviousWinnerComparison,
    extract_previous_winner,
    require_pair_market_snapshot,
    validate_evidence_packet,
)


class TopFiveComparisonTests(unittest.TestCase):
    def test_extracts_previous_first_from_current_result_shape(self):
        symbol, context = extract_previous_winner({
            "status": "COMPLETE",
            "active_selection": "MPC",
            "ranking": {"ranking": [{"rank": 1, "symbol": "mpc"}, {"rank": 2, "symbol": "VLO"}]},
        })
        self.assertEqual(symbol, "MPC")
        self.assertEqual(context["winner_record"]["rank"], 1)

    def test_extracts_previous_first_from_legacy_nested_shape(self):
        symbol, context = extract_previous_winner({
            "status": "COMPLETE",
            "active_selection": "MPC",
            "result": {
                "deep_research": {
                    "ranking": {"ranking": [{"rank": 1, "symbol": "MPC"}]},
                },
            },
        })
        self.assertEqual(symbol, "MPC")
        self.assertEqual(context["winner_record"]["symbol"], "MPC")

    def test_previous_result_must_be_completed_and_stock_selected(self):
        for payload in (
            {"status": "FAILED", "selected_symbol": "MPC"},
            {"status": "COMPLETE", "selected_symbol": "CASH"},
            {"status": "COMPLETE", "ranking": {"ranking": []}},
        ):
            with self.assertRaises(ValueError):
                extract_previous_winner(payload)

    def test_comparison_schema_has_only_directional_research_fields(self):
        comparison = PreviousWinnerComparison.model_validate({
            "new_first_symbol": "VEEV",
            "previous_first_symbol": "MPC",
            "rebalance_decision": "SWITCH_TO_NEW_FIRST",
            "alpha_gap": None,
            "alpha_gap_status": "UNKNOWN",
            "why_new_beats_previous": ["更强的已核验盈利催化剂"],
            "why_keep_previous": ["旧首选的行业证据仍然有效"],
            "evidence_refs": ["new_first.evidence.1", "previous_first.evidence.2"],
            "confidence": 0.72,
            "confidence_reducers": ["固定期限一致预期仍未完全取得"],
            "thesis_invalidation_conditions": ["关键催化剂被公司公告否定"],
        })
        dumped = comparison.model_dump(mode="json")
        self.assertEqual(dumped["rebalance_decision"], "SWITCH_TO_NEW_FIRST")
        self.assertNotIn("target_weight", dumped)
        self.assertNotIn("shares", dumped)
        self.assertNotIn("account_position", dumped)
        for action in ("KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"):
            low_confidence = {**dumped, "rebalance_decision": action, "confidence": 0.25}
            self.assertEqual(PreviousWinnerComparison.model_validate(low_confidence).rebalance_decision, action)
        with self.assertRaises(ValueError):
            PreviousWinnerComparison.model_validate({**dumped, "rebalance_decision": "REVIEW_REQUIRED"})

    def test_evidence_packet_requires_provenance_and_search_audit(self):
        packet = {"evidence": [{"status": "VERIFIED"}]}
        with self.assertRaises(ValueError):
            validate_evidence_packet(packet)

    def test_evidence_packet_does_not_require_a_financial_checklist(self):
        validate_evidence_packet({
            "evidence": [{"symbol": "TEST", "status": "UNKNOWN"}],
            "as_of_basis": {"research_date": "2026-09-13"},
            "gap_audit": {},
            "company_specific_question": [{"question": "Company-specific evidence chosen by the research model"}],
        })

    def test_evidence_packet_allows_explicit_unknowns(self):
        packet = {
            "evidence": [{"status": "VERIFIED"}],
            "as_of_basis": {"market_snapshot_date": "2026-09-09"},
            "same_date_market_snapshot": [{"status": "UNAVAILABLE"}],
            "earnings_estimate_revision_snapshot": [{"status": "UNAVAILABLE"}],
            "cash_flow_and_official_financials": [{"status": "UNAVAILABLE"}],
            "catalysts_and_events": [{"status": "UNAVAILABLE"}],
            "peer_comparison": [{"status": "UNAVAILABLE"}],
            "gap_audit": {"still_unknown": ["peer metrics"]},
        }
        validate_evidence_packet(packet)

    def test_pair_snapshot_requires_current_packet_coverage_for_both_symbols(self):
        packet = {
            "as_of_basis": {"market_snapshot_date": "2026-09-15"},
            "project_same_basis_market_data": [
                {"symbol": "CRM", "status": "OBSERVED_SAME_SESSION", "source": "Yahoo", "last_bar_at": "2026-09-15T20:00:00Z", "facts": {"price": 259.43}},
                {"symbol": "MPC", "status": "OBSERVED_SAME_SESSION", "source": "Yahoo", "last_bar_at": "2026-09-15T20:00:00Z", "facts": {"price": 396.45}},
            ],
        }
        aligned = require_pair_market_snapshot(packet, "CRM", "MPC")
        self.assertEqual(aligned["section"], "project_same_basis_market_data")
        self.assertEqual(aligned["as_of_date"], "2026-09-15")
        self.assertEqual(set(aligned["symbols"]), {"CRM", "MPC"})

    def test_pair_snapshot_rejects_missing_or_misaligned_incumbent(self):
        missing = {
            "as_of_basis": {"market_snapshot_date": "2026-09-15"},
            "same_date_market_snapshot": [
                {"symbol": "CRM", "status": "VERIFIED", "observed_at": "2026-09-15T20:00:00Z", "facts": {"price": 259.43}},
            ],
        }
        with self.assertRaises(PairMarketSnapshotRequired):
            require_pair_market_snapshot(missing, "CRM", "MPC")
        misaligned = {
            "as_of_basis": {"market_snapshot_date": "2026-09-15"},
            "same_date_market_snapshot": [
                {"symbol": "CRM", "status": "VERIFIED", "observed_at": "2026-09-15T20:00:00Z", "facts": {"price": 259.43}},
                {"symbol": "MPC", "status": "VERIFIED", "observed_at": "2026-09-14T20:00:00Z", "facts": {"price": 396.45}},
            ],
        }
        with self.assertRaises(PairMarketSnapshotRequired):
            require_pair_market_snapshot(misaligned, "CRM", "MPC")


if __name__ == "__main__":
    unittest.main()
