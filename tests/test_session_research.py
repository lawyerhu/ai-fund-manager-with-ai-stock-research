"""Offline tests for the session handoff; no project runtime imports or network."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("session_research", Path(__file__).parents[1] / "src" / "session_research.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SessionResearchTests(unittest.TestCase):
    def test_keep_uses_incumbent_not_winner(self):
        self.assertEqual(m.resolve_selection({"status": "COMPLETE", "new_first_symbol": "CRM", "previous_first_symbol": "MPC", "rebalance_decision": "KEEP_PREVIOUS"}), "MPC")

    def test_switch_uses_new_winner(self):
        self.assertEqual(m.resolve_selection({"status": "COMPLETE", "new_first_symbol": "CRM", "previous_first_symbol": "MPC", "rebalance_decision": "SWITCH_TO_NEW_FIRST"}), "CRM")

    def test_ranking_cannot_initialize_selection(self):
        with self.assertRaises(ValueError):
            m.resolve_selection({"status": "COMPLETE", "selected_symbol": "CRM"})

    def test_conflicting_alias_rejected(self):
        with self.assertRaises(ValueError):
            m.resolve_selection({"status": "COMPLETE", "new_first_symbol": "CRM", "previous_first_symbol": "MPC", "rebalance_decision": "KEEP_PREVIOUS", "active_selection": "CRM"})

    def test_failed_state_rejected(self):
        with self.assertRaises(ValueError):
            m.resolve_selection({"status": "FAILED", "active_selection": "CRM"})

    def test_cash_rejected(self):
        with self.assertRaises(ValueError):
            m.symbol("CASH")

    def test_duplicate_coverage_rejected(self):
        with self.assertRaises(ValueError):
            m.require_exact([{"symbol": "CRM"}, {"symbol": "CRM"}], ["CRM", "MPC"])

    def test_compact_removes_sensitive_fields_and_series(self):
        self.assertEqual(m.compact({"prices": [1], "api_key": "secret", "facts": {"revenue": 2, "source_url": "https://example.com"}}), {"facts": {"revenue": 2, "source_url": "https://example.com"}})

    def test_session_provenance_required(self):
        with self.assertRaises(ValueError):
            m.validate_decision({"initial_top_five": [], "incoming_active_selection": "MPC"}, {"research_provider": "API"}, {})

    def test_same_stock_switch_rejected(self):
        with self.assertRaises(ValueError):
            m.resolve_selection({"status": "COMPLETE", "new_first_symbol": "MPC", "previous_first_symbol": "MPC", "rebalance_decision": "SWITCH_TO_NEW_FIRST"})


if __name__ == "__main__":
    unittest.main()
