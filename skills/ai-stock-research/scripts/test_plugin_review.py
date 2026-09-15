import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
args, rest = parser.parse_known_args()
sys.path.insert(0, str(args.project.resolve()))
from plugin_review import validate_snapshot, review_snapshot


def sample():
    return {"source": "IBKR_CONNECTOR", "fetched_at": datetime.now(timezone.utc).isoformat(),
            "summary": {"currency": "USD", "net_liquidation": 1000, "total_cash_value": 400},
            "positions": [{"contract_id": i, "contract_description": f"Label{i}", "position": 2,
                           "currency": "USD", "asset_class": "STK", "market_price": 100, "market_value": 200}
                          for i in (1, 2, 3)]}


class PluginReviewTests(unittest.TestCase):
    def test_joint_selection_review_and_empty_account(self):
        data = sample()
        data["positions"] = []
        data["selection"] = {"symbol": "TEST", "contract_id": 99, "currency": "USD", "asset_class": "STK",
            "research_result_path": "isolated/result.json", "researched_at": data["fetched_at"],
            "research": {"conclusion": "test evidence"}}
        response = {"portfolio_findings": ["Compare with cash"], "holdings": [], "candidate_advice": {
            "contract_id": 99, "action": "BUY", "reasons": ["test"], "comparison_with_holdings": ["Cash only"],
            "funding_basis": "Cash; sizing not approved", "prerequisites": ["Fresh quote and risk approval"],
            "evidence_refs": ["selection", "summary"]}}
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps(response))
        provider = SimpleNamespace(create_response=create)
        runtime = SimpleNamespace(sol_model="gpt-6-astra")
        result = review_snapshot(provider, runtime, validate_snapshot(data))
        self.assertEqual(result.candidate_advice.contract_id, 99)
        self.assertIn('"research_result_path": "isolated/result.json"', calls[0]["input"])
        self.assertEqual(calls[0]["reasoning"], {"effort": "medium"})
        response["candidate_advice"]["contract_id"] = 100
        with self.assertRaises(ValueError):
            review_snapshot(provider, runtime, validate_snapshot(data))
        response["candidate_advice"]["contract_id"] = 99
        data["selection"]["currency"] = "HKD"
        with self.assertRaises(ValueError):
            review_snapshot(provider, runtime, validate_snapshot(data))
        data["selection"]["currency"] = "USD"
        data["positions"] = [{**sample()["positions"][0], "contract_id": 99}]
        response["holdings"] = [{"contract_id": 99, "action": "REVIEW_REQUIRED", "reasons": ["Missing"],
            "weight_basis": "UNKNOWN", "incremental_reason": "UNKNOWN", "evidence_refs": ["positions"]}]
        with self.assertRaises(ValueError):
            review_snapshot(provider, runtime, validate_snapshot(data))
        response["candidate_advice"]["action"] = "KEEP_EXISTING"
        self.assertEqual(review_snapshot(provider, runtime, validate_snapshot(data)).candidate_advice.action, "KEEP_EXISTING")

    def test_three_positions_no_fake_entry_history(self):
        context = validate_snapshot(sample()).context()
        self.assertEqual(len(context["positions"]), 3)
        self.assertEqual(context["positions"][0]["weight"], .2)
        self.assertEqual(context["entry_dates"], "UNKNOWN")
        self.assertEqual(context["account_type"], "UNKNOWN")
        self.assertEqual(context["quote_timestamp"], "UNKNOWN")

    def test_currency_mismatch_weight_unknown(self):
        data = sample()
        data["positions"][0]["currency"] = "HKD"
        self.assertIsNone(validate_snapshot(data).context()["positions"][0]["weight"])

    def test_stale_and_duplicate_rejected(self):
        data = sample()
        data["fetched_at"] = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
        with self.assertRaises(ValueError):
            validate_snapshot(data)
        data = sample()
        data["positions"].append(data["positions"][0])
        with self.assertRaises(ValueError):
            validate_snapshot(data)

    def test_missing_fields_not_zero_cash(self):
        data = sample()
        del data["summary"]["total_cash_value"]
        with self.assertRaises(ValueError):
            validate_snapshot(data)

    def test_review_uses_astra_and_requires_all_positions(self):
        snapshot = validate_snapshot(sample())
        response = {"portfolio_findings": ["Research data missing"], "holdings": [
            {"contract_id": i, "action": "REVIEW_REQUIRED", "reasons": ["Missing research"],
             "weight_basis": "UNKNOWN", "incremental_reason": "UNKNOWN", "evidence_refs": ["positions"]}
            for i in (1, 2, 3)]}
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps(response))
        provider = SimpleNamespace(create_response=create)
        runtime = SimpleNamespace(sol_model="gpt-6-astra")
        result = review_snapshot(provider, runtime, snapshot)
        self.assertEqual(len(result.holdings), 3)
        self.assertEqual(calls[0]["model"], "gpt-6-astra")
        self.assertEqual(calls[0]["reasoning"], {"effort": "medium"})
        response["holdings"].pop()
        with self.assertRaises(ValueError):
            review_snapshot(provider, runtime, snapshot)

    def test_missing_research_cannot_fabricate_reduction_target(self):
        data = sample()
        data["positions"] = data["positions"][:1]
        response = {"portfolio_findings": ["test"], "holdings": [{"contract_id": 1, "action": "REDUCE",
            "target_weight": .1, "reasons": ["Missing data"], "weight_basis": "Unsupported",
            "incremental_reason": "Unsupported", "evidence_refs": ["positions"]}]}
        provider = SimpleNamespace(create_response=lambda **k: SimpleNamespace(output_text=json.dumps(response)))
        with self.assertRaises(ValueError):
            review_snapshot(provider, SimpleNamespace(sol_model="gpt-6-astra"), validate_snapshot(data))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *rest])
