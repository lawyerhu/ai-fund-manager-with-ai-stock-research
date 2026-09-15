"""Offline behavior tests: run with project Python and --project <root>."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(args.project.resolve()))

from research import ASTRA_MODEL, NoTradingImports, execute_research, require_models, research_only_context, research_runtime, source_context
from src.llm_agent import CCSwitchProvider, LLMRuntimeConfig, LunaSolPipeline
from src.data_provider import MockDataProvider
from src.models import ManagedPosition, PortfolioState
from src.storage import SQLiteStore
from tests.test_sol_deep_research import ranking_payload, deep_payload, bear_payload, final_payload


class ResearchSkillTests(unittest.TestCase):
    def runtime(self):
        return research_runtime(LLMRuntimeConfig(base_url="http://invalid.test", api_key="offline-test",
                                                deep_research_enabled=True, api_protocol="RESPONSES"))

    def provider(self, payloads, calls):
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id=f"test-{len(calls)}", output=[], output_text=json.dumps(payloads.pop(0)), usage=None)
        return CCSwitchProvider(runtime=self.runtime(), client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep_fn=lambda _: None)

    def test_all_six_cio_stages_really_request_astra_medium(self):
        calls = []
        provider = self.provider([ranking_payload(), deep_payload("NVDA"), deep_payload("META"),
                                  deep_payload("AVGO"), bear_payload(), final_payload()], calls)
        pipeline = LunaSolPipeline(MockDataProvider(), provider, self.runtime())
        result = execute_research("candidates", pipeline, PortfolioState(equity=1000, peak_equity=1000, cash=1000),
                                  None, ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"], {}, {})
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(c["model"] == ASTRA_MODEL and c["reasoning"] == {"effort": "medium"} for c in calls))
        self.assertEqual(result["deep_research"]["final_decision"]["selected_symbol"], "NVDA")
        self.assertGreater(len(result["tool_calls"]), 0)

    def test_verified_external_evidence_is_passed_to_astra_research_stages(self):
        calls = []
        provider = self.provider([ranking_payload(), deep_payload("NVDA"), deep_payload("META"),
                                  deep_payload("AVGO"), bear_payload(), final_payload()], calls)
        pipeline = LunaSolPipeline(MockDataProvider(), provider, self.runtime())
        packet = {"evidence": [{"symbol": "NVDA", "status": "VERIFIED", "source_url": "https://example.test/filing",
                                "facts": {"reported_period": "2026-Q2"}}]}
        execute_research("candidates", pipeline, PortfolioState(equity=1000, peak_equity=1000, cash=1000),
                         None, ["NVDA", "META", "AVGO", "MSFT", "MPC", "VLO"],
                         {"verified_external_evidence": packet}, {})
        prompts = [call.get("input", "") for call in calls]
        self.assertTrue(prompts)
        self.assertTrue(all("VERIFIED_EXTERNAL_EVIDENCE_JSON" in prompt for prompt in prompts))
        self.assertTrue(all("https://example.test/filing" in prompt for prompt in prompts))

    def test_position_review_and_repair_use_astra_medium(self):
        calls = []
        review = dict(action="HOLD", thesis_status="INTACT", current_holding_score=80, confidence=.7,
                      new_horizon_days=20, reason=["Unchanged evidence"], target_weight=.5,
                      sizing_audit=dict(weight_basis="Unchanged", incremental_reason="No new facts",
                                        risk_budget_basis="UNKNOWN", downside_scenario="Qualitative",
                                        evidence_refs=["holding_fundamentals"]))
        pipeline = LunaSolPipeline(MockDataProvider(), self.provider([{}, review], calls), self.runtime())
        position = ManagedPosition(position_id="p", run_id="r", decision_id="d", symbol="NVDA",
            entry_time="2026-09-01T00:00:00+00:00", entry_price=100, current_price=100, current_weight=.5,
            original_thesis_horizon_days=20, current_thesis_horizon_days=20, original_thesis=["test"])
        result = execute_research("review", pipeline, PortfolioState(equity=1000, peak_equity=1000, cash=500,
            current_symbol="NVDA", current_weight=.5), position, [], {}, {})
        self.assertEqual(result["review"]["action"], "HOLD")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c["model"] == ASTRA_MODEL and c["reasoning"]["effort"] == "medium" for c in calls))

    def test_luna_is_unchanged(self):
        original = replace(self.runtime(), sol_model="gpt-5.6-sol", sol_reasoning_effort="high")
        selected = research_runtime(original)
        self.assertEqual(original.sol_model, "gpt-5.6-sol")
        self.assertEqual(selected.luna_model, original.luna_model)
        self.assertEqual(selected.luna_screen_batch_size, original.luna_screen_batch_size)
        calls = []
        self.provider([{"ok": True}], calls).create_response(model=selected.luna_model, input="test")
        self.assertEqual(calls[0]["reasoning"]["effort"], "max")

    def test_discovery_does_not_fallback_to_sol(self):
        provider = SimpleNamespace(client=SimpleNamespace(models=SimpleNamespace(list=lambda: {"data": [{"id": "gpt-5.6-sol"}]})))
        with self.assertRaisesRegex(ValueError, ASTRA_MODEL):
            require_models(provider, self.runtime(), False)

    def test_exact_discovery(self):
        provider = SimpleNamespace(client=SimpleNamespace(models=SimpleNamespace(list=lambda: {"data": [{"id": ASTRA_MODEL}]})))
        require_models(provider, self.runtime(), False)
        with self.assertRaises(ValueError):
            require_models(provider, self.runtime(), True)

    def test_broker_worker_and_risk_imports_are_blocked(self):
        guard = NoTradingImports()
        for name in ("src.main", "src.runner", "src.service", "src.execution.ibkr", "src.scheduler", "src.risk_engine"):
            with self.assertRaises(ImportError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec("src.llm_agent"))

    def test_source_reads_do_not_write_commands_or_orders(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite3"
            with SQLiteStore(path) as store:
                store.save_portfolio(PortfolioState(equity=1000, peak_equity=1000, cash=1000), [])
            with SQLiteStore(path, read_only=True) as store:
                portfolio, position, candidates, source, context = source_context(store)
                self.assertEqual(portfolio.equity, 1000)
                self.assertEqual(candidates, [])
                self.assertEqual(store.pending_commands(), [])
                self.assertEqual(store.recent("order_records"), [])
                with self.assertRaises(Exception):
                    store.set_runtime("execution_mode", "PAPER")

    def test_research_only_context_never_reads_portfolio_snapshot(self):
        class Connection:
            def execute(self, query, params=()):
                self.query = query
                return self

            def fetchone(self):
                return {"decision_id": "d1", "recorded_at": "2026-09-10T00:00:00Z",
                        "candidate_symbols_json": '["MPC", "VLO"]'}

        class Store:
            connection = Connection()

            def latest_portfolio(self):
                raise AssertionError("research-only context must not read portfolio state")

        portfolio, position, candidates, source, context = research_only_context(Store(), "d1")
        self.assertIsNone(position)
        self.assertEqual(candidates, ["MPC", "VLO"])
        self.assertEqual(portfolio.as_of, "RESEARCH_ONLY")
        self.assertEqual(context, {})

    def test_no_saved_portfolio_fails_closed(self):
        with SQLiteStore(":memory:") as store:
            with self.assertRaisesRegex(ValueError, "No persisted portfolio"):
                source_context(store)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
