"""Offline integration checks using the real project provider and structured stage."""
import argparse
from copy import deepcopy
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

from evidence_discipline import ResearchPending, install_evidence_discipline
from equal_depth_candidates import read_event_state
from research import execute_research, research_runtime
from src.data_provider import MockDataProvider
from src.llm_agent import CCSwitchProvider, CrossSectionalRanking, DeepDiveResearch, LLMRuntimeConfig, LunaSolPipeline
from src.models import PortfolioState
from tests.test_sol_deep_research import ranking_payload, deep_payload, bear_payload, final_payload
from top_five_deep_research import PreviousWinnerComparison


def assessment(symbol, confidence=0.3, **changes):
    return {
        "symbol": symbol, "investment_confidence": confidence,
        "evidence_completeness": "Partial: a material future outcome remains unknown after source searches",
        "confidence_basis": "Investment judgment remains uncertain; completeness is not a score deduction",
        "research_requests": [], "unresolved_information_gaps": [], **changes,
    }


def audited(payload, symbols=None):
    result = deepcopy(payload)
    if "ranking" in payload:
        confidences = {row["symbol"]: row["confidence"] for row in payload["ranking"]}
    else:
        confidences = {}
    symbols = symbols or [payload["symbol"]]
    result["evidence_assessments"] = [assessment(s, confidences.get(s, payload.get("confidence", 0.3))) for s in symbols]
    return result


class EvidenceDisciplineTests(unittest.TestCase):
    def pipeline(self, payloads, *, stages=None):
        self.calls, self.files = [], {}
        runtime = research_runtime(LLMRuntimeConfig(
            base_url="http://invalid.test", api_key="offline-test", deep_research_enabled=True, api_protocol="RESPONSES"))

        def create(**request):
            self.calls.append(request)
            return SimpleNamespace(id=f"test-{len(self.calls)}", output=[],
                                   output_text=json.dumps(payloads.pop(0)), usage=None)

        provider = CCSwitchProvider(runtime=runtime,
            client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep_fn=lambda _: None)
        pipeline = LunaSolPipeline(MockDataProvider(), provider, runtime)
        install_evidence_discipline(pipeline.sol, lambda name, value: self.files.__setitem__(name, deepcopy(value)), stages=stages)
        return pipeline

    def test_all_existing_stages_and_ranking_scores_are_preserved(self):
        initial = ranking_payload()
        symbols = [row["symbol"] for row in initial["ranking"]] + ["EXTRA"]
        top3 = symbols[:3]
        payloads = [audited(initial, symbols), *[audited(deep_payload(s)) for s in top3],
                    audited(bear_payload(), top3), audited(final_payload(), top3)]
        pipeline = self.pipeline(payloads)
        result = execute_research("candidates", pipeline,
            PortfolioState(equity=1000, peak_equity=1000, cash=1000), None, symbols, {}, {})
        self.assertEqual(len(self.calls), 6)
        self.assertTrue(all(c["model"] == "gpt-6-astra" and c["reasoning"] == {"effort": "medium"} for c in self.calls))
        rows = result["deep_research"]["ranking"]["ranking"]
        self.assertEqual([(r["symbol"], r["preliminary_alpha_score"], r["confidence"]) for r in rows],
                         [(r["symbol"], r["preliminary_alpha_score"], r["confidence"]) for r in initial["ranking"]])
        self.assertEqual(result["deep_research"]["final_decision"]["selected_symbol"], final_payload()["selected_symbol"])
        self.assertEqual(len(self.files["evidence_assessments.json"]), 6)
        self.assertNotIn("research_pending.json", self.files)

    def test_material_gap_outside_top5_stops_before_elimination(self):
        initial = ranking_payload()
        symbols = [row["symbol"] for row in initial["ranking"]] + ["EXTRA"]
        payload = audited(initial, symbols)
        payload["evidence_assessments"][-1]["research_requests"] = [{
            "question": "Does the issuer have the approval needed for its catalyst?",
            "decision_impact": "Could move EXTRA above the initial winner",
            "query": "EXTRA issuer approval original announcement",
            "sources": ["issuer and relevant regulator"], "research_depth": "Read originals and reconcile conflicting dates",
        }]
        pipeline = self.pipeline([payload])
        with self.assertRaises(ResearchPending):
            execute_research("candidates", pipeline,
                PortfolioState(equity=1000, peak_equity=1000, cash=1000), None, symbols, {}, {})
        self.assertEqual(len(self.calls), 1)
        pending = self.files["research_pending.json"]
        self.assertEqual(pending["status"], "NEEDS_RESEARCH")
        self.assertEqual(pending["research_requests"][0]["symbol"], "EXTRA")
        self.assertEqual(pending["provisional_result"]["ranking"][0]["preliminary_alpha_score"], initial["ranking"][0]["preliminary_alpha_score"])
        self.assertFalse(hasattr(pipeline.sol, "last_deep_research") and pipeline.sol.last_deep_research)

    def test_low_confidence_and_partial_evidence_allow_either_model_action(self):
        for action in ("KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"):
            payload = dict(new_first_symbol="NEW", previous_first_symbol="OLD", rebalance_decision=action,
                alpha_gap=None, alpha_gap_status="UNKNOWN", why_new_beats_previous=["New catalyst"],
                why_keep_previous=["Incumbent thesis remains plausible"], evidence_refs=["packet"],
                confidence=0.2, confidence_reducers=["Genuine outcome uncertainty"], thesis_invalidation_conditions=["Catalyst fails"])
            pipeline = self.pipeline([audited(payload, ["NEW", "OLD"])])
            result = pipeline.sol._structured_deep_stage(PreviousWinnerComparison, "pair", "PAIR", "Compare pair", dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0))
            self.assertEqual(result.rebalance_decision, action)
            self.assertEqual(result.confidence, 0.2)
            self.assertNotIn("research_pending.json", self.files)

    def test_unresolved_gap_must_reference_an_actual_search_record(self):
        gap = dict(question="Approval terms", search_record_refs=["/gap_audit/searches/0"],
                   stopping_reason="Original and alternative sources do not publish terms", decision_impact="Outcome remains uncertain")
        for have_record in (False, True):
            payload = audited(deep_payload("NVDA"))
            payload["evidence_assessments"][0]["unresolved_information_gaps"] = [gap]
            pipeline = self.pipeline([payload])
            if have_record:
                pipeline.sol.external_verification = {"gap_audit": {"searches": [{
                    "query": "issuer approval terms", "source_url": "https://example.test/original",
                    "retrieved_at": "2026-09-13", "outcome": "Terms not publicly disclosed"}]}}
                result = pipeline.sol._structured_deep_stage(DeepDiveResearch, "deep", "DEEP", "Research", dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0), symbol="NVDA")
                self.assertEqual(result.symbol, "NVDA")
            else:
                with self.assertRaisesRegex(ValueError, "unavailable search record"):
                    pipeline.sol._structured_deep_stage(DeepDiveResearch, "deep", "DEEP", "Research", dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0), symbol="NVDA")

    def test_repair_keeps_source_context_and_existing_retry_path(self):
        pipeline = self.pipeline([{}, audited(deep_payload("NVDA"))])
        pipeline.sol._structured_deep_stage(DeepDiveResearch, "deep", "DEEP", "SOURCE_RECORD_42", dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0), symbol="NVDA")
        self.assertEqual(len(self.calls), 2)
        self.assertIn("SOURCE_RECORD_42", self.calls[1]["input"])
        self.assertIn("EVIDENCE DISCIPLINE", self.calls[1]["input"])

    def test_incomplete_audit_cannot_skip_an_unranked_candidate(self):
        initial = ranking_payload()
        symbols = [row["symbol"] for row in initial["ranking"]]
        pipeline = self.pipeline([audited(initial, symbols)])
        pipeline.sol._candidate_symbols = symbols + ["EXTRA"]
        with self.assertRaisesRegex(ValueError, "every evaluated stock"):
            pipeline.sol._structured_deep_stage(CrossSectionalRanking, "rank", "RANK", "Rank", dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0))

    def test_pending_completed_event_is_not_reused_as_finished_research(self):
        pending = audited(deep_payload("NVDA"))
        pending["evidence_assessments"][0]["research_requests"] = [{"question": "Pending"}]
        complete = audited(deep_payload("META"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            events = [{"event_type": "EQUAL_DEPTH_DEEP_DIVE_COMPLETED", "symbol": p["symbol"],
                       "metadata": {"result": p}} for p in (pending, complete, audited(deep_payload("AVGO")))]
            events.append({"event_type": "SKILL_EVIDENCE_AUDIT_PASSED", "symbol": "META",
                           "metadata": {"stage": "EQUAL_DEPTH_DEEP_DIVE"}})
            path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            _, resumed = read_event_state(path)
        self.assertNotIn("NVDA", resumed)
        self.assertNotIn("AVGO", resumed)
        self.assertEqual(DeepDiveResearch.model_validate(resumed["META"]).symbol, "META")

    def test_full_mode_preserves_pending_status_through_project_error_wrapper(self):
        pipeline = SimpleNamespace(sol=SimpleNamespace(pending_research={"status": "NEEDS_RESEARCH"}))
        def wrapped(*args):
            raise RuntimeError("Project wrapper around a pending evidence request")
        pipeline.decide = wrapped
        with self.assertRaises(ResearchPending):
            execute_research("full", pipeline, None, None, [], {}, {})

    def test_stage_d_scope_leaves_saved_stage_schemas_unchanged(self):
        initial = ranking_payload()
        symbols = [row["symbol"] for row in initial["ranking"]]
        top3 = symbols[:3]
        payloads = [initial, *[deep_payload(s) for s in top3], bear_payload(), audited(final_payload(), top3)]
        pipeline = self.pipeline(payloads, stages={"SOL_STAGE_D_IC_DECISION"})
        execute_research("candidates", pipeline,
            PortfolioState(equity=1000, peak_equity=1000, cash=1000), None, symbols, {}, {})
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(len(self.files["evidence_assessments.json"]), 1)
        for call in self.calls[:5]:
            self.assertNotIn("evidence_assessments", call["text"]["format"]["schema"]["properties"])

    def test_pending_result_saves_prior_stages_for_targeted_continuation(self):
        complete = audited(deep_payload("NVDA"))
        pending = audited(deep_payload("META"))
        pending["evidence_assessments"][0]["research_requests"] = [{
            "question": "Material fact", "decision_impact": "Could change rank", "query": "issuer fact",
            "sources": ["issuer"], "research_depth": "Read original disclosure"}]
        pipeline = self.pipeline([complete, pending])
        totals = dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0)
        pipeline.sol._structured_deep_stage(DeepDiveResearch, "deep", "DEEP", "NVDA evidence", totals, symbol="NVDA")
        with self.assertRaises(ResearchPending):
            pipeline.sol._structured_deep_stage(DeepDiveResearch, "deep", "DEEP", "META evidence", totals, symbol="META")
        saved = self.files["research_pending.json"]
        self.assertEqual(saved["completed_stages"][0]["result"]["symbol"], "NVDA")
        self.assertEqual(saved["stage_context"]["prompt"], "META evidence")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
