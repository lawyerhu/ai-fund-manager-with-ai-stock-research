"""Offline tests for the session handoff; no project runtime imports or network.

Run with the project Python and --project <root>:
    python -X utf8 scripts/test_session_handoff.py --project '<项目>'
"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).parent))

from research_model import DEFAULT_SESSION_MODEL, RESEARCH_MODEL, api_effort, api_model, session_model  # noqa: E402
from session_handoff import NoProjectRuntimeImports, load_handoff_module  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
args, remaining = parser.parse_known_args()
PROJECT = args.project.resolve()
SESSION_MODEL = "TEST-SESSION-MODEL"

# This round's unified research ranks CRM first; the incumbent AAPL comes from the
# previous round, where AAPL was the new first and MPC was switched out.
CANDIDATES = ["CRM", "DELL", "VLO", "TRGP", "AAPL"]
NEW_FIRST = "CRM"
INCUMBENT = "AAPL"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_result():
    ranking = [{"rank": index + 1, "symbol": symbol} for index, symbol in enumerate(CANDIDATES)]
    return {"status": "COMPLETE", "research_model": "legacy-api", "source": "offline-fixture",
            "candidate_symbols": CANDIDATES, "candidate_count": len(CANDIDATES),
            "ranking": {"ranking": ranking}}


def previous_result():
    return {"status": "COMPLETE", "new_first_symbol": INCUMBENT, "previous_first_symbol": "MPC",
            "rebalance_decision": "SWITCH_TO_NEW_FIRST"}


def packet():
    return {"evidence": [{"id": "ev1", "status": "VERIFIED", "source_url": "https://example.test/filing",
                          "retrieved_at": "2026-09-15T00:00:00+00:00", "facts": {"revenue": 1},
                          "source_opened": True}]}


def row(symbol):
    return {"symbol": symbol, "investment_confidence": 0.6, "evidence_completeness": "partial",
            "confidence_basis": "offline fixture", "evidence_refs": ["ev1"],
            "alpha_thesis": "UNKNOWN", "market_expectation": "UNKNOWN",
            "remaining_alpha_view": "UNKNOWN"}


def decision(action="SWITCH_TO_NEW_FIRST"):
    return {
        "research_provider": "CURRENT_CONVERSATION", "research_model": SESSION_MODEL, "status": "COMPLETE",
        "supplemental_research": [row(symbol) for symbol in CANDIDATES],
        "final_ranking": [{**row(symbol), "rank": index + 1} for index, symbol in enumerate(CANDIDATES)],
        "incumbent_research": row(INCUMBENT),
        "selection_rationale": {
            "why_final_first": "第一名从当前价格出发具有更强的未来相对收益证据",
            "why_not_finalists": [{"symbol": symbol, "reason": "该候选的当前预期差或兑现路径相对较弱"}
                                  for symbol in CANDIDATES[1:]],
            "why_first_over_second": "第一名的盈利预期差和兑现路径相对第二名更有支撑",
            "remaining_alpha": "市场预期尚未完全反映当前经营证据，当前价格仍有相对机会",
            "incumbent_comparison": "相对 AAPL，当前新候选的前瞻预期差更具吸引力",
            "remaining_alpha_comparison": "新候选相对原有效首选的预期差足以支持本轮比较",
            "core_catalysts": ["经营证据继续改善并推动预期修订"],
            "main_risks": ["经营改善不及预期导致相对机会收窄"],
            "thesis_invalidation_conditions": ["关键经营证据反转并否定当前论点"],
            "why_keep": ["若新旧差异落入判断误差范围，继续保留原有效首选"],
            "why_switch": ["若前瞻优势可信且足以覆盖研究层摩擦，则切换"],
            "core_reason": "本轮根据当前价格和已核验证据选择相对机会更高的股票",
            "biggest_risk": "未来经营结果与当前前瞻判断出现显著偏离",
        },
        "comparison": {"new_first_symbol": NEW_FIRST, "previous_first_symbol": INCUMBENT,
                       "rebalance_decision": action, "why_keep": "incumbent still valid",
                       "why_switch": "new first has better verified evidence",
                       "decision_reason": "offline fixture decision", "evidence_refs": ["ev1"]},
    }


class SessionHandoffTests(unittest.TestCase):
    def fixtures(self, directory, action="SWITCH_TO_NEW_FIRST"):
        root = Path(directory)
        source, previous, evidence = root / "source.json", root / "previous.json", root / "packet.json"
        decision_path = root / "decision.json"
        write_json(source, source_result())
        write_json(previous, previous_result())
        write_json(evidence, packet())
        write_json(decision_path, decision(action))
        return source, previous, evidence, decision_path, root / "runs"

    def prepare(self, directory, action="SWITCH_TO_NEW_FIRST"):
        source, previous, evidence, decision_path, output_root = self.fixtures(directory, action)
        from session_handoff import main
        code = main(["prepare", "--project", str(PROJECT), "--session-model", SESSION_MODEL,
                     "--source-result", str(source), "--previous-result", str(previous),
                     "--verification-packet", str(evidence), "--output-root", str(output_root)])
        self.assertEqual(code, 0)
        runs = sorted(output_root.iterdir())
        self.assertEqual(len(runs), 1)
        return runs[0], decision_path, evidence

    def test_project_runtime_imports_are_blocked(self):
        guard = NoProjectRuntimeImports()
        for name in ("src", "src.main", "src.llm_agent", "src.runner", "src.risk_engine", "src.execution.ibkr"):
            with self.assertRaises(ImportError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec("json"))

    def test_missing_project_module_fails_closed(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "session research module"):
                load_handoff_module(Path(directory))

    def test_session_model_is_declared_not_api_verified(self):
        self.assertEqual(session_model(None), DEFAULT_SESSION_MODEL)
        self.assertEqual(session_model(SESSION_MODEL), SESSION_MODEL)
        self.assertEqual(session_model("  "), DEFAULT_SESSION_MODEL)

    def test_legacy_api_model_is_override_only(self):
        self.assertEqual(api_model(), RESEARCH_MODEL)
        self.assertEqual(api_effort(), "medium")

    def test_prepare_writes_immutable_baseline_and_waits_for_session(self):
        with TemporaryDirectory() as directory:
            run, _, _ = self.prepare(directory)
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "WAITING_SESSION_RESEARCH")
            self.assertEqual(manifest["research_provider"], "CURRENT_CONVERSATION")
            self.assertEqual(manifest["research_model"], SESSION_MODEL)
            self.assertEqual(manifest["model_provenance"], "SESSION_DECLARED_NOT_API_VERIFIED")
            self.assertIsNone(manifest["reasoning_effort_requested"])
            self.assertEqual(manifest["reasoning_effort_effective"], "NOT_EXPOSED_BY_SESSION")
            self.assertFalse(manifest["standalone_model_api"])
            self.assertEqual(manifest["api_calls"], 0)
            self.assertEqual(manifest["incoming_active_selection"], INCUMBENT)
            self.assertEqual(manifest["initial_top_five"], CANDIDATES)
            self.assertIn("current executable price", manifest["objective"])
            self.assertIn("highest expected forward relative return", manifest["objective"])
            self.assertEqual(manifest["active_selection_meaning"], "RESEARCH_LAYER_CURRENT_AI_PREFERENCE; NOT_BROKER_POSITION")
            self.assertEqual(manifest["safety"]["orders"], "NOT_SENT")
            self.assertFalse((run / "result.json").exists())

    def test_finalize_switch_records_new_first(self):
        with TemporaryDirectory() as directory:
            run, decision_path, evidence = self.prepare(directory)
            from session_handoff import main
            code = main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                         "--decision", str(decision_path), "--evidence", str(evidence)])
            self.assertEqual(code, 0)
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["rebalance_decision"], "SWITCH_TO_NEW_FIRST")
            self.assertEqual(result["new_first_symbol"], NEW_FIRST)
            self.assertEqual(result["outgoing_active_selection"], NEW_FIRST)
            self.assertEqual(result["active_selection"], NEW_FIRST)
            self.assertEqual(result["selection_scope"], "RESEARCH_ONLY_NOT_ACCOUNT_HOLDING")
            active = json.loads((run / "active_selection.json").read_text(encoding="utf-8"))
            self.assertEqual(active["research_provider"], "CURRENT_CONVERSATION")
            self.assertEqual(active["active_selection"], NEW_FIRST)

    def test_finalize_keep_records_incumbent(self):
        with TemporaryDirectory() as directory:
            run, decision_path, evidence = self.prepare(directory, action="KEEP_PREVIOUS")
            from session_handoff import main
            main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                  "--decision", str(decision_path), "--evidence", str(evidence)])
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["outgoing_active_selection"], INCUMBENT)
            self.assertEqual(result["active_selection"], INCUMBENT)
            self.assertEqual(result["new_first_symbol"], NEW_FIRST)

    def test_finalize_rejects_api_provenance(self):
        with TemporaryDirectory() as directory:
            run, decision_path, evidence = self.prepare(directory)
            payload = decision()
            payload["research_provider"] = "API"
            write_json(decision_path, payload)
            from session_handoff import main
            with self.assertRaisesRegex(ValueError, "current-conversation provenance"):
                main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                      "--decision", str(decision_path), "--evidence", str(evidence)])

    def test_finalize_requires_decision_time_remaining_alpha_rationale(self):
        with TemporaryDirectory() as directory:
            run, decision_path, evidence = self.prepare(directory)
            payload = decision()
            del payload["selection_rationale"]["remaining_alpha"]
            write_json(decision_path, payload)
            from session_handoff import main
            with self.assertRaisesRegex(ValueError, "remaining_alpha"):
                main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                      "--decision", str(decision_path), "--evidence", str(evidence)])

    def test_finalize_refuses_to_overwrite(self):
        with TemporaryDirectory() as directory:
            run, decision_path, evidence = self.prepare(directory)
            from session_handoff import main
            main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                  "--decision", str(decision_path), "--evidence", str(evidence)])
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                main(["finalize", "--project", str(PROJECT), "--run-directory", str(run),
                      "--decision", str(decision_path), "--evidence", str(evidence)])

    def test_incomplete_source_cannot_open_a_run(self):
        with TemporaryDirectory() as directory:
            source, previous, evidence, _, output_root = self.fixtures(directory)
            payload = source_result()
            payload["status"] = "FAILED"
            write_json(source, payload)
            from session_handoff import main
            with self.assertRaisesRegex(ValueError, "not COMPLETE"):
                main(["prepare", "--project", str(PROJECT), "--session-model", SESSION_MODEL,
                      "--source-result", str(source), "--previous-result", str(previous),
                      "--verification-packet", str(evidence), "--output-root", str(output_root)])

    def test_ranking_only_history_cannot_initialize_selection(self):
        with TemporaryDirectory() as directory:
            source, previous, evidence, _, output_root = self.fixtures(directory)
            write_json(previous, {"status": "COMPLETE", "selected_symbol": INCUMBENT})
            from session_handoff import main
            with self.assertRaisesRegex(ValueError, "ranking is not selection"):
                main(["prepare", "--project", str(PROJECT), "--session-model", SESSION_MODEL,
                      "--source-result", str(source), "--previous-result", str(previous),
                      "--verification-packet", str(evidence), "--output-root", str(output_root)])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
