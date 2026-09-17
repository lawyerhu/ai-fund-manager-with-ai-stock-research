"""Offline state-chain, decision-time rationale and Chinese report integration tests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(args.project.resolve()))

from selection_state import SelectionTransition, resolve_active_selection, transition_fields
from selection_rationale import TopFiveRationale
from research_report import BUNDLED_PYTHON, report_blocks
from top_five_deep_research import extract_previous_winner, main as run_top5
from evidence_discipline import install_evidence_discipline
from research import ASTRA_MODEL, source_universe
from src import config, llm_agent
from src.data_provider import MockDataProvider
from tests.test_sol_deep_research import ranking_payload, deep_payload


def top_rationale(symbols, others):
    return {"why_top5": [{"symbol": s, "reason": f"{s} 的已核验催化剂和相对收益前景优于其余候选"} for s in symbols],
            "why_not_others": [{"symbol": s, "reason": f"{s} 的本轮机会缺乏足够相对优势，未进入前五名"} for s in others]}


def final_rationale(symbols):
    return {"why_final_first": "第一名具备本轮最明确的公司催化剂与相对机会", "why_not_finalists": [
        {"symbol": s, "reason": f"{s} 的短期催化剂相比第一名较弱"} for s in symbols[1:]],
        "why_first_over_second": "第一名相对第二名具有更明确的新增订单兑现路径",
        "remaining_alpha": "当前价格尚未完全反映盈利与现金流改善，市场预期和已核验证据之间仍有相对机会",
        "core_catalysts": ["公司披露新增订单预计本期交付"], "main_risks": ["订单交付可能推迟"],
        "thesis_invalidation_conditions": ["公司公告取消主要订单"], "maximum_risk": "订单无法兑现导致投资论点失效"}


def comparison_rationale():
    return {"incumbent_comparison": "原有效首选的兑现路径更稳定，本轮新冠军的优势尚不足以替代它",
            "why_keep": ["保留原首选，因为其当前催化剂更有支撑"],
            "why_switch": ["新冠军有潜在增长优势，但本轮证据尚不足以支持切换"],
            "remaining_alpha_comparison": "新旧首选的当前价格预期差仍需结合兑现可信度和研究层摩擦判断",
            "core_reason": "本轮维持原有效首选，保留其更明确的机会", "biggest_risk": "原首选的催化剂可能被新公告否定"}


def audit(payload, symbols, rationale=None):
    result = deepcopy(payload)
    confidences = {r["symbol"]: r["confidence"] for r in payload.get("ranking", [])}
    result["evidence_assessments"] = [{"symbol": s, "investment_confidence": confidences.get(s, 0.3),
        "evidence_completeness": "关键公告已核验，未来兑现结果仍不确定", "confidence_basis": "对兑现时点仍有真实不确定性",
        "research_requests": [], "unresolved_information_gaps": []} for s in symbols]
    if rationale is not None:
        result["selection_rationale"] = rationale
    return result


class SelectionReportingTests(unittest.TestCase):
    def test_three_round_keep_chain_retains_original_active_selection(self):
        week1 = {"status": "COMPLETE", "active_selection": "A"}
        week2 = {"status": "COMPLETE", "selected_symbol": "B",
                 **transition_fields(resolve_active_selection(week1), "B", "KEEP_PREVIOUS")}
        week3 = transition_fields(resolve_active_selection(week2), "C", "SWITCH_TO_NEW_FIRST")
        self.assertEqual(week2["outgoing_active_selection"], "A")
        self.assertEqual(week3["incoming_active_selection"], "A")
        self.assertEqual(week3["outgoing_active_selection"], "C")

    def test_switch_chain_and_same_symbol_keep(self):
        state = {"status": "COMPLETE", **transition_fields("A", "B", "SWITCH_TO_NEW_FIRST")}
        incoming = resolve_active_selection(state)
        self.assertEqual(incoming, "B")
        self.assertEqual(transition_fields(incoming, "B", "KEEP_PREVIOUS")["active_selection"], "B")
        with self.assertRaises(ValueError):
            transition_fields("B", "B", "SWITCH_TO_NEW_FIRST")

    def test_legacy_keep_resolves_incumbent_not_ranking_winner(self):
        old = {"status": "COMPLETE", "selected_symbol": "B", "previous_first_symbol": "A",
               "rebalance_decision": "KEEP_PREVIOUS", "previous_first_supplemental_research": {"symbol": "A", "evidence": "原首选证据"},
               "supplemental_research": {"B": {"symbol": "B", "evidence": "新冠军证据"}}}
        symbol, context = extract_previous_winner(old)
        self.assertEqual(symbol, "A")
        self.assertEqual(context["research_record"]["evidence"], "原首选证据")
        old["rebalance_decision"] = "SWITCH_TO_NEW_FIRST"
        self.assertEqual(extract_previous_winner(old)[0], "B")

    def test_conflicting_state_failed_result_and_ranking_only_are_rejected(self):
        for value in ({"status": "COMPLETE", "selected_symbol": "B"},
                      {"status": "NEEDS_RESEARCH", "active_selection": "A"},
                      {"status": "COMPLETE", **transition_fields("A", "B", "KEEP_PREVIOUS"), "outgoing_active_selection": "B"}):
            with self.assertRaises(ValueError):
                resolve_active_selection(value)
        with self.assertRaises(ValueError):
            SelectionTransition(incoming_active_selection="A", new_first_symbol="B", rebalance_decision="KEEP_PREVIOUS", outgoing_active_selection="B")

    def test_initial_rationale_is_captured_in_existing_ranking_request(self):
        initial = ranking_payload()
        top5 = [r["symbol"] for r in initial["ranking"]]
        candidates = top5 + [f"X{i:02d}" for i in range(15)]
        rationale = top_rationale(top5, candidates[5:])
        payload = audit(initial, candidates, rationale)
        calls = []
        runtime = llm_agent.LLMRuntimeConfig(base_url="http://invalid.test", api_key="offline-test", sol_model=ASTRA_MODEL, api_protocol="RESPONSES")
        def create(**request):
            calls.append(request)
            return SimpleNamespace(id="offline", output=[], output_text=json.dumps(payload), usage=None)
        provider = llm_agent.CCSwitchProvider(runtime=runtime, client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep_fn=lambda _: None)
        agent = llm_agent.SolResearchCIOAgent(ASTRA_MODEL, MockDataProvider(), provider=provider)
        agent._candidate_symbols = candidates
        install_evidence_discipline(agent, lambda *a: None, capture_selection_path=True)
        totals = dict(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0)
        result = agent._structured_deep_stage(llm_agent.CrossSectionalRanking, "rank", "EQUAL_DEPTH_RANKING", "本次初选", totals)
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.selection_rationale["EQUAL_DEPTH_RANKING"], rationale)
        self.assertEqual(result.ranking[0].symbol, top5[0])
        self.assertEqual(len(rationale["why_not_others"]), 15)
        with self.assertRaises(ValueError):
            TopFiveRationale.model_validate({**rationale, "why_top5": [{"symbol": s, "reason": "English only rationale"} for s in top5]})

    def test_complete_top5_run_persists_active_state_and_generates_saved_chinese_report(self):
        initial = ranking_payload()
        top5 = [r["symbol"] for r in initial["ranking"]]
        candidates = top5 + [f"X{i:02d}" for i in range(15)]
        final_order = top5[1:] + top5[:1]
        final = deepcopy(initial)
        by_symbol = {row["symbol"]: row for row in final["ranking"]}
        final["ranking"] = [{**by_symbol[s], "rank": i, "preliminary_alpha_score": 96 - i * 4,
                              "alpha_thesis": f"{s} 的经营变化可能尚未被价格完全反映",
                              "market_expectation": f"OBSERVED: {s} 的一致预期与公司公开信息",
                              "remaining_alpha_view": f"{s} 从当前价格仍有可研究的剩余预期差"}
                            for i, s in enumerate(final_order, 1)]
        source = {"status": "COMPLETE", "ranking": initial,
                  "deep_research": {s: deep_payload(s) for s in candidates}, "candidate_count": 20,
                  "candidate_symbols": candidates, "universe": {"count": 30, "symbols": candidates + [f"U{i}" for i in range(10)]},
                  "selection_rationale": top_rationale(top5, candidates[5:])}
        findings = {"positive_evidence": ["公司新增披露确认订单交付计划"], "negative_evidence": ["公司提示交付可能延迟"], "unresolved_gaps": []}
        comparison = dict(new_first_symbol=final_order[0], previous_first_symbol="OLD", rebalance_decision="KEEP_PREVIOUS",
            alpha_gap=None, alpha_gap_status="UNKNOWN", why_new_beats_previous=["新候选有增长机会"],
            why_keep_previous=["原首选的机会更明确"], evidence_refs=["packet"], confidence=0.3,
            confidence_reducers=["兑现存在不确定性"], thesis_invalidation_conditions=["原首选催化剂失效"],
            pair_comparison_complete=True, decision_basis_sufficient=True, material_asymmetry_resolved=True)
        old_research = deep_payload("OLD")
        old_research.update({"alpha_thesis": "原首选的经营预期仍有支持",
                             "market_expectation": "OBSERVED: 原首选的公开一致预期",
                             "remaining_alpha_view": "原首选从当前价格仍有未充分计价的预期差"})
        payloads = [*[audit(deep_payload(s), [s], findings) for s in top5],
                    audit(final, top5, final_rationale(final_order)), audit(old_research, ["OLD"]),
                    audit(comparison, [final_order[0], "OLD"], comparison_rationale())]
        calls = []
        runtime = llm_agent.LLMRuntimeConfig(base_url="http://invalid.test", api_key="offline-test", sol_model=ASTRA_MODEL, api_protocol="RESPONSES")
        provider_class = llm_agent.CCSwitchProvider
        def provider_factory(**kwargs):
            def create(**request):
                calls.append(request)
                return SimpleNamespace(id=f"offline-{len(calls)}", output=[], output_text=json.dumps(payloads.pop(0)), usage=None)
            client = SimpleNamespace(responses=SimpleNamespace(create=create), models=SimpleNamespace(list=lambda: {"data": [{"id": ASTRA_MODEL}]}))
            return provider_class(runtime=kwargs["runtime"], client=client, event_sink=kwargs.get("event_sink"), sleep_fn=lambda _: None)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/llm_agent.py").write_text("", encoding="utf-8")
            for name, value in (("source.json", source), ("previous.json", {"status": "COMPLETE", "active_selection": "OLD"}),
                                ("packet.json", {"evidence": [{"status": "VERIFIED"}], "as_of_basis": {}, "gap_audit": {}})):
                (root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with patch.object(config, "load_project_env"), patch.object(config, "load_config", return_value={"data": {"provider": "yahoo"}}), \
                 patch.object(llm_agent.LLMRuntimeConfig, "from_mapping", return_value=runtime), patch.object(llm_agent, "CCSwitchProvider", side_effect=provider_factory):
                code = run_top5(["--project", str(root), "--source-result", str(root / "source.json"), "--previous-result", str(root / "previous.json"),
                                 "--verification-packet", str(root / "packet.json"), "--run"])
            self.assertEqual(code, 0)
            result_path = next(root.glob("outputs/skill-research/*/result.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["candidate_count"], 20)
            self.assertEqual(result["incoming_active_selection"], "OLD")
            self.assertEqual(result["outgoing_active_selection"], "OLD")
            self.assertEqual(resolve_active_selection(result), "OLD")
            self.assertEqual(result["new_first_symbol"], final_order[0])
            self.assertEqual(result["selection_rationale"]["why_top5"], source["selection_rationale"]["why_top5"])
            self.assertEqual(result["selection_rationale"]["why_final_first"], final_rationale(final_order)["why_final_first"])
            self.assertEqual(len(calls), 8)
            with ZipFile(result_path.parent / "report.docx") as archive:
                xml = ET.fromstring(archive.read("word/document.xml"))
                text = "\n".join(xml.itertext())
            self.assertIn(comparison_rationale()["core_reason"], text)
            self.assertIn("最终候选数量：20", text)
            self.assertIn("X14 的本轮机会", text)
            self.assertNotIn("KEEP_PREVIOUS", text)
            self.assertNotIn("investment_confidence", text)
            self.assertIn("投资论点失效条件", text)
            self.assertIn("剩余 Alpha 与当前价格机会", text)
            self.assertIn("新旧首选的剩余 Alpha 比较", text)
            self.assertIn("Alpha Thesis", text)
            self.assertIn(f"{final_order[0]} 的经营变化可能尚未被价格完全反映", text)
            self.assertIn("Incumbent Remaining Alpha", text)
            self.assertIn("Challenger Remaining Alpha", text)
            self.assertIn("为什么 KEEP / SWITCH", text)
            legacy_result = deepcopy(result)
            legacy_result["selection_rationale"].pop("remaining_alpha", None)
            legacy_result["selection_rationale"].pop("remaining_alpha_comparison", None)
            for row in legacy_result["final_ranking"]["ranking"]:
                for field in ("alpha_thesis", "market_expectation", "remaining_alpha_view"):
                    row.pop(field, None)
            for field in ("alpha_thesis", "market_expectation", "remaining_alpha_view"):
                legacy_result["previous_first_supplemental_research"].pop(field, None)
            legacy_text = "\n".join(block[1] for block in report_blocks(legacy_result))
            self.assertIn("NOT_RECORDED", legacy_text)
            self.assertEqual(result["reports"]["pdf_status"], "COMPLETE")
            self.assertTrue((result_path.parent / "report.pdf").stat().st_size > 1000)
            self.assertTrue((result_path.parent / "events.jsonl").is_file())
            self.assertEqual(resolve_active_selection(json.loads((result_path.parent / "active_selection.json").read_text(encoding="utf-8"))), "OLD")
            # Optional fixture copy for local visual QA only; never a real research report.
            if getattr(self, "fixture_output", None):
                import shutil
                shutil.copytree(result_path.parent, self.fixture_output, dirs_exist_ok=True)
            # A PDF failure must preserve DOCX and the original model decision.
            code = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); import research_report as r; from pathlib import Path; from unittest.mock import patch; import json\nwith patch.object(r, 'write_pdf', side_effect=RuntimeError('pdf unavailable')):\n print(json.dumps(r.render_reports(Path({str(result_path)!r}))))"
            failed_pdf = subprocess.run([str(BUNDLED_PYTHON), "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8", check=True)
            self.assertEqual(json.loads(failed_pdf.stdout)["pdf_status"], "FAILED")
            self.assertTrue((result_path.parent / "report.docx").is_file())

    def test_universe_comes_from_the_source_screen_not_current_market_data(self):
        class Connection:
            def execute(self, query, parameters):
                self.parameters = parameters
                return self
            def fetchall(self):
                return [{"timestamp": "2026-09-14", "event_type": "AI_PROGRESS",
                         "metadata_json": json.dumps({"universe_total": 519, "universe_processed": 519})}]
        store = SimpleNamespace(connection=Connection())
        self.assertEqual(source_universe(store, "source-decision")["count"], 519)
        self.assertEqual(store.connection.parameters, ("source-decision",))
        self.assertIsNone(source_universe(store, None))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
