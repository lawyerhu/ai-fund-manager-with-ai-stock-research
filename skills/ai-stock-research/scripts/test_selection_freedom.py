"""Offline regression tests for model-owned investment judgment boundaries."""

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from recheck_selection_pair import CandidateReScore, extract_pair_context  # noqa: E402
from selection_rationale import (FinalSelectionRationale, IncumbentRationale,
                                  validate_alpha_fields)  # noqa: E402
from selection_state import resolve_active_selection, transition_fields  # noqa: E402
from top_five_deep_research import EVIDENCE_PACKET_INSTRUCTIONS, PreviousWinnerComparison  # noqa: E402


def comparison_payload(challenger, incumbent, action, *, gap=None, gap_status="UNKNOWN"):
    return {
        "new_first_symbol": challenger,
        "previous_first_symbol": incumbent,
        "rebalance_decision": action,
        "alpha_gap": gap,
        "alpha_gap_status": gap_status,
        "why_new_beats_previous": ["当前价格后的相对经营机会更强"],
        "why_keep_previous": ["原有效首选仍有可信的兑现路径"],
        "evidence_refs": ["pair-evidence"],
        "confidence": 0.51,
        "confidence_reducers": ["未来结果仍有不确定性"],
        "thesis_invalidation_conditions": ["关键经营证据反转"],
    }


class SelectionFreedomTests(unittest.TestCase):
    def test_alpha_discipline_does_not_turn_good_news_into_an_automatic_rank_boost(self):
        row = {
            "alpha_thesis": "UNKNOWN",
            "market_expectation": "OBSERVED: 高一致预期和已实现的财报结果",
            "remaining_alpha_view": "原有利好已大部分计价，剩余预期差较弱",
        }
        validate_alpha_fields(row, require_recorded=True)
        prompt = EVIDENCE_PACKET_INSTRUCTIONS
        self.assertIn("Business Quality is not Alpha", prompt)
        self.assertIn("Good news is not Alpha", prompt)
        self.assertIn("Mispricing is the potential source of Alpha", prompt)

    def test_no_near_term_catalyst_does_not_block_revision_supported_selection(self):
        validate_alpha_fields({
            "alpha_thesis": "盈利预期持续改善而价格反应有限，市场可能低估修订持续性",
            "market_expectation": "OBSERVED: EPS 与收入一致预期连续上修",
            "remaining_alpha_view": "没有近期事件，但仍有未充分计价的经营修订",
        }, require_recorded=True)
        self.assertIn("near-term catalyst is not required", EVIDENCE_PACKET_INSTRUCTIONS)

    def test_price_up_fifty_and_eps_up_sixty_remains_model_eligible(self):
        validate_alpha_fields({
            "alpha_thesis": "盈利修订快于价格，过去涨幅未必耗尽预期差",
            "market_expectation": "OBSERVED: EPS 预期上修 60% 且估值倍数基本稳定",
            "remaining_alpha_view": "基本面改善仍可能支持从当前价格继续相对占优",
        }, require_recorded=True)
        self.assertNotIn("price movement has consumed the gap", EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("Past gains do not imply poor future opportunity", EVIDENCE_PACKET_INSTRUCTIONS)

    def test_multiple_expansion_is_evidence_not_an_automatic_exclusion(self):
        validate_alpha_fields({
            "alpha_thesis": "价格上涨主要由倍数扩张推动，后续预期差需要重新验证",
            "market_expectation": "OBSERVED: EPS 预期变化有限而估值倍数扩张",
            "remaining_alpha_view": "倍数扩张可能已计价较多，但是否放弃仍由模型判断",
        }, require_recorded=True)
        self.assertIn("multiple expansion", EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("never apply a hard filter", EVIDENCE_PACKET_INSTRUCTIONS)

    def test_unknown_market_expectation_is_explicit_and_not_fabricated(self):
        row = {
            "alpha_thesis": "UNKNOWN",
            "market_expectation": "UNKNOWN",
            "remaining_alpha_view": "UNKNOWN",
        }
        self.assertIs(validate_alpha_fields(row, require_recorded=True), row)
        with self.assertRaises(ValueError):
            validate_alpha_fields({**row, "market_expectation": "市场隐含 EPS 为 10%"}, require_recorded=True)

    def test_inferred_market_expectation_is_labeled_as_inference(self):
        row = {
            "alpha_thesis": "价格和公开叙事可能隐含较高增长，但直接一致预期不可得",
            "market_expectation": "INFERRED: 根据当前价格、估值和公开叙事反推，非直接验证事实",
            "remaining_alpha_view": "模型只能保留定性预期差判断",
        }
        self.assertIs(validate_alpha_fields(row, require_recorded=True), row)
        with self.assertRaises(ValueError):
            validate_alpha_fields({**row, "market_expectation": "市场预期增长 20%"}, require_recorded=True)

    def test_a_b_h_p_price_move_and_risk_are_not_schema_vetoes(self):
        result = CandidateReScore.model_validate({
            "symbol": "NVDA",
            "alpha_score": 99,
            "confidence": 0.42,
            "score_basis": ["过去涨幅较大但盈利预期上修", "高估值和高波动作为研究证据"],
            "score_interpretation": "当前前瞻相对机会判断",
            "evidence_strengths": ["盈利与收入预期同步改善"],
            "evidence_limitations": ["短期价格延伸风险较高"],
            "expected_alpha_basis": ["盈利修订支持价格"],
            "bull_case": "经营结果继续超预期",
            "base_case": "经营改善部分兑现",
            "bear_case": "预期修订反转",
            "thesis_invalidation_conditions": ["盈利预期显著下修"],
            "evidence_refs": ["price-and-revision"],
        })
        self.assertEqual(result.symbol, "NVDA")
        self.assertEqual(result.alpha_score, 99)
        self.assertIn("price", EVIDENCE_PACKET_INSTRUCTIONS.lower())
        self.assertIn("hard filter", EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("near-term catalyst is not required", EVIDENCE_PACKET_INSTRUCTIONS)

    def test_c_n_small_or_unknown_gap_allows_either_model_action(self):
        for action in ("KEEP_PREVIOUS", "SWITCH_TO_NEW_FIRST"):
            payload = comparison_payload("DELL", "NVDA", action, gap=0.1, gap_status="COMPARABLE")
            self.assertEqual(PreviousWinnerComparison.model_validate(payload).rebalance_decision, action)
        unknown = PreviousWinnerComparison.model_validate(comparison_payload("VRT", "DELL", "SWITCH_TO_NEW_FIRST"))
        self.assertIsNone(unknown.alpha_gap)

    def test_d_e_f_sunk_cost_and_elapsed_time_are_not_state_inputs(self):
        state = transition_fields("AAPL", "CRM", "SWITCH_TO_NEW_FIRST")
        state["days_held"] = 3
        self.assertEqual(resolve_active_selection({"status": "COMPLETE", **state}), "CRM")
        state = transition_fields("CRM", "NVDA", "KEEP_PREVIOUS")
        state["days_held"] = 200
        self.assertEqual(resolve_active_selection({"status": "COMPLETE", **state}), "CRM")

    def test_g_luna_and_history_are_not_investment_priors(self):
        prompt = EVIDENCE_PACKET_INSTRUCTIONS
        self.assertIn("Luna ranking is retrieval context only", prompt)
        self.assertIn("historical rank/score", prompt)
        self.assertIn("sunk entry cost", prompt)

    def test_i_o_cross_style_and_no_event_requirement_remain_model_owned(self):
        for challenger, incumbent in (("DELL", "NVDA"), ("NVDA", "VRT"), ("VRT", "DELL")):
            result = PreviousWinnerComparison.model_validate(
                comparison_payload(challenger, incumbent, "SWITCH_TO_NEW_FIRST")
            )
            self.assertEqual(result.new_first_symbol, challenger)
        self.assertIn("style prior", prompt := EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("industry", prompt)
        self.assertIn("event is not alpha", prompt.lower())

    def test_j_k_l_integrity_controls_are_separate_from_action(self):
        self.assertIn("RETRIEVAL_FAILED", EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("UNKNOWN", EVIDENCE_PACKET_INSTRUCTIONS)
        self.assertIn("schema", EVIDENCE_PACKET_INSTRUCTIONS.lower())
        with self.assertRaises(ValueError):
            PreviousWinnerComparison.model_validate(comparison_payload("DELL", "NVDA", "INVALID"))

    def test_q_pair_recheck_source_has_no_historical_symbol_dependency(self):
        source_text = (SCRIPT_DIR / "recheck_selection_pair.py").read_text(encoding="utf-8")
        self.assertNotIn("CRM", source_text)
        self.assertNotIn("MPC", source_text)
        source = {
            "status": "COMPLETE",
            "active_selection": "NVDA",
            "final_ranking": {"ranking": [{"symbol": "DELL", "preliminary_alpha_score": 80}]},
            "supplemental_research": {"DELL": {"symbol": "DELL"}},
            "previous_first_supplemental_research": {"symbol": "NVDA"},
        }
        context = extract_pair_context(source, "DELL", "NVDA")
        self.assertEqual((context["challenger_symbol"], context["incumbent_symbol"]), ("DELL", "NVDA"))

    def test_r_old_rationale_fields_default_without_rewriting_history(self):
        final = FinalSelectionRationale.model_validate({
            "why_final_first": "第一名的当前前瞻机会更强",
            "why_not_finalists": [{"symbol": f"X{i}", "reason": "相对机会较弱"} for i in range(4)],
            "why_first_over_second": "第一名的兑现证据更强",
            "core_catalysts": ["经营改善"],
            "main_risks": ["结果偏离预期"],
            "thesis_invalidation_conditions": ["核心证据反转"],
            "maximum_risk": "经营结果显著不及预期",
        })
        incumbent = IncumbentRationale.model_validate({
            "incumbent_comparison": "原有效首选仍有相对机会",
            "why_keep": ["证据仍然支持"], "why_switch": ["新候选可能更强"],
            "core_reason": "当前比较仍需模型判断", "biggest_risk": "预期差消失",
        })
        self.assertEqual(final.remaining_alpha, "NOT_RECORDED")
        self.assertEqual(incumbent.remaining_alpha_comparison, "NOT_RECORDED")


if __name__ == "__main__":
    unittest.main()
