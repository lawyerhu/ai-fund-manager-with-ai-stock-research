"""Offline tests for generic challenger/incumbent pair replay."""

import unittest
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from recheck_selection_pair import CandidateReScore, extract_pair_context  # noqa: E402
from top_five_deep_research import PreviousWinnerComparison  # noqa: E402


class PairRecheckTests(unittest.TestCase):
    def test_rescore_schema_is_compact_and_judgmental(self):
        result = CandidateReScore.model_validate({
            "symbol": "DELL",
            "alpha_score": 81,
            "confidence": 0.68,
            "score_basis": ["盈利增长", "估值", "催化剂"],
            "score_interpretation": "同一证据包内的主观相对机会评分",
            "evidence_strengths": ["收入与cRPO增长", "现金流改善"],
            "evidence_limitations": ["未来固定期限一致预期未完全核验"],
            "expected_alpha_basis": ["earnings", "valuation", "catalyst"],
            "estimate_type": "SOL_MODEL_ESTIMATE",
            "bull_case": "催化剂带来持续盈利重估",
            "base_case": "高位震荡",
            "bear_case": "催化剂不及预期导致回撤",
            "thesis_invalidation_conditions": ["收入增长明显放缓"],
            "evidence_refs": ["same_date_market_snapshot", "cash_flow_and_official_financials"],
        })
        self.assertEqual(result.symbol, "DELL")
        self.assertEqual(result.estimate_type, "SOL_MODEL_ESTIMATE")
        self.assertNotIn("target_weight", result.model_dump())

    def test_pair_context_accepts_arbitrary_challenger_and_incumbent(self):
        source = {
            "status": "COMPLETE",
            "active_selection": "NVDA",
            "final_ranking": {"ranking": [{"symbol": "DELL", "preliminary_alpha_score": 76}]},
            "supplemental_research": {"DELL": {"symbol": "DELL"}},
            "previous_first_symbol": "NVDA",
            "previous_first_supplemental_research": {"symbol": "NVDA"},
            "previous_winner_record": {"winner_record": {"symbol": "NVDA", "preliminary_alpha_score": 86}},
        }
        context = extract_pair_context(source, challenger_symbol="DELL", incumbent_symbol="NVDA")
        self.assertEqual(context["incumbent_research"]["symbol"], "NVDA")
        self.assertEqual(context["challenger_symbol"], "DELL")
        self.assertEqual(context["incumbent_symbol"], "NVDA")
        self.assertEqual(context["previous_score"], 76)

    def test_comparison_schema_rejects_review_required(self):
        payload = {
            "new_first_symbol": "DELL",
            "previous_first_symbol": "NVDA",
            "rebalance_decision": "REVIEW_REQUIRED",
            "alpha_gap": None,
            "alpha_gap_status": "UNKNOWN",
            "why_new_beats_previous": ["CRM的已核验盈利增长更强"],
            "why_keep_previous": ["MPC的周期催化剂仍可能持续"],
            "evidence_refs": ["same_date_market_snapshot"],
            "confidence": 0.5,
            "confidence_reducers": ["未来一致预期仍有缺口"],
            "thesis_invalidation_conditions": ["新证据否定增长逻辑"],
        }
        with self.assertRaises(ValueError):
            PreviousWinnerComparison.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
