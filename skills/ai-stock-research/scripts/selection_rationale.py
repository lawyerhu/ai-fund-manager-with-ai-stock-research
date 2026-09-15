"""Decision-time Chinese explanations for deterministic report generation."""
import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def chinese_text(value):
    narrative = re.sub(r"https?://[^\s\u3400-\u9fff]+", "", value)
    if not re.search(r"[\u3400-\u9fff]", narrative) or re.search(r"[a-z]", narrative):
        raise ValueError("Report narrative must be written in Chinese at decision time")
    return value


ChineseText = Annotated[str, Field(min_length=1), AfterValidator(chinese_text)]


class StockReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    reason: ChineseText


class TopFiveRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    why_top5: list[StockReason] = Field(min_length=5, max_length=5)
    why_not_others: list[StockReason]


class SupplementalFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive_evidence: list[ChineseText]
    negative_evidence: list[ChineseText]
    unresolved_gaps: list[ChineseText]


class FinalSelectionRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    why_final_first: ChineseText
    why_not_finalists: list[StockReason] = Field(min_length=4, max_length=4)
    why_first_over_second: ChineseText
    core_catalysts: list[ChineseText] = Field(min_length=1)
    main_risks: list[ChineseText] = Field(min_length=1)
    thesis_invalidation_conditions: list[ChineseText] = Field(min_length=1)
    maximum_risk: ChineseText


class IncumbentRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incumbent_comparison: ChineseText
    why_keep: list[ChineseText] = Field(min_length=1)
    why_switch: list[ChineseText] = Field(min_length=1)
    core_reason: ChineseText
    biggest_risk: ChineseText


RATIONALE_STAGES = {
    "SOL_STAGE_A_RANKING": TopFiveRationale,
    "EQUAL_DEPTH_RANKING": TopFiveRationale,
    "SOL_TOP5_SUPPLEMENTAL": SupplementalFindings,
    "SOL_TOP5_FINAL_RANKING": FinalSelectionRationale,
    "SOL_NEW_VS_PREVIOUS": IncumbentRationale,
}

RATIONALE_INSTRUCTIONS = (
    "\n请在本次实际研究或决策时填写 selection_rationale，所有自然语言理由、证据完整程度、"
    "置信度说明均使用简体中文；股票代码和来源网址可保留原样，枚举代码遵守原 schema。"
    "这是本次实际判断的可审计摘要，不是隐藏思维链，也不是事后根据胜者编写的解释。"
    "初选时逐一记录五只股票的核心逻辑及优于其他候选的依据，并覆盖每只未入围股票的主要原因。"
    "补充深研只记录相对初选新增的正面证据、负面证据和未解决缺口；没有相应新增事实可留空列表。"
    "最终排名时解释为何第一名胜过其余四只，单独比较第二名，保存催化剂、主要风险、论点失效条件和最大风险。"
    "新旧首选比较时保存维持理由、换股理由、最终动作的核心依据及最大风险；"
    "未采用动作的理由应清楚表达其支持因素与为何本轮未采用，不能与实际动作矛盾。"
    "这些是输出说明要求，不是固定投资指标、打分因子或新的筛选步骤。"
)


def check_coverage(rationale, payload, candidates):
    def same(rows, expected):
        symbols = [row["symbol"].upper() for row in rows]
        if len(symbols) != len(set(symbols)) or set(symbols) != expected:
            raise ValueError("Decision rationale must cover the exact relevant candidate set")
    if "why_top5" in rationale:
        selected = {row["symbol"].upper() for row in payload["ranking"]}
        same(rationale["why_top5"], selected)
        same(rationale["why_not_others"], {s.upper() for s in candidates} - selected)
    if "why_not_finalists" in rationale:
        same(rationale["why_not_finalists"], {row["symbol"].upper() for row in payload["ranking"][1:]})
