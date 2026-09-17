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
    # Plain text plus a sentinel keeps older saved results readable.  New
    # selection stages are checked by evidence_discipline before completion.
    remaining_alpha: str = "NOT_RECORDED"
    core_catalysts: list[ChineseText] = Field(min_length=1)
    main_risks: list[ChineseText] = Field(min_length=1)
    thesis_invalidation_conditions: list[ChineseText] = Field(min_length=1)
    maximum_risk: ChineseText


class IncumbentRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incumbent_comparison: ChineseText
    why_keep: list[ChineseText] = Field(min_length=1)
    why_switch: list[ChineseText] = Field(min_length=1)
    # Kept separate from ``remaining_alpha`` because the final result stores
    # both stage objects in one flat, backward-compatible rationale object.
    remaining_alpha_comparison: str = "NOT_RECORDED"
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
    "最终排名和新旧首选比较都必须保存一段剩余 Alpha 判断：从当前可执行价格出发，"
    "说明市场预期、模型判断、预期差、盈利/收入/现金流修订、估值变化是经营增长还是倍数扩张、"
    "历史涨跌是否消耗预期差、催化剂是未计价/部分计价/基本充分计价、相对其他候选的优势以及最关键的下行路径。"
    "这是一段开放式研究叙述，不是固定指标清单；过去涨幅、估值、波动、技术位置、事件和入场延伸风险只能作为证据，"
    "不得自动否决、固定扣分或设定冠军资格。事件本身不是 Alpha，也不要求近期必须有事件。"
    "新旧首选比较时保存维持理由、换股理由、最终动作的核心依据及最大风险；"
    "比较 challenger 与 incumbent 时不设置固定分差、置信度或持有期限门槛，模型可基于自身判断误差和研究层交易摩擦自主决定。"
    "历史买入成本、浮盈浮亏和持有天数属于账户/审计信息，不得成为研究层理由；"
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
