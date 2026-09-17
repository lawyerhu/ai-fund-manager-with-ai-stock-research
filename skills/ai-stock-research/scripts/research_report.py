"""Render saved decision-time Chinese rationale without any model or market calls."""
import argparse
import json
from pathlib import Path
import subprocess


BUNDLED_PYTHON = Path("C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
ACTIONS = {"KEEP_PREVIOUS": "维持原有效首选", "SWITCH_TO_NEW_FIRST": "切换至本轮第一名"}
GAP_STATUS_LABELS = {
    "VERIFIED": "已核实",
    "NOT_PUBLIC": "未公开",
    "NOT_YET_OCCURRED": "尚未发生",
    "RETRIEVAL_FAILED": "本轮检索失败",
    "PAID_DATA_REQUIRED": "需要付费数据",
    "DERIVATION_REQUIRED": "需要确定性计算",
    "INSUFFICIENT_SPECIFICITY": "问题不够具体",
    "CONFLICTING_EVIDENCE": "证据冲突",
    "STALE": "数据过旧",
    "NOT_APPLICABLE": "不适用",
    "UNAVAILABLE": "旧版不可得",
    "CONFLICT": "旧版冲突",
    "NOT_RECORDED": "未记录",
}


def _gap_rows(result):
    """Collect saved structured gaps; never infer a gap from a score or count."""
    from evidence_discipline import iter_gap_records

    rows = []
    seen = set()
    for stage in result.get("evidence_assessments", []) or []:
        if not isinstance(stage, dict):
            continue
        assessments = stage.get("evidence_assessments", [])
        # Session handoff results persist one assessment per stage directly;
        # the skill runner persists a wrapper with an inner assessment list.
        if not assessments and any(key in stage for key in (
                "evidence_gaps", "important_evidence_gaps", "unresolved_information_gaps")):
            assessments = [stage]
        for assessment in assessments:
            if not isinstance(assessment, dict):
                continue
            for gap in iter_gap_records(assessment):
                key = (str(gap.get("symbol", "")).upper(), gap.get("field_name"),
                       gap.get("gap_status"), gap.get("reason"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({**gap, "stage": stage.get("stage")})
            for derived in assessment.get("resolved_derivations", []) or []:
                calculation = derived.get("calculation", {}) if isinstance(derived, dict) else {}
                rows.append({
                    "symbol": assessment.get("symbol", "NOT_RECORDED"),
                    "field_name": (derived.get("gap", {}) or {}).get("field_name", "NOT_RECORDED"),
                    "gap_status": "VERIFIED",
                    "criticality": "NOT_RECORDED",
                    "reason": "已根据公开原始输入完成确定性计算",
                    "source_required": calculation.get("evidence_refs", []),
                    "last_checked_at": calculation.get("as_of"),
                    "retrieval_attempts": 0,
                    "evidence_refs": calculation.get("evidence_refs", []),
                    "decision_impact": "已从未决计算转为可追溯结果",
                    "blocking_research": False,
                    "stage": stage.get("stage"),
                    "deterministic_calculation": calculation,
                })
    return rows


def _render_gap_detail(gap):
    status = str(gap.get("gap_status", "NOT_RECORDED")).upper()
    status_text = GAP_STATUS_LABELS.get(status, "未记录")
    attempts = gap.get("retrieval_attempts", 0)
    if isinstance(attempts, list):
        attempt_text = str(len(attempts))
    else:
        attempt_text = str(attempts)
    checked = "是" if gap.get("last_checked_at") or gap.get("search_record_refs") or attempt_text not in {"0", "None"} else "未记录"
    sources = gap.get("source_required") or "未记录"
    if isinstance(sources, list):
        sources = "、".join(str(item) for item in sources) or "未记录"
    return (
        f"{status}（{status_text}）；是否影响本轮判断：{gap.get('decision_impact', '未记录')}；"
        f"是否已经补查：{checked}；检索次数：{attempt_text}；最后核验：{gap.get('last_checked_at') or '未记录'}；"
        f"所需来源：{sources}；停止/原因：{gap.get('stopping_reason') or gap.get('reason') or '未记录'}"
    )


def report_blocks(result):
    """Each narrative below is copied from its saved decision field, never synthesized."""
    from selection_state import resolve_active_selection
    from selection_rationale import (TopFiveRationale, FinalSelectionRationale, IncumbentRationale,
                                     SupplementalFindings, chinese_text, check_coverage)
    resolve_active_selection(result)
    rationale = result["selection_rationale"]
    for schema in (TopFiveRationale, FinalSelectionRationale, IncumbentRationale):
        schema.model_validate({key: rationale[key] for key in schema.model_fields})
    check_coverage({key: rationale[key] for key in ("why_top5", "why_not_others")}, result["initial_ranking"], result["candidate_symbols"])
    check_coverage({"why_not_finalists": rationale["why_not_finalists"]}, result["final_ranking"], result["top_five"])
    blocks = [("title", "人工智能选股研究报告"),
              ("p", f"本轮结论：{ACTIONS[result['rebalance_decision']]}，继续有效的首选为 {result['outgoing_active_selection']}。"),
              ("p", rationale["core_reason"])]

    def heading(text):
        blocks.append(("h1", text))

    def para(label, value):
        blocks.append(("p", f"{label}：{value}"))

    def points(values, empty="本轮未记录此类新增事项"):
        for value in values or [empty]:
            blocks.append(("p", value))

    heading("一 本轮研究范围")
    universe = result.get("universe") or {}
    para("股票池规模", universe.get("count") if universe.get("count") is not None else "来源结果未记录，未推定股票池数量")
    if universe.get("symbols"):
        para("股票池名单", "、".join(universe["symbols"]))
    para("最终候选数量", result["candidate_count"])
    para("最终候选名单", "、".join(result["candidate_symbols"]))
    para("候选来源记录", result.get("source_decision_id") or "本轮保存的统一候选研究")
    para("进入本轮的有效首选", result["incoming_active_selection"])

    heading("二 从最终候选中选出前五名")
    for row in result["initial_ranking"]["ranking"]:
        para(f"初选第{row['rank']}名 {row['symbol']}",
             f"初选分数 {row['preliminary_alpha_score']}；投资判断置信程度 {row['confidence']:.1%}")
    for row in rationale["why_top5"]:
        para(row["symbol"], row["reason"])
    blocks.append(("h2", "其他候选未进入前五名的原因"))
    for row in rationale["why_not_others"]:
        para(row["symbol"], row["reason"])
    if not rationale["why_not_others"]:
        points([], "本轮最终候选全部进入前五名")

    heading("三 前五名补充深研的新证据")
    for symbol in result["top_five"]:
        blocks.append(("h2", symbol))
        findings = result["supplemental_findings"][symbol]
        SupplementalFindings.model_validate(findings)
        for key, label in (("positive_evidence", "新增正面证据"), ("negative_evidence", "新增负面证据"), ("unresolved_gaps", "未解决的信息缺口")):
            values = findings[key] or ["本轮未记录此类事项"]
            para(label, values[0])
            for value in values[1:]:
                blocks.append(("p", value))
    blocks.append(("h2", "结构化信息缺口及处理状态"))
    gaps = _gap_rows(result)
    if gaps:
        for gap in gaps:
            blocks.append(("p", f"{gap.get('symbol', '未记录')} · {gap.get('field_name', '未记录')}：{_render_gap_detail(gap)}"))
    else:
        blocks.append(("p", "本轮没有保存结构化信息缺口；历史结果缺少新字段时不从分数反推。"))

    heading("四 最终第一名的选择依据")
    for row in result["final_ranking"]["ranking"]:
        para(f"最终第{row['rank']}名 {row['symbol']}", f"最终分数 {row['preliminary_alpha_score']}")
    para("本轮第一名与第二名的模型分差", result["final_alpha_gap"])
    para("本轮第一名", result["new_first_symbol"])
    points([rationale["why_final_first"]])
    blocks.append(("h2", "为什么不选择另外四只股票"))
    for row in rationale["why_not_finalists"]:
        para(row["symbol"], row["reason"])
    para("相对第二名的优势", rationale["why_first_over_second"])
    for key, label in (("core_catalysts", "核心催化剂"), ("main_risks", "主要风险"), ("thesis_invalidation_conditions", "投资论点失效条件")):
        blocks.append(("h2", label))
        points(rationale[key])
    audit = next(record for record in reversed(result["evidence_assessments"]) if record["stage"] == "SOL_TOP5_FINAL_RANKING")
    for row in audit["evidence_assessments"]:
        chinese_text(row["evidence_completeness"])
        chinese_text(row["confidence_basis"])
        para(row["symbol"], f"投资判断置信程度 {row['investment_confidence']:.1%}；"
             f"关键证据完整程度：{row['evidence_completeness']}；置信度依据：{row['confidence_basis']}")
    para("第一名最大风险", rationale["maximum_risk"])

    heading("五 本轮第一名与原有效首选比较")
    para("比较对象", f"{result['new_first_symbol']} 与 {result['incoming_active_selection']}")
    pair_audit = result.get("pair_evidence_audit") or {}
    if pair_audit:
        para("成对证据审计状态", pair_audit.get("status", "未记录"))
        para("市场时点是否接近", (pair_audit.get("market_as_of_basis") or {}).get("status", "未记录"))
        para("关键证据不对称是否已处理", "是" if pair_audit.get("material_asymmetry_resolved") else "否")
        for item in pair_audit.get("material_asymmetries", []) or []:
            blocks.append(("p", f"不对称变量：{item.get('symbol')} 的 {item.get('field_name')} 为 {item.get('gap_status')}，"
                           f"另一腿已有 {item.get('other_evidence_status')}；可能改变方向：{item.get('could_change_direction')}。"))
    points([rationale["incumbent_comparison"]])
    blocks.append(("h2", "维持原首选的理由与取舍"))
    points(rationale["why_keep"])
    blocks.append(("h2", "切换至新首选的理由与取舍"))
    points(rationale["why_switch"])

    heading("六 最终结论")
    for key, label in (("new_first_symbol", "本轮研究第一名"), ("incoming_active_selection", "进入本轮的有效首选"), ("outgoing_active_selection", "本轮结束后的有效首选")):
        para(label, result[key])
    para("最终决策", ACTIONS[result["rebalance_decision"]])
    para("核心理由", rationale["core_reason"])
    para("最大风险", rationale["biggest_risk"])
    points(["仅为研究建议；未读取实际账户持仓；未执行风控审批；未发送或取消订单。"])
    return blocks


def write_docx(blocks, path):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2.3)
    for name, size, font in (("Normal", 11, "宋体"), ("Title", 21, "黑体"), ("Heading 1", 15, "黑体"), ("Heading 2", 12, "黑体")):
        style = doc.styles[name]
        style.font.name, style.font.size, style.font.color.rgb = font, Pt(size), RGBColor(0, 0, 0)
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = 1.3
    for kind, text in blocks:
        paragraph = doc.add_paragraph(text, {"title": "Title", "h1": "Heading 1", "h2": "Heading 2", "p": "Normal"}[kind])
        paragraph.paragraph_format.widow_control = True
        if kind != "p":
            paragraph.paragraph_format.keep_with_next = True
    doc.core_properties.title = "人工智能选股研究报告"
    doc.core_properties.language = "zh-CN"
    doc.save(path)


def write_pdf(blocks, path):
    """Use the same ordered text as DOCX; PDF support is optional."""
    from html import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    font = Path("C:/Windows/Fonts/simsun.ttc")
    pdfmetrics.registerFont(TTFont("ReportChinese", str(font), subfontIndex=0))
    styles = {kind: ParagraphStyle(kind, fontName="ReportChinese", fontSize=size,
              leading=size * 1.4, spaceAfter=8, wordWrap="CJK", textColor=colors.black,
              keepWithNext=kind != "p") for kind, size in (("title", 21), ("h1", 15), ("h2", 12), ("p", 11))}
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=65, rightMargin=65, topMargin=55, bottomMargin=55,
                      title="人工智能选股研究报告").build([Paragraph(escape(text), styles[kind]) for kind, text in blocks])


def render_reports(result_path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    blocks = report_blocks(result)
    write_docx(blocks, result_path.parent / "report.docx")
    outcome = {"docx": "report.docx", "docx_status": "COMPLETE"}
    try:
        write_pdf(blocks, result_path.parent / "report.pdf")
        outcome.update(pdf="report.pdf", pdf_status="COMPLETE")
    except Exception as exc:
        outcome.update(pdf=None, pdf_status="FAILED", pdf_error=f"{type(exc).__name__}: {exc}")
    return outcome


def generate_reports(result_path):
    """Run document libraries in the bundled runtime, not the research venv."""
    completed = subprocess.run([str(BUNDLED_PYTHON), "-X", "utf8", str(Path(__file__).resolve()),
                                "--result", str(result_path), "--render"],
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
    if completed.returncode:
        raise RuntimeError("Chinese report generation failed: " + completed.stderr[-1500:])
    return json.loads(completed.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    outcome = render_reports(args.result.resolve()) if args.render else generate_reports(args.result.resolve())
    if not args.render:
        saved = json.loads(args.result.read_text(encoding="utf-8"))
        saved["reports"] = outcome
        saved["report_status"] = "COMPLETE"
        args.result.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False))
