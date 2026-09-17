---
name: ai-stock-research
description: 使用现有 AI Fund Manager 做美股选股，由Luna筛选、当前对话模型完成候选深研，并将本轮第一名与上一轮有效首选同场比较，输出是否换股/调仓方向。仅研究，不执行交易。
---

# AI 选股研究：Luna → 当前对话模型

复用现有项目，不重写选股策略。原 Sol 是代码中的角色名；本技能该角色的研究请求由**当前对话模型**执行，即发起本技能的会话所使用的模型（具体模型名由会话在本次运行时声明，不在技能内硬编码），通过 `scripts/session_handoff.py` 的交互式交接完成。**不再固定调用 `gpt-6-astra`，也不再固定 reasoning effort `medium`**：会话的 reasoning effort 不对外暴露，统一记录为 `NOT_EXPOSED_BY_SESSION`；模型身份由会话声明并标记 `SESSION_DECLARED_NOT_API_VERIFIED`，不得通过 `/models` 冒充已被服务端确认。包括横向排名、深度研究、对抗审查、委员会决策、结构修复与上一轮有效首选比较。默认不读取 IBKR 实际持仓，也不调用持仓复核插件。Luna模型及批处理参数继承项目设置。

## 入口与边界

## 本实验的投资目标：排除主动持币

用户以小额实验资金检验 AI 自主选股的扣费后超额收益，接受集中持仓及较高波动。正常完成研究时，现金不属于可选投资标的，也不能建议清仓后长期等待。当前对话模型 应在已核验的候选中自主选择相对最佳股票，优先提出满仓首选、继续持有首选或清仓换股。满仓指已确认实验资金范围内扣除必要费用和交易单位余款后的可投资金额，不意味着融资或动用其他资金。不得自行推定整个插件账户都是实验资金。

每次向 当前对话模型 提交选股或新旧首选比较请求时，必须在研究上下文中带入此目标；仅修改报告措辞不算应用。新旧比较若触发换股，必须说明换入新首选及换出上一轮有效首选的相对依据，不能以降低股票总敞口、转为现金为目的。不得仅因非关键资料缺失机械调仓。股票排名第一仅表示本轮相对优选，不能承诺正收益或真实 Alpha。

此约束优先于下文允许主动选择现金的旧流程描述。已有候选结果若为 CASH/WAIT，应作为旧结果保留，重新让 当前对话模型 按本目标研究，不能由宿主强改成 BUY。API 请求失败、结果无法解析或股票身份无法确认时，应如实报告技术失败；投资资料补查后仍有缺口时，必须基于现有证据选择维持或换股；这不是现金投资建议，也不能伪造首选。保留既有研究隔离，不发送订单。本技能不读取账户数值仓位、不计算账户金额或股数，也不虚构损失预算。

默认项目：本仓库根目录；执行时请使用项目根目录的绝对路径传给 `--project`，不要依赖固定的本机路径。

使用项目 `.venv/Scripts/python.exe` 启动本技能 `scripts/research.py`，传入绝对 `--project` 路径。不复制API Key，不要求重新输入现有密钥。不修改生产 `.env`、数据库、交易模式或正在运行的服务。

**不得使用 Dashboard 的 RUN_AI_RESEARCH、src.main、StrategyRunner、backend command queue、IBKR connector 或 scripts/plugin_review.py 来执行本技能。** 当前生产可能是 PAPER，该入口可能下单。辅助脚本直接使用研究类：生产SQLite只读，研究结果进入独立目录，并阻止导入执行器/worker。TradeIntent仅作为研究返回值导出，不提交、不写回生产。没有实际Risk Engine审批；不得报告 WOULD_* 已获批准或声称已成交。

用户要求交易时，说明本技能仅研究，交易必须回到单独授权的正常风控流程，不扩展此脚本为下单入口。

## 投资判断与证据缺失分离

本节只修正证据处理，适用于候选初选、深研、对抗审查、最终排名和新旧首选比较；不改变 Universe、Luna 筛选参数、候选集合、筛选轮次、Top 5 或当前对话模型的决策归属。

- 当前对话模型 根据公司行业、商业模式、当前市场环境、催化剂和风险，自主确定真正影响本次决策的信息及其权重。下文及旧输出 schema 中的财务维度、packet 分区和数据工具只是可用线索或兼容输出槽位，不是固定因子模型、固定研究模板或买入前置指标清单；不适用的槽位可标 `NOT_APPLICABLE`，未知可标 `UNKNOWN`，不得据字段数量评分。保留原有数据采集供模型使用。
- 投资价值与证据状态分开：不得仅因信息未检索到而断言基本面差、价值低或机械降低投资分数；搜索失败不证明事实不存在，未知不自动构成负面，不编造或为提高置信度猜测数据。不同来源冲突时，由 当前对话模型 判断可靠性、时间和口径；优先公司公告、监管/SEC 披露、财报和 IR 等原始来源，具体来源由模型自主选择。
- 当前对话模型 认为缺失或冲突事实可能实质改变排名、冠军或换股方向时，必须提出具体补查问题、决策影响、关键词、来源方向和研究深度，由 Codex 实际只读检索并反馈。最终胜负依赖未知事实时优先查该事实；未补查的临时分数不能用于淘汰候选或宣布冠军。该义务同样覆盖初选 Top 5 之外可能受信息缺失影响的候选，不新增全池深研轮次。
- 保留现有最多两轮定向追加研究。每轮的检索范围、来源、关键词和深度由 当前对话模型 决定；保留实际查询、读取结果、日期、失败原因、替代来源尝试及停止依据，不能把“达到两轮”或单一接口失败当作已经合理研究。仅经合理补查仍无法解决的问题才列入 `unresolved_information_gaps`；之后仍按原规则由 当前对话模型 基于可信证据选择，允许真实的低置信度，不以完整度公式或阈值改变排名、保留或换股。

输出分别记录 `investment_confidence`（对投资判断的主观置信程度）与 `evidence_completeness`（当前对话模型 自主认定的关键证据覆盖情况及剩余缺口，用文字说明，不计算固定清单完成率）。旧 `confidence` 保留为同一投资判断置信度的兼容字段；两种概念不能互相推导。用 `confidence_basis` 解释真正影响判断的不确定性，不要求补查后置信度提高。

`scripts/evidence_discipline.py` 在 Skill 的现有当前对话模型结构化请求上附加 `evidence_assessments`，将其独立保存为 `evidence_assessments.json`，不改生产模型或评分算法。出现模型的关键 `research_requests` 时保存 `research_pending.json` 和 `status=NEEDS_RESEARCH`，其 `provisional_result` 仅供恢复与审计，不是最终排名/冠军/换股结论。Codex 必须继续执行检索并更新 packet，复用已有有效阶段，只重评受影响部分；不得把脚本退出当作整个任务完成或向用户重复索取研究授权。恢复时沿用原阶段的候选集合、上下文和同一研究角色（会话模型；仅显式选择 legacy API 路径时才用同一 CCSwitchProvider），使用相同审计；不能把待补查的原始 `_COMPLETED` 工具事件当作有效完成结果。尚未执行合理检索属于阶段未完成，不属于模型最终拒绝判断。旧结果缺少新增字段时标 `NOT_RECORDED`，不能从旧分数猜出证据完整度。

### 精细化证据缺口与成对审计

旧的 `VERIFIED`、`CONFLICT`、`UNAVAILABLE`、`NOT_APPLICABLE` 结果必须继续可读；新产生的 gap 使用 `gap_status`：`VERIFIED`、`NOT_PUBLIC`、`NOT_YET_OCCURRED`、`RETRIEVAL_FAILED`、`PAID_DATA_REQUIRED`、`DERIVATION_REQUIRED`、`INSUFFICIENT_SPECIFICITY`、`CONFLICTING_EVIDENCE`、`STALE` 或 `NOT_APPLICABLE`。其中 `NOT_PUBLIC` 是权威检索后发行人/监管机构未公开，`NOT_YET_OCCURRED` 是未来事件尚未发生，`RETRIEVAL_FAILED` 是本轮未成功取得公开资料，`PAID_DATA_REQUIRED` 是当前环境无权限的付费数据，`DERIVATION_REQUIRED` 是已有原始数据但尚未计算，`INSUFFICIENT_SPECIFICITY` 是问题范围过宽，`CONFLICTING_EVIDENCE` 是来源实质冲突，`STALE` 是时点不足。旧 `UNAVAILABLE` 不得被反推成其中任何一种新状态。

每个重要 gap 至少保存 `field_name`、`symbol`、`gap_status`、`criticality`、`reason`、`source_required`、`last_checked_at`、`retrieval_attempts`、`evidence_refs`、`decision_impact`、`blocking_research`。`criticality` 只能由当前对话模型按公司、行业、商业模式和当前 thesis 判断为 `CRITICAL`、`IMPORTANT` 或 `NON_CRITICAL`；不按缺失字段数量降低分数，也不使用固定财务指标清单。`RETRIEVAL_FAILED`/`STALE` 表示研究可能尚未完成，优先补查；`NOT_PUBLIC`/`PAID_DATA_REQUIRED`/`NOT_YET_OCCURRED` 在合理检索边界后可以保留，不得无限等待或程序默认 `KEEP_PREVIOUS`。

在最终 `KEEP_PREVIOUS` / `SWITCH_TO_NEW_FIRST` 之前执行 `pair_evidence_audit`，检查 incumbent 与 challenger 的市场 `as_of_basis` 是否合理接近、公司经营证据是否足够新，以及一方关键变量为 `RETRIEVAL_FAILED`/`STALE` 而另一方已经 `VERIFIED` 时是否可能改变换股方向。若存在这种 material asymmetry，必须保存 `status=NEEDS_RESEARCH`，只生成受影响两腿的定向 `research_requests`，不重跑 Luna、不重跑整个 Top 5，并继续遵守最多两轮定向补查。比较记录新增 `pair_comparison_complete`、`decision_basis_sufficient`、`material_asymmetry_resolved`；只有模型完成实质新旧比较且关键检索失败已处理到合理边界，比较阶段才可 COMPLETE。不得把 challenger 资料更新更完整本身当作优势；CRM 有最新 Investor Day 而 MPC 的裂解价差/利润捕获率/盈利修订为 `RETRIEVAL_FAILED` 时，必须先补查 MPC，不能直接 SWITCH CRM。

`DERIVATION_REQUIRED` 应进入 `scripts/deterministic_calculations.py` 的确定性计算，保存原始输入、公式、结果、`as_of`、来源引用和 `estimate_type=DETERMINISTIC`，解决后不继续列为 unresolved gap；不可确定计算的中周期盈利等才明确标为 `MODEL_ESTIMATE`。若 refiner/energy thesis 实质依赖周期变量，研究问题可考虑 crack spread、product inventories、refinery utilization、throughput、capture rate、maintenance/outage、earnings revisions 和 mid-cycle earnings，重点区分当前利润高位与市场对持续时间预期的上修/下修；不建立固定因子模型，也不因股价上涨或利润峰值机械扣分。

## 跨轮有效首选状态

`active_selection` 是上一轮最终 KEEP / SWITCH 决策结束后仍有效的 AI 首选，不是账户实际持仓，也不一定是上一轮研究排名第一名。本轮比较对象始终为 `new_first_symbol` 与 `incoming_active_selection`。

每轮最终决策完成后，在同一输出目录的 `result.json` 与 `active_selection.json` 显式保存以下状态，后者同时记录结果文件路径：

- `incoming_active_selection`：从上轮 `outgoing_active_selection` 继承。
- `new_first_symbol`：本轮 Top 5 最终比较的第一名；原 `selected_symbol` 继续表示该排名第一名。
- `rebalance_decision`：仍只能由 当前对话模型 输出 `KEEP_PREVIOUS` 或 `SWITCH_TO_NEW_FIRST`。
- `outgoing_active_selection`：KEEP 时等于 incoming；SWITCH 时等于 new first。`active_selection` 为 outgoing 的同值别名。

`scripts/selection_state.py` 只校验并落实已经由模型选择的动作，不根据分数、置信度或固定阈值选择 KEEP / SWITCH。新旧相同仍须 KEEP。例：第一周 A；第二周 B 与 A 比较后 KEEP；第三周必须比较 C 与 A。未完成的研究、待补查或比较失败不得推进该状态。

`--previous-result` 仍指向上轮结果，优先继承 outgoing。为兼容完整的旧比较结果，可用其保存的比较双方和最终动作还原有效首选；显式状态与动作冲突时报错。首次启动或只有排名的历史结果，需要显式的初始 `active_selection` 状态（例如已确认初始选择的 COMPLETE 状态文件），不能机械拿第一名初始化。旧 `previous_first_symbol` 仅保留为 incoming 的兼容字段，不再具有“上轮排名第一名”的含义。下一轮优先读取 `active_selection_research` 中与有效首选身份一致的研究记录。

## 选择操作

默认“帮我选股/运行选股技能”的交付是下面的联合流程，不再停在单只股票推荐。用户明确只要选股、只做新旧首选比较或只看结果时遵从该范围。创建/修改技能不启动付费研究。

### 研究角色：当前对话模型（默认）

横向排名、Top 5 补充深研、对抗审查、最终排名和新旧首选比较全部由**当前对话模型在本次会话中直接完成**，不经过独立模型 API。Codex 负责只读检索、结构化校验、状态记录和报告排版。交互式交接：

```powershell
& '<项目>/.venv/Scripts/python.exe' '<技能>/scripts/session_handoff.py' prepare `
  --project '<项目>' `
  --source-result '<本轮统一研究 result.json>' `
  --previous-result '<上一轮 COMPLETE result.json>' `
  --verification-packet '<同日核验 packet.json>' `
  --output-root '<项目>/outputs/skill-research' `
  --session-model '<会话在本次运行时声明的模型名>'
```

`prepare` 只导出不可变的历史基线与新运行目录，写出 `status=WAITING_SESSION_RESEARCH`、`research_provider=CURRENT_CONVERSATION`、`model_provenance=SESSION_DECLARED_NOT_API_VERIFIED`、`api_calls=0`，不选股。当前对话模型完成研究、把决策与证据写入 JSON 后：

```powershell
& '<项目>/.venv/Scripts/python.exe' '<技能>/scripts/session_handoff.py' finalize `
  --project '<项目>' `
  --run-directory '<prepare 输出的目录>' `
  --decision '<会话产出的 decision.json>' `
  --evidence '<会话产出的 evidence packet.json>'
```

`finalize` 只校验候选/证据覆盖、证据引用、`research_provider=CURRENT_CONVERSATION` 与最终动作一致性，再写入 `result.json`、`active_selection.json`、`evidence_assessments.json`、`verification_packet.json`。它不选股、不改写动作、拒绝覆盖已有结果，也不发送订单。该入口不导入项目运行时、不读 `.env`、不打开生产库、不调用 Risk Engine 或 broker；`--session-model` 是会话声明，不是 API 模型校验。

`session_handoff.py` 的离线回归：`scripts/test_session_handoff.py --project '<项目>'`。

### Legacy API 路径（仅在显式要求时）

下面以 `CCSwitchProvider` 请求 `gpt-6-astra` 的脚本是历史路径，**不再是本技能声明的研究模型**，默认不运行。确需手动 API 复算时才执行，且必须显式说明本轮使用了 legacy API 而非当前对话模型；模型与 reasoning effort 已不再硬编码为 `gpt-6-astra`/`medium`，可用 `AI_RESEARCH_API_MODEL`、`AI_RESEARCH_API_EFFORT` 覆盖。上游持续 503 时应停止重试并回到会话内研究，不要伪造首选。

使用该路径前先运行不联网、不调用LLM的检查：

```powershell
& '<项目>/.venv/Scripts/python.exe' '<技能>/scripts/research.py' inspect --project '<项目>'
```

它显示候选来源与模型配置。不要把上一轮结果直接当作本轮研究；检查输出中的执行隔离状态和实际模型角色设置。

- **完整选股**：用户明确要求全量才运行 `full --run`。沿用实际Universe与现有Luna完整性校验，不能硬写518成功，也不能遗漏后进入研究阶段。关闭从Luna失败自动跳到全池CIO的fallback。
- **复用候选研究**：`candidates --run --decision-id <来源决策ID>`；无ID时使用最新已保存候选并报告来源/时间。不重新跑Luna，但重新获取研究数据，不宣称复用了旧证据。
- **统一研究 → Top 5补充深研（默认必做）**：full 完成 Luna 后及 candidates 复用候选时，先对全部最终候选使用相同证据标准统一研究，再选出初选 Top 5，对这5只逐一检索原文、补齐关键证据并重新比较，才能确定最终首选。Top 5 交给当前对话模型前必须建立同一检索日期的 `verification_packet`，包含 `evidence`、`as_of_basis` 和 `gap_audit`；其他分区按当前对话模型认定的关键问题组织，既有市场快照、盈利修正、财务、催化剂和同行分区可继续使用；资料不可得时保留 `UNKNOWN`，不能用模型猜测填充。`scripts/equal_depth_candidates.py --project '<项目>' --decision-id '<真实ID>' --run` 可用于逐股分析和统一初步排名；需要恢复时才加 `--resume-events '<含真实完成结果的events.jsonl>'`。脚本的 COMPLETE 只证明其结构化分析完成；其初步排名不能直接充当最终投资结论，也不能跳过 Top 5 的补充核实。

恢复前检查事件文件中的实际完成结果与候选集合。仅有 `EQUAL_DEPTH_DEEP_DIVE_REUSED` 标记不包含可恢复正文，必须追溯 manifest 指向的原始完成事件或结果文件。校验实际复用数量后才发请求；只补失败、过期或受新增证据影响的部分，不重跑已有效完成的候选。
- **上一轮有效首选比较（默认）**：使用下面的“上一轮有效首选比较来源”。最终 Top 1 完成后补齐上一轮有效首选的同等证据，并由同一次当前对话模型上下文输出 `KEEP_PREVIOUS` 或 `SWITCH_TO_NEW_FIRST`。不读取账户持仓，不调用 IBKR 插件，不生成账户目标仓位。
- **只看进度/结果**：读取独立输出目录的 `events.jsonl`、`result.json`。不要为了查看状态重新启动研究。长期监控使用产品原生heartbeat，锁定该输出目录，结束后停止监控。
- **最终委员会失败后重放**：本轮A/B/C结果和工具事件完整时，使用 `scripts/replay_stage_d.py --project '<项目>' --source '<失败输出目录>'` 做离线预检，实际重放加 `--run`。只重新请求Stage D及其有限修复，复用原始历史组合和已有证据；原先未保存的比较上下文标UNKNOWN。输出为原始委员会研究，不执行强制选股投影或交易。缺少阶段即停止，不自动重跑全池；成功后按默认流程比较上一轮有效首选，不读取插件账户。
- **指定新旧首选重评分**：用户只要求重新评估某个新候选并与已保存的上一轮有效首选比较时，使用 `scripts/recheck_selection_pair.py --project '<项目>' --source-result '<本轮已完成结果.json>' --verification-packet '<同口径核验packet.json>' --run`。该入口只向同一个 `CCSwitchProvider` 发起候选重评分和新旧比较请求；当前脚本固定输出 `CRM` 与 `MPC`，仅用于本次精确复核，不重跑Luna、不读取账户、不运行Risk Engine、不发送订单。比较动作仍只能是 `KEEP_PREVIOUS` 或 `SWITCH_TO_NEW_FIRST`。

完整命令示例：

```powershell
& '<项目>/.venv/Scripts/python.exe' '<技能>/scripts/research.py' candidates --project '<项目>' --decision-id '<真实ID>' --run
```

完成候选统一研究后，用 Top 5 辅助脚本完成五只补充深研，并将最终第一名与上一轮 COMPLETE 结果中的 `outgoing_active_selection` 比较：

```powershell
& '<项目>/.venv/Scripts/python.exe' '<技能>/scripts/top_five_deep_research.py' `
  --project '<项目>' `
  --source-result '<本轮统一研究 result.json>' `
  --verification-packet '<本轮核验 packet.json>' `
  --previous-result '<上一轮 COMPLETE result.json>' `
  --run
```

`--previous-result` 是新旧首选比较的必需输入；脚本会补查旧首选，并在研究结果中写出 `rebalance_decision`。该方向只表示证券选择层面的“换股/保留”，不读取账户持仓、不计算账户仓位、不发送订单。

`--run`才授权脚本发起研究API请求。创建/检查技能本身不等于用户要求跑一次付费全量研究。脚本启动后立即打印本轮输出目录：`outputs/skill-research/<时间-UUID>/`。保留该路径用于进度查询。

## 上一轮有效首选比较来源

1. 优先使用调用方提供的 `--previous-result <绝对路径>`。该文件必须是上一轮已完成的独立研究结果；不能把用户记忆中的股票名、旧报告标题或旧分数当作已确认的上一轮有效首选。
2. 从上一轮已完成结果解析 `outgoing_active_selection`（`active_selection` 是同值别名），不能使用上一轮 `selected_symbol` 或排名第一名替代。旧格式若有完整 `rebalance_decision` 和比较双方，可按动作还原；只有排名的旧结果不能自动初始化。要求状态为 `COMPLETE`、状态与动作一致且结果可追溯；解析失败则比较未完成，不猜测。
3. 上一轮有效首选不是实际持仓事实。此流程不读取 IBKR connector、账户余额、持仓、合约快照或 `scripts/plugin_review.py`，也不生成账户目标权重、金额或股数。
4. 对旧首选执行与本轮 Top 5 相同的证据纪律；具体研究问题由 当前对话模型 按其公司、行业和投资论点自主确定，使用的证据须保留来源、期间和口径。旧结果中的分数只能作审计记录，不能直接和新分数相减。
5. 将“本轮新第一名”和“上一轮有效首选”的同日/同口径证据放入一次当前对话模型请求，要求直接返回：`new_first_symbol`、`previous_first_symbol`、`rebalance_decision`、`alpha_gap`（无法同步比较则 UNKNOWN）、`why_new_beats_previous`、`why_keep_previous`、`evidence_refs`、`confidence`、`confidence_reducers`、`thesis_invalidation_conditions`；另在 `evidence_assessments` 中分别记录 `investment_confidence` 和 `evidence_completeness`。不输出账户仓位建议。
6. 调仓方向只允许两种：`SWITCH_TO_NEW_FIRST`（建议换到新首选）、`KEEP_PREVIOUS`（建议维持上一轮有效首选）。补查后仍有未知或冲突，按现有可用证据作出选择并说明假设、反证和改变结论的条件；不能拒绝判断，也不能程序默认维持。它表达证券选择轮换方向，不表示账户已交易。

## 默认流程：选股 → 旧首选比较 → 调仓方向

执行本流程前阅读 [主动核实与新旧首选比较](references/verification-and-sizing.md)。Codex 必须实际执行其中的只读检索，给当前对话模型提供核验后的资料；当前对话模型本身不执行联网检索，不能只在提示词中要求它“自行联网”并宣称已经核实。创建或修改技能不自动启动付费研究。

1. 按下文“分阶段完成标准”执行全部候选统一研究、初选 Top 5 的补充检索深研及最终比较。复用候选时披露来源和日期；候选过期或无候选时说明情况，不用旧结果冒充本轮研究。初选阶段完成不代表最终首选完成。
2. 最终 Top 1 确定后解析上一轮 COMPLETE 结果，识别旧首选；若新旧相同，记录 `KEEP_PREVIOUS`，仍保留本轮核验，不重复制造换股理由。
3. 对旧首选补齐与 Top 5 同等的关键证据。不能只读取价格就结束为“资料不足”；记录实际检索来源、时间、核验结果和仍不可取得的资料。若补查反证改变新旧判断，允许更新比较结论。
4. 将新第一名、旧第一名和各自带来源/日期的结构化研究资料交给同一次当前对话模型比较，明确要求回答“为什么换、为什么不换、哪些证据使差距不够”。不传账户持仓、现金、入场日、历史目标仓位或 IBKR 数据。
5. 输出 `rebalance_decision` 和方向依据。只给证券选择层面的换股/保留结论，不给账户百分比、参考金额、股数或订单指令；不运行 Risk Engine，不调用 broker。
6. 对模型提出的关键补资料要求最多实际追加两轮；无需重跑 Luna。中文报告分别展示初选 Top 5、补查记录、最终 Top 5、新旧首选比较、调仓方向及仍未解决缺口。

任何一阶段未完成，明确标记阶段未完成；不能把新旧比较失败包装成“保持上一轮”或现金建议。保留选股结果、旧首选来源、比较输出路径便于追溯。

## 新旧首选的调仓方向归属

Luna 只做广度筛选；最终首选和是否轮换必须由当前对话模型在同一次横向研究/旧首选比较上下文中直接给出。宿主只做 schema 校验、证据引用校验、分数/日期可比性检查和安全隔离，不根据分数差、置信度或固定阈值自行生成调仓结论。

`SWITCH_TO_NEW_FIRST` 是 当前对话模型 建议证券选择从旧首选转向新首选；`KEEP_PREVIOUS` 是继续以旧首选为研究基准；资料不足时先补查，仍不可得则由模型依据现有信息在两种动作中选择。新旧首选相同必须是 `KEEP_PREVIOUS`。这些状态都不是订单、成交或 Risk Engine 批准。

本技能不输出账户目标仓位、金额、股数、实际买卖清单或持仓动作。若用户日后需要账户级执行，必须另行授权并回到项目的独立对账、Risk Engine 和 OBSERVE/PAPER 安全流程。

## 模型验证

研究角色是**当前对话模型**：具体模型名由发起本技能的会话在本次运行时声明（例如本次任务为 DeepSeek-V4.1-Flash），并记录 `research_provider=CURRENT_CONVERSATION`、`model_provenance=SESSION_DECLARED_NOT_API_VERIFIED`、`reasoning_effort_effective=NOT_EXPOSED_BY_SESSION`。**不得把会话模型写成经 `/models` 验证过的 API 模型，不得再要求精确 `gpt-6-astra`，也不得声称 reasoning effort 已被服务端确认。**

仅当用户显式要求 legacy API 路径时才做 API 模型校验：通过共用 CCSwitchProvider 的 `/models` 检查实际请求的模型（默认 `gpt-6-astra`，可用 `AI_RESEARCH_API_MODEL` 覆盖），全量模式还检查当前 Luna 模型。不得按“sol”模糊匹配回旧模型。不可用就停止、报告错误并回到会话内研究；用户明确指定网关别名后才能改别名。

沿用Provider协议适配、SDK max_retries=0和显式有限重试，不另造HTTP实现。模型列表出现不等于已验证工具能力。真实研究阶段的工具和结构化验证必须通过才报告成功；会话路径没有 API 遥测，缺失的 reasoning/usage 一律写 UNAVAILABLE / NOT_EXPOSED_BY_SESSION。上游连续 503 时停止重试、如实报告技术失败，不得无限重跑整轮。

## 当前对话模型选股前主动核实

### 分阶段完成标准（当前默认流程）

1. **全部候选统一研究**：对实际最终候选集合（当前20只）以相同基础核验标准比较，实际使用的横比证据须检查日期和口径。要求 当前对话模型 返回全部候选的初选分数、排名、置信度与理由，并单独标记初选 Top 5。输出遗漏时只补做此阶段的完整比较，不能从旧分数或隐藏推理推断遗漏值。
2. **Top 5补充深研**：对初选5只全部执行额外的主动检索和原文核验，由 当前对话模型 自主确定每只股票的关键问题，针对可能改变投资判断的证据逐项补查。实际使用市场快照横比时检查日期及口径可比性；复用有效证据，保存新增来源、日期与缺口查询记录。基础研究内容重复提交一次不算补充深研。
3. **最终选优**：将补充后的5只证据放在同一次当前对话模型上下文，完成横向比较与对抗审查，输出全部5只最终分数、排名、首选、第二名、同轮模型分差及选择依据。分别保留初选/最终分数和置信度；两阶段分数不可混用。证据可使置信度升高或降低，不能设定必须提高的目标。新证据使候选不成立时可从统一研究名单补入，并完成同等补查后再排名。
4. **与上一轮有效首选比较**：最终首选确定后解析上一轮 COMPLETE 结果，补充核验上一轮有效首选，即使它不在本轮 Top 5，也须达到此次5只的核验标准。把新首选和上一轮有效首选放入同一次 当前对话模型 复核，比较未来机会、原论点是否失效、新增证据、相对风险和可验证催化剂；旧分数不能与本轮分数直接相减。
5. **调仓方向与报告**：明确 `KEEP_PREVIOUS` 或 `SWITCH_TO_NEW_FIRST`，列出新旧首选事实、相对差异及依据，并说明为什么换或不换。新首选排名第一不自动触发调仓。报告分别展示初选 Top 5、补查记录、最终 Top 5、新旧首选比较和调仓方向；不输出账户仓位、金额、股数或订单。任何一阶段未完成，明确标记阶段未完成。

以下全部候选核验规则用于第一阶段的统一基础研究；额外检索深研集中于初选 Top 5及上一轮有效首选，不再要求池中每只候选完成相同数量的追加研究。该阶段划分优先于下文旧的全候选深研措辞。现有脚本仅覆盖其中部分步骤，必须核对实际输出，不能将单个脚本 COMPLETE 宣称为整个联合流程完成。

当前对话模型不具备独立联网检索能力；不能只在提示词中写“请自行查资料”后把结果当作已核实。凡是要让当前对话模型参与候选排名、深度研究、对抗审查或最终决策，必须先由 Codex 实际执行只读检索，再把结构化、脱敏的 `verification_packet` 传给同一次当前对话模型研究上下文（仅 legacy API 路径为同一次 `CCSwitchProvider` 请求）。该要求适用于 `full` 完成 Luna 后的候选研究和 `candidates` 复用研究；只做 Luna 广度筛选时不额外深挖全部股票。

核实范围按“会不会改变排名、调仓方向或风险”排序：

1. Luna 只对实际 Universe 做广度筛选；之后对全部最终候选逐一核实并深研。默认不得因旧排名、偏好或节省 token 而将某些候选降为浅层分析；通过缓存、共享行业资料和精简证据减少重复消耗。用户明确缩小本轮范围时才按该范围执行并披露。
2. 每只最终候选及上一轮有效首选由 当前对话模型 自主选择关键研究问题。可用线索包括最新价格/已完成K线及时间戳、公司IR公告与SEC原文、收入/EPS/现金流/债务/资本开支、估值分子分母、固定期限EPS/收入一致预期修正、公司催化剂/事件风险、行业供需/价格/库存/停产、波动率、流动性、同行和基准数据。若采用 Top 5 市场快照横比，应核对日期和来源口径；财务报表可以因财年不同而使用各自最新期间，但必须标出期间，禁止把不同期间当作同口径。行业字段按行业选择适用来源，例如能源可参考EIA，宏观事件可参考官方日历；不相关字段标记 `NOT_APPLICABLE`。
3. 优先使用公司IR、SEC、交易所、政府/监管机构和官方统计机构等原始来源，其他来源由 当前对话模型 按问题自主选择并判断可靠性；搜索摘要只能作为线索，聚合数据须核对其来源和口径，不能只凭搜索摘要标 `VERIFIED`。目标价变化不能冒充EPS/收入修正，行业库存/开工率不能冒充公司实现价差，原油价格也不能直接冒充炼化利润。
4. 证据必须保存 `source_url`、`published_at`、报告期间或 `as_of`、`retrieved_at`、结构化事实、适用限制和状态（旧版 `VERIFIED`、`CONFLICT`、`UNAVAILABLE`、`NOT_APPLICABLE` 可继续读取；新 gap 另存精确 `gap_status`）。搜索过但未打开的来源是线索；访问失败、没有公开数据或无法对齐口径必须区分 `RETRIEVAL_FAILED`、`NOT_PUBLIC`、`STALE` 等状态，不得让 当前对话模型 补猜。
5. 候选之间必须尽量使用相同日期、相同口径和相同来源比较。为 Top 5 实际采用的横向比较证据保存日期、来源和口径；具体指标由 当前对话模型 选择，不要求凑齐固定字段。未对齐的回报、估值、预测期或行业指标不能形成数值 Alpha gap；无法得到同步因子暴露时，`expected_alpha_vs_spy/qqq` 应为 `UNKNOWN`，而不是把原始涨幅称为回归 Alpha。
6. 把缺口、冲突和已核实证据一并交给 当前对话模型，明确禁止虚构数字、来源、事件日期、停产状态、盈利修正和隐藏推理。资料缺失不是立即结束研究的理由：先按 `gap_audit` 实际查询，必要时更换权威来源、补做可追溯计算，并在 当前对话模型 提出具体缺口后最多追加两轮有针对性的核实；同一不可得来源不无限重复请求。补查结束后，即使仍有关键缺口或冲突，模型也必须根据现有可信证据选择维持或换股；不允许以资料不足拒绝判断。仍未知的字段应明确说明其对新旧选择和置信度的影响；不得把未知自动当成否定理由，也不得强迫模型为了避免复核失败而编造证据。置信度由 当前对话模型 综合投资判断自主给出；模型自行决定证据权重、缺口的重要性及其是否影响判断，不预设扣分、上限、目标或升降方向。

核实结束后，报告实际覆盖的候选数、每项证据状态、来源日期、冲突与未解决缺口。当前对话模型 的评分、置信度和 Alpha 仍是模型估计；只有可追溯、同步且口径一致的数据才可显示为数值，其他一律 `UNKNOWN`。

每只候选承担相同的证据核验责任；研究问题和重要信息由 当前对话模型 按公司及本轮投资论点自主决定，不设置固定研究维度清单。相同深度指相同核验标准与分析责任，不要求不同行业拥有相同指标或来源数量。公司公告/SEC链接清单与新闻摘要不能冒充已阅读的原文；未查询的字段不能标为 UNAVAILABLE。

交付前列出逐股覆盖矩阵：symbol、原文来源及日期、已核实维度、缺口查询记录、深研状态与结果路径。必须核对候选总数、完成数、Missing、Duplicate；任一候选分析失败或尚未执行必要核实则报告部分完成并恢复该部分，不宣称全候选结论完成。已实际穷尽合理来源仍未知的字段可保留 UNKNOWN 并说明影响，不要求伪造完备数据。最后把全部候选的核实证据和逐股结论交给 当前对话模型 同一次比较，保留可比数值及日期，避免摘要压缩丢失已有证据后被误判为未知；输出 Top 5、首选/第二名比较及其余候选未入选理由，不预设 MPC 或其他旧首选获胜。

## 每轮正式中文报告与决策理由留存

完整研究的交付包含同一输出目录内的 `result.json`、`events.jsonl`、`evidence_assessments.json`、`active_selection.json` 和 `report.docx`；环境支持时同步生成 `report.pdf`。`top_five_deep_research.py` 在最终比较和状态保存后自动调用 `scripts/research_report.py`。报告的 `unresolved_information_gaps` 必须按股票和 field 展示精确 `gap_status`、是否影响本轮判断、是否已补查及停止继续检索的原因；旧结果缺少这些字段时显示 `NOT_RECORDED`，不能补猜。仅初选、仅看结果或指定股票对复核不冒充完整选股报告。

会话交接路径（`scripts/session_handoff.py`）与 legacy 路径有两处差异，排版前必须处理：其一，事件文件名为 `events.json`（legacy 为 `events.jsonl`）；其二，会话校验要求 `final_ranking` 为有序列表（`require_exact` 逐行读取），而 `scripts/research_report.py` 读取的是映射形态 `final_ranking.ranking`。因此会话路径须先在同一输出目录生成报告视图（把 `final_ranking` 包装为 `{"ranking": [...]}`，原始列表另存 `final_ranking_rows` 以保留可追溯性），再对该视图执行 `scripts/research_report.py --result '<报告视图>'`；**已定稿的 `result.json` 不得改写**，它是校验通过的不可变记录。同理，会话决策文件需自带 `selection_rationale`、`supplemental_findings`、`initial_ranking`、`top_five`、`final_alpha_gap` 与 `evidence_assessments`（含 `SOL_TOP5_SUPPLEMENTAL` 与 `SOL_TOP5_FINAL_RANKING` 两个 stage），否则报告排版会缺少理由字段。会话路径的 `selection_rationale` 受 `selection_rationale.py::chinese_text` 约束：正文不得出现小写拉丁字母，技术名词须写中文或大写缩写（如 `cRPO` 写“当期合同剩余履约义务”、`iPhone` 写“苹果手机”、`Pro Max` 写“高配大屏机型”）。

指定股票对复核（`scripts/recheck_selection_pair.py`，或手工序列化的等价产物）沿用旧版 `PreviousWinnerComparison` 结构，该结构的字段含义与直觉相反，必须按语义而非按字面赋值：**`previous_first_symbol` 指“被复核的现任首选”，即 incoming active selection，不是“被换出的那一只”**。`scripts/selection_state.py::resolve_active_selection` 正是用 `previous_first_symbol` 反推 incoming；若把被换出的股票写进该字段（现任为 CRM 时写成 MPC），校验会抛 `Legacy previous_first_symbol disagrees with incoming active selection`，之后任何扫描 `outputs/skill-research/*/result.json` 的轮次都无法解析有效首选。现任为 CRM、指定比较对象为 MPC 时的正确赋值是：`incoming_active_selection`、`previous_first_symbol`、`new_first_symbol`、`outgoing_active_selection`、`active_selection` 全部为 `'CRM'`，另用显式字段（如 `challenger_symbol`／`compared_against_previous_first`）承载 `'MPC'`，并保留 `pair` 列表说明比较对象。序列化后必须自检 `resolve_active_selection(result.json)` 的返回值等于预期首选再收尾。该模式在 legacy 路径会经 `transition_fields` 写入 `active_selection.json`；手工序列化时应显式声明 `selection_written` 与是否改写状态，避免被误认为已完成状态写入。

指定股票对复核的证据包须写明来源：逐条标出本轮新取的证据与从上一包结转的证据（结转项加 `carried_from` / `carried_from_as_of_basis` 标记，包级写 `evidence_provenance`），并在结转前校验上一包的 `as_of_basis` 与本轮一致——基准日不同的证据不能混入同一包，否则两条腿不可比。结转是复用同一会话内已打开的来源，不是重新取证，报告中应如实表述。

同一逻辑轮次重新生成时不得不断新建目录：重复生成会留下多个近乎相同的 `*-crm-mpc-pair-recheck-*` 目录，后续扫描 `outputs/skill-research/*/result.json` 时极易取到被废弃的那一个。应就地覆盖既有轮次目录（本仓库脚本用 `PAIR_RECHECK_REUSE_DIR=<dir>` 环境变量实现），只保留一个可解析目录；确需保留旧目录时写入 `SUPERSEDED.md` 说明修正原因与取代它的目录名，不要删除，也不要让旧目录看起来仍然有效。

**同一基准日上先后完成完整选股研究与指定股票对复核时，应合并为一份完整报告，而不是并列交付两份。** 合并规则：采用「部分 → 章节 → 子标题」三级结构，章节编号跨两轮连续（完整选股研究占一至六，指定股票对复核接七至十三，统一结论为十四，附录不编号）；前置封面（标题、副标题，以及基准日／研究角色／报告范围／两部分结论／最终有效首选的信息表）、摘要表、目录。合并必须新增两样东西：其一，摘要表与状态链表，把两轮的进入时有效首选、本轮第一名、最终动作、结束后有效首选并排列出，使两轮结论是否一致一眼可见；其二，明确声明两轮分差口径不同、不可相加或互相替代（完整选股研究的分差是同一轮内第一名与第二名的模型分差，指定股票对复核的分差是同一基准日上两个指定标的的模型分差）。合并只做排版与再组织，正文仍须逐段来自已保存字段，不得改写或新写理由；产物写入完整选股研究那一轮的目录（`complete_report.docx`、`complete_report.pdf`、`complete_report_source.json`），source 文件记录两轮各自的 run 目录与 sha256，且不覆盖该轮已定稿的 `report.docx`。

**排版要求（阅读格式）**：正文段落首行缩进 2 字符、行距 1.5；标题分三级；摘要、同基准对比与状态链用表格，表头跨页重复；页脚居中显示「第 X 页 / 共 N 页」；每个部分另起一页。标题不得单独留在页脚——reportlab 的 `keepWithNext` 只对段落生效，标题后紧接表格或目录列表时必须显式 `KeepTogether`，否则会出现「目录」标题在上一页、目录内容在下一页的断裂。合并报告的表格与摘要由脚本从已保存字段生成，不额外调用模型。

理由必须在对应的原 当前对话模型 请求中以简体中文生成并保存，不新增事后“解释为何获胜”的请求：初选保存五只股票的核心逻辑、优于其他候选的理由，以及每只未入围候选的主要原因；Top 5 逐股补充深研保存新增正面证据、负面证据和未解决缺口；最终排名保存第一名与其余四只的比较、相对第二名优势、催化剂、主要风险、论点失效条件；最终新旧比较保存维持/换股的支持因素、实际选择的核心理由和最大风险。

`scripts/selection_rationale.py` 定义这些输出字段，附加在已有结构化请求中，不改变研究指标、股票数、排名或模型分工。`selection_rationale` 在最终结果中保存 `why_top5`、`why_not_others`、`why_final_first`、`why_not_finalists`、`why_first_over_second`、`incumbent_comparison`、`why_keep`、`why_switch` 及相应催化剂/风险/失效条件；`supplemental_findings` 保存逐股新增证据。校验理由的股票覆盖范围，不允许把五只补查股票误当成全部候选数量。

正式报告全部标题、说明、理由和动作名称均用中文，股票代码及必要来源网址保留原样；不得把英文枚举和字段名直接当作正文。报告依次说明：

1. 本轮股票池（Universe）的真实规模与记录、最终候选数量/名单和进入本轮的有效首选。上游股票池未保存时明确“来源结果未记录”，不能硬填 518 或用候选数代替；全量入口保存实际股票池快照，复用候选应保留可追溯来源。
2. 从全部最终候选选入前五名的逐股理由，以及每只其余候选未入围的主要原因。
3. 五只股票补充深研的新正面证据、负面证据、未解决缺口。
4. 最终第一名为何胜出、为何不选另外四只、相对第二名的优势、核心催化剂、主要风险、投资论点失效条件、投资判断置信程度和关键证据完整程度。
5. 本轮第一名与 incoming 的直接比较，以及为何维持或切换。
6. 本轮第一名、incoming、最终动作、outgoing、核心理由和最大风险。

报告程序只读取已保存字段并排版，不调用模型、不根据结果推测理由、不翻译后扩写。旧结果缺少当时的完整路径理由时，不补写成完整报告；先恢复并执行缺失的原研究/比较阶段，不重跑 Luna。DOCX 生成失败须明确报告交付未完成，可单独重试 `scripts/research_report.py --result '<本轮 result.json>'`（会话路径为 `--result '<报告视图>'`）；已完成决策及 active 状态保留。PDF 失败不影响研究完成，保留 DOCX 并记录失败原因。文档库使用 Codex 已配置的工作区运行时，不修改研究项目依赖；该运行时含 `python-docx`、`reportlab` 与 `pydantic`，但不含 `pyyaml`，而技能自测与 quick_validate 需要 `pyyaml`，故排版用捆绑运行时、自测用项目 `.venv`。

## 研究与报告

`investment_confidence`（旧字段 `confidence`）表示“在指定研究期限内，对当前相对选股/换股判断的主观把握”，不代表经历史校准的上涨概率。证据权重、缺口重要性、数据是否适用、差异成因和置信度全部由 当前对话模型 自主判断；不以字段完整度、日期不同或窗口不同预设扣分、上限或升降方向。模型可判断差异来自时间、方法、经营变化或真实矛盾，并依据有来源的输入作必要调和或归一化，说明重要调整。简要说明决定选择的正反证据和实质不确定性，区分事实、假设与模型估计，不虚构缺失数据。补查结束后仍须自主选择维持或换股。

保留已有A/B/C/D研究约束：同一次横向排名、Top5、第二名、评分差、为什么不选竞争者、Bull/Base/Bear、失效条件、置信度依据、预期超额依据。评分与置信度是模型估计，不是胜率/回归Alpha；数据不存在标UNKNOWN。不保存或展示隐藏思维链。

新旧首选比较要区分新增负面证据与一直缺失的数据；不得将“涨多了”或同一个未知因素反复作为机械换股理由。说明与委员会/上一轮建议相比的新增调整理由；不为了激进选股强制换股，不放宽研究安全边界。持有天数沿用项目日历日口径，是复核期限，不是机械卖出日。

最终用中文报告：模式、来源与结果时间、上一轮有效首选来源、实际股票池覆盖（若有）、候选数、Top5/首选/第二名/差距、核心比较、情景/失效条件、调仓方向依据、逐股 investment_confidence 与 evidence_completeness、关键补查及未解决缺口、研究角色与模型来源（当前对话模型会话声明，或显式指定的 legacy API）、token/latency（会话路径为 UNAVAILABLE / NOT_EXPOSED_BY_SESSION）、输出路径。固定注明：**仅研究建议；未调用Risk Engine审批；未发送/取消订单；未读取IBKR实际持仓。**

技能更新验证：使用项目Python执行 `scripts/test_session_handoff.py --project '<项目>'`（会话研究交接的离线回归，不联网、不读生产库），以及 legacy API 路径的 `scripts/test_research.py --project '<项目>'`、`scripts/test_plugin_review.py --project '<项目>'`、Top 5 比较测试、CRM/MPC 测试、`scripts/test_evidence_discipline.py --project '<项目>'` 和 `scripts/test_evidence_gap_upgrade.py`，最后跑 skill-creator 的 quick_validate（Windows中文文件使用 `python -X utf8`）。`test_plugin_review.py` 仅保留为旧的独立插件安全回归测试，不属于本技能默认流程。不要为验证技能而启动生产worker或重新跑518只。

状态与报告更新另执行 `scripts/test_selection_reporting.py --project '<项目>'`，用离线三轮状态链及固定的中文决策记录验证报告；不得为验证而运行付费选股。
