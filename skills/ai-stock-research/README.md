# ai-stock-research

这是本仓库附带的 Codex skill，用于按 `SKILL.md` 中的流程进行 AI 选股研究。

## 在 Codex 中使用

把本目录安装到 Codex 的 skills 目录，或从本仓库安装 `skills/ai-stock-research`。安装后新建一个任务并明确写出：

> 使用 `ai-stock-research`，仅进行研究，不发送订单。

运行脚本时，`--project` 应指向本仓库根目录的绝对路径。

## 安全边界

- `.env`、数据库、日志、缓存和研究输出不应提交到 Git。
- API Key、券商登录信息和账户信息必须由使用者在本地配置。
- 本 skill 的研究结论不是交易执行，也不替代独立的风险审批。
