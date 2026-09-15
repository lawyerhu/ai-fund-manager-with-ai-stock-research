# AI Fund Manager V1

AI research and risk-gated paper trading terminal for a single long US stock or cash. The backend runs independently of the Streamlit dashboard and stores the audit trail in SQLite.

This is an experiment, not investment advice. It does not guarantee returns or prevent losses through gaps, outages, bad data, or broker errors. Real-money trading is not implemented.

## Safety boundary

V1 has two independent selections:

- `BROKER_SOURCE=LOCAL` or `IBKR_PAPER`.
- `EXECUTION_MODE=OBSERVE` or `PAPER`.

`LOCAL + OBSERVE` records hypothetical fills, `LOCAL + PAPER` uses the persistent local simulator, and `IBKR_PAPER + OBSERVE` connects to a verified Paper account for read-only reconciliation and IBKR quotes while recording `WOULD_*` actions without broker mutation. `IBKR_PAPER + PAPER` is the separate Paper order path.

The legacy `TRADING_MODE=OBSERVE|LOCAL_PAPER|IBKR_PAPER` setting remains accepted for compatibility. `TRADING_MODE=IBKR_PAPER` selects the Paper order path unless `EXECUTION_MODE=OBSERVE` is explicitly set.

There is no `LIVE`, `IBKR_LIVE`, or `REAL_MONEY` mode. The CLI accepts only `OBSERVE`, `LOCAL_PAPER`, and `IBKR_PAPER`.

## 普通 Windows 用户首次使用

第一次只需双击 `setup.bat`。向导会自动检查 Python、建立 `.venv`、安装普通依赖、检查官方 IBKR API、检测安全 Paper 端口、生成本地 `.env`、检测 CC Switch 能力、创建诊断报告，并进行 `FIRST_RUN_OBSERVE` 只读联调。

向导不会替用户填写 IBKR 用户名、密码或 2FA，也不会关闭 Read-Only API。请在提示后手动登录 TWS 的 `PAPER TRADING` 账户，保持 `Read-Only API = ON`，并开启 `Enable ActiveX and Socket Clients`。若官方 API 或 TWS 尚未安装，向导会打开官方页面并显示下一步。

首次检查完成后，Dashboard 的 Setup / System Health 页应显示 `READY FOR OBSERVE`。首次 AI 研究必须由用户点击 `Run First AI Research (No Order)`，它只记录 `WOULD_*` 结果，不调用 broker mutation。以后只需双击 `start.bat`。遇到问题双击 `troubleshoot.bat`，然后查看 `diagnostic_report.txt`；报告不会包含 API Key、账户号、密码或 token。

`FIRST_RUN_MODE=FIRST_RUN_OBSERVE` 是一次性安全引导。首次观察研究完成并持久化为 `OBSERVE_RESEARCH_COMPLETED` 后，后续启动自动进入正常后台模式，并按配置启动 Scheduler；无需手工删除该环境变量。

## Developer start on Windows

1. Create the environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Run `start.bat`. It starts the backend service, scheduler, and Streamlit dashboard in local headless mode, then opens the browser only after the dashboard health endpoint responds:

`http://127.0.0.1:8501`

The setup wizard creates `.streamlit/config.toml` with telemetry and the first-run email prompt disabled. The Windows launcher also applies the same headless flags and creates an empty Streamlit credentials file when the user profile does not have one, so startup never waits for an email. If the dashboard does not answer within 30 seconds, the launcher prints `Dashboard startup FAILED`, shows the last 40 lines of both dashboard logs, stops the just-started backend, and does not open a browser.

`setup.bat` creates `.env` locally. The first-run defaults are `IBKR_PAPER + OBSERVE` with `DATA_PROVIDER=yahoo` and `LLM_PIPELINE=LUNA_SOL`; the wizard asks only for the local CC Switch token using hidden input and never prints it. It does not ask for or handle an upstream provider secret. `start.bat` opens the Dashboard after the backend starts.

The backend uses an OS-level singleton lock next to the SQLite database. Starting a second backend cannot create a second trading process. The dashboard is a monitor and control surface; it does not contain manual BUY or SELL inputs.

`run.ps1` starts the service with `--mode llm`. If TWS is not logged in, the backend remains visible as `WAITING_FOR_TWS_PAPER_LOGIN`; failed CC Switch capability or account checks enter `SAFE_MODE` and do not trade. Use the offline command below for a no-key demonstration.

## Offline Local Paper demonstration

Run the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe -m src.main --service --mode mock --trading-mode LOCAL_PAPER
```

Run the dashboard in another terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

For a deterministic end-to-end acceptance run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -s tests\test_local_paper_scenario.py
```

The scenario verifies CASH, BUY, HOLD without a duplicate BUY, target reduction, SWITCH with the old leg fully sold before the new leg, CASH liquidation, a later BUY, 20% drawdown reduction, 25% drawdown liquidation, and `RISK_HALTED`. The complete test suite is run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The database defaults to `data/ai_fund_manager.sqlite3`. Reusing it preserves local broker state, portfolio history, orders, peak equity, risk state, journal observations, and scheduler timestamps across restarts. Set `DATABASE_PATH` to use a separate experiment.

## Scheduler and controls

All execution clocks use `America/New_York`. The defaults run Full Research once on Monday at 10:00 ET and run an intraday Risk Cycle every 15 minutes during the NYSE regular session. Each Risk Cycle reconciles the broker, refreshes the execution quote, recalculates drawdown, evaluates the Risk Engine, and reduces or liquidates when required. The Position Monitor runs every 30 minutes during the regular session. In the Paper alpha-validation profile, a 10% position drawdown triggers a Sol emergency review without mechanically selling, a 15% drawdown triggers the deterministic hard stop through the Risk Engine, and fixed take-profit selling is disabled so winners can continue. The 16:05 ET Daily Position Review does not rerun the full universe screening.

Weekly outcomes are persisted as `COMPLETED`, `WAITING_FOR_MARKET`, or `FAILED`. A market-closed outcome does not consume the weekly execution. The pending intent is retried in the next valid session after reconciliation, a fresh quote, and a new Risk Engine evaluation. An expired intent is discarded and full AI research runs again.

TWS/IB Gateway login remains manual. The project never stores the IBKR password or bypasses 2FA. If TWS exits, the backend keeps the Command Worker alive and disables trading. After the user logs back into the Paper session, Broker Monitor automatically reconnects, completes account/position/account-wide order/execution reconciliation, and restarts Scheduler. It never automatically clears `MANUAL_HALT`, `RISK_HALTED`, or an unknown broker-order safety failure.

The Dashboard controls are queued to SQLite and handled by the backend:

- Start or stop the scheduler.
- Run reconciliation now.
- Run a risk check now.
- Run AI research now.
- Emergency Stop.
- Resume From Manual Halt.

Emergency Stop cancels outstanding BUY orders, liquidates positions, and enters `MANUAL_HALT`. It never resumes automatically. `SAFE_MODE` and `RISK_HALTED` block new AI transaction cycles. No Dashboard action bypasses the Risk Engine.

## Dashboard

The home status strip shows system, trading mode, broker source, execution permission, broker, LLM, execution market data, macro data, scheduler, risk state, and the last reconciliation and AI decision times. `IBKR_PAPER + OBSERVE` is shown as `OBSERVE — ORDERS DISABLED`. Macro status is explicit: `AVAILABLE` or `UNKNOWN — POSITION CAPPED` under the default Paper policy. Red alerts are shown for `SAFE_MODE`, `RISK_HALTED`, broker disconnects, and system errors; yellow alerts are shown for reduced risk and stale or failed market data.

The tabs expose:

- Overview: equity, cash, invested value, position details, weights, average cost, current price, market value, PnL, drawdown, peak equity, and historical maximum drawdown.
- Live AI Activity: the current `run_id`, pipeline stage status, Luna universe/candidates/rationales, Sol research tool calls and structured results, final AI decision, Risk Engine approval, and OBSERVE hypothetical execution. Prompt text and hidden model reasoning are not displayed.
- System Timeline: backend-written chronological runtime events with timestamp, run ID, decision ID, component, event type, symbol, and short description.
- Risk Engine: current drawdown, requested weight, drawdown, volatility, event, liquidity, confidence, and final approved limits, plus rejection reasons.
- AI Decisions: CC Switch gateway, `LUNA_SOL` or `SOL_ONLY` pipeline, actual Luna screening model, actual Sol Research + CIO model, per-stage status, decision ID, prompt versions, action, symbols, weights, confidence, thesis, risk factors, invalidation conditions, evidence, tool results, token usage, latency, fallback events, and estimated cost.
- Decision Journal: entry and fill prices, 1D/5D/20D stock and benchmark observations, realized alpha versus SPY and QQQ, and the structured decision rationale.
- Benchmarks: normalized Net AI Strategy, SPY Buy & Hold, and QQQ Buy & Hold series; Gross Strategy Return, Trading Costs, LLM Costs, Net Strategy Return, SPY Return, QQQ Return, and Net Excess Return versus each benchmark. `Net Excess vs QQQ` is the primary relative result. Excess return is not labeled statistical regression alpha.
- Orders: requested and filled quantities, average fill price, status, timestamps, broker order ID, IBKR permId, and rejection message.
- Errors: timestamp, severity, component, message, decision ID, and order ID.

API keys and broker credentials are never rendered. The Dashboard only exposes configured or missing status through backend runtime state.

The Dashboard uses Streamlit native fragments instead of browser reloads: system status refreshes every 3 seconds by default, Live AI Activity and System Timeline every 2 seconds, portfolio/risk every 4 seconds, and performance every 15 seconds. `DASHBOARD_REFRESH_SECONDS` controls the system-status interval. Fragments open SQLite read-only and only reread backend state; `Refresh Now` is a user action and does not call the LLM, broker, command queue, or order functions. The Dashboard remains bound to `127.0.0.1:8501` by the Windows launcher.

## Configuration

The main configuration is in `config.yaml`; environment variables override the database, dashboard refresh interval, data provider, broker source, execution permission, legacy trading mode, CC Switch gateway, Luna/Sol model IDs and reasoning efforts, pipeline, LLM protocol, IBKR connection, and optional model cost rates. The runtime reads `DASHBOARD_REFRESH_SECONDS`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_LUNA_MODEL`, `LLM_SOL_MODEL`, `LLM_LUNA_REASONING_EFFORT`, `LLM_SOL_REASONING_EFFORT`, `LLM_PIPELINE`, `LLM_API_PROTOCOL`, `LLM_PROTOCOL_FALLBACK`, and `LLM_TIMEOUT_SECONDS`. Setup persists the protocol actually validated by the capability test, including an explicit `CHAT_COMPLETIONS` fallback when needed. Keep `.env` local and never commit it.

## LLM gateway and pipeline

CC Switch is the configured API gateway. The compatibility client first uses the OpenAI Responses API shape and, for a 404/502/unsupported Responses route, performs bounded retries before an explicit `API_PROTOCOL_FALLBACK` to Chat Completions. When a root gateway URL is configured, the Chat adapter also checks its `/v1` API route; it never targets `api.openai.com` implicitly. `LUNA_SOL` sends the compact deduplicated S&P 500/Nasdaq-100 snapshot to Luna for candidate screening, then gives candidate symbols to Sol, which independently uses the raw research tools and makes the only final `BUY`, `HOLD`, `SWITCH`, or `CASH` decision. `SOL_ONLY` skips screening. A configured Luna failure may fall back to Sol only with an explicit `PIPELINE_FALLBACK` event; a Sol failure creates no new `TradeIntent`.

Each completed decision stores pipeline metadata, candidate symbols, Sol tool calls and evidence, model IDs, prompt versions, per-stage token usage, latency, cost, and fallback events in SQLite. A change to the gateway/model signature is recorded as `MODEL_CHANGE`. Structured output is validated locally with Pydantic and has only a limited repair retry that includes the validation error and complete schema before failing closed. Setup diagnostics report the gateway, model discovery, Luna basic/structured, Sol basic/JSON/tool/decision, protocol, reasoning metadata, and token usage statuses separately; unavailable telemetry is non-fatal.

Risk controls include stale data, missing price, trading halt, liquidity, abnormal gap, maximum volatility, earnings and macro-event caps, drawdown tiers, and a hard 25% drawdown halt. At 20% drawdown, existing positions are reduced to the configured cap. At or above 25%, outstanding BUY orders are cancelled, positions are liquidated, the result is reconciled, and `RISK_HALTED` is persisted. The risk engine, not the LLM, owns these decisions.

Research and execution data are separate contracts. Yahoo/Wikipedia supplies universe, daily history, fundamentals, earnings, revisions, news, SEC data, and benchmarks. In `IBKR_PAPER`, IBKR supplies the final bid, ask, midpoint/last price, broker timestamp, and market-data type used for valuation, target delta, and Risk Engine evaluation. A Yahoo daily bar timestamp is never treated as a fresh execution quote.

`MacroEventProvider` is replaceable and is designed for FOMC, CPI, PCE, and NFP events. V1 does not bundle a provider claimed to be reliable. The Paper default is therefore `macro_unknown_policy: cap` with `macro_unknown_max_weight: 0.50`; the Dashboard never silently treats missing macro data as available.

## IBKR Paper setup

1. Download the official TWS API that matches the installed TWS/Gateway. Do not install `ibapi` from PyPI.
2. Install the downloaded official Python client into this project's `.venv`, then run `.\.venv\Scripts\python.exe -m src.ibkr_diagnostics`.
3. Log in to an IBKR Paper account in TWS or IB Gateway and enable API socket clients. Keep TWS `Read-Only API` enabled for the first integration run.
4. Use the Paper port configured in TWS API Settings, recommended `7947` for TWS or `4002` for Gateway. The adapter also accepts a user-configured legacy Paper port such as `7497`; it rejects live ports `7496`, `7946`, and `4001`.
5. Set `BROKER_SOURCE=IBKR_PAPER`, `EXECUTION_MODE=OBSERVE`, `DATA_PROVIDER=yahoo`, `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID` in `.env`.
6. Start the backend and confirm account reconciliation, fresh IBKR quote status, and `OBSERVE — ORDERS DISABLED` in the Dashboard. This mode does not place or cancel broker orders.

Use a dedicated Paper account for this experiment and do not trade it manually or from another application at the same time. The adapter rejects common live ports `7496`, `7946`, and `4001`, requires exactly one account whose identifier starts with `DU`, blocks orders until the complete broker snapshot is known, and maps client order IDs, broker order IDs, and permIds for restart-safe idempotency.

Reconciliation uses `reqAllOpenOrders`, so an active order from another API client that is not known to this system causes `SAFE_MODE` with `UNKNOWN BROKER OPEN ORDER`. IBKR does not guarantee that ordinary `reqAllOpenOrders` exposes every manually entered TWS order unless it has been bound through the applicable client-0 workflow. V1 does not bind or control those manual orders. This is why a dedicated Paper account with no simultaneous manual trading is a safety requirement; the adapter fails closed for every order it can observe.

`reqAccountSummary` and `reqPositions` are subscriptions and are cancelled after each completed or timed-out reconciliation snapshot. Account-wide open orders and executions are callback-complete snapshot requests. Disconnect, error 1100, and reconnect messages 1101/1102 discard broker readiness. A new socket handshake is not enough: account, positions, account-wide open orders, and executions must all reconcile again before quotes or orders are allowed.

Execution quotes use short-lived IBKR streaming requests, because snapshot requests cannot be combined with generic ticks 165/233. The request is cancelled as soon as bid+ask or last, a broker timestamp, and the market-data type are known; it does not wait for `tickSnapshotEnd`. `allow_delayed_quotes: false` is the default and rejects delayed data. Set it to `true` only when delayed Paper data is knowingly acceptable; it remains labeled `DELAYED` and must still pass the configured freshness limit. Errors 354/414, missing price, missing timestamp, and timeout all fail closed and cancel the request.

Every target symbol is resolved through `reqContractDetails` before market data or order submission. Only one verified SMART-tradable, US-listed `STK` contract in USD with a positive conId is accepted and cached. Share classes use explicit identities such as canonical `BRK.B`, Yahoo `BRK-B`, and the verified IBKR contract/localSymbol; generic string replacement is not used. Error 200, no match, or multiple eligible matches blocks both quote and order.

IBKR errors are persisted with request/order ID, broker error time, local receive time, error code, message, advanced reject JSON, category, severity, and timestamp. Order error 201 becomes a linked rejected Execution Report. Connection-farm information messages remain informational; error 1100 or fatal broker uncertainty places the runtime in `SAFE_MODE`. Reconnect statuses 1101/1102 invalidate broker readiness and disable execution until a full reconciliation succeeds. Any intraday Risk Cycle exception also enters `SAFE_MODE`, disables new AI execution, and stops scheduled trading tasks.

IBKR Paper still requires an actual TWS/Gateway session and end-to-end validation of callback timing, market data availability, partial fills, cancellation races, reconnects, account currencies, and broker-specific fields. The Yahoo/Wikipedia provider is unofficial and may be stale, rate-limited, or incomplete. Do not treat the adapter as production-ready solely because deterministic tests pass.

## Known limitations

- Mock data and mock decisions are for plumbing tests, not alpha evaluation.
- Yahoo Finance and Wikipedia are low-cost sources with variable reliability; a supervised production data source is still needed.
- Yahoo mode has no reliable macro-calendar provider by default. Paper therefore caps exposure at 50% while macro status is unknown; `reject_new` and `allow` remain explicit alternatives for controlled tests.
- Market execution is gated by the NYSE regular session when `execution.enforce_market_hours: true`. Decisions and quotes also have independent freshness limits.
- Dashboard performance uses one official daily observation keyed by the last common SPY/QQQ trading date. AI NAV, SPY, and QQQ therefore share exactly one date; weekends and US market holidays only update the preceding trading date and cannot create fake observations.
- `LOCAL_PAPER` cost defaults are simulation assumptions: $0.005 per share, $1.00 minimum commission, and 5 bps slippage. IBKR uses broker-reported commission when available. Net Portfolio Equity and Net Excess returns include trading costs; LLM API cost is reported separately because it is not debited from the brokerage account.
- Operational reconciliation snapshots are separate from daily performance observations; repeated same-day cycles update one date rather than creating extra trading days.
- A market gap can exceed any pre-trade drawdown limit. The 25% rule is a forced response, not a loss guarantee.
- Forward Paper Trading is required to evaluate alpha; historical LLM backtesting is intentionally not included.

## Project layout

`src/service.py` owns the long-lived backend and SAFE_MODE gate. `src/first_run.py` owns FIRST_RUN_OBSERVE and health checks. `src/setup_wizard.py` owns safe local configuration and redaction. `src/scheduler.py` owns configured daily and weekly timing. `src/runner.py` coordinates research, Risk Engine, execution, reconciliation, journal, and state events. `src/storage.py` owns SQLite persistence. `dashboard.py` is the local Streamlit monitor. `setup.bat`, `start.bat`, and `troubleshoot.bat` are the ordinary Windows entry points.
