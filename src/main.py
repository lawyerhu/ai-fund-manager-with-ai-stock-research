from __future__ import annotations

import argparse

from .config import env, load_config
from .data_provider import MockDataProvider, YahooFinanceDataProvider
from .execution.ibkr import IBKRConfig, IBKRPaperExecutor
from .execution_quotes import IBKRExecutionQuoteProvider
from .execution.observe import ObserveExecutor, ReadOnlyBroker
from .execution.paper import LocalPaperExecutor
from .scheduler import PaperScheduler, SchedulerConfig
from .runner import StrategyRunner
from .service import BackendService
from .storage import SQLiteStore


def build_provider(cfg):
    provider_name = cfg.get("data", {}).get("provider", "mock").lower()
    if provider_name == "mock":
        return MockDataProvider()
    if provider_name in {"yahoo", "yahoo_finance"}:
        return YahooFinanceDataProvider()
    raise ValueError(f"Unknown data provider: {provider_name}")


def build_llm_agent(cfg, data):
    """Build the configured CC Switch pipeline without binding business code to an upstream vendor."""
    from .llm_agent import CCSwitchProvider, LLMRuntimeConfig, LunaSolPipeline

    runtime = LLMRuntimeConfig.from_mapping(cfg)
    provider = CCSwitchProvider(runtime=runtime)
    agent_cfg = cfg.get("agent", {})
    return LunaSolPipeline(
        data,
        provider,
        runtime,
        max_tool_rounds=int(agent_cfg.get("max_tool_rounds", 12)),
        max_candidates=int(agent_cfg.get("max_candidates", 25)),
    )


def _execution_selection(cfg, trading_mode=None, broker_source=None, execution_mode=None):
    execution = cfg.get("execution", {})
    legacy_mode = (trading_mode or env("TRADING_MODE") or execution.get("mode"))
    source = broker_source or env("BROKER_SOURCE") or execution.get("broker_source")
    permission = execution_mode or env("EXECUTION_MODE") or execution.get("execution_mode")

    legacy_mapping = {
        "OBSERVE": ("LOCAL", "OBSERVE"),
        "LOCAL_PAPER": ("LOCAL", "PAPER"),
        "IBKR_PAPER": ("IBKR_PAPER", "PAPER"),
    }
    if legacy_mode:
        legacy_key = str(legacy_mode).upper()
        if legacy_key not in legacy_mapping:
            raise ValueError(f"Unsupported trading mode: {legacy_key}. V1 permits only OBSERVE, LOCAL_PAPER, or IBKR_PAPER")
        legacy_source, legacy_permission = legacy_mapping[legacy_key]
        source = source or legacy_source
        permission = permission or legacy_permission

    source = (source or "LOCAL").upper()
    permission = (permission or "OBSERVE").upper()
    if source not in {"LOCAL", "IBKR_PAPER"}:
        raise ValueError(f"Unsupported broker source: {source}. V1 permits only LOCAL or IBKR_PAPER")
    if permission not in {"OBSERVE", "PAPER"}:
        raise ValueError(f"Unsupported execution mode: {permission}. V1 permits only OBSERVE or PAPER")
    return source, permission


def build_executor(cfg, store, trading_mode=None, broker_source=None, execution_mode=None):
    source, permission = _execution_selection(cfg, trading_mode, broker_source, execution_mode)
    if source == "LOCAL" and permission == "OBSERVE":
        return ObserveExecutor(starting_cash=float(cfg.get("portfolio", {}).get("starting_equity", 1000.0)))
    if source == "LOCAL" and permission == "PAPER":
        return LocalPaperExecutor(
            fractional_shares=bool(cfg.get("execution", {}).get("fractional_shares", True)),
            minimum_order_notional=float(cfg.get("execution", {}).get("minimum_order_notional", 1.0)),
            starting_cash=float(cfg.get("portfolio", {}).get("starting_equity", 1000.0)),
            store=store,
            commission_per_order=float(cfg.get("execution", {}).get("commission_per_order", 0.0)),
            commission_per_share=float(cfg.get("execution", {}).get("commission_per_share", 0.0)),
            minimum_commission=float(cfg.get("execution", {}).get("minimum_commission", 0.0)),
            slippage_bps=float(cfg.get("execution", {}).get("slippage_bps", 0.0)),
        )
    if source == "IBKR_PAPER":
        execution = cfg.get("execution", {})
        ibkr_config = IBKRConfig.from_env()
        ibkr_config.order_timeout_seconds = float(execution.get("order_timeout_seconds", 30))
        ibkr_config.quote_timeout_seconds = float(execution.get("quote_timeout_seconds", 10))
        ibkr_config.allow_delayed_quotes = bool(execution.get("allow_delayed_quotes", False))
        broker = IBKRPaperExecutor(
            ibkr_config,
            store=store,
        )
        return ReadOnlyBroker(broker) if permission == "OBSERVE" else broker
    raise ValueError(f"Unsupported broker source: {source}")


def run(use_llm: bool, config_path=None, trading_mode=None, broker_source=None, execution_mode=None):
    cfg = load_config(config_path)
    cfg.setdefault("data", {})["provider"] = env("DATA_PROVIDER", cfg["data"].get("provider", "mock"))
    cfg.setdefault("portfolio", {})["database_path"] = env("DATABASE_PATH", cfg["portfolio"].get("database_path", "data/ai_fund_manager.sqlite3"))
    data = build_provider(cfg)
    store = SQLiteStore(cfg["portfolio"]["database_path"])
    executor = build_executor(cfg, store, trading_mode, broker_source, execution_mode)
    quote_provider = IBKRExecutionQuoteProvider(executor) if getattr(executor, "mode", None) == "IBKR_PAPER" else None
    agent = None
    if use_llm:
        agent = build_llm_agent(cfg, data)
    runner = StrategyRunner(cfg, data, executor=executor, store=store, agent=agent, quote_provider=quote_provider)
    result = runner.run(use_llm=use_llm)
    print("TRADE_INTENT", result["intent"].model_dump(mode="json"))
    print("RISK_DECISION", result["risk"].model_dump(mode="json"))
    print("TRADING_STATE", result["state"])
    return result


def build_backend_service(cfg, use_llm: bool = True, store=None, trading_mode=None, broker_source=None, execution_mode=None, first_run_observe=False):
    data = build_provider(cfg)
    owned_store = store or SQLiteStore(cfg["portfolio"].get("database_path", "data/ai_fund_manager.sqlite3"))
    executor = build_executor(cfg, owned_store, trading_mode, broker_source, execution_mode)
    agent = None
    agent_error = None
    if use_llm:
        try:
            agent = build_llm_agent(cfg, data)
        except Exception as exc:
            agent_error = exc
    quote_provider = IBKRExecutionQuoteProvider(executor) if getattr(executor, "mode", None) == "IBKR_PAPER" else None
    runner = StrategyRunner(cfg, data, executor=executor, store=owned_store, agent=agent, quote_provider=quote_provider)
    first_run_requested = str(env("FIRST_RUN_MODE", "")).upper() == "FIRST_RUN_OBSERVE"
    first_run_completed = owned_store.get_runtime("first_run_state") == "OBSERVE_RESEARCH_COMPLETED"
    first_run_enabled = bool(first_run_observe) or (first_run_requested and not first_run_completed)
    service = BackendService(
        cfg,
        runner,
        owned_store,
        use_llm=use_llm,
        initialization_error=agent_error,
        first_run_observe=first_run_enabled,
    )
    scheduler_values = dict(cfg.get("scheduler") or {})
    scheduler_values["position_monitor_interval_minutes"] = int(cfg.get("monitoring", {}).get("position_check_minutes", 30))
    scheduler = PaperScheduler(
        SchedulerConfig.from_mapping(scheduler_values),
        weekly_callback=lambda: owned_store.enqueue_command("RUN_FULL_AI_RESEARCH", source="SCHEDULER", trigger_reason="WEEKLY_FULL_RESEARCH"),
        pre_execution_callback=lambda: owned_store.enqueue_command("RUN_PRE_EXECUTION_REVALIDATION", source="SCHEDULER", trigger_reason="PRE_MARKET_REVALIDATION"),
        daily_callback=lambda: owned_store.enqueue_command("RUN_DAILY_POSITION_REVIEW", source="SCHEDULER", trigger_reason="DAILY_POSITION_REVIEW"),
        risk_callback=lambda: owned_store.enqueue_command("RUN_FULL_RISK_CHECK", source="SCHEDULER", trigger_reason="15_MINUTE_FULL_RISK_CHECK"),
        entry_callback=lambda: owned_store.enqueue_command("RUN_PENDING_ENTRIES", source="SCHEDULER", trigger_reason="ENTRY_EXECUTION_MONITOR") if (
            owned_store.pending_entries(("PENDING_ENTRY",))
            or owned_store.get_runtime("pending_execution_intent")
        ) else None,
        position_callback=lambda: owned_store.enqueue_command("RUN_POSITION_MONITOR", source="SCHEDULER", trigger_reason="30_MINUTE_POSITION_MONITOR") if owned_store.active_managed_positions() else None,
        store=owned_store,
    )
    service.scheduler = scheduler
    return service


def run_service(use_llm: bool, config_path=None, trading_mode=None, broker_source=None, execution_mode=None, first_run_observe=False):
    cfg = load_config(config_path)
    cfg.setdefault("data", {})["provider"] = env("DATA_PROVIDER", cfg["data"].get("provider", "mock"))
    cfg.setdefault("portfolio", {})["database_path"] = env("DATABASE_PATH", cfg["portfolio"].get("database_path", "data/ai_fund_manager.sqlite3"))
    service = build_backend_service(
        cfg,
        use_llm=use_llm,
        trading_mode=trading_mode,
        broker_source=broker_source,
        execution_mode=execution_mode,
        first_run_observe=first_run_observe,
    )
    service.start()
    try:
        while True:
            service._stop_event.wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        service.store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fund Manager V1")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock", help="mock is offline; llm uses the configured CC Switch gateway and LLM_* settings")
    parser.add_argument("--config", default=None, help="optional YAML configuration path")
    parser.add_argument("--trading-mode", choices=["OBSERVE", "LOCAL_PAPER", "IBKR_PAPER"], default=None)
    parser.add_argument("--broker-source", choices=["LOCAL", "IBKR_PAPER"], default=None)
    parser.add_argument("--execution-mode", choices=["OBSERVE", "PAPER"], default=None)
    parser.add_argument("--first-run-observe", action="store_true", help="run the read-only first integration checks before scheduling")
    parser.add_argument("--service", action="store_true", help="run the persistent backend supervisor")
    args = parser.parse_args()
    if args.service:
        run_service(use_llm=args.mode == "llm", config_path=args.config, trading_mode=args.trading_mode, broker_source=args.broker_source, execution_mode=args.execution_mode, first_run_observe=args.first_run_observe)
    else:
        run(use_llm=args.mode == "llm", config_path=args.config, trading_mode=args.trading_mode, broker_source=args.broker_source, execution_mode=args.execution_mode)
