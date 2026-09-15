from pathlib import Path

from src import config as project_config
from src.first_run import FirstRunResult, HealthCheck, local_diagnostic_checks, persist_first_run_result, run_first_run_observe
from src.models import BrokerSnapshot
from src.setup_wizard import (
    DEFAULT_MODEL,
    LIVE_PORTS,
    SAFE_PAPER_PORTS,
    discover_paper_port,
    is_paper_account,
    persist_detected_llm_config,
    redact_sensitive,
    validate_paper_port,
    write_diagnostic_report,
    write_observe_env,
)
from src.storage import SQLiteStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_env_loading_does_not_depend_on_process_working_directory(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=ccswitch\nLLM_PIPELINE=LUNA_SOL\n", encoding="utf-8")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PIPELINE", raising=False)
    (tmp_path / "nested").mkdir()
    monkeypatch.chdir(tmp_path / "nested")

    project_config.load_project_env(env_path, override=True)

    assert project_config.env("LLM_PROVIDER") == "ccswitch"
    assert project_config.env("LLM_PIPELINE") == "LUNA_SOL"


def test_setup_files_are_one_click_windows_entrypoints():
    for name in ("setup.bat", "setup.ps1", "troubleshoot.bat", "troubleshoot.ps1"):
        assert (PROJECT_ROOT / name).is_file()
    assert "setup.ps1" in (PROJECT_ROOT / "setup.bat").read_text(encoding="utf-8-sig")
    assert "troubleshoot.ps1" in (PROJECT_ROOT / "troubleshoot.bat").read_text(encoding="utf-8-sig")
    setup_script = (PROJECT_ROOT / "setup.ps1").read_text(encoding="utf-8-sig")
    assert "persist_detected_llm_config" in setup_script
    assert "AI_FUND_DETECTED_LUNA_MODEL" in setup_script
    assert "AI_FUND_DETECTED_API_PROTOCOL" in setup_script
    assert '$executionMode = Get-DotEnvValue "EXECUTION_MODE"' in setup_script
    assert 'if ($executionMode -ne "PAPER") { $executionMode = "OBSERVE" }' in setup_script
    assert '("EXECUTION_MODE=" + $executionMode)' in setup_script


def test_setup_configuration_forces_ibkr_paper_observe(tmp_path):
    env_path = tmp_path / ".env"

    write_observe_env(env_path, port=7947, api_key="sk-test-secret")

    content = env_path.read_text(encoding="utf-8")
    assert "BROKER_SOURCE=IBKR_PAPER" in content
    assert "EXECUTION_MODE=OBSERVE" in content
    assert "IBKR_HOST=127.0.0.1" in content
    assert "IBKR_PORT=7947" in content
    assert "DATA_PROVIDER=yahoo" in content
    assert "FIRST_RUN_MODE=FIRST_RUN_OBSERVE" in content
    assert f"OPENAI_MODEL={DEFAULT_MODEL}" in content
    assert "OPENAI_API_KEY=sk-test-secret" in content


def test_setup_defaults_to_long_enough_llm_timeout_for_ccswitch(tmp_path):
    env_path = tmp_path / ".env"

    write_observe_env(env_path, port=7947)

    assert "LLM_TIMEOUT_SECONDS=300" in env_path.read_text(encoding="utf-8")


def test_setup_persists_detected_model_ids_without_replacing_existing_secret(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_BASE_URL=http://127.0.0.1:18080\n"
        "LLM_API_KEY=existing-secret\n"
        "LLM_LUNA_MODEL=gpt-5.6-luna\n"
        "LLM_SOL_MODEL=gpt-5.6-sol\n"
        "UNRELATED_SETTING=keep\n",
        encoding="utf-8",
    )

    persist_detected_llm_config(
        env_path,
        luna_model="provider-luna",
        sol_model="provider-sol",
    )

    content = env_path.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=ccswitch" in content
    assert "LLM_LUNA_MODEL=provider-luna" in content
    assert "LLM_SOL_MODEL=provider-sol" in content
    assert "LLM_PIPELINE=LUNA_SOL" in content
    assert "LLM_BASE_URL=http://127.0.0.1:18080" in content
    assert "LLM_API_KEY=existing-secret" in content
    assert "UNRELATED_SETTING=keep" in content


def test_paper_port_discovery_skips_live_ports_and_returns_first_open():
    attempts = []

    class Connection:
        def close(self):
            pass

    def connector(address, timeout):
        attempts.append(address[1])
        if address[1] == 7497:
            return Connection()
        raise OSError("closed")

    assert discover_paper_port(ports=(7946, 7947, 7497), connector=connector) == 7497
    assert attempts == [7947, 7497]
    assert set(LIVE_PORTS) == {7496, 7946, 4001}
    assert tuple(SAFE_PAPER_PORTS) == (7947, 7497, 4002)


def test_live_ports_are_rejected_even_when_supplied_to_setup():
    for port in LIVE_PORTS:
        try:
            validate_paper_port(port)
        except ValueError:
            pass
        else:
            raise AssertionError(f"live port {port} was accepted")


def test_du_account_detection_is_strict():
    assert is_paper_account(["DU123456"]) is True
    assert is_paper_account(["U123456"]) is False
    assert is_paper_account(["DU123456", "DU654321"]) is False
    assert is_paper_account([]) is False


def test_troubleshooter_redacts_api_keys_and_account_identifiers(tmp_path):
    report_path = tmp_path / "diagnostic_report.txt"
    write_diagnostic_report(
        report_path,
        [{"name": "OpenAI", "status": "PASS", "message": "OPENAI_API_KEY=sk-live-secret account=DU123456"}],
    )

    report = report_path.read_text(encoding="utf-8")
    assert "sk-live-secret" not in report
    assert "DU123456" not in report
    assert "REDACTED" in report
    assert redact_sensitive("token=abc123 password=hello") == "token=[REDACTED] password=[REDACTED]"


class _ObserveBroker:
    mode = "IBKR_PAPER"
    broker_source = "IBKR_PAPER"
    execution_mode = "OBSERVE"
    mutations_allowed = False
    connected = False

    def connect(self):
        self.connected = True
        return {"connected": True}

    def position_symbols(self):
        return []

    def reconcile(self, prices=None):
        return BrokerSnapshot(equity=1000, cash=1000, positions=[], open_orders=[], source="IBKR_PAPER")

    def account_summary(self):
        return {"NetLiquidation": "1000", "TotalCashValue": "1000"}

    def resolve_instrument(self, symbol):
        return {"symbol": symbol}

    def get_execution_quote(self, symbol):
        return {
            "symbol": symbol,
            "price": 100.0,
            "bid": 99.0,
            "ask": 101.0,
            "mid": 100.0,
            "timestamp": "2026-08-30T14:00:00+00:00",
            "data_type": "REALTIME",
            "source": "IBKR",
        }

    def submit_order(self, request):
        raise AssertionError("FIRST_RUN_OBSERVE must not submit orders")

    def cancel_order(self, client_order_id):
        raise AssertionError("FIRST_RUN_OBSERVE must not cancel orders")


def test_first_run_observe_reads_state_quote_and_reports_ready(tmp_path):
    result = run_first_run_observe(
        _ObserveBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "first-run.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert isinstance(result, FirstRunResult)
    assert result.ready is True
    assert result.state == "READY_FOR_OBSERVE"
    assert result.check("Broker Mutation") == "PASS"
    assert result.check("Market Data") == "PASS"
    assert result.check("LLM API") == "PASS"


def test_first_run_observe_uses_market_data_probe_and_reports_frozen_quote(tmp_path):
    class ProbeBroker(_ObserveBroker):
        def probe_market_data(self, symbol):
            return {
                "symbol": symbol,
                "probes": [
                    {"requested_type": 1, "callback_market_data_type": 1, "result": "NO_CURRENT_TICK", "received_tick": True},
                    {"requested_type": 2, "callback_market_data_type": 2, "data_type": "FROZEN", "result": "PASS", "bid": 769.28, "ask": 769.42, "last": 769.33, "close": 769.33, "timestamp": "2026-08-28T20:00:00+00:00", "received_tick": True},
                ],
                "selected": {"requested_type": 2, "callback_market_data_type": 2, "data_type": "FROZEN", "result": "PASS", "bid": 769.28, "ask": 769.42, "last": 769.33, "close": 769.33, "timestamp": "2026-08-28T20:00:00+00:00", "received_tick": True},
                "final_state": "MARKET_CLOSED / FROZEN",
            }

    result = run_first_run_observe(
        ProbeBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "probe.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.ready is True
    assert result.details["market_data_state"] == "MARKET_CLOSED"
    assert result.details["market_data_probe"]["selected"]["data_type"] == "FROZEN"
    assert result.check("Market Data") == "WARN"


def test_first_run_probe_preserves_all_market_data_errors_and_fails_closed(tmp_path):
    class FailedProbeBroker(_ObserveBroker):
        def probe_market_data(self, symbol):
            return {
                "symbol": symbol,
                "probes": [
                    {"requested_type": 1, "result": "ERROR", "error_code": 354, "error_message": "Requested market data is not subscribed"},
                    {"requested_type": 2, "result": "ERROR", "error_code": 414, "error_message": "Snapshot market data subscription is not applicable to generic ticks"},
                ],
                "selected": None,
                "final_state": "UNAVAILABLE",
            }

    result = run_first_run_observe(
        FailedProbeBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "failed-probe.sqlite3",
    )

    assert result.state == "SAFE_MODE"
    assert result.details["market_data_state"] == "PERMISSION_DENIED"
    assert result.details["market_data_probe"]["probes"][1]["error_code"] == 414
    assert result.check("Market Data") == "FAIL"


class _FixedMarketClock:
    def __init__(self, is_open):
        self._is_open = is_open

    def is_open(self):
        return self._is_open

    def latest_completed_session(self):
        return {"date": "2026-08-28", "close": "2026-08-28T20:00:00+00:00"}


class _ClosedNoTickBroker(_ObserveBroker):
    def __init__(self, bar_date="2026-08-28"):
        self.bar_date = bar_date
        self.historical_calls = 0

    def probe_market_data(self, symbol):
        return {
            "symbol": symbol,
            "probes": [{"requested_type": 3, "callback_market_data_type": 3, "data_type": "DELAYED", "result": "NO_CURRENT_TICK", "received_tick": False}],
            "selected": None,
            "final_state": "UNAVAILABLE",
        }

    def get_historical_fallback(self, symbol):
        self.historical_calls += 1
        return {"symbol": symbol, "bar_date": self.bar_date, "close": 769.33, "source": "IBKR historical"}


def test_first_run_closed_market_uses_recent_historical_fallback(tmp_path):
    broker = _ClosedNoTickBroker()

    result = run_first_run_observe(
        broker,
        llm_check=lambda: True,
        database_path=tmp_path / "closed-historical.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.ready is True
    assert result.details["market_session"] == "CLOSED"
    assert result.details["streaming_quote"] == "NO_CURRENT_TICK"
    assert result.details["historical_fallback"]["result"] == "PASS"
    assert result.details["historical_fallback"]["last_close"] == 769.33
    assert result.details["historical_fallback"]["last_bar_time"] == "2026-08-28T20:00:00+00:00"
    assert result.details["market_data_state"] == "MARKET_CLOSED"
    assert result.details["quote"]["source"] == "IBKR historical"
    assert broker.historical_calls == 1
    assert result.check("Market Data") == "WARN"


def test_first_run_open_market_does_not_use_historical_fallback(tmp_path):
    broker = _ClosedNoTickBroker()

    result = run_first_run_observe(
        broker,
        llm_check=lambda: True,
        database_path=tmp_path / "open-no-tick.sqlite3",
        market_clock=_FixedMarketClock(True),
    )

    assert result.state == "SAFE_MODE"
    assert result.details["market_session"] == "OPEN"
    assert result.details["market_data_state"] == "UNAVAILABLE"
    assert broker.historical_calls == 0


def test_first_run_closed_market_rejects_stale_historical_fallback(tmp_path):
    broker = _ClosedNoTickBroker(bar_date="2026-08-27")

    result = run_first_run_observe(
        broker,
        llm_check=lambda: True,
        database_path=tmp_path / "stale-historical.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.state == "SAFE_MODE"
    assert result.details["historical_fallback"]["result"] == "FAIL"
    assert result.details["market_data_state"] == "UNAVAILABLE"


def test_diagnostic_report_includes_market_data_probe_details(tmp_path):
    report_path = tmp_path / "diagnostic_report.txt"
    from src.first_run import format_market_data_probe_report

    write_diagnostic_report(
        report_path,
        [{"name": "Market Data", "status": "WARN", "message": "FROZEN"}],
        details=format_market_data_probe_report({
            "symbol": "SPY",
            "probes": [{"requested_type": 1, "callback_market_data_type": 1, "bid": -1, "ask": -1, "last": -1, "close": -1, "received_tick": True, "error_code": None, "error_message": "", "warning_code": 2186, "warning_message": "Delayed fallback", "elapsed_ms": 12.3, "result": "NO_CURRENT_TICK"}],
            "final_state": "MARKET_CLOSED / FROZEN",
        }),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "SPY" in report
    assert "requested_type=1" in report
    assert "callback_type=1" in report
    assert "bid=-1" in report
    assert "result=NO_CURRENT_TICK" in report
    assert "warning_code=2186" in report
    assert "FINAL MARKET DATA STATE: MARKET_CLOSED / FROZEN" in report


def test_diagnostic_report_includes_closed_session_historical_fallback():
    from src.first_run import format_market_data_probe_report

    report = format_market_data_probe_report({
        "symbol": "SPY",
        "probes": [{"requested_type": 3, "callback_market_data_type": 3, "result": "NO_CURRENT_TICK"}],
        "market_session": "CLOSED",
        "streaming_quote": "NO_CURRENT_TICK",
        "historical_fallback": {"result": "PASS", "last_close": 769.33, "last_bar_time": "2026-08-28T20:00:00+00:00"},
        "final_state": "MARKET_CLOSED",
    })

    assert "Market Session: CLOSED" in report
    assert "Streaming Quote: NO_CURRENT_TICK" in report
    assert "Historical Fallback: PASS" in report
    assert "Last Close: 769.33" in report
    assert "Last Bar Time: 2026-08-28T20:00:00+00:00" in report
    assert "FINAL MARKET DATA STATE: MARKET_CLOSED" in report


def test_first_run_observe_allows_delayed_quote_as_health_warning(tmp_path):
    class DelayedBroker(_ObserveBroker):
        def get_execution_quote(self, symbol, allow_delayed=False):
            return {
                "symbol": symbol,
                "price": 100.0,
                "timestamp": "2026-08-30T14:00:00+00:00",
                "data_type": "DELAYED",
                "source": "IBKR",
            }

    result = run_first_run_observe(
        DelayedBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "delayed.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.ready is True
    assert result.check("Market Data") == "WARN"


def test_first_run_observe_allows_closed_frozen_quote_without_safe_mode(tmp_path):
    class ClosedFrozenBroker(_ObserveBroker):
        def get_execution_quote(self, symbol, allow_delayed=False):
            return {
                "symbol": symbol,
                "price": 100.0,
                "timestamp": "2026-08-28T20:00:00+00:00",
                "data_type": "FROZEN",
                "market_status": "CLOSED",
                "source": "IBKR",
            }

    result = run_first_run_observe(
        ClosedFrozenBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "closed-frozen.sqlite3",
        market_clock=_FixedMarketClock(False),
    )

    assert result.ready is True
    assert result.state == "READY_FOR_OBSERVE"
    assert result.details["market_data_state"] == "MARKET_CLOSED"
    assert result.details["quote"]["data_type"] == "FROZEN"
    assert result.check("Market Data") == "WARN"


def test_first_run_observe_permission_denied_enters_safe_mode(tmp_path):
    class PermissionDeniedBroker(_ObserveBroker):
        def get_execution_quote(self, symbol, allow_delayed=False):
            raise RuntimeError("IBKR error 354: market data is not subscribed")

    result = run_first_run_observe(
        PermissionDeniedBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "permission-denied.sqlite3",
    )

    assert result.state == "SAFE_MODE"
    assert result.details["market_data_state"] == "PERMISSION_DENIED"
    assert result.check("Market Data") == "FAIL"


def test_first_run_observe_unavailable_quote_enters_safe_mode(tmp_path):
    class UnavailableBroker(_ObserveBroker):
        def get_execution_quote(self, symbol, allow_delayed=False):
            raise TimeoutError("IBKR execution quote timed out")

    result = run_first_run_observe(
        UnavailableBroker(),
        llm_check=lambda: True,
        database_path=tmp_path / "unavailable.sqlite3",
    )

    assert result.state == "SAFE_MODE"
    assert result.details["market_data_state"] == "UNAVAILABLE"
    assert result.check("Market Data") == "FAIL"


def test_first_run_waits_for_tws_without_starting_scheduler_or_showing_stack():
    class OfflineBroker(_ObserveBroker):
        def connect(self):
            raise ConnectionRefusedError("socket refused")

    result = run_first_run_observe(OfflineBroker(), llm_check=lambda: True)

    assert result.ready is False
    assert result.state == "WAITING_FOR_TWS_PAPER_LOGIN"
    assert result.check("TWS") == "WARN"
    assert "ConnectionRefusedError" not in result.user_message
    assert "WAITING FOR TWS PAPER LOGIN" in result.user_message


def test_failed_first_run_result_persists_safe_mode_and_disables_scheduler(tmp_path):
    result = FirstRunResult(
        "SAFE_MODE",
        [HealthCheck("IBKR API", "FAIL", "missing"), HealthCheck("Broker Mutation", "PASS")],
        "SAFE_MODE. Setup checks failed; trading is disabled.",
        {},
    )
    with SQLiteStore(tmp_path / "diagnostics.sqlite3") as store:
        persist_first_run_result(store, result)

        assert store.get_runtime("service_status") == "SAFE_MODE"
        assert store.get_runtime("safe_mode") is True
        assert store.get_runtime("trading_enabled") is False
        assert store.get_runtime("scheduler_status") == "STOPPED"


def test_persisted_llm_runtime_separates_gateway_capability_and_agent_states(tmp_path):
    result = FirstRunResult(
        "SAFE_MODE",
        [
            HealthCheck("CC Switch Gateway", "PASS"),
            HealthCheck("Luna Basic Call", "PASS"),
            HealthCheck("Sol Basic Call", "PASS"),
            HealthCheck("LLM API", "FAIL"),
        ],
        "capabilities incomplete",
        {"llm_capabilities": {"luna_model": "provider-luna", "sol_model": "provider-sol"}},
    )

    with SQLiteStore(tmp_path / "llm-runtime.sqlite3") as store:
        persist_first_run_result(store, result)

        assert store.get_runtime("llm_gateway_status") == "PASS"
        assert store.get_runtime("llm_capability_status") == "FAIL"
        assert store.get_runtime("llm_luna_status") == "ONLINE"
        assert store.get_runtime("llm_sol_status") == "ONLINE"
        assert store.get_runtime("llm_luna_model") == "provider-luna"
        assert store.get_runtime("llm_sol_model") == "provider-sol"


def test_troubleshooter_reports_required_local_checks_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    checks = {item["name"]: item for item in local_diagnostic_checks(tmp_path)}

    assert {"Python", "Dependencies", "IBKR API", "TWS", "Socket", "LLM API", "Database", "Risk Engine", "Execution Mode", "Broker Mutation", "Process Lock"} <= set(checks)
    assert checks["Execution Mode"]["status"] == "PASS"
    assert checks["Broker Mutation"]["status"] == "PASS"
    assert checks["LLM API"]["status"] == "WARN"
