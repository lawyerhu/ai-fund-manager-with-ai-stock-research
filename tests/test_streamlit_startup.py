import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _powershell_quote(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def test_project_streamlit_config_is_local_headless_and_non_interactive():
    import tomllib

    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["server"]["headless"] is True
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["port"] == 8501
    assert config["server"]["showEmailPrompt"] is False
    assert config["browser"]["gatherUsageStats"] is False


def test_streamlit_starts_without_email_prompt_in_a_fresh_user_directory(tmp_path):
    pytest.importorskip("streamlit")
    port = _free_local_port()
    app_root = tmp_path / "project"
    app_root.mkdir()
    (app_root / ".streamlit").mkdir()
    config_text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "port = 8501", f"port = {port}"
    )
    (app_root / ".streamlit" / "config.toml").write_text(config_text, encoding="utf-8")
    (app_root / "app.py").write_text(
        "import streamlit as st\n\nst.write('ready')\n", encoding="utf-8"
    )

    fresh_home = tmp_path / "fresh-home"
    fresh_home.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(fresh_home),
            "USERPROFILE": str(fresh_home),
            "HOMEDRIVE": "",
            "HOMEPATH": "",
            "PYTHONFAULTHANDLER": "0",
        }
    )
    stdout_path = app_root / "streamlit.stdout.log"
    stderr_path = app_root / "streamlit.stderr.log"
    response_seen = False
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            cwd=app_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                response_seen = probe.connect_ex(("127.0.0.1", port)) == 0
            if response_seen:
                break
            if process.poll() is not None:
                break
            time.sleep(0.1)

        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    output = f"{stdout_path.read_text(encoding='utf-8')}\n{stderr_path.read_text(encoding='utf-8')}"
    assert response_seen, output
    assert "Welcome to Streamlit!" not in output
    assert "Email:" not in output


def test_streamlit_bootstrap_creates_project_config_and_empty_credentials(tmp_path):
    if os.name != "nt":
        pytest.skip("the bootstrap helper is a Windows launcher boundary")
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is required for the Windows launcher test")

    project_root = tmp_path / "project"
    user_home = tmp_path / "user-home"
    project_root.mkdir()
    user_home.mkdir()
    helper = PROJECT_ROOT / "streamlit_bootstrap.ps1"
    command = (
        f". {_powershell_quote(helper)}; "
        f"Initialize-StreamlitHeadless -ProjectRoot {_powershell_quote(project_root)} "
        f"-UserHome {_powershell_quote(user_home)}"
    )
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    config_path = project_root / ".streamlit" / "config.toml"
    credentials_path = user_home / ".streamlit" / "credentials.toml"
    assert config_path.is_file()
    assert credentials_path.read_text(encoding="utf-8-sig").strip() == '[general]\nemail = ""'


def test_windows_launcher_forces_headless_and_gates_browser_on_health_check():
    run_script = (PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8-sig")
    setup_script = (PROJECT_ROOT / "setup.ps1").read_text(encoding="utf-8-sig")
    start_script = (PROJECT_ROOT / "start.bat").read_text(encoding="utf-8-sig")

    assert "streamlit_bootstrap.ps1" in setup_script
    assert "Initialize-StreamlitHeadless -ProjectRoot $ProjectRoot" in setup_script
    assert "--server.headless=true" in run_script
    assert "--server.showEmailPrompt=false" in run_script
    assert "--browser.gatherUsageStats=false" in run_script
    assert '"--broker-source", "IBKR_PAPER"' in run_script
    assert '$executionMode = Get-DotEnvValue "EXECUTION_MODE"' in run_script
    assert '"--execution-mode", $executionMode' in run_script
    assert "http://127.0.0.1:8501/_stcore/health" in run_script
    assert "Dashboard startup FAILED" in run_script
    assert "dashboard.err.log" in run_script
    assert "dashboard.out.log" in run_script
    assert "-Tail 40" in run_script
    assert "set \"EXIT_CODE=%ERRORLEVEL%\"" in start_script
    assert "if not \"%EXIT_CODE%\"==\"0\" (" in start_script
    assert "pause" in start_script
    assert "endlocal & exit /b %EXIT_CODE%" in start_script

    health_check = run_script.index("/_stcore/health")
    browser_open = run_script.index('Start-Process "http://127.0.0.1:8501"')
    failure_message = run_script.index("Dashboard startup FAILED")
    assert health_check < browser_open
    assert failure_message < browser_open


def test_startup_contract_keeps_ibkr_paper_observe_defaults():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "BROKER_SOURCE=IBKR_PAPER" in env_example
    assert "EXECUTION_MODE=OBSERVE" in env_example
