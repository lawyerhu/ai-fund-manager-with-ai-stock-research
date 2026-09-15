$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$streamlitBootstrap = Join-Path $PSScriptRoot "streamlit_bootstrap.ps1"
if (-not (Test-Path -LiteralPath $streamlitBootstrap)) {
    throw "Streamlit bootstrap helper is missing: $streamlitBootstrap"
}
. $streamlitBootstrap
Initialize-StreamlitHeadless -ProjectRoot $PSScriptRoot | Out-Null

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Write-Host "首次配置尚未完成，正在打开 setup wizard..."
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Setup did not finish. Double-click setup.bat after completing the displayed steps."
    }
}
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    throw "Setup is incomplete. Double-click setup.bat to continue."
}

$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$backendOutLog = Join-Path $logs "backend.out.log"
$backendErrLog = Join-Path $logs "backend.err.log"
$dashboardOutLog = Join-Path $logs "dashboard.out.log"
$dashboardErrLog = Join-Path $logs "dashboard.err.log"

function Show-LogTail([string]$Path) {
    Write-Host ("--- " + $Path + " (last 40 lines) ---")
    if (Test-Path -LiteralPath $Path) {
        Get-Content -LiteralPath $Path -Tail 40
    } else {
        Write-Host "Log file not found."
    }
}

function Stop-ProcessIfRunning($Process) {
    if ($null -ne $Process) {
        try {
            if (-not $Process.HasExited) {
                Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
            }
        } catch { }
    }
}

function Stop-ExistingAppInstances {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = [string]$_.CommandLine
        if ($cmd -match 'src\.main.*service' -or $cmd -match 'streamlit.*dashboard\.py') {
            try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
    Start-Sleep -Milliseconds 500
}

Stop-ExistingAppInstances

function Fail-DashboardStartup([string]$Reason, $DashboardProcess, $BackendProcess) {
    Write-Host "Dashboard startup FAILED" -ForegroundColor Red
    if ($Reason) { Write-Host $Reason -ForegroundColor Red }
    Show-LogTail $dashboardErrLog
    Show-LogTail $dashboardOutLog
    Stop-ProcessIfRunning $DashboardProcess
    Stop-ProcessIfRunning $BackendProcess
    exit 1
}

& $python -m src.worker_health --invalidate
if ($LASTEXITCODE -ne 0) {
    throw "Could not initialize command worker health state."
}
$executionMode = Get-DotEnvValue "EXECUTION_MODE"
if ($executionMode -notin @("PAPER", "OBSERVE")) {
    $executionMode = "PAPER"
}
$backendArguments = @(
    "-m", "src.main", "--service", "--mode", "llm",
    "--broker-source", "IBKR_PAPER",
    "--execution-mode", $executionMode
)
$backend = Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot -ArgumentList $backendArguments -RedirectStandardOutput $backendOutLog -RedirectStandardError $backendErrLog -PassThru

$workerReady = $false
$workerDeadline = [DateTime]::UtcNow.AddSeconds(90)
while ([DateTime]::UtcNow -lt $workerDeadline) {
    & $python -m src.worker_health
    if ($LASTEXITCODE -eq 0) {
        $workerReady = $true
        break
    }
    Start-Sleep -Milliseconds 500
}
if (-not $workerReady) {
    Write-Host "Backend command worker startup FAILED" -ForegroundColor Red
    Show-LogTail $backendErrLog
    Show-LogTail $backendOutLog
    Stop-ProcessIfRunning $backend
    exit 1
}

$env:STREAMLIT_SERVER_HEADLESS = "true"
$env:STREAMLIT_SERVER_SHOW_EMAIL_PROMPT = "false"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_ADDRESS = "127.0.0.1"
$env:STREAMLIT_SERVER_PORT = "8501"
$dashboardArguments = @(
    "-m", "streamlit", "run", "dashboard.py",
    "--server.headless=true",
    "--server.showEmailPrompt=false",
    "--browser.gatherUsageStats=false",
    "--server.address=127.0.0.1",
    "--server.port=8501"
)
try {
    $dashboard = Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot -ArgumentList $dashboardArguments -RedirectStandardOutput $dashboardOutLog -RedirectStandardError $dashboardErrLog -PassThru
} catch {
    Fail-DashboardStartup $_.Exception.Message $null $backend
}

$dashboardReady = $false
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            $dashboardReady = $true
            break
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}

if (-not $dashboardReady) {
    Fail-DashboardStartup "Dashboard did not answer the local health check within 30 seconds." $dashboard $backend
}

Start-Process "http://127.0.0.1:8501"
Write-Host "AI Fund Manager started. Backend PID: $($backend.Id); Dashboard PID: $($dashboard.Id)"
