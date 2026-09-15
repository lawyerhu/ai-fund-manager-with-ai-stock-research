[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OfficialApiUrl = "https://www.interactivebrokers.com/en/trading/tws-api.php"
$TwsUrl = "https://www.interactivebrokers.com/en/trading/tws.php"
$StreamlitBootstrapPath = Join-Path $ProjectRoot "streamlit_bootstrap.ps1"

if (-not (Test-Path -LiteralPath $StreamlitBootstrapPath)) {
    throw "Streamlit bootstrap helper is missing: $StreamlitBootstrapPath"
}
. $StreamlitBootstrapPath

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ("== " + $Message + " ==") -ForegroundColor Cyan
}

function Find-CompatiblePython {
    $candidates = @()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { $candidates += [pscustomobject]@{ File = $pyLauncher.Source; Prefix = @("-3") } }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += [pscustomobject]@{ File = $pythonCommand.Source; Prefix = @() } }
    foreach ($candidate in $candidates) {
        try {
            $versionText = (& $candidate.File @($candidate.Prefix) --version 2>&1 | Out-String).Trim()
            if ($versionText -match "Python\s+(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if (($major -gt 3) -or ($major -eq 3 -and $minor -ge 11)) {
                    return $candidate
                }
            }
        } catch { }
    }
    return $null
}

function Try-InstallPython {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { return $false }
    Write-Host "No compatible Python found. Trying the Windows winget installer for Python 3.12."
    try {
        & $winget.Source install --id Python.Python.3.12 --exact --scope user --silent --accept-source-agreements --accept-package-agreements
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-DotEnvValue([string]$Name) {
    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envPath)) { return "" }
    foreach ($line in (Get-Content -LiteralPath $envPath -ErrorAction SilentlyContinue)) {
        if ($line -match ("^" + [regex]::Escape($Name) + "=(.*)$")) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

function Add-OfficialApiPath {
    $possibleRoots = @(
        $env:IBKR_TWS_API_PATH,
        (Join-Path $ProjectRoot "TWS API\source\pythonclient"),
        "C:\TWS API\source\pythonclient",
        "C:\Program Files\TWS API\source\pythonclient",
        "C:\Program Files (x86)\TWS API\source\pythonclient"
    )
    $systemPython = Find-CompatiblePython
    if ($systemPython) {
        try {
            $systemModuleRoot = (& $systemPython.File @($systemPython.Prefix) -c "import pathlib, ibapi; print(pathlib.Path(ibapi.__file__).resolve().parent.parent)" 2>$null | Select-Object -Last 1).Trim()
            if ($systemModuleRoot) { $possibleRoots += $systemModuleRoot }
        } catch { }
    }
    $candidate = $possibleRoots | Where-Object {
        $_ -and (Test-Path -LiteralPath (Join-Path $_ "ibapi"))
    } | Select-Object -First 1
    if (-not $candidate -or -not (Test-Path -LiteralPath $VenvPython)) { return $false }
    $sitePackages = (& $VenvPython -c 'import site; print(site.getsitepackages()[0])').Trim()
    if (-not $sitePackages) { return $false }
    Set-Content -LiteralPath (Join-Path $sitePackages "ibkr_official_api.pth") -Value $candidate -Encoding UTF8
    return $true
}

function Test-OfficialApi {
    if (-not (Test-Path -LiteralPath $VenvPython)) { return $false }
    & $VenvPython -m src.ibkr_diagnostics *> $null
    return $LASTEXITCODE -eq 0
}

function Find-TwsExecutable {
    $running = Get-Process -Name tws,ibgateway -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($running) { return [pscustomobject]@{ Path = $null; Running = $true } }
    $candidates = @(
        "C:\Jts\tws.exe",
        "C:\Jts\ibgateway\ibgateway.exe",
        (Join-Path ${env:ProgramFiles} "Trader Workstation\tws.exe"),
        (Join-Path ${env:ProgramFiles} "IB Gateway\ibgateway.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Trader Workstation\tws.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "IB Gateway\ibgateway.exe")
    )
    $registryRoots = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    foreach ($registryRoot in $registryRoots) {
        try {
            $entries = Get-ItemProperty -Path $registryRoot -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match "Trader Workstation|IB Gateway" }
            foreach ($entry in $entries) {
                if ($entry.InstallLocation) {
                    $candidates += (Join-Path $entry.InstallLocation "tws.exe")
                    $candidates += (Join-Path $entry.InstallLocation "ibgateway.exe")
                }
                if ($entry.DisplayIcon) {
                    $candidates += ($entry.DisplayIcon -replace ',\d+$', '')
                }
            }
        } catch { }
    }
    try {
        if (Test-Path -LiteralPath "C:\Jts") {
            $candidates += @(Get-ChildItem -LiteralPath "C:\Jts" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -in @("tws.exe", "ibgateway.exe") } | Select-Object -ExpandProperty FullName)
        }
    } catch { }
    $path = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if ($path) { return [pscustomobject]@{ Path = $path; Running = $false } }
    return $null
}

function Read-HiddenValue([string]$Prompt) {
    $secureValue = Read-Host $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

Write-Host "AI Fund Manager V1 First Run Setup" -ForegroundColor Green
Write-Host "Safety: IBKR PAPER + OBSERVE + Read-Only. LIVE and broker orders are disabled."

Write-Step "Configure Streamlit"
Initialize-StreamlitHeadless -ProjectRoot $ProjectRoot | Out-Null
Write-Host "Streamlit: PASS - headless local configuration ready"

Write-Step "Check Python"
$python = Find-CompatiblePython
if (-not $python) {
    $null = Try-InstallPython
    $python = Find-CompatiblePython
}
if (-not $python) {
    Write-Host "No compatible Python found. Opening the official download page." -ForegroundColor Red
    Start-Process "https://www.python.org/downloads/windows/"
    Write-Host "Install Python 3.11 or newer, then double-click setup.bat again."
    exit 1
}
Write-Host "Python: PASS"

Write-Step "Create Python environment"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $python.File @($python.Prefix) -m venv (Join-Path $ProjectRoot ".venv")
}
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
Write-Host "Dependencies: PASS"

Write-Step "Check official IBKR TWS API"
$apiReady = Test-OfficialApi
if (-not $apiReady) { $apiReady = Add-OfficialApiPath -and (Test-OfficialApi) }
while (-not $apiReady) {
    Write-Host "Official IBKR TWS API not found. Unknown ibapi packages from PyPI will not be installed." -ForegroundColor Yellow
    Start-Process $OfficialApiUrl
    Write-Host "Install the official IBKR TWS API that matches TWS, then press Enter to check again."
    Write-Host "If skipped, OBSERVE settings remain and the system stays in SAFE_MODE."
    $answer = Read-Host "Press Enter after installation, or type N to skip"
    if ($answer -match "^[Nn]") { break }
    $apiReady = Add-OfficialApiPath -and (Test-OfficialApi)
}
if ($apiReady) { Write-Host "IBKR API: PASS" } else { Write-Host "IBKR API: WARN - install the official API later" -ForegroundColor Yellow }

Write-Step "Check Trader Workstation"
$tws = Find-TwsExecutable
if (-not $tws) {
    Write-Host "Trader Workstation was not found. Opening the official IBKR download page." -ForegroundColor Yellow
    Start-Process $TwsUrl
    Write-Host "Install Trader Workstation, then double-click setup.bat again."
} elseif (-not $tws.Running) {
    try {
        Start-Process -FilePath $tws.Path
        Start-Sleep -Seconds 3
        Write-Host "Tried to start Trader Workstation."
    } catch {
        Write-Host "Please open Trader Workstation." -ForegroundColor Yellow
    }
} else {
    Write-Host "TWS: PASS - process is running"
}
Write-Host "Log in to the PAPER TRADING account in TWS. Do not select LIVE TRADING."
Write-Host "Keep Read-Only API = ON. In API Settings enable Enable ActiveX and Socket Clients."

Write-Step "Find safe Paper Socket Port"
$portText = (& $VenvPython -m src.setup_wizard --discover-port 2>$null | Select-Object -Last 1).Trim()
$port = 0
if ($portText -match "^\d+$") { $port = [int]$portText }
if ($port -eq 0) {
    $port = 7947
    $configuredPort = Get-DotEnvValue "IBKR_PORT"
    if ($configuredPort -match "^\d+$" -and ([int]$configuredPort -notin @(7496, 7946, 4001)) -and ([int]$configuredPort -ge 1) -and ([int]$configuredPort -le 65535)) {
        $port = [int]$configuredPort
    }
    Write-Host ("No logged-in TWS socket found. Keeping safe port " + $port + ".") -ForegroundColor Yellow
    Write-Host "If TWS uses another port, find it at TWS -> Global Configuration -> API -> Settings."
    $customPort = Read-Host ("Enter another Paper Socket Port if known; press Enter for " + $port)
    if ($customPort -match "^\d+$") {
        $candidatePort = [int]$customPort
        if ($candidatePort -in @(7496, 7946, 4001)) {
            Write-Host ("Live port rejected. Keeping safe port " + $port + ".") -ForegroundColor Red
        } elseif ($candidatePort -ge 1 -and $candidatePort -le 65535) {
            $port = $candidatePort
        } else {
            Write-Host ("Invalid port. Keeping safe port " + $port + ".") -ForegroundColor Yellow
        }
    }
} else {
    Write-Host ("Paper Socket Port: " + $port)
}

Write-Step "Save local safe configuration"
$baseUrl = Get-DotEnvValue "LLM_BASE_URL"
if (-not $baseUrl) {
    $baseUrl = Read-Host "Enter the CC Switch gateway URL (blank keeps AI disabled)"
}
$apiKey = Get-DotEnvValue "LLM_API_KEY"
if (-not $apiKey) {
    $apiKey = Read-HiddenValue "Enter the local CC Switch token (hidden; blank is allowed)"
}
$lunaModel = Get-DotEnvValue "LLM_LUNA_MODEL"
if (-not $lunaModel) { $lunaModel = "gpt-5.6-luna" }
$solModel = Get-DotEnvValue "LLM_SOL_MODEL"
if (-not $solModel) {
    Write-Host "Sol model: 1 = recommended GPT-5.6 Sol; 2 = custom model."
    $modelChoice = Read-Host "Choose (press Enter for 1)"
    if ($modelChoice -eq "2") {
        $solModel = Read-Host "Enter custom Sol model name"
    } else {
        $solModel = "gpt-5.6-sol"
    }
}
if (-not $solModel) { $solModel = "gpt-5.6-sol" }
$apiProtocol = Get-DotEnvValue "LLM_API_PROTOCOL"
if (-not $apiProtocol) { $apiProtocol = "RESPONSES" }
$protocolFallback = Get-DotEnvValue "LLM_PROTOCOL_FALLBACK"
if (-not $protocolFallback) { $protocolFallback = "NO" }
$executionMode = Get-DotEnvValue "EXECUTION_MODE"
if ($executionMode -ne "PAPER") { $executionMode = "OBSERVE" }
$envLines = @(
    "# Generated by AI Fund Manager setup wizard. Keep this file private.",
    "TRADING_MODE=IBKR_PAPER",
    "BROKER_SOURCE=IBKR_PAPER",
    ("EXECUTION_MODE=" + $executionMode),
    "FIRST_RUN_MODE=FIRST_RUN_OBSERVE",
    "IBKR_HOST=127.0.0.1",
    ("IBKR_PORT=" + $port),
    "IBKR_CLIENT_ID=41",
    "DATA_PROVIDER=yahoo",
    "LLM_PROVIDER=ccswitch",
    ("LLM_BASE_URL=" + $baseUrl),
    ("LLM_API_KEY=" + $apiKey),
    ("LLM_LUNA_MODEL=" + $lunaModel),
    ("LLM_SOL_MODEL=" + $solModel),
    "LLM_LUNA_REASONING_EFFORT=max",
    "LLM_SOL_REASONING_EFFORT=medium",
    "LLM_PIPELINE=LUNA_SOL",
    ("LLM_API_PROTOCOL=" + $apiProtocol),
    ("LLM_PROTOCOL_FALLBACK=" + $protocolFallback),
    "LLM_TIMEOUT_SECONDS=300",
    "DATABASE_PATH=data/ai_fund_manager.sqlite3"
)
$temporaryEnv = Join-Path $ProjectRoot ".env.setup.tmp"
Set-Content -LiteralPath $temporaryEnv -Value $envLines -Encoding UTF8
Move-Item -LiteralPath $temporaryEnv -Destination (Join-Path $ProjectRoot ".env") -Force
Write-Host "Configuration saved to local .env (the key is never displayed)."

Write-Step "Check CC Switch capabilities"
$env:LLM_BASE_URL = $baseUrl
$env:LLM_API_KEY = $apiKey
$env:LLM_LUNA_MODEL = $lunaModel
$env:LLM_SOL_MODEL = $solModel
$env:LLM_API_PROTOCOL = $apiProtocol
$env:LLM_PROTOCOL_FALLBACK = $protocolFallback
$gatewayCheck = & $VenvPython -c "import json, os; from src.setup_wizard import ccswitch_health_check; print(json.dumps(ccswitch_health_check(os.getenv('LLM_BASE_URL', ''), os.getenv('LLM_API_KEY', ''), luna_model=os.getenv('LLM_LUNA_MODEL', 'gpt-5.6-luna'), sol_model=os.getenv('LLM_SOL_MODEL', 'gpt-5.6-sol'), api_protocol=os.getenv('LLM_API_PROTOCOL'), protocol_fallback=os.getenv('LLM_PROTOCOL_FALLBACK'))))" 2>$null | Select-Object -Last 1
$gateway = $null
try {
    $gateway = $gatewayCheck | ConvertFrom-Json
    if ($gateway.endpoint -eq "PASS") {
        Write-Host "CC Switch Gateway: PASS" -ForegroundColor Green
        Write-Host ("Model Discovery: " + $gateway.model_discovery)
        Write-Host ("Luna Basic Call: " + $gateway.luna_basic_call + " | Luna Structured Output: " + $gateway.luna_structured_output)
        Write-Host ("Sol Basic Call: " + $gateway.sol_basic_call + " | Sol Structured JSON: " + $gateway.sol_structured_json)
        Write-Host ("Sol Tool Calling: " + $gateway.sol_tool_calling + " | Sol Decision: " + $gateway.sol_decision)
        Write-Host ("API Protocol: " + $gateway.api_protocol + " | Protocol Fallback: " + $gateway.protocol_fallback)
        Write-Host ("Reasoning Metadata: " + $gateway.reasoning_metadata + " | Token Usage: " + $gateway.token_usage)
        if (-not $gateway.ok) {
            Write-Host "LLM capability health is incomplete; automatic AI execution remains fail-closed." -ForegroundColor Yellow
        }
    } elseif ($gateway.endpoint -eq "WARN") {
        Write-Host "CC Switch Gateway: WARN - gateway URL is not configured" -ForegroundColor Yellow
    } else {
        Write-Host ("CC Switch Gateway: FAIL - " + $gateway.error) -ForegroundColor Red
    }
} catch {
    Write-Host "CC Switch Gateway: WARN - capability check did not return a valid result" -ForegroundColor Yellow
}

if ($null -ne $gateway -and $gateway.endpoint -eq "PASS" -and $gateway.model_discovery -eq "PASS") {
    $detectedLunaModel = [string]$gateway.luna_model
    $detectedSolModel = [string]$gateway.sol_model
    if ($detectedLunaModel -and $detectedSolModel) {
        $lunaModel = $detectedLunaModel
        $solModel = $detectedSolModel
        $env:LLM_LUNA_MODEL = $lunaModel
        $env:LLM_SOL_MODEL = $solModel
        $env:AI_FUND_PROJECT_ROOT = $ProjectRoot
        $env:AI_FUND_DETECTED_LUNA_MODEL = $lunaModel
        $env:AI_FUND_DETECTED_SOL_MODEL = $solModel
        $env:AI_FUND_DETECTED_API_PROTOCOL = [string]$gateway.api_protocol
        $env:AI_FUND_DETECTED_PROTOCOL_FALLBACK = [string]$gateway.protocol_fallback
        & $VenvPython -c "import os; from pathlib import Path; from src.setup_wizard import persist_detected_llm_config; persist_detected_llm_config(Path(os.environ['AI_FUND_PROJECT_ROOT']) / '.env', luna_model=os.environ['AI_FUND_DETECTED_LUNA_MODEL'], sol_model=os.environ['AI_FUND_DETECTED_SOL_MODEL'], api_protocol=os.environ['AI_FUND_DETECTED_API_PROTOCOL'], protocol_fallback=os.environ['AI_FUND_DETECTED_PROTOCOL_FALLBACK'])" *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "CC Switch model configuration could not be persisted. SAFE_MODE remains active."
        }
        Remove-Item Env:AI_FUND_PROJECT_ROOT -ErrorAction SilentlyContinue
        Remove-Item Env:AI_FUND_DETECTED_LUNA_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:AI_FUND_DETECTED_SOL_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:AI_FUND_DETECTED_API_PROTOCOL -ErrorAction SilentlyContinue
        Remove-Item Env:AI_FUND_DETECTED_PROTOCOL_FALLBACK -ErrorAction SilentlyContinue
        Write-Host "Detected CC Switch model IDs saved to .env (the key is never displayed)." -ForegroundColor Green
    }
}

Write-Step "First read-only integration check"
$diagnosticExit = 1
try {
    & $VenvPython -m src.first_run --first-run-observe --report diagnostic_report.txt
    $diagnosticExit = $LASTEXITCODE
} catch {
    Write-Host "First integration check is not ready. SAFE_MODE remains active." -ForegroundColor Yellow
}
if ($diagnosticExit -eq 0) {
    Write-Host "READY FOR OBSERVE" -ForegroundColor Green
    Write-Host "From now on, double-click start.bat. Start the first AI research from the Dashboard."
} else {
    Write-Host "SAFE_MODE / WAITING FOR TWS PAPER LOGIN" -ForegroundColor Yellow
    Write-Host "See diagnostic_report.txt. After TWS login or official API installation, run setup.bat again."
}
Write-Host "Diagnostic report: $(Join-Path $ProjectRoot 'diagnostic_report.txt')"
exit 0
