[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { (Get-Command python -ErrorAction SilentlyContinue).Source }
$Report = Join-Path $ProjectRoot "diagnostic_report.txt"

if ($Python) {
    & $Python -m src.first_run --troubleshoot --report diagnostic_report.txt
}
else {
    Set-Content -LiteralPath $Report -Encoding UTF8 -Value @(
        "AI Fund Manager diagnostic report",
        "Python: FAIL - Python was not found",
        "Venv: FAIL - .venv was not found",
        "TWS: UNKNOWN",
        "OpenAI API: UNKNOWN",
        "Broker Mutation: DISABLED",
        "SAFE_MODE: ON"
    )
}

$extra = @(
    "",
    "Windows checks",
    ("Venv: " + $(if (Test-Path -LiteralPath $VenvPython) { "PASS" } else { "FAIL" })),
    ("TWS process: " + $(if (Get-Process -Name tws,ibgateway -ErrorAction SilentlyContinue) { "PASS" } else { "WARN - TWS is not running" })),
    ("Config file: " + $(if (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env")) { "PASS" } else { "FAIL" })),
    ("OpenAI API Key: " + $(
        $envFile = Join-Path $ProjectRoot ".env"
        if ((Test-Path -LiteralPath $envFile) -and (Select-String -LiteralPath $envFile -Pattern "^OPENAI_API_KEY=.+" -Quiet)) { "CONFIGURED" } else { "MISSING" }
    )),
    ("Backend lock: " + $(if (Test-Path -LiteralPath (Join-Path $ProjectRoot "data\backend.lock")) { "PRESENT" } else { "NOT PRESENT" })),
    ("Database: " + $(if (Test-Path -LiteralPath (Join-Path $ProjectRoot "data\ai_fund_manager.sqlite3")) { "PRESENT" } else { "NOT CREATED" })),
    "Live ports checked: NONE (7496, 7946, 4001 are never probed)",
    "Broker Mutation: DISABLED",
    "Sensitive values are intentionally omitted from this report."
)
Add-Content -LiteralPath $Report -Value $extra -Encoding UTF8
Write-Host "Diagnostics complete: $Report"
