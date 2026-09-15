param(
    [string]$Python = "python"
)

Push-Location $PSScriptRoot
try {
    & $Python -m src.ibkr_diagnostics
    if ($LASTEXITCODE -ne 0) {
        Write-Error "OFFICIAL IBKR TWS API NOT INSTALLED OR INCOMPATIBLE"
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
