[CmdletBinding()]

param()



function Get-DotEnvValue([string]$Name) {

    $rootVariable = Get-Variable -Name ProjectRoot -Scope 1 -ErrorAction SilentlyContinue

    $projectRoot = if ($rootVariable) { [string]$rootVariable.Value } else { $PSScriptRoot }

    $envPath = Join-Path $projectRoot ".env"

    if (-not (Test-Path -LiteralPath $envPath)) { return "" }

    foreach ($line in (Get-Content -LiteralPath $envPath -ErrorAction SilentlyContinue)) {

        if ($line -match ("^" + [regex]::Escape($Name) + "=(.*)$")) {

            return $Matches[1].Trim().Trim('"').Trim("'")

        }

    }

    return ""

}


function Initialize-StreamlitHeadless {

    [CmdletBinding()]

    param(

        [Parameter(Mandatory = $true)]

        [string]$ProjectRoot,

        [string]$UserHome = ""

    )



    $streamlitDirectory = Join-Path $ProjectRoot ".streamlit"

    New-Item -ItemType Directory -Force -Path $streamlitDirectory | Out-Null



    $configPath = Join-Path $streamlitDirectory "config.toml"

    if (-not (Test-Path -LiteralPath $configPath)) {

        $config = @'

[server]

headless = true

showEmailPrompt = false

address = "127.0.0.1"

port = 8501

enableWebsocketCompression = false

websocketPingInterval = 20



[browser]

gatherUsageStats = false



[global]

showWarningOnDirectExecution = false

'@

        Set-Content -LiteralPath $configPath -Value $config -Encoding UTF8

    }



    if (-not $UserHome) {

        $UserHome = $env:USERPROFILE

        if (-not $UserHome) {

            $UserHome = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)

        }

    }



    $credentialsPath = $null

    if ($UserHome) {

        $credentialsDirectory = Join-Path $UserHome ".streamlit"

        $credentialsPath = Join-Path $credentialsDirectory "credentials.toml"

        if (-not (Test-Path -LiteralPath $credentialsPath)) {

            New-Item -ItemType Directory -Force -Path $credentialsDirectory | Out-Null

            $credentials = @'

[general]
email = ""

'@

            Set-Content -LiteralPath $credentialsPath -Value $credentials -Encoding UTF8

        }

    }



    return [pscustomobject]@{

        ConfigPath = $configPath

        CredentialsPath = $credentialsPath

    }

}
