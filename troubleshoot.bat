@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0troubleshoot.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Diagnostic report: %~dp0diagnostic_report.txt
pause
endlocal & exit /b %EXIT_CODE%
