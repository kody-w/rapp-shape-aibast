@echo off
setlocal
REM RAPP Brainstem Installer for Windows CMD
REM Launches the PowerShell installer

echo.
echo   RAPP Brainstem Installer
echo   ========================
echo.
echo   Launching installer...
echo.

REM Tell install.ps1 it may exit with a real code on failure — under irm^|iex it
REM otherwise swallows errors and our ERRORLEVEL check below always reads 0.
set "BRAINSTEM_INSTALL_EXIT=1"
powershell -ExecutionPolicy Bypass -Command "& { irm https://raw.githubusercontent.com/microsoft/aibast-agents-library/main/install.ps1 | iex }"

if %ERRORLEVEL% neq 0 (
    echo.
    echo   Installation failed. Try running install.ps1 directly in PowerShell.
    echo.
    pause
    exit /b 1
)

echo.
echo   Installation complete!
echo   Open a new terminal and run: brainstem
echo.
pause
