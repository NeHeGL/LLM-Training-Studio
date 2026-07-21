@echo off
setlocal enabledelayedexpansion
title LLM Training Studio

:: Run from the project root so relative paths in server.py work correctly
cd /d "%~dp0"

set PYTHONPYCACHEPREFIX=%~dp0__pycache__

:: ── Pick the right Python (prefer .venv if present) ──────────
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

:: ── Check if server is already running on port 5001 ──────────
:: Use socket connect — works even if HTTP returns an error code.
:: If the port is open, just open the browser and close this window silently.
%PYTHON% -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',5001)); s.close()" >nul 2>&1
if not errorlevel 1 (
    start "" "http://localhost:5001"
    exit /b 0
)

:: ── Server not running — start it. Re-launch minimized so the console ────────
:: sits in the background rather than taking focus on the desktop.
if not defined LLM_MINIMIZED (
    set LLM_MINIMIZED=1
    start /min "LLM Training Studio" "%~f0"
    exit /b
)

:: ── Quick dependency check ───────────────────────────────────
echo.
echo  ============================================================
echo   LLM Training Studio - Web Browser Mode
echo  ============================================================
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo  [INFO] Virtual environment not found. Running installer...
    echo.
    set LLM_AUTO_INSTALL=1
    call "%~dp0install.bat"
    set LLM_AUTO_INSTALL=
    if errorlevel 1 (
        echo  [ERROR] Installation failed. Fix the errors above and try again.
        pause
        exit /b 1
    )
)

set PYTHON="%~dp0.venv\Scripts\python.exe"

:: ── Start the server ──────────────────────────────────────────
echo  [OK] Starting LLM Training Studio...
echo       URL: http://localhost:5001
echo.
echo  Press Ctrl+C to stop the server.
echo.

%PYTHON% train\server.py

:: If we get here, the server exited
echo.
echo  [INFO] Server stopped.
pause
