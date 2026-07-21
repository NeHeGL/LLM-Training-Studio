@echo off
setlocal enabledelayedexpansion
title LLM Training Studio

:: Run from the project root so relative paths in server.py work correctly
cd /d "%~dp0"

set PYTHONPYCACHEPREFIX=%~dp0__pycache__

:: ── Pick the right Python (prefer .venv if present) ──────────
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
    set PYTHONW="%~dp0.venv\Scripts\pythonw.exe"
) else (
    set PYTHON=python
    set PYTHONW=pythonw
)

:: ── Check if server is already running on port 5001 ──────────
:: Use socket connect — works even if HTTP returns an error code.
:: If the port is open, just open the desktop app and close this window silently.
%PYTHON% -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',5001)); s.close()" >nul 2>&1
if not errorlevel 1 (
    start "LLM Training Studio" %PYTHONW% desktop-view\launch_app.py
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
echo   LLM Training Studio - PyQt6 Desktop App
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
set PYTHONW="%~dp0.venv\Scripts\pythonw.exe"

:: ── Check if PyQt6 is installed ───────────────────────────────
%PYTHON% -c "import PyQt6.QtWebEngineWidgets" >nul 2>&1
if errorlevel 1 (
    echo  [WARN] PyQt6-WebEngine not found. Installing now...
    echo.
    %PYTHON% -m pip install PyQt6 PyQt6-WebEngine
    if errorlevel 1 (
        echo.
        echo  [ERROR] Installation failed. See errors above.
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Installation complete.
    echo.
) else (
    echo  [OK] PyQt6 already installed.
    echo.
)

:: ── Start the server + desktop app ───────────────────────────
echo  [OK] Starting LLM Training Studio...
echo       URL: http://localhost:5001
echo.
echo  Press Ctrl+C to stop the server.
echo.

:: Start the Qt desktop app in background (no console, pythonw)
:: It polls the server and shows the window once it's ready.
start "LLM Training Studio" %PYTHONW% desktop-view\launch_app.py

:: Run the server in the foreground — this console becomes the server log
%PYTHON% train\server.py --no-open

:: If we get here, the server exited
echo.
echo  [INFO] Server stopped.
pause
