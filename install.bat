@echo off
setlocal enabledelayedexpansion
title LLM Training Studio - Installer

echo.
echo  ============================================================
echo   LLM Training Studio Installer - NeHe Productions
echo  ============================================================
echo.

:: Run from the project root
cd /d "%~dp0"

:: -- Check Python ---------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    echo          from https://www.python.org/downloads/
    echo          Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] %PYVER%

:: -- Create virtual environment --------------------------------
echo.
echo  [0/5] Setting up virtual environment...
if not exist "%~dp0.venv\Scripts\python.exe" (
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created at .venv\
) else (
    echo  [OK] Virtual environment already exists.
)

:: Use the venv Python from here on
set PYTHON="%~dp0.venv\Scripts\python.exe"
set PIP=%PYTHON% -m pip

:: -- Upgrade pip first ----------------------------------------
echo.
echo  [1/5] Upgrading pip...
%PIP% cache purge >nul 2>&1
%PIP% install --upgrade pip --quiet
if errorlevel 1 (
    echo  [WARN] pip upgrade failed - continuing anyway
)
echo  [OK] pip up to date

:: -- Pick the right PyTorch build for this GPU ----------------
:: tools\select_torch.ps1 looks at the card's compute capability and the
:: driver's CUDA version (see that file for the rules).
echo.
echo  [2/5] Detecting GPU / CUDA version...
set "GPU_NAME=" & set "CUDA_VER=" & set "GPU_TAG=" & set "TORCH_INDEX=" & set "TORCH_SPEC=" & set "ERROR="
for /f "usebackq tokens=1,* delims==" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\select_torch.ps1"`) do set "%%a=%%b"
if defined ERROR (
    echo.
    echo  [ERROR] !ERROR!
    pause
    exit /b 1
)
if "!GPU_TAG!"=="cpu" (
    echo  [INFO] No NVIDIA GPU detected - will install CPU-only PyTorch.
    echo         Training on a CPU is extremely slow; an NVIDIA GPU is strongly recommended.
) else (
    echo  [OK] GPU: !GPU_NAME!
    echo  [OK] Driver supports CUDA !CUDA_VER! - using PyTorch build !GPU_TAG!
)

:install_torch
echo.
echo  [3/5] Installing PyTorch (%GPU_TAG%)...
echo        Index URL: !TORCH_INDEX!
echo.
%PIP% install !TORCH_SPEC! --index-url !TORCH_INDEX!
if errorlevel 1 (
    echo.
    echo  [ERROR] PyTorch install failed.
    echo          Check your internet connection and try again.
    pause
    exit /b 1
)
echo.
echo  [OK] PyTorch installed
%PYTHON% "%~dp0tools\check_torch.py"
if errorlevel 1 (
    echo  [WARN] The GPU check above failed - training may not work until this is fixed.
)

:: -- Install requirements.txt ---------------------------------
echo.
echo  [4/5] Installing Python dependencies (requirements.txt)...
echo.
%PIP% install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo          See errors above - you may need to run as Administrator
    echo          or check your internet connection.
    pause
    exit /b 1
)

:: -- Download llama-quantize (needed for q4_k_m GGUF export) --
echo.
echo  [5/5] Installing llama-quantize (needed for q4_k_m GGUF export)...
echo.

set LLAMA_DIR=%~dp0tools\llama
set LLAMA_EXE=%LLAMA_DIR%\llama-quantize.exe

if exist "!LLAMA_EXE!" (
    echo  [OK] llama-quantize already installed at !LLAMA_EXE!
    goto :llama_done
)

if not exist "!LLAMA_DIR!" mkdir "!LLAMA_DIR!"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_llama_quantize.ps1"
if errorlevel 1 (
    echo.
    echo  [WARN] Could not auto-install llama-quantize.
    echo         q4_k_m export will fall back to f16 GGUF until it is installed.
    echo         To install manually, download llama.cpp from:
    echo         https://github.com/ggerganov/llama.cpp/releases
    echo         and place llama-quantize.exe in: !LLAMA_DIR!\
    echo.
) else (
    echo  [OK] llama-quantize installed - q4_k_m GGUF export is ready
)

:llama_done
echo.
echo  ============================================================
echo   Installation complete!
echo  ============================================================
echo.
echo   Virtual environment: .venv\
echo   To start LLM Training Studio, run:  start_web.bat
echo   Or desktop app:                     start_app.bat
echo.
echo   GPU:      !GPU_NAME!
echo   CUDA:     !CUDA_VER!  (!GPU_TAG!)
echo   PyTorch:  installed from !TORCH_INDEX!
echo.
if not defined LLM_AUTO_INSTALL pause
