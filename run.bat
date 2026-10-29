@echo off
title Smart Study Monitor - Phase 2
echo =====================================================================
echo           AI-Powered Smart Study Monitor - Phase 2
echo =====================================================================
echo.

set VENV_PYTHON=%~dp0venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at: %~dp0venv
    echo.
    echo Please create the Python 3.11 virtual environment first:
    echo    py -3.11 -m venv venv
    echo    .\venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [INFO] Found project virtual environment.
echo [INFO] Launching Smart Study Monitor (Phase 2)...
echo.

"%VENV_PYTHON%" "%~dp0main.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
