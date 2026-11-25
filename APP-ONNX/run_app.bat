@echo off
setlocal enabledelayedexpansion
title NutraVision AI - Food & Nutrition Platform
cd /d "%~dp0"

echo =================================================================
echo        NUTRAVISION AI - FOOD & NUTRITION PLATFORM
echo =================================================================
echo Initializing services (Vision API :8001, ML & Frontend :8000)...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_services.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =================================================================
    echo Service launcher exited with code %ERRORLEVEL%.
    echo =================================================================
    pause
)
