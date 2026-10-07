@echo off
title Stretch v2.0 - Launcher
cd /d "%~dp0"

echo ===================================================
echo   Memulai Aplikasi Stretch v2.0...
echo ===================================================

if exist "C:\python\python.exe" (
    "C:\python\python.exe" "stretch.py"
) else (
    python "stretch.py"
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Terjadi kendala saat menjalankan aplikasi.
    pause
)
